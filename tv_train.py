"""torchvision detectors for HW2: Faster R-CNN (two-stage) and RetinaNet (one-stage, anchor-based).

They train on exactly the frames built by common.build_yolo_dataset (same subsampling, same masking of
ignored regions) and are evaluated with common.coco_evaluate, so results are comparable with YOLO.
Reference: torchvision detection tutorial and reference scripts (github.com/pytorch/vision/tree/main/references/detection).
"""
import math, random, time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.anchor_utils import AnchorGenerator

import common as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data

def resize_long_side(img, imgsz):
    H, W = img.shape[:2]
    s = imgsz / max(H, W)
    if s != 1:
        img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return img, s


class FrameDataset(Dataset):
    """Frames resized so that the long side is `imgsz` (same as YOLO's letterbox), uint8 CHW tensors.
    Labels are read from the YOLO folder layout; torchvision class ids are 1..8 (0 = background)."""

    def __init__(self, image_paths, imgsz, train=False, with_labels=True):
        self.paths = [str(p) for p in image_paths]
        self.imgsz, self.train, self.with_labels = imgsz, train, with_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        img, s = resize_long_side(img, self.imgsz)
        target = {"key": C.frame_key(path), "scale": s}
        if self.with_labels:
            lab = C.read_txt(path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
            boxes = C.xywhn_to_xyxy(lab[:, 1:], W, H) * s if lab.size else np.zeros((0, 4))
            cls = lab[:, 0].astype(int) + 1 if lab.size else np.zeros(0, int)
            h, w = img.shape[:2]
            boxes = np.clip(boxes, 0, [w, h, w, h])
            keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
            boxes, cls = boxes[keep], cls[keep]
            if self.train and torch.rand(1).item() < 0.5:  # horizontal flip, the only augmentation
                img = img[:, ::-1]
                boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1) if len(boxes) else boxes
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.as_tensor(cls, dtype=torch.int64)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), target


def collate(batch):
    return tuple(zip(*batch))


def seed_everything(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    np.random.seed(C.SEED + worker_id); random.seed(C.SEED + worker_id)


# ----------------------------------------------------------------------------- models

_ORIGINAL_FASTRCNN_LOSS = tv_roi_heads.fastrcnn_loss


def giou_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """Same as torchvision's fastrcnn_loss but the box term is GIoU loss instead of smooth L1.
    Predicted and target deltas are both decoded relative to one unit reference box; IoU and GIoU do not change
    under shifting and per-axis scaling, so the result equals GIoU between the real boxes (arXiv:1902.09630)."""
    labels = torch.cat(labels, 0)
    regression_targets = torch.cat(regression_targets, 0)
    cls_loss = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    n = class_logits.shape[0]
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    coder = BoxCoder((10.0, 10.0, 5.0, 5.0))
    ref = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=labels.device).expand(len(pos), 4)
    pred = coder.decode_single(box_regression[pos, labels[pos]].float(), ref)
    tgt = coder.decode_single(regression_targets[pos].float(), ref)
    box_loss = torchvision.ops.generalized_box_iou_loss(pred, tgt, reduction="sum") / labels.numel()
    return cls_loss, box_loss


def build_faster_rcnn(imgsz, num_classes=len(C.CLASSES) + 1, box_loss="smooth_l1", **kw):
    """Faster R-CNN ResNet-50 FPN v2 with COCO weights; only the class/box predictor is replaced.
    kw go to torchvision's FasterRCNN (e.g. rpn_post_nms_top_n_test, box_batch_size_per_image)."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    tv_roi_heads.fastrcnn_loss = giou_fastrcnn_loss if box_loss == "giou" else _ORIGINAL_FASTRCNN_LOSS
    args = dict(min_size=imgsz, max_size=imgsz, box_detections_per_img=500, box_score_thresh=0.001)
    args.update(kw)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **args)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def anchor_generator(base_sizes, ratios=(0.5, 1.0, 2.0)):
    """RetinaNet-style anchors: per FPN level, 3 scales (x1, x2^(1/3), x2^(2/3)) of the base size times 3 ratios."""
    sizes = tuple(tuple(round(s * 2 ** (k / 3)) for k in range(3)) for s in base_sizes)
    return AnchorGenerator(sizes, (tuple(ratios),) * len(base_sizes))


def build_retinanet(imgsz, num_classes=len(C.CLASSES) + 1, anchor_sizes=None, **kw):
    """RetinaNet ResNet-50 FPN v2 (P3-P7, focal loss, GIoU box loss) with COCO weights; the classification head is
    replaced. `anchor_sizes` = base anchor size per level (default 32, 64, 128, 256, 512)."""
    from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    args = dict(min_size=imgsz, max_size=imgsz, detections_per_img=500, score_thresh=0.001, topk_candidates=1000)
    args.update(kw)
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, **args)
    if anchor_sizes is not None:  # same number of anchors per location, so the pretrained heads still fit
        model.anchor_generator = anchor_generator(anchor_sizes)
    n_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head.classification_head = RetinaNetClassificationHead(256, n_anchors, num_classes,
                                                                 norm_layer=partial(nn.GroupNorm, 32))
    return model


# ----------------------------------------------------------------------------- training

def train_model(model, train_paths, imgsz, epochs, batch, lr, out_dir, val_paths=None, root=None,
                workers=4, log_every=200):
    """SGD + momentum, linear warm-up then cosine decay, mixed precision. After every epoch the model is
    evaluated on `val_paths` (a val subset) to get a learning curve. Returns the per-epoch history."""
    seed_everything()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.to(DEVICE)
    g = torch.Generator(); g.manual_seed(C.SEED)
    loader = DataLoader(FrameDataset(train_paths, imgsz, train=True), batch_size=batch, shuffle=True,
                        num_workers=workers, collate_fn=collate, worker_init_fn=_worker_init, generator=g,
                        pin_memory=DEVICE.type == "cuda", drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total, warmup = epochs * len(loader), min(500, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * it / total)))
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, it = [], 0
    for epoch in range(1, epochs + 1):
        model.train(); t0 = time.time(); running = {}
        for imgs, targets in loader:
            imgs = [im.to(DEVICE, non_blocking=True).float().div_(255) for im in imgs]
            targets = [{"boxes": t["boxes"].to(DEVICE), "labels": t["labels"].to(DEVICE)} for t in targets]
            with torch.autocast(DEVICE.type, enabled=use_amp):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                print("non-finite loss, batch skipped"); sched.step(); it += 1; continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step(); it += 1
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            if it % log_every == 0:
                done = it - (epoch - 1) * len(loader)
                eta = (time.time() - t0) / done * (len(loader) - done) / 60
                print(f"  epoch {epoch} it {done}/{len(loader)} loss {loss.item():.3f} | ~{eta:.1f} min left in epoch")
        row = {"epoch": epoch, "minutes": round((time.time() - t0) / 60, 1),
               **{k: round(v / len(loader), 4) for k, v in running.items()}}
        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_paths is not None:
            preds, _ = predict(model, val_paths, imgsz, workers=workers)
            row.update({f"val_{k}": v for k, v in C.coco_evaluate(preds, root, "val")[0].items()})
        history.append(row)
        print(row)
    return history


@torch.no_grad()
def predict(model, image_paths, imgsz, batch=4, workers=4):
    """Returns ({key: array[N, 6] = cls(0-based), score, x, y, w, h in original pixels}, ms per image)."""
    model.to(DEVICE).eval()
    loader = DataLoader(FrameDataset(image_paths, imgsz, with_labels=False), batch_size=batch,
                        num_workers=workers, collate_fn=collate)
    preds, t0 = {}, time.time()
    for imgs, metas in loader:
        imgs = [im.to(DEVICE).float().div_(255) for im in imgs]
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            outs = model(imgs)
        for out, meta in zip(outs, metas):
            b = out["boxes"].float().cpu().numpy() / meta["scale"]
            preds[meta["key"]] = np.column_stack([out["labels"].cpu().numpy() - 1, out["scores"].float().cpu().numpy(),
                                                  b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return preds, 1000 * (time.time() - t0) / max(len(image_paths), 1)


# ----------------------------------------------------------------------------- analysis

@torch.no_grad()
def rpn_vs_final_recall(model, root, image_paths, imgsz, score_thr=0.3, iou_thr=0.5):
    """Recall of the RPN proposals vs recall of the final detections, by object size.
    Shows what each stage of the two-stage detector contributes."""
    model.to(DEVICE).eval()
    ds = FrameDataset(image_paths, imgsz, with_labels=False)
    _, anns = C.load_coco_frames(root, "val")
    anns = anns[anns["iscrowd"] == 0]
    by_key = {k: g for k, g in anns.groupby("key")}
    rows = []
    for i in range(len(ds)):
        img, meta = ds[i]
        g = by_key.get(meta["key"])
        if g is None:
            continue
        gt = torch.tensor(np.c_[g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]] * meta["scale"], dtype=torch.float32)
        x = [img.to(DEVICE).float() / 255]
        images, _ = model.transform(x)
        feats = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, feats)
        det = model(x)[0]
        r = images.image_sizes[0][0] / img.shape[1]  # transform may rescale slightly; bring boxes back to `img` scale
        prop = proposals[0].cpu() / r
        final = det["boxes"][det["scores"] >= score_thr].cpu()
        hit_p = torchvision.ops.box_iou(gt, prop).max(1).values >= iou_thr if len(prop) else torch.zeros(len(gt), dtype=torch.bool)
        hit_f = torchvision.ops.box_iou(gt, final).max(1).values >= iou_thr if len(final) else torch.zeros(len(gt), dtype=torch.bool)
        rows.append(np.c_[g["w"] * g["h"], hit_p.numpy(), hit_f.numpy()])
    a = np.concatenate(rows)
    import pandas as pd
    df = pd.DataFrame(a, columns=["area", "rpn_hit", "final_hit"])
    df["size"] = pd.cut(df["area"], [0, 32**2, 96**2, np.inf], labels=["small", "medium", "large"], right=False)
    out = df.groupby("size", observed=False)[["rpn_hit", "final_hit"]].mean().mul(100).round(1)
    out.loc["all"] = df[["rpn_hit", "final_hit"]].mean().mul(100).round(1)
    out.columns = [f"RPN recall@{iou_thr} (top {model.rpn._post_nms_top_n['testing']}), %",
                   f"final recall@{iou_thr} (score>={score_thr}), %"]
    return out
