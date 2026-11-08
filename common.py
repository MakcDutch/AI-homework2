"""Shared helpers for HW2 (VisDrone-VID detection): paths, dataset building, COCO evaluation, submission.

Every notebook imports this file so that all models use the same data split, the same
preprocessing and the same metric.
"""
import contextlib, glob, io, json, os, pickle, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
SEED = 42
MASK_VALUE = 114  # grey used by Ultralytics for letterbox padding


# ----------------------------------------------------------------------------- paths

def find_data_root():
    """Folder that holds train/, val/, test/ and test_images.csv (Kaggle input or DATA_ROOT env)."""
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"])
    for pattern in ["/kaggle/input/*/test_images.csv", "/kaggle/input/*/*/test_images.csv",
                    "/kaggle/input/*/*/*/test_images.csv", "./data/test_images.csv"]:
        hits = glob.glob(pattern)
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError("test_images.csv not found: attach the competition data or set DATA_ROOT")


def frame_key(path):
    """'clip/frame' key used everywhere to join images, predictions and COCO entries."""
    p = Path(str(path))
    return f"{p.parent.name}/{p.stem}"


def list_frames(root, split):
    """One row per frame: clip, frame number, image path and (if present) label / ignore paths."""
    rows = []
    for img in sorted(glob.glob(str(Path(root) / split / "images" / "*" / "*.jpg"))):
        p = Path(img)
        lbl = Path(root) / split / "labels" / p.parent.name / f"{p.stem}.txt"
        ign = Path(root) / split / "ignore" / p.parent.name / f"{p.stem}.txt"
        rows.append(dict(key=frame_key(p), clip=p.parent.name, frame=int(p.stem), image=img,
                         label=str(lbl) if lbl.exists() else None, ignore=str(ign) if ign.exists() else None))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- annotations

def read_txt(path):
    if not isinstance(path, str) or not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 0))
    return np.loadtxt(path, ndmin=2)


def load_coco_frames(root, split):
    """COCO annotations as two DataFrames: images and boxes (pixel units, iscrowd=1 means ignored region)."""
    with open(Path(root) / split / "annotations_coco.json") as fh:
        coco = json.load(fh)
    imgs = pd.DataFrame(coco["images"])
    imgs["key"] = imgs["file_name"].map(frame_key)
    anns = pd.DataFrame(coco["annotations"])
    anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index)
    anns = anns.merge(imgs[["id", "key", "width", "height"]].rename(columns={"id": "image_id"}), on="image_id")
    anns["clip"] = anns["key"].str.split("/").str[0]
    anns["frame"] = anns["key"].str.split("/").str[1].astype(int)
    anns["class"] = anns["category_id"].map(lambda c: CLASSES[c - 1])
    anns["size"] = np.sqrt(anns["w"] * anns["h"])
    return imgs, anns


def xywhn_to_xyxy(a, W, H):
    cx, cy, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def inside_fraction(boxes, regions):
    """Fraction of each box area covered by the union-approximation (max over regions)."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    ix1 = np.maximum(boxes[:, None, 0], regions[None, :, 0]); iy1 = np.maximum(boxes[:, None, 1], regions[None, :, 1])
    ix2 = np.minimum(boxes[:, None, 2], regions[None, :, 2]); iy2 = np.minimum(boxes[:, None, 3], regions[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (inter / np.maximum(area[:, None], 1e-9)).max(1)


# ----------------------------------------------------------------------------- dataset for Ultralytics

def mask_regions(img, regions):
    """Paints ignored regions grey so the model is not taught that unlabelled objects there are background."""
    img = img.copy()
    for x1, y1, x2, y2 in np.asarray(regions).round().astype(int):
        img[max(y1, 0):y2, max(x1, 0):x2] = MASK_VALUE
    return img


def _prepare_frame(row, out_img, out_lbl, mask_ignore):
    name = f"{row.clip}__{row.frame:07d}"
    dst_img, dst_lbl = Path(out_img) / f"{name}.jpg", Path(out_lbl) / f"{name}.txt"
    labels = read_txt(row.label)
    if mask_ignore and isinstance(row.ignore, str):
        img = cv2.imread(row.image)
        H, W = img.shape[:2]
        regions = xywhn_to_xyxy(read_txt(row.ignore), W, H)
        img = mask_regions(img, regions)
        cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if labels.size:  # drop boxes that are mostly painted over
            keep = inside_fraction(xywhn_to_xyxy(labels[:, 1:], W, H), regions) < 0.5
            labels = labels[keep]
    else:
        if not dst_img.exists():
            os.symlink(os.path.abspath(row.image), dst_img)
    with open(dst_lbl, "w") as fh:
        for c, *b in (labels if labels.size else []):
            fh.write(f"{int(c)} " + " ".join(f"{v:.6f}" for v in b) + "\n")


def build_yolo_dataset(root, out_dir, train_step=5, val_step=5, mask_ignore=True, workers=8):
    """Writes a YOLO dataset: every `train_step`-th train frame (ignored regions painted grey)
    and every `val_step`-th val frame (untouched, used only for Ultralytics' internal monitoring).
    Returns the path of the data.yaml."""
    out = Path(out_dir)
    if (out / "data.yaml").exists():
        print("dataset already built:", out)
        return str(out / "data.yaml")
    for split, step, mask in [("train", train_step, mask_ignore), ("val", val_step, False)]:
        frames = list_frames(root, split)
        frames = frames[(frames["frame"] - frames.groupby("clip")["frame"].transform("min")) % step == 0]
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda r: _prepare_frame(r, img_dir, lbl_dir, mask), frames.itertuples()))
        print(f"{split}: {len(frames)} frames from {frames['clip'].nunique()} clips (step {step})")
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                         f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    return str(yaml_path)


# ----------------------------------------------------------------------------- predictions

def ultralytics_predict(model, image_paths, imgsz, conf=0.001, iou=0.7, max_det=500, batch=16, **kw):
    """Runs an Ultralytics model and returns ({key: array[N, 6] = cls, score, x, y, w, h}, ms per image).
    Boxes are in original image pixels, (x, y) = top-left corner."""
    preds, t0 = {}, time.time()
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        for path, r in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                                verbose=False, **kw)):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy()
            preds[frame_key(path)] = np.column_stack([b.cls.cpu().numpy(), b.conf.cpu().numpy(), xyxy[:, 0], xyxy[:, 1],
                                                      xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
    return preds, 1000 * (time.time() - t0) / max(len(image_paths), 1)


def save_preds(preds, path):
    with open(path, "wb") as fh:
        pickle.dump(preds, fh)


def load_preds(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------- evaluation

def coco_evaluate(preds, root, split="val", max_dets=(1, 100, 500)):
    """COCO mAP on frames present in `preds`. Ignored regions are iscrowd=1 in the GT json, so
    pycocotools does not penalise detections inside them. Returns overall metrics and per-class AP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(Path(root) / split / "annotations_coco.json"))
    key2id = {frame_key(im["file_name"]): im["id"] for im in gt.dataset["images"]}
    dets = []
    for key, arr in preds.items():
        for c, s, x, y, w, h in arr:
            dets.append(dict(image_id=key2id[key], category_id=int(c) + 1, bbox=[float(x), float(y), float(w), float(h)],
                             score=float(s)))
    img_ids = [key2id[k] for k in preds]
    if not dets:
        return dict(mAP=0.0, AP50=0.0, AP75=0.0, AP_small=0.0, AP_medium=0.0, AP_large=0.0), {}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = img_ids
        ev.params.maxDets = list(max_dets)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    metrics = dict(mAP=s[0], AP50=s[1], AP75=s[2], AP_small=s[3], AP_medium=s[4], AP_large=s[5])
    prec = ev.eval["precision"][:, :, :, 0, -1]  # [IoU, recall, class] at area=all, maxDets=last
    per_class = {CLASSES[cid - 1]: float(np.mean(prec[:, :, k][prec[:, :, k] > -1])) if (prec[:, :, k] > -1).any() else float("nan")
                 for k, cid in enumerate(ev.params.catIds)}
    return {k: round(float(v), 4) for k, v in metrics.items()}, per_class


# ----------------------------------------------------------------------------- submission

def make_submission(preds, root, out_path, max_det=500):
    """Kaggle format: image_id, 'cls score x y w h ...' with (x, y) = top-left corner in pixels."""
    test = pd.read_csv(Path(root) / "test_images.csv")
    strings = []
    for fn in test["file_name"]:
        arr = preds.get(frame_key(fn), np.zeros((0, 6)))
        arr = arr[np.argsort(-arr[:, 1])][:max_det]
        strings.append(" ".join(f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" for c, s, x, y, w, h in arr))
    sub = pd.DataFrame({"image_id": test["image_id"], "PredictionString": strings})
    sub.to_csv(out_path, index=False)
    print(f"saved {out_path}: {len(sub)} rows, {sum(map(bool, strings))} with detections")
    return sub


def results_table(path="results.csv", row=None):
    """Appends one experiment row to a CSV log and returns the whole log."""
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if row is not None:
        df = pd.concat([df[df.get("exp", pd.Series(dtype=str)) != row["exp"]] if len(df) else df,
                        pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)
    return df


# ----------------------------------------------------------------------------- visualisation

COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def draw_frame(ax, root, split, key, gt=True, preds=None, score_thr=0.3, title=None):
    """Draws one frame: ground truth (thin, class colours), ignored regions (red hatch) and predictions (dashed white)."""
    import matplotlib.patches as patches
    clip, stem = key.split("/")
    img = cv2.cvtColor(cv2.imread(str(Path(root) / split / "images" / clip / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    ax.imshow(img)
    if gt and split != "test":
        lab = read_txt(str(Path(root) / split / "labels" / clip / f"{stem}.txt"))
        if lab.size:
            for c, (x1, y1, x2, y2) in zip(lab[:, 0].astype(int), xywhn_to_xyxy(lab[:, 1:], W, H)):
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1,
                                               edgecolor=np.array(COLORS[c]) / 255))
        ign = read_txt(str(Path(root) / split / "ignore" / clip / f"{stem}.txt"))
        if ign.size:
            for x1, y1, x2, y2 in xywhn_to_xyxy(ign, W, H):
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1, edgecolor="red", hatch="//"))
    if preds is not None and key in preds:
        for c, s, x, y, w, h in preds[key]:
            if s >= score_thr:
                ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, lw=1, ls="--", edgecolor="white"))
    ax.set_title(title or key, fontsize=9)
    ax.axis("off")
