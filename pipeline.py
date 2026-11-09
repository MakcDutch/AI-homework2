"""Inference strategies and post-processing for the final pipeline (06_pipeline).

- tiled_predict: full frame + overlapping tiles, merged with class-wise NMS (idea from SAHI, arXiv:2202.06934)
- temporal_rescore: raises the score of detections that are confirmed in neighbouring frames and lowers
  flickering ones (simplified Seq-NMS idea, arXiv:1602.08465)
- wbf_ensemble: Weighted Boxes Fusion of several models (arXiv:1910.13302, `ensemble-boxes` package)
All functions use the prediction format of common.py: {key: array[N, 6] = cls, score, x, y, w, h}.
"""
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

import common as C


def find_input(pattern):
    """First file under /kaggle/input (or ./inputs) matching a recursive glob, e.g. '**/val_preds_frcnn_base.pkl'."""
    for base in ["/kaggle/input", "inputs"]:
        hits = sorted(glob.glob(f"{base}/{pattern}", recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{pattern} not found: attach the notebook output that contains it (Add Input)")


def image_sizes(root, split):
    """{key: (W, H)} for every frame of a split."""
    if split == "test":
        t = pd.read_csv(Path(root) / "test_images.csv")
        return {C.frame_key(f): (w, h) for f, w, h in zip(t["file_name"], t["width"], t["height"])}
    imgs, _ = C.load_coco_frames(root, split)
    return dict(zip(imgs["key"], zip(imgs["width"], imgs["height"])))


def _nms_merge(arr, iou, max_det):
    if len(arr) == 0:
        return arr
    b = torch.as_tensor(np.c_[arr[:, 2], arr[:, 3], arr[:, 2] + arr[:, 4], arr[:, 3] + arr[:, 5]], dtype=torch.float32)
    keep = torchvision.ops.batched_nms(b, torch.as_tensor(arr[:, 1], dtype=torch.float32),
                                       torch.as_tensor(arr[:, 0], dtype=torch.int64), iou).numpy()
    return arr[keep[:max_det]]


def tiled_predict(model, image_paths, imgsz, grid=(2, 2), overlap=0.2, conf=0.001, merge_iou=0.6, max_det=500, **kw):
    """Runs the model on the full frame and on a grid of overlapping tiles (each tile is resized to `imgsz`, so small
    objects are seen ~1.8x larger), shifts tile boxes back to frame coordinates and merges everything with NMS."""
    import time
    preds, t0 = {}, time.time()
    gx, gy = grid
    for path in image_paths:
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        tw, th = int(W / (gx - (gx - 1) * overlap)), int(H / (gy - (gy - 1) * overlap))
        crops, offsets = [img], [(0, 0)]
        for i in range(gx):
            for j in range(gy):
                x0, y0 = min(int(i * tw * (1 - overlap)), W - tw), min(int(j * th * (1 - overlap)), H - th)
                crops.append(img[y0:y0 + th, x0:x0 + tw]); offsets.append((x0, y0))
        parts = []
        for r, (ox, oy) in zip(model.predict(crops, imgsz=imgsz, conf=conf, max_det=max_det, verbose=False, **kw), offsets):
            xyxy = r.boxes.xyxy.cpu().numpy()
            parts.append(np.column_stack([r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), xyxy[:, 0] + ox,
                                          xyxy[:, 1] + oy, xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]]))
        preds[C.frame_key(path)] = _nms_merge(np.concatenate(parts), merge_iou, max_det)
    return preds, 1000 * (time.time() - t0) / max(len(image_paths), 1)


def _xyxy(a):
    return torch.as_tensor(np.c_[a[:, 2], a[:, 3], a[:, 2] + a[:, 4], a[:, 3] + a[:, 5]], dtype=torch.float32)


def temporal_rescore(preds, alpha=0.4, iou_thr=0.5):
    """new score = (1 - alpha) * score + alpha * support, where support is the mean over the previous and next frame of
    the best score of a same-class detection overlapping the box with IoU >= iou_thr (0 if none). Frames of one clip
    must be consecutive (true for val and test)."""
    out = {}
    df = pd.DataFrame({"key": list(preds)})
    df["clip"] = df["key"].str.split("/").str[0]
    df["frame"] = df["key"].str.split("/").str[1].astype(int)
    for _, g in df.sort_values(["clip", "frame"]).groupby("clip"):
        keys, frames = g["key"].tolist(), g["frame"].tolist()
        for i, key in enumerate(keys):
            cur = preds[key]
            if len(cur) == 0:
                out[key] = cur; continue
            supports = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(keys) and abs(frames[j] - frames[i]) == 1:
                    nb = preds[keys[j]]
                    s = np.zeros(len(cur))
                    if len(nb):
                        iou = torchvision.ops.box_iou(_xyxy(cur), _xyxy(nb)).numpy()
                        iou[cur[:, 0][:, None] != nb[:, 0][None, :]] = 0  # same class only
                        match = iou >= iou_thr
                        s = np.where(match.any(1), (match * nb[:, 1][None, :]).max(1), 0.0)
                    supports.append(s)
            new = cur.copy()
            if supports:
                new[:, 1] = (1 - alpha) * cur[:, 1] + alpha * np.mean(supports, 0)
            out[key] = new
    return out


def wbf_ensemble(pred_list, weights, sizes, iou_thr=0.55, skip_box_thr=0.001, top_k=300, max_det=500):
    """Weighted Boxes Fusion of several prediction dicts (same keys). Boxes are normalised by the frame size."""
    from ensemble_boxes import weighted_boxes_fusion
    out = {}
    for key in pred_list[0]:
        W, H = sizes[key]
        boxes, scores, labels = [], [], []
        for p in pred_list:
            a = p.get(key, np.zeros((0, 6)))
            a = a[np.argsort(-a[:, 1])][:top_k]
            b = np.c_[a[:, 2] / W, a[:, 3] / H, (a[:, 2] + a[:, 4]) / W, (a[:, 3] + a[:, 5]) / H].clip(0, 1)
            ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])  # boxes lying outside the frame become empty after clipping
            a, b = a[ok], b[ok]
            boxes.append(b.tolist()); scores.append(a[:, 1].tolist()); labels.append(a[:, 0].tolist())
        b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        if len(b):
            b = b * [W, H, W, H]
            arr = np.c_[l, s, b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]
            out[key] = arr[np.argsort(-arr[:, 1])][:max_det]
        else:
            out[key] = np.zeros((0, 6))
    return out
