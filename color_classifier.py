"""
背心颜色分类器 v2.6.4
使用远景更稳的多区域上半身 ROI，近景颜色抖动交给人脸识别模块处理
业务语义: 红底+黄条=监火员, 黄底=动火员
返回: {'color':'red'/'yellow'/None, 'red_ratio':, 'yellow_ratio':, 'score':}
"""
import cv2, numpy as np


def _make_weight_map(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dx = (xx - cx) / max(w / 2.0, 1)
    dy = (yy - cy) / max(h / 2.0, 1)
    dist = (dx / 0.75) ** 2 + (dy / 0.90) ** 2
    return (0.10 + 0.90 * np.exp(-1.8 * dist)).astype(np.float32)


def _row_coverage(mask, thres=0.15):
    m = (mask > 0).astype(np.float32)
    if m.size == 0:
        return 0.0
    row_density = m.mean(axis=1)
    return float((row_density > thres).mean())


def _grid_coverage(mask, gh=3, gw=3, thres=0.15):
    m = (mask > 0).astype(np.float32)
    h, w = m.shape[:2]
    if h < gh or w < gw:
        return 0.0
    hit = 0
    for i in range(gh):
        for j in range(gw):
            y1, y2 = int(h * i / gh), int(h * (i + 1) / gh)
            x1, x2 = int(w * j / gw), int(w * (j + 1) / gw)
            cell = m[y1:y2, x1:x2]
            if cell.size > 0 and cell.mean() > thres:
                hit += 1
    return hit / (gh * gw)


def _outer_density(mask):
    """背心外框(上下边+左右边)中颜色密度: 高→颜色是背心底色而非装饰"""
    m = (mask > 0).astype(np.float32)
    h, w = m.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    outer = np.zeros_like(m, dtype=bool)
    outer[:int(h * 0.25), :] = True
    outer[int(h * 0.75):, :] = True
    outer[:, :int(w * 0.22)] = True
    outer[:, int(w * 0.78):] = True
    return float(m[outer].mean()) if outer.any() else 0.0


def _score_roi(roi_bgr, wmap=None):
    """对一个ROI计算红/黄特征"""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    valid = hsv[:, :, 2] < 245

    red1 = cv2.inRange(hsv, np.array([0, 80, 70]), np.array([8, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([172, 80, 70]), np.array([180, 255, 255]))
    red_hsv = cv2.bitwise_or(red1, red2) & valid

    b, g, r = cv2.split(roi_bgr)
    red_bgr = (r.astype(np.float32) > g.astype(np.float32) * 1.25) & \
              (r.astype(np.float32) > b.astype(np.float32) * 1.4)

    red_mask = red_hsv & red_bgr
    # 宽松红色只用于判断红色底色是否存在, 不直接替代严格红色比例.
    red_lenient1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([12, 255, 255]))
    red_lenient2 = cv2.inRange(hsv, np.array([168, 40, 40]), np.array([180, 255, 255]))
    red_lenient_mask = cv2.bitwise_or(red_lenient1, red_lenient2) & valid
    yellow_mask = cv2.inRange(hsv, np.array([15, 45, 45]), np.array([55, 255, 255])) & valid

    if wmap is None:
        rh, rw = roi_bgr.shape[:2]
        wmap = _make_weight_map(rh, rw)

    total_weight = wmap.sum() + 1e-6
    red_ratio = float((red_mask.astype(np.float32) * wmap).sum() / total_weight)
    red_lenient_ratio = float((red_lenient_mask.astype(np.float32) * wmap).sum() / total_weight)
    yellow_ratio = float((yellow_mask.astype(np.float32) * wmap).sum() / total_weight)

    b, g, r = cv2.split(roi_bgr.astype(np.float32))
    rgb_sum = r + g + b + 1e-6
    non_yellow = valid & (yellow_mask == 0)
    if non_yellow.any():
        non_yellow_red_share = float(
            ((r / rgb_sum)[non_yellow] * wmap[non_yellow]).sum() /
            (wmap[non_yellow].sum() + 1e-6)
        )
    else:
        non_yellow_red_share = 0.0

    return {
        'red_ratio': red_ratio,
        'red_lenient_ratio': red_lenient_ratio,
        'non_yellow_red_share': non_yellow_red_share,
        'yellow_ratio': yellow_ratio,
        'yellow_row_cov': _row_coverage(yellow_mask, 0.15),
        'yellow_grid_cov': _grid_coverage(yellow_mask, 3, 3, 0.15),
        'red_row_cov': _row_coverage(red_mask, 0.10),
        'red_grid_cov': _grid_coverage(red_mask, 3, 3, 0.10),
        'red_outer_density': _outer_density(red_mask),
        'yellow_outer_density': _outer_density(yellow_mask),
        'mask_yellow': yellow_mask,
        'mask_red': red_mask,
    }


# v2.6.4 upper-body vest ROIs. Keep these for distant red/yellow vest recall.
ALL_ROIS = [
    (0.18, 0.65, 0.20, 0.80),
    (0.10, 0.50, 0.18, 0.82),
    (0.15, 0.60, 0.08, 0.55),
    (0.15, 0.60, 0.45, 0.92),
]


def classify(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 10.0:
        return None

    # 多ROI评分, 按底色可信度选最优(不再偏黄)
    best = None
    roi_scores = []
    for y1f, y2f, x1f, x2f in ALL_ROIS:
        y1, y2 = max(0, int(h * y1f)), min(h, int(h * y2f))
        x1, x2 = max(0, int(w * x1f)), min(w, int(w * x2f))
        if y2 <= y1 or x2 <= x1:
            continue
        roi = crop_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        wmap_local = _make_weight_map(roi.shape[0], roi.shape[1])
        s = _score_roi(roi, wmap_local)
        roi_scores.append(s)

        if best is None:
            best = s
            continue

        # 选黄色ratio更高的ROI
        if s['yellow_ratio'] > best['yellow_ratio'] * 1.2:
            best = s

    if best is None:
        return None

    red_ratio = best['red_ratio']
    red_lenient_ratio = best['red_lenient_ratio']
    non_yellow_red_share = best['non_yellow_red_share']
    yellow_ratio = best['yellow_ratio']
    red_outer_density = best['red_outer_density']

    # 黄色是底色: 覆盖多行多格
    yellow_is_base = (
        yellow_ratio > 0.10 and
        best['yellow_row_cov'] > 0.35 and
        best['yellow_grid_cov'] > 0.30
    )

    # 黄色是装饰: 只在少量行/格出现
    yellow_is_decoration = (
        yellow_ratio > 0.08 and
        (best['yellow_row_cov'] < 0.30 or best['yellow_grid_cov'] < 0.22)
    )

    # 红色底色
    red_is_base = (
        red_ratio > 0.08 and
        best['red_row_cov'] > 0.25 and
        best['red_grid_cov'] > 0.22
    )

    red_roi_base_score = 0.0
    red_roi_base_hits = 0
    strong_red_roi_base = False
    for s in roi_scores:
        red_roi_is_base = (
            s['red_ratio'] > 0.16 and
            s['red_row_cov'] > 0.28 and
            s['red_grid_cov'] >= 0.33 and
            s['yellow_row_cov'] < 0.35 and
            s['yellow_grid_cov'] <= 0.56 and
            (
                s['red_ratio'] > s['yellow_ratio'] * 1.05 or
                s['red_lenient_ratio'] > s['yellow_ratio'] * 1.30
            )
        )
        if red_roi_is_base:
            red_roi_base_hits += 1
            strong_red_roi_base = strong_red_roi_base or (
                s['red_ratio'] > 0.32 and
                s['red_ratio'] > s['yellow_ratio'] * 1.45 and
                s['yellow_row_cov'] < 0.30
            )
            red_roi_base_score = max(
                red_roi_base_score,
                s['red_ratio'] - s['yellow_ratio'] * 0.25,
            )
    red_roi_base = red_roi_base_hits >= 2 or strong_red_roi_base

    # 低分辨率下严格红色可能被过滤掉; 只在红色底色证据强于黄色主底时辅助判红.
    red_under_yellow_stripes = (
        yellow_ratio > 0.08 and
        red_lenient_ratio > 0.05 and
        red_lenient_ratio > yellow_ratio * 0.85 and
        non_yellow_red_share > 0.39
        and not yellow_is_base
        and (
            yellow_is_decoration or
            best['yellow_row_cov'] < 0.45 or
            best['yellow_grid_cov'] <= 0.35
        )
        and best['red_row_cov'] > 0.25
        and best['red_grid_cov'] > 0.22
    )

    # ---- 判色 ----
    # 1. 红底 + 黄装饰(反光条/文字) → 红色
    if red_is_base and yellow_is_decoration:
        color = 'red'
        score = red_ratio - yellow_ratio * 0.3

    # 2. 红底在低分辨率下被黄反光条压制 → 红色
    elif red_under_yellow_stripes or red_roi_base:
        color = 'red'
        score = max(red_roi_base_score, max(red_ratio, red_lenient_ratio) - yellow_ratio * 0.2)

    # 3. 真正黄底 → 黄色
    elif yellow_is_base and yellow_ratio > max(red_ratio, 0.05) * 0.50:
        color = 'yellow'
        score = yellow_ratio - red_ratio * 0.5

    # 4. 黄明显但非底色 → 红底黄条 → 红色
    elif yellow_ratio > 0.10 and yellow_is_decoration and red_ratio > 0.03:
        color = 'red'
        score = red_ratio - yellow_ratio * 0.2

    # 5. 红足够强 → 红色
    elif red_ratio > 0.10 and red_ratio > yellow_ratio * 1.20:
        color = 'red'
        score = red_ratio - yellow_ratio

    # 6. 黄足够 → 黄色
    elif yellow_ratio > 0.10 and yellow_ratio > red_ratio * 0.55:
        color = 'yellow'
        score = yellow_ratio - red_ratio * 0.5

    else:
        color = None
        score = 0.0

    return {
        'color': color,
        'red_ratio': round(red_ratio, 4),
        'red_lenient_ratio': round(red_lenient_ratio, 4),
        'non_yellow_red_share': round(non_yellow_red_share, 4),
        'yellow_ratio': round(yellow_ratio, 4),
        'score': round(score, 4),
        'red_row_cov': round(best['red_row_cov'], 4),
        'red_grid_cov': round(best['red_grid_cov'], 4),
        'yellow_row_cov': round(best['yellow_row_cov'], 4),
        'yellow_grid_cov': round(best['yellow_grid_cov'], 4),
    }


def classify_batch(crops_bgr):
    results = []
    for c in crops_bgr:
        r = classify(c)
        results.append(r['color'] if r else None)
    return results


def classify_batch_scored(crops_bgr):
    return [classify(c) for c in crops_bgr]
