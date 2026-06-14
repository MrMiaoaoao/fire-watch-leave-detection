"""
监火员离岗检测系统 v2.0 (2026-06-11)
YOLOv8n + HSV颜色面积占比 + ByteTrack追踪 + 离岗监控
用法: python full_pipeline.py
"""

import json, time, sys, os
from pathlib import Path
from collections import deque
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from color_classifier import classify_batch_scored
from leave_detector import LeaveDetector

# ===== 配置 =====
if len(sys.argv) > 1:
    VIDEO = sys.argv[1]
else:
    VIDEO = str(ROOT / "data" / "监火员离岗测试2.mp4")
YOLO_WEIGHTS = str(ROOT / "models" / "yolov8s.pt")
OUTPUT = ROOT / "outputs"
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0") or "0")
LEAVE_TIMEOUT = 10.0
CONF_THRES = 0.45
MIN_BOX_W = 50
MIN_BOX_H = 100

DETECT_CLASSES = ["person"]

# ByteTrack追踪 + 颜色滑动投票
known = {}  # {tid: [x1,y1,x2,y2, last_frame, color_history(deque), locked_color, last_final, stable_count]}
VOTE_WINDOW = 12      # 滑动窗口帧数
VOTE_THRES = 7         # 确认票数阈值
LOCK_CONSEC = 8        # 连续N帧final_color同色 → 锁定
UNLOCK_CONSEC = 15     # 锁定后连续N帧相反色+分数优势 → 解锁
SWITCH_MARGIN = 0.08   # 解锁需相反色score高出此边际

# ===== 中文标签 =====
LABEL_CACHE = {}

def get_label(text, color):
    k = (text, color)
    if k not in LABEL_CACHE:
        for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttf"]:
            try:
                font = ImageFont.truetype(fp, 18)
                break
            except Exception:
                font = ImageFont.load_default()
        d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bb = d.textbbox((0, 0), text, font=font)
        w, h = bb[2] - bb[0] + 4, bb[3] - bb[1] + 4
        img = Image.new("RGB", (w, h), (0, 0, 0))
        ImageDraw.Draw(img).text((2, 0), text, font=font, fill=color)
        LABEL_CACHE[k] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return LABEL_CACHE[k]


def paste_label(frame, text, x, y, color=(0, 0, 255)):
    lbl = get_label(text, color)
    lh, lw = lbl.shape[:2]
    y = y + 20 if y - lh < 0 else y
    x1, y1 = max(0, x), max(0, y - lh)
    x2, y2 = min(frame.shape[1], x + lw), min(frame.shape[0], y1 + lh)
    roi = frame[y1:y2, x1:x2]
    lc = lbl[0:roi.shape[0], 0:roi.shape[1]]
    if roi.shape == lc.shape and roi.size > 0:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, lc, 0.7, 0)


# ===== 标准IoU+包含率去重 =====
def box_iou_contain(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou = inter / (area_a + area_b - inter + 1e-6)
    contain = inter / (min(area_a, area_b) + 1e-6)
    return iou, contain


# ===== 重叠框检查 (debug) =====
def find_overlaps(items, box_key="bbox", iou_thres=0.15, contain_thres=0.40):
    overlaps = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            if isinstance(items[i], dict):
                bi, bj = items[i][box_key], items[j][box_key]
                ii, ij = items[i].get("track_id"), items[j].get("track_id")
                ci, cj = items[i].get("color"), items[j].get("color")
            else:
                bi, bj = items[i][:4], items[j][:4]
                ii, ij = items[i][5], items[j][5]
                ci, cj = None, None
            iou, contain = box_iou_contain(bi, bj)
            if iou > iou_thres or contain > contain_thres:
                overlaps.append({
                    "i": i, "j": j,
                    "tid_i": int(ii) if ii is not None else None,
                    "tid_j": int(ij) if ij is not None else None,
                    "col_i": ci, "col_j": cj,
                    "iou": round(float(iou), 4), "contain": round(float(contain), 4),
                    "box_i": [int(x) for x in bi], "box_j": [int(x) for x in bj],
                })
    return overlaps


# ===== 颜色滑动投票 + 角色锁定 =====
def vote_color(hist):
    """纯投票, 不处理锁定逻辑"""
    red_votes = hist.count('red')
    yellow_votes = hist.count('yellow')
    valid = red_votes + yellow_votes
    if valid < 4:
        return None
    if red_votes >= VOTE_THRES:
        return 'red'
    if yellow_votes >= VOTE_THRES:
        return 'yellow'
    return None


# ===== 防御性检查 =====
import torch
if not os.path.exists(YOLO_WEIGHTS):
    raise FileNotFoundError(f"模型权重不存在: {YOLO_WEIGHTS}")
if not os.path.exists(VIDEO) and not str(VIDEO).isdigit():
    raise FileNotFoundError(f"视频文件不存在: {VIDEO}")
if not torch.cuda.is_available():
    print("警告: CUDA不可用, 将使用CPU推理(速度较慢)")
try:
    from ultralytics.trackers import register_tracker
except ImportError:
    raise ImportError("ByteTrack依赖缺失, 请运行: pip install lap")

print("加载 YOLOv8s + ByteTrack...")
model = YOLO(YOLO_WEIGHTS)
# YOLOv8n: classes=[0] 过滤 person 类

OUTPUT.mkdir(exist_ok=True)
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total == 0:
    cap.release()
    raise RuntimeError(f"视频读取失败或帧数为0: {VIDEO}")

detector = LeaveDetector(timeout=LEAVE_TIMEOUT, fps=fps, warmup=5.0)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_name = Path(VIDEO).stem
out_stem = f"{vid_name}{OUTPUT_SUFFIX}"
vout = cv2.VideoWriter(
    str(OUTPUT / f"{out_stem}_output.mp4"),
    cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h),
)
jsonl_path = OUTPUT / f"{out_stem}_result.jsonl"
jsonl_file = open(jsonl_path, "w", encoding="utf-8")

print(f"视频: {total}帧, {fps}fps, {total/fps:.0f}s")
t0 = time.time()
idx = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MAX_FRAMES > 0 and idx >= MAX_FRAMES:
            break
        ts = idx / fps

        results = model.track(
            frame, conf=CONF_THRES, iou=0.35, classes=[0],
            verbose=False, persist=True, tracker="bytetrack.yaml",
        )[0]
        boxes = results.boxes
        red_count = yellow_count = 0
        detections = []
        raw_pc = dedup_pc = 0
        raw_ov = dedup_ov = final_ov = []

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            tids = boxes.id.cpu().numpy().astype(int)
            person_data = []
            for (x1, y1, x2, y2), cf, tid in zip(xyxy, confs, tids):
                if tid < 0:
                    continue
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                # 边界裁剪 (修改2)
                ix1 = max(0, min(ix1, w - 1))
                iy1 = max(0, min(iy1, h - 1))
                ix2 = max(0, min(ix2, w))
                iy2 = max(0, min(iy2, h))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) >= MIN_BOX_W and (iy2 - iy1) >= MIN_BOX_H:
                    person_data.append((ix1, iy1, ix2, iy2, cf, tid))

            # debug: 记录原始检测是否有重叠
            raw_pc = len(person_data)
            raw_ov = find_overlaps(person_data, iou_thres=0.15, contain_thres=0.40)

            # 标准IoU+包含率去重 (修改1)
            if len(person_data) > 1:
                person_data.sort(key=lambda x: x[4], reverse=True)
                keep = []
                for p in person_data:
                    dup = False
                    for k in keep:
                        iou, contain = box_iou_contain(p[:4], k[:4])
                        if iou > 0.20 or contain > 0.45:
                            dup = True
                            break
                    if not dup:
                        keep.append(p)
                person_data = keep
            dedup_pc = len(person_data)
            dedup_ov = find_overlaps(person_data, iou_thres=0.15, contain_thres=0.40)

            if person_data:
                crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2, _, _ in person_data]
                scored = classify_batch_scored(crops)
                raw_dets = []  # 先收集原始结果(含分数)
                for (ix1, iy1, ix2, iy2, cf, tid), s in zip(person_data, scored):
                    if s is None:
                        color = None
                    else:
                        color = s['color']
                    if tid not in known:
                        # [x1,y1,x2,y2, last_frame, color_history, locked_color, last_final, stable_count, unlock_count]
                        known[tid] = [ix1, iy1, ix2, iy2, idx, deque(maxlen=VOTE_WINDOW), None, None, 0, 0]
                    else:
                        t = known[tid]
                        t[0:4] = [ix1, iy1, ix2, iy2]
                        t[4] = idx
                    known[tid][5].append(color)
                    hist = list(known[tid][5])

                    # 投票得出本帧颜色
                    final_c = vote_color(hist)

                    # ---- 稳定计数 (基于 final_color 非 raw_color) ----
                    if final_c is not None:
                        if final_c == known[tid][7]:  # 与上一帧 final 相同
                            known[tid][8] += 1
                        else:
                            known[tid][8] = 1
                        known[tid][7] = final_c

                    # ---- 锁定 ----
                    locked = known[tid][6]
                    if locked is None and known[tid][8] >= LOCK_CONSEC and final_c is not None:
                        known[tid][6] = final_c
                        known[tid][9] = 0  # 重置解锁计数

                    # ---- 解锁: 连续UNLOCK_CONSEC帧相反色+分数优势 ----
                    if locked is not None and final_c is not None:
                        sc = s['score'] if s else 0
                        if final_c != locked:
                            # 用原始分类中相反色的累积分数来判断
                            opposite_raw = sum(1 for c in hist if c == final_c)
                            same_raw = sum(1 for c in hist if c == locked)
                            if opposite_raw > same_raw and sc > SWITCH_MARGIN:
                                known[tid][9] += 1
                            else:
                                known[tid][9] = 0
                            if known[tid][9] >= UNLOCK_CONSEC:
                                known[tid][6] = None  # 解锁
                                known[tid][9] = 0
                                known[tid][8] = 1      # 用当前色重计
                        else:
                            known[tid][9] = 0  # 回归锁定色, 重置

                    # 已锁定 → 用锁定色覆盖 final_c
                    if known[tid][6] is not None:
                        final_c = known[tid][6]
                    if final_c is not None:
                        sc = s['score'] if s else 0
                        raw_dets.append({
                            "bbox": [ix1, iy1, ix2, iy2], "color": final_c,
                            "track_id": int(tid), "conf": round(float(cf), 2),
                            "score": sc, "red_ratio": s['red_ratio'] if s else 0,
                            "yellow_ratio": s['yellow_ratio'] if s else 0,
                        })

                # 后处理去重合并: 保留置信度高、框更完整的
                if len(raw_dets) > 1:
                    raw_dets.sort(key=lambda x: (x['conf'], (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1])), reverse=True)
                    merged = []
                    for d in raw_dets:
                        dup = False
                        for m in merged:
                            iou, contain = box_iou_contain(d['bbox'], m['bbox'])
                            if iou > 0.15 or contain > 0.40:
                                dup = True  # 同一人,跳过
                                break
                        if not dup:
                            merged.append(d)
                    raw_dets = merged

                final_ov = find_overlaps(raw_dets, box_key="bbox", iou_thres=0.15, contain_thres=0.40)
                if final_ov:
                    print(f"  [重叠框警告] frame={idx}, time={ts:.2f}s, n={len(final_ov)}")

                for d in raw_dets:
                    if d['color'] == 'red':
                        red_count += 1
                    elif d['color'] == 'yellow':
                        yellow_count += 1
                    detections.append(d)

        # 清理失联track
        gone = [tid for tid, t in known.items() if t[4] < idx - 60]
        for tid in gone:
            del known[tid]

        # 离岗检测
        alert = detector.update(red_count)
        alerts = [alert] if alert else []
        if alert:
            print(f"  [{alert['time_s']}s] {alert['content']}")

        # 画框
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            if d["color"] == 'red':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                paste_label(frame, f"监火员 {d['conf']:.2f}", x1, y1 - 5, (0, 0, 255))
            elif d["color"] == 'yellow':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                paste_label(frame, f"动火员 {d['conf']:.2f}", x1, y1 - 5, (0, 255, 255))

        paste_label(frame, f"Frame:{idx} Time:{ts:.1f}s 监火员:{red_count} 动火员:{yellow_count}", 10, 30, (255, 255, 255))
        for i, a in enumerate(alerts):
            paste_label(frame, a["content"], 10, 60 + i * 25, (0, 0, 255))

        frame_data = {
            "frame": idx, "timestamp": round(ts, 2),
            "supervisor_count": red_count, "worker_count": yellow_count,
            "alerts": alerts,
            "raw_pc": raw_pc, "dedup_pc": dedup_pc,
            "raw_ov": raw_ov, "dedup_ov": dedup_ov,
            "final_ov": final_ov,
            "detections": detections,
        }
        jsonl_file.write(json.dumps(frame_data, ensure_ascii=False) + "\n")
        jsonl_file.flush()

        vout.write(frame)
        idx += 1
        if idx % 500 == 0:
            print(f"  {idx}/{total} ({100*idx/total:.0f}%) {idx/(time.time()-t0):.1f}fps")

except KeyboardInterrupt:
    print("\n用户中断")
except Exception as e:
    print(f"\n运行错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    vout.release()

elapsed = time.time() - t0
jsonl_file.close()

leave_report = detector.report()
summary_path = OUTPUT / f"{out_stem}_result.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "video": VIDEO, "fps": fps, "total_frames": total,
            "leave_timeout_s": LEAVE_TIMEOUT, "conf_thres": CONF_THRES,
            "model": "YOLOv8s + HSV颜色面积占比 + ByteTrack",
            "inference_fps": round(idx / elapsed, 1) if idx > 0 else 0,
        },
        "leave_monitor": leave_report,
    }, f, ensure_ascii=False, indent=2)

print(f"\n完成! {elapsed:.0f}s ({idx/elapsed:.1f}fps)" if idx > 0 else f"\n完成! {elapsed:.0f}s")
print(f"告警: {leave_report['summary']['total_alerts']}次")
print(f"JSONL(逐帧): {jsonl_path}")
print(f"摘要: {summary_path}")
print(f"视频: {OUTPUT / f'{out_stem}_output.mp4'}")
