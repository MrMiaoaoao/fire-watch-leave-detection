"""
监火员离岗检测系统 v2.6.4

YOLOv8s + ByteTrack + HSV 背心颜色分类 + 离岗告警。
当前优化边界：只关注远景红/黄背心识别和离岗检测泛化，近景交给人脸识别模块。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from leave_detector import LeaveDetector


@dataclass(frozen=True)
class PipelineConfig:
    video: str
    weights: str
    output_dir: str
    output_suffix: str = ""
    max_frames: int = 0
    leave_timeout: float = 10.0
    conf_thres: float = 0.45
    min_box_w: int = 50
    min_box_h: int = 100
    vote_window: int = 12
    vote_thres: int = 7
    lock_consec: int = 8
    unlock_consec: int = 15
    switch_margin: float = 0.08
    warmup: float = 5.0
    tracker: str = "bytetrack.yaml"
    device: str = ""


LABEL_CACHE = {}


def parse_args() -> PipelineConfig:
    default_video = ROOT / "data" / "监火员离岗测试2.mp4"
    default_weights = ROOT / "models" / "yolov8s.pt"
    default_output = ROOT / "outputs"

    parser = argparse.ArgumentParser(description="监火员离岗检测主流程")
    parser.add_argument("video", nargs="?", default=str(default_video), help="输入视频路径或摄像头编号")
    parser.add_argument("--weights", default=str(default_weights), help="YOLO 权重路径")
    parser.add_argument("--output-dir", default=str(default_output), help="输出目录")
    parser.add_argument(
        "--output-suffix",
        default=os.environ.get("OUTPUT_SUFFIX", ""),
        help="输出文件名后缀，也可用环境变量 OUTPUT_SUFFIX",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=int(os.environ.get("MAX_FRAMES", "0") or "0"),
        help="最多处理帧数，0 表示处理完整视频；也可用环境变量 MAX_FRAMES",
    )
    parser.add_argument("--leave-timeout", type=float, default=10.0, help="离岗告警超时秒数")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="person 检测置信度阈值")
    parser.add_argument("--min-box-w", type=int, default=50, help="最小 person 框宽度")
    parser.add_argument("--min-box-h", type=int, default=100, help="最小 person 框高度")
    parser.add_argument("--device", default="", help="推理设备，例如 cpu、0；空字符串表示由 Ultralytics 自动选择")
    args = parser.parse_args()
    return PipelineConfig(
        video=args.video,
        weights=args.weights,
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
        max_frames=max(0, args.max_frames),
        leave_timeout=args.leave_timeout,
        conf_thres=args.conf_thres,
        min_box_w=args.min_box_w,
        min_box_h=args.min_box_h,
        device=args.device,
    )


def get_label(text, color):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    key = (text, color)
    if key not in LABEL_CACHE:
        font = ImageFont.load_default()
        for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf"]:
            try:
                font = ImageFont.truetype(fp, 18)
                break
            except Exception:
                continue
        drawer = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = drawer.textbbox((0, 0), text, font=font)
        label_w, label_h = bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4
        image = Image.new("RGB", (label_w, label_h), (0, 0, 0))
        ImageDraw.Draw(image).text((2, 0), text, font=font, fill=color)
        LABEL_CACHE[key] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return LABEL_CACHE[key]


def paste_label(frame, text, x, y, color=(0, 0, 255)):
    import cv2

    label = get_label(text, color)
    label_h, label_w = label.shape[:2]
    y = y + 20 if y - label_h < 0 else y
    x1, y1 = max(0, x), max(0, y - label_h)
    x2, y2 = min(frame.shape[1], x + label_w), min(frame.shape[0], y1 + label_h)
    roi = frame[y1:y2, x1:x2]
    label_crop = label[0:roi.shape[0], 0:roi.shape[1]]
    if roi.shape == label_crop.shape and roi.size > 0:
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, label_crop, 0.7, 0)


def box_iou_contain(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou = inter / (area_a + area_b - inter + 1e-6)
    contain = inter / (min(area_a, area_b) + 1e-6)
    return iou, contain


def find_overlaps(items, box_key="bbox", iou_thres=0.15, contain_thres=0.40):
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if isinstance(items[i], dict):
                box_i, box_j = items[i][box_key], items[j][box_key]
                tid_i, tid_j = items[i].get("track_id"), items[j].get("track_id")
                color_i, color_j = items[i].get("color"), items[j].get("color")
            else:
                box_i, box_j = items[i][:4], items[j][:4]
                tid_i, tid_j = items[i][5], items[j][5]
                color_i, color_j = None, None
            iou, contain = box_iou_contain(box_i, box_j)
            if iou > iou_thres or contain > contain_thres:
                overlaps.append({
                    "i": i,
                    "j": j,
                    "tid_i": int(tid_i) if tid_i is not None else None,
                    "tid_j": int(tid_j) if tid_j is not None else None,
                    "col_i": color_i,
                    "col_j": color_j,
                    "iou": round(float(iou), 4),
                    "contain": round(float(contain), 4),
                    "box_i": [int(x) for x in box_i],
                    "box_j": [int(x) for x in box_j],
                })
    return overlaps


def vote_color(hist, vote_thres):
    red_votes = hist.count("red")
    yellow_votes = hist.count("yellow")
    valid = red_votes + yellow_votes
    if valid < 4:
        return None
    if red_votes >= vote_thres:
        return "red"
    if yellow_votes >= vote_thres:
        return "yellow"
    return None


def validate_inputs(config: PipelineConfig):
    weights = Path(config.weights)
    if not weights.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights}")
    if not Path(config.video).exists() and not str(config.video).isdigit():
        raise FileNotFoundError(f"视频文件不存在: {config.video}")
    try:
        from ultralytics.trackers import register_tracker  # noqa: F401
    except ImportError as exc:
        raise ImportError("ByteTrack 依赖缺失，请运行: pip install lap") from exc


def deduplicate_persons(person_data):
    if len(person_data) <= 1:
        return person_data
    person_data.sort(key=lambda x: x[4], reverse=True)
    keep = []
    for person in person_data:
        duplicate = False
        for kept in keep:
            iou, contain = box_iou_contain(person[:4], kept[:4])
            if iou > 0.20 or contain > 0.45:
                duplicate = True
                break
        if not duplicate:
            keep.append(person)
    return keep


def deduplicate_detections(raw_dets):
    if len(raw_dets) <= 1:
        return raw_dets
    raw_dets.sort(
        key=lambda x: (x["conf"], (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])),
        reverse=True,
    )
    merged = []
    for det in raw_dets:
        duplicate = False
        for kept in merged:
            iou, contain = box_iou_contain(det["bbox"], kept["bbox"])
            if iou > 0.15 or contain > 0.40:
                duplicate = True
                break
        if not duplicate:
            merged.append(det)
    return merged


def run_pipeline(config: PipelineConfig):
    validate_inputs(config)
    import cv2
    import torch
    from ultralytics import YOLO

    from color_classifier import classify_batch_scored

    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU 推理，速度会明显变慢")

    print("加载 YOLOv8s + ByteTrack...")
    model = YOLO(config.weights)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(config.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {config.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError(f"视频读取失败或帧数为 0: {config.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = Path(config.video).stem if not str(config.video).isdigit() else f"camera_{config.video}"
    out_stem = f"{video_name}{config.output_suffix}"
    output_video_path = output_dir / f"{out_stem}_output.mp4"
    jsonl_path = output_dir / f"{out_stem}_result.jsonl"
    summary_path = output_dir / f"{out_stem}_result.json"

    vout = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    known = {}
    detector = LeaveDetector(timeout=config.leave_timeout, fps=fps, warmup=config.warmup)
    print(f"视频: {total}帧, {fps}fps, {total / fps:.0f}s")

    start = time.time()
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if config.max_frames > 0 and idx >= config.max_frames:
                break

            ts = idx / fps
            track_kwargs = {
                "conf": config.conf_thres,
                "iou": 0.35,
                "classes": [0],
                "verbose": False,
                "persist": True,
                "tracker": config.tracker,
            }
            if config.device:
                track_kwargs["device"] = config.device
            results = model.track(frame, **track_kwargs)[0]
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
                for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, tids):
                    if tid < 0:
                        continue
                    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                    ix1 = max(0, min(ix1, width - 1))
                    iy1 = max(0, min(iy1, height - 1))
                    ix2 = max(0, min(ix2, width))
                    iy2 = max(0, min(iy2, height))
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    if (ix2 - ix1) >= config.min_box_w and (iy2 - iy1) >= config.min_box_h:
                        person_data.append((ix1, iy1, ix2, iy2, conf, tid))

                raw_pc = len(person_data)
                raw_ov = find_overlaps(person_data, iou_thres=0.15, contain_thres=0.40)
                person_data = deduplicate_persons(person_data)
                dedup_pc = len(person_data)
                dedup_ov = find_overlaps(person_data, iou_thres=0.15, contain_thres=0.40)

                if person_data:
                    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2, _, _ in person_data]
                    scored = classify_batch_scored(crops)
                    raw_dets = []
                    for (ix1, iy1, ix2, iy2, conf, tid), score_data in zip(person_data, scored):
                        if tid not in known:
                            known[tid] = [
                                ix1,
                                iy1,
                                ix2,
                                iy2,
                                idx,
                                deque(maxlen=config.vote_window),
                                None,
                                None,
                                0,
                                0,
                            ]
                        else:
                            track = known[tid]
                            track[0:4] = [ix1, iy1, ix2, iy2]
                            track[4] = idx

                        color = score_data["color"] if score_data else None
                        known[tid][5].append(color)
                        hist = list(known[tid][5])
                        final_color = vote_color(hist, config.vote_thres)

                        if final_color is not None:
                            if final_color == known[tid][7]:
                                known[tid][8] += 1
                            else:
                                known[tid][8] = 1
                            known[tid][7] = final_color

                        locked = known[tid][6]
                        if locked is None and known[tid][8] >= config.lock_consec and final_color is not None:
                            known[tid][6] = final_color
                            known[tid][9] = 0

                        if locked is not None and final_color is not None:
                            score = score_data["score"] if score_data else 0
                            if final_color != locked:
                                opposite_raw = sum(1 for c in hist if c == final_color)
                                same_raw = sum(1 for c in hist if c == locked)
                                if opposite_raw > same_raw and score > config.switch_margin:
                                    known[tid][9] += 1
                                else:
                                    known[tid][9] = 0
                                if known[tid][9] >= config.unlock_consec:
                                    known[tid][6] = None
                                    known[tid][9] = 0
                                    known[tid][8] = 1
                            else:
                                known[tid][9] = 0

                        if known[tid][6] is not None:
                            final_color = known[tid][6]
                        if final_color is not None:
                            score = score_data["score"] if score_data else 0
                            raw_dets.append({
                                "bbox": [ix1, iy1, ix2, iy2],
                                "color": final_color,
                                "track_id": int(tid),
                                "conf": round(float(conf), 2),
                                "score": score,
                                "red_ratio": score_data["red_ratio"] if score_data else 0,
                                "yellow_ratio": score_data["yellow_ratio"] if score_data else 0,
                            })

                    raw_dets = deduplicate_detections(raw_dets)
                    final_ov = find_overlaps(raw_dets, box_key="bbox", iou_thres=0.15, contain_thres=0.40)
                    if final_ov:
                        print(f"  [重复框警告] frame={idx}, time={ts:.2f}s, n={len(final_ov)}")

                    for det in raw_dets:
                        if det["color"] == "red":
                            red_count += 1
                        elif det["color"] == "yellow":
                            yellow_count += 1
                        detections.append(det)

            gone = [tid for tid, track in known.items() if track[4] < idx - 60]
            for tid in gone:
                del known[tid]

            alert = detector.update(red_count)
            alerts = [alert] if alert else []
            if alert:
                print(f"  [{alert['time_s']}s] {alert['content']}")

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                if det["color"] == "red":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    paste_label(frame, f"监火员 {det['conf']:.2f}", x1, y1 - 5, (0, 0, 255))
                elif det["color"] == "yellow":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    paste_label(frame, f"动火员 {det['conf']:.2f}", x1, y1 - 5, (0, 255, 255))

            paste_label(
                frame,
                f"Frame:{idx} Time:{ts:.1f}s 监火员:{red_count} 动火员:{yellow_count}",
                10,
                30,
                (255, 255, 255),
            )
            for i, alert_data in enumerate(alerts):
                paste_label(frame, alert_data["content"], 10, 60 + i * 25, (0, 0, 255))

            frame_data = {
                "frame": idx,
                "timestamp": round(ts, 2),
                "supervisor_count": red_count,
                "worker_count": yellow_count,
                "alerts": alerts,
                "raw_pc": raw_pc,
                "dedup_pc": dedup_pc,
                "raw_ov": raw_ov,
                "dedup_ov": dedup_ov,
                "final_ov": final_ov,
                "detections": detections,
            }
            jsonl_file.write(json.dumps(frame_data, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            vout.write(frame)
            idx += 1
            if idx % 500 == 0:
                print(f"  {idx}/{total} ({100 * idx / total:.0f}%) {idx / (time.time() - start):.1f}fps")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as exc:
        print(f"\n运行错误: {exc}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        cap.release()
        vout.release()
        jsonl_file.close()

    elapsed = time.time() - start
    leave_report = detector.report()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "video": config.video,
                "fps": fps,
                "total_frames": total,
                "processed_frames": idx,
                "leave_timeout_s": config.leave_timeout,
                "conf_thres": config.conf_thres,
                "model": "YOLOv8s + HSV背心颜色分类 + ByteTrack",
                "weights": config.weights,
                "inference_fps": round(idx / elapsed, 1) if idx > 0 else 0,
                "config": asdict(config),
            },
            "leave_monitor": leave_report,
        }, f, ensure_ascii=False, indent=2)

    if idx > 0:
        print(f"\n完成! {elapsed:.0f}s ({idx / elapsed:.1f}fps)")
    else:
        print(f"\n完成! {elapsed:.0f}s")
    print(f"告警: {leave_report['summary']['total_alerts']}次")
    print(f"JSONL(逐帧): {jsonl_path}")
    print(f"摘要: {summary_path}")
    print(f"视频: {output_video_path}")


def main():
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
