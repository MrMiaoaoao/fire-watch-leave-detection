"""监火员离岗检测模块。

输入逐帧红色监火员数量，输出离岗告警、返回清除事件和结构化报告。
"""

from __future__ import annotations

import json


class LeaveDetector:
    """基于连续无红色监火员时长的离岗检测器。"""

    def __init__(
        self,
        timeout=10.0,
        fps=24.0,
        warmup=5.0,
        present_confirm_s=0.3,
        clear_confirm_s=0.5,
    ):
        """
        timeout: 离岗超时秒数。
        fps: 视频帧率，由调用方传入实际值。
        warmup: 启动预热秒数，预热期内不触发告警。
        present_confirm_s: 连续检测到红色达到该时长才刷新在岗时间。
        clear_confirm_s: 告警后连续检测到红色达到该时长才清除告警。
        """
        self.timeout = timeout
        self.fps = fps
        self.warmup = warmup
        self.present_confirm_frames = max(1, int(round(present_confirm_s * fps)))
        self.clear_confirm_frames = max(self.present_confirm_frames, int(round(clear_confirm_s * fps)))

        self.frame_idx = 0
        self.last_seen_ts = 0.0
        self.first_frame_ts = None
        self.alert_active = False
        self.alerts = []
        self.present_streak = 0
        self.absent_streak = 0

    def update(self, red_count):
        """处理一帧红色监火员数量，触发告警时返回 warn dict，否则返回 None。"""
        ts = self.frame_idx / self.fps
        if self.first_frame_ts is None:
            self.first_frame_ts = ts

        result = None

        if red_count > 0:
            self.present_streak += 1
            self.absent_streak = 0

            if self.present_streak >= self.present_confirm_frames:
                self.last_seen_ts = ts

            if self.alert_active and self.present_streak >= self.clear_confirm_frames:
                self.alerts.append({
                    "type": "clear",
                    "content": "监火员返回岗位",
                    "time_s": round(ts, 1),
                })
                self.alert_active = False

        else:
            self.absent_streak += 1
            self.present_streak = 0

        if red_count <= 0 and ts - self.first_frame_ts > self.warmup:
            gap = ts - max(self.last_seen_ts, self.first_frame_ts)
            if gap > self.timeout and not self.alert_active:
                self.alert_active = True
                result = {
                    "type": "warn",
                    "content": "警告:监火员离岗",
                    "time_s": round(ts, 1),
                    "leave_duration_s": round(gap, 1),
                }
                self.alerts.append(result)

        self.frame_idx += 1
        return result

    def status(self):
        """返回当前状态快照。"""
        ts = self.frame_idx / self.fps if self.fps > 0 else 0
        gap = ts - max(self.last_seen_ts, self.first_frame_ts or 0)
        return {
            "frame": self.frame_idx,
            "time_s": round(ts, 1),
            "supervisor_present": not self.alert_active and (gap < self.timeout),
            "alert_active": self.alert_active,
            "seconds_since_last_seen": round(gap, 1) if gap > 0 else None,
        }

    def report(self):
        """生成结构化 JSON 报告。"""
        total_duration = self.frame_idx / self.fps if self.fps > 0 else 0
        warn_events = [a for a in self.alerts if a["type"] == "warn"]
        return {
            "summary": {
                "total_frames": self.frame_idx,
                "fps": self.fps,
                "total_duration_s": round(total_duration, 1),
                "timeout_s": self.timeout,
                "warmup_s": self.warmup,
                "total_alerts": len(warn_events),
            },
            "alerts": self.alerts,
        }

    def save_report(self, path):
        """保存结构化报告到 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report(), f, ensure_ascii=False, indent=2)
        return path


if __name__ == "__main__":
    detector = LeaveDetector(timeout=30.0, fps=30.0, warmup=3.0)

    for _ in range(150):
        detector.update(red_count=1)

    for _ in range(1050):
        alert = detector.update(red_count=0)
        if alert:
            print(f"  [{alert['time_s']}s] {alert['content']}")

    for _ in range(300):
        alert = detector.update(red_count=1)
        if alert:
            print(f"  [{alert['time_s']}s] {alert['content']}")

    print(f"\n报告: {json.dumps(detector.report(), ensure_ascii=False, indent=2)}")
