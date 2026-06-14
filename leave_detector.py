"""
监火员离岗检测模块 v2.0 (2026-06-06)
输入: 逐帧监火员数量 → 超时告警 + 结构化JSON报告
支持: 预热期/告警/解除/状态快照/报告导出
"""

import json, time
from pathlib import Path


class LeaveDetector:
    """监火员离岗检测器"""

    def __init__(self, timeout=10.0, fps=24.0, warmup=5.0,
                 present_confirm_s=0.3, clear_confirm_s=0.5):
        """
        timeout: 离岗超时(秒), 默认10
        fps: 视频帧率, 由调用方传入实际值, 24仅为兜底
        warmup: 启动预热(秒), 前N秒不触发告警
        present_confirm_s: 连续检测到红色达到该时长才刷新在岗时间
        clear_confirm_s: 告警后连续检测到红色达到该时长才清除告警
        """
        self.timeout = timeout
        self.fps = fps
        self.warmup = warmup
        self.present_confirm_frames = max(1, int(round(present_confirm_s * fps)))
        self.clear_confirm_frames = max(self.present_confirm_frames, int(round(clear_confirm_s * fps)))

        # 内部状态
        self.frame_idx = 0
        self.last_seen_ts = 0.0
        self.first_frame_ts = None
        self.alert_active = False
        self.alerts = []  # 所有告警记录
        self.gap_history = []  # (ts, gap) 采样
        self.present_streak = 0
        self.absent_streak = 0

    def update(self, red_count):
        """
        每帧调用一次.
        red_count: 当前帧检测到的监火员(红马甲)数量
        返回: dict if 触发告警 else None
        """
        ts = self.frame_idx / self.fps
        if self.first_frame_ts is None:
            self.first_frame_ts = ts

        result = None

        if red_count > 0:
            self.present_streak += 1
            self.absent_streak = 0

            if self.present_streak >= self.present_confirm_frames:
                self.last_seen_ts = ts

            # 告警清除需要更强的连续在岗证据, 防止短时误红打断离岗事件.
            if self.alert_active and self.present_streak >= self.clear_confirm_frames:
                # 告警解除
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
        """当前状态快照 (与update逻辑同步, 修改6)"""
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
        """生成结构化JSON报告"""
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
        """保存报告到JSON文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report(), f, ensure_ascii=False, indent=2)
        return path


# ===== 独立测试 =====
if __name__ == "__main__":
    # 模拟测试: 30fps, 前5秒监火员在场, 然后离开35秒, 再回来
    detector = LeaveDetector(timeout=30.0, fps=30.0, warmup=3.0)

    # 0-5s: 在场
    for _ in range(150):  # 150帧 = 5秒
        r = detector.update(red_count=1)

    # 5-40s: 离开 (35秒, 超过30秒阈值)
    for i in range(1050):  # 1050帧 = 35秒
        r = detector.update(red_count=0)
        if r:
            print(f"  [{r['time_s']}s] {r['content']}")

    # 40-50s: 返回
    for _ in range(300):
        r = detector.update(red_count=1)
        if r:
            print(f"  [{r['time_s']}s] {r['content']}")

    report = detector.report()
    print(f"\n报告: {json.dumps(report, ensure_ascii=False, indent=2)}")
