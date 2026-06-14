# Fire Watch Leave Detection

监火员离岗检测项目，当前版本重点关注远景红/黄背心识别和离岗检测泛化。近景身份确认交给人脸识别模块，不继续优化近景颜色抖动。

## 当前版本

- 颜色分类器：`color_classifier.py`
- 当前关键版本：v2.6.4 ROI restore
- 离岗检测：`leave_detector.py`
- 主流程：`full_pipeline.py`
- 检测模型：YOLOv8s + ByteTrack + HSV 背心颜色分类

当前 `ALL_ROIS` 使用远景更稳的 v2.6.4 多区域上半身 ROI：

```python
(0.18, 0.65, 0.20, 0.80)
(0.10, 0.50, 0.18, 0.82)
(0.15, 0.60, 0.08, 0.55)
(0.15, 0.60, 0.45, 0.92)
```

`leave_detector.py` 保留连续红色确认：

- `present_confirm_s=0.3`
- `clear_confirm_s=0.5`

该逻辑用于防止短时误红清除离岗告警。

## 目录

```text
full_v2/
  full_pipeline.py
  color_classifier.py
  leave_detector.py
  requirements.txt
  操作说明.md
  data/       # 本地放测试视频，不提交 mp4
  models/     # 本地放 YOLO 权重，不提交 pt
  outputs/    # 提交 JSON/JSONL 结果，不提交输出视频
```

## 环境

推荐使用已有 `yolo` conda 环境：

```powershell
conda activate yolo
pip install -r requirements.txt
```

本项目当前验证使用：

- `ultralytics 8.4.60`
- `opencv-python 4.13.0`
- CUDA 可用

## 运行

```powershell
python full_pipeline.py "data\监火员离岗测试.mp4"
python full_pipeline.py "data\监火员离岗测试2.mp4"
python full_pipeline.py "data\监火员离岗测试3.mp4"
python full_pipeline.py "data\监火员离岗测试4.mp4"
```

输出：

- `outputs/*_result.json`
- `outputs/*_result.jsonl`
- `outputs/*_output.mp4`

注意：`*_output.mp4` 不提交到 GitHub。

## 当前四个测试结果

重新生成时间：2026-06-14

| 测试视频 | 告警次数 | 告警时间 | 清除时间 |
| --- | ---: | --- | --- |
| `监火员离岗测试.mp4` | 1 | `96.6s` | `104.3s` |
| `监火员离岗测试2.mp4` | 1 | `55.3s` | `94.0s` |
| `监火员离岗测试3.mp4` | 0 | 无 | 无 |
| `监火员离岗测试4.mp4` | 4 | `13.5s`, `81.3s`, `123.5s`, `152.3s` | `34.2s`, `100.9s`, `138.4s`, `166.1s` |

## 大文件说明

以下文件不进入普通 git 仓库：

- `data/*.mp4`
- `models/*.pt`
- `outputs/*_output.mp4`

如果需要完整复现实验，请从外部归档位置恢复视频和权重到对应目录。
