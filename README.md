# Fire Watch Leave Detection

监火员离岗检测项目。当前版本只关注远景红/黄背心识别和离岗检测泛化；近景身份确认交给人脸识别模块，不继续优化近景颜色抖动。

## 当前版本

- 版本标记：v2.6.4 ROI restore
- 主流程：`full_pipeline.py`
- 颜色分类：`color_classifier.py`
- 离岗检测：`leave_detector.py`
- 检测方案：YOLOv8s + ByteTrack + HSV 背心颜色分类 + 离岗告警

当前 `color_classifier.py` 使用远景更稳的 v2.6.4 上半身多 ROI：

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

## 模型权重

模型权重不提交到普通 GitHub 仓库，需要本地自行放入 `models/` 目录。

当前主流程默认读取：

```text
models/yolov8s.pt
```

可选备用权重：

```text
models/yolov8n.pt
models/yolov8l.pt
models/yolov8l-worldv2_vest_finetuned.pt
```

注意：

- `models/*.pt` 已被 `.gitignore` 忽略。
- 如果缺少 `models/yolov8s.pt`，运行 `full_pipeline.py` 会直接报 `FileNotFoundError`。
- 建议将权重文件放在网盘、服务器或 GitHub Release 中归档，不要直接提交到 git。

## 数据与输出

测试视频不提交到普通 GitHub 仓库，需要本地放入 `data/` 目录：

```text
data/监火员离岗测试.mp4
data/监火员离岗测试2.mp4
data/监火员离岗测试3.mp4
data/监火员离岗测试4.mp4
```

输出目录：

```text
outputs/
```

会生成：

- `*_result.json`：摘要报告
- `*_result.jsonl`：逐帧结果
- `*_output.mp4`：标注视频

其中 `outputs/*_output.mp4` 不提交到 GitHub；当前仓库保留 JSON/JSONL 结果用于复查。

## 环境

推荐使用已有 `yolo` conda 环境：

```powershell
conda activate yolo
pip install -r requirements.txt
```

当前验证环境：

- `ultralytics 8.4.60`
- `opencv-python 4.13.0`
- CUDA 可用

## 运行

在项目根目录运行：

```powershell
python full_pipeline.py "data\监火员离岗测试.mp4"
python full_pipeline.py "data\监火员离岗测试2.mp4"
python full_pipeline.py "data\监火员离岗测试3.mp4"
python full_pipeline.py "data\监火员离岗测试4.mp4"
```

常用参数：

```powershell
python full_pipeline.py "data\监火员离岗测试2.mp4" --max-frames 5
python full_pipeline.py "data\监火员离岗测试2.mp4" --output-suffix "_probe"
python full_pipeline.py "data\监火员离岗测试2.mp4" --weights "models\yolov8s.pt"
python full_pipeline.py "data\监火员离岗测试2.mp4" --device cpu --max-frames 1
```

如果只做快速连通性验证：

```powershell
$env:MAX_FRAMES="5"
python full_pipeline.py "data\监火员离岗测试2.mp4"
Remove-Item Env:MAX_FRAMES
```

一键复跑四个测试：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all_tests.ps1 -Python "C:\Users\22260\.conda\envs\yolo\python.exe"
```

显存紧张时可用 CPU 做 smoke test：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all_tests.ps1 -Python "C:\Users\22260\.conda\envs\yolo\python.exe" -Device cpu -MaxFrames 1 -OutputSuffix "_probe"
```

汇总并校验当前四测结果：

```powershell
python scripts\summarize_results.py --strict-current
```

## 当前四个测试结果

重新生成时间：2026-06-14

| 测试视频 | 告警次数 | 告警时间 | 清除时间 |
| --- | ---: | --- | --- |
| `监火员离岗测试.mp4` | 1 | `96.6s` | `104.3s` |
| `监火员离岗测试2.mp4` | 1 | `55.3s` | `94.0s` |
| `监火员离岗测试3.mp4` | 0 | 无 | 无 |
| `监火员离岗测试4.mp4` | 4 | `13.5s`, `81.3s`, `123.5s`, `152.3s` | `34.2s`, `100.9s`, `138.4s`, `166.1s` |

## GitHub 收录范围

提交到 GitHub：

- 代码：`*.py`
- 文档：`README.md`、`操作说明.md`、`AGENTS.md`
- 环境：`requirements.txt`
- 结果摘要：`outputs/*_result.json`
- 逐帧结果：`outputs/*_result.jsonl`
- 占位说明：`data/README.md`、`models/README.md`

不提交：

- `data/*.mp4`
- `models/*.pt`
- `outputs/*_output.mp4`
- `__pycache__/`
