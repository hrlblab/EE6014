# LAB 05 — Precision, Quantization & Compression Trade-offs

**ECE 6041: Embedded AI & Edge Computing**  
**Weeks:** 9–10 | **Total:** 70 points | **Alignment:** CLO 4, 6  
**Product:** GestureAI — Touchless Computer Control

## Mission and deliverable
Create **FP32/INT8 comparisons across different classifier capacities**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick win
Run provided ONNX Runtime quantization for Small model and display before/after bytes in 20 minutes.

## Two-week student instructions
- W1: Compute a toy INT8 scale/zero point by hand; distinguish static calibration from dynamic weight quantization.
- W1: Quantize Small ONNX model with ONNX Runtime dynamic quantization; verify that the quantized runtime is supported on the CPU environment.
- W2: Quantize Tiny and Large as well where the runtime supports them; record original and quantized file bytes, batch=1 median latency and held-out session accuracy.
- W2: Graph the 3-model accuracy–file-size–latency Pareto trade-off. Explain why tiny MLPs may show negligible speedup or larger quantized file overhead.
- W2: Read short instructor examples of pruning and knowledge distillation; select one as an optional extension, not a mandatory training pipeline.

## Required individual modification
**Change model capacity before quantization and compare all other conditions consistently.** Running the instructor's code without modification cannot earn full credit.

## Submit
Quantized artifacts, per-model comparison table, trade-off chart and engineering selection memo.

## Explain it — answer in your own words
Why is INT8 not always faster? Why is quantized ONNX file size different from weight-byte estimates? Can the full hand-landmark stage run on an MCU?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. Submit a short (3–6 sentence) reflection showing one AI suggestion and how you **tested** it. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
CPU execution only; if a quantized operator is unsupported, report the incompatibility and use teacher-provided benchmark artifact to analyze.

## Score /70
| Criterion | Points | Evidence required |
|---|---:|---|
| Working system | 15 | Reproducible run; starter is correctly extended |
| Engineering modification | 20 | A meaningful change with an A/B comparison |
| Measurements / artifacts | 15 | Authentic numbers, tables, screenshots and files |
| Technical explanation | 15 | Correct constraints, pipeline, limitations and decisions |
| Reproducibility / AI verification | 5 | Clean files, environment notes, AI-assisted checks |
| **Total** | **70** | |

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
