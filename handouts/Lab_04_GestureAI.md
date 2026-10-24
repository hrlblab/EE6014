# LAB 04 — Portable Deployment: ONNX, CPU & AI Accelerators

**ECE 6041: Embedded AI & Edge Computing**  
**Weeks:** 7–8 | **Total:** 70 points | **Alignment:** CLO 4, 5  
**Product:** GestureAI — Touchless Computer Control

## Mission and deliverable
Create **Validated ONNX CPU deployment and engine comparison**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick win
Run selected model in PyTorch and ONNX Runtime CPU and display predicted label parity in 20 minutes.

## Two-week student instructions
- W1: Export chosen classifier as ONNX with batch dimension; run ONNX checker and compare logits against PyTorch for at least 20 held-out samples.
- W1: Record maximum absolute output difference, predicted-class agreement, and any opset/runtime errors.
- W2: Benchmark PyTorch CPU and ONNX Runtime CPU for batch=1 and batch=32; use warm-up, 100 iterations, median/p95 and consistent thread settings.
- W2: Explain CPU vs GPU vs NPU and TensorRT engine-building on NVIDIA targets; use instructor-provided GPU/TensorRT benchmark data if compatible hardware is unavailable.
- W2: Week 7 project proposal: target, model, privacy boundary, timing/accuracy/size metrics and deployment risks.

## Required individual modification
**Change batch size or ORT CPU thread setting, rerun benchmarks, explain latency versus throughput.** Running the instructor's code without modification cannot earn full credit.

## Submit
ONNX file, parity table, full benchmark CSV, annotated runtime diagram, proposal checkpoint.

## Explain it — answer in your own words
Why does ONNX portability not guarantee execution on every accelerator? What is the difference between batch latency and throughput? Why is TensorRT not mandatory here?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. Submit a short (3–6 sentence) reflection showing one AI suggestion and how you **tested** it. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
CPU-only execution earns full credit; no TensorRT install required.

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
