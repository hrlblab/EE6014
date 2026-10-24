# LAB 01 — Sensor-to-Action Pipeline & PyTorch Foundations

**ECE 6041: Embedded AI & Edge Computing**  
**Weeks:** 1–2 | **Total:** 70 points | **Alignment:** CLO 1, 4  
**Product:** GestureAI — Touchless Computer Control

## Mission and deliverable
Create **Live webcam landmarks and a tiny PyTorch inference demo**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick win
Open provided local webcam page, see 21 hand landmarks and raw pipeline timing within 20 minutes.

## Two-week student instructions
- W1: Draw sensing → preprocessing → pretrained hand detector → student classifier → decision → action. Identify what happens on-device and what is downloaded only during app setup.
- W1: In a Colab notebook create a 42-element tensor and run a supplied two-layer PyTorch model; explain inputs, logits, training vs inference.
- W1: Run the browser camera app, record 30 consecutive frames of hand-detector time and delivered FPS; annotate camera availability and privacy boundaries.
- W2: Change hand-detection confidence threshold (e.g., 0.5 vs 0.7) and try normal vs dim lighting; record detection success out of 20 intentional poses per setting.
- W2: Create a stage-by-stage timing table (sensor/frame delivery; hand detector; classifier placeholder; action placeholder). Mark unavailable stage measurements as N/A, not 0.

## Required individual modification
**Change detection threshold; compare measured detection stability and latency under controlled conditions.** Running the instructor's code without modification cannot earn full credit.

## Submit
Camera overlay screenshot; PyTorch tensor notebook; sensing/inference/action diagram; 2-condition comparison table.

## Explain it — answer in your own words
Why is landmark detection an inference task? Which parts run in the browser? What does FPS measure compared with one-stage latency?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. Submit a short (3–6 sentence) reflection showing one AI suggestion and how you **tested** it. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
No camera: view provided recorded demo and analyze instructor-supplied stage timing CSV; explicitly label as reference data.

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
