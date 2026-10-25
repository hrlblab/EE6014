# LAB 07 — Production-Like Edge AI Web App & Deployment Evidence

**ECE 6041: Embedded AI & Edge Computing**  
**Weeks:** 13–14 | **Total:** 70 points | **Alignment:** CLO 4, 8, 9  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `web/` | The GestureAI app you finish and deploy |
| `web/models/` | Instructor reference model for the quick win; replace it with your `gesture_model.onnx` and `classes.json` |
| `sample_data/gesture_demo_stream.csv` | Recorded 42-feature stream (~18 s, 7 gestures) for **Replay CSV** when you have no webcam |
| `requirements.txt` | Python packages (only needed to re-export or verify your model) |

**You need:** Your verified FP32 `gesture_model.onnx` and `classes.json` from Labs 3–4. Copy both into `web/models/`, replacing the reference model (see `web/models/PUT_YOUR_MODEL_HERE.txt`).  
**You produce:** Deployed or locally demonstrated app, README, benchmark table, and demo video.

## How to start

First time? Set up Python once using [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). Then, with your environment active, run `python -m pip install -r requirements.txt` from this folder.

**Run the webcam app** from this folder (`Lab7/`):

```bash
python -m http.server 8000
```

Then open **http://localhost:8000/web/** in Chrome or Edge and click **Start Camera**. Press **Ctrl+C** to stop the server.

## Mission and deliverable
Create **Portfolio-grade privacy-first GestureAI local-inference demo**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick win
Open the starter UI, load supplied tested ONNX model and actuate one slide-control command within 20 minutes.

## Two-week student instructions
- W1: Integrate verified student ONNX model (FP32 browser path), class order, exact 42-feature normalization, and local browser inference via ONNX Runtime Web/WASM.
- W1: Display live camera, hand landmark skeleton, label/confidence, visible current slide and at least three static gesture controls.
- W2: Add performance dashboard: landmark inference time, classifier inference time, end-to-end FPS and command responsiveness; include session-independent accuracy from Lab 3.
- W2: Compare two scheduling/interaction settings and identify at least one failure case (occlusion, lighting, two hands, false trigger). Verify no webcam frames are uploaded to a remote inference endpoint; initial model/CDN download does not imply offline availability.
- W2: Publish on HTTPS static hosting or demonstrate locally; deliver README, architecture diagram, model/benchmark table, 60–90 second demo video, privacy notice, and final engineering justification.

## Required individual modification
**Choose and implement one product-level improvement (debouncing, UI feedback, keyboard fallback, accessible controls, safe pause state) and measure/usability-test it.** Running the instructor's code without modification cannot earn full credit.

## Submit
Live URL or localhost screencast, source repo/ZIP, verified model, README, demo, end-to-end profiling and deployment explanation.

## Explain it — answer in your own words
Where does computation run? What are the complete system bottlenecks? What would change for phone, MCU, FPGA or NPU?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. Submit a short (3–6 sentence) reflection showing one AI suggestion and how you **tested** it. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Simulated feature-stream mode must remain usable without webcam; publish only consented footage or hand-only recordings.

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
