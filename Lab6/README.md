# LAB 06 — Real-Time Scheduling, Memory & MCU Feasibility

**ECE 6041: Embedded AI & Edge Computing**  
**Weeks:** 11–12 | **Total:** 70 points | **Alignment:** CLO 4, 7, 8  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `Lab06_Optimization_and_MCU.ipynb` | ONNX Runtime graph-optimization benchmark and MCU memory estimate |
| `web/` | Webcam app; modify `web/app.js` for frame skipping, voting, and cooldown |
| `requirements.txt` | Python packages |

**You need:** `gesture_model.onnx` and `classes.json` from Lab 3. Copy both into `web/models/` so the app can **Load Classifier**.  
**You produce:** Scheduler comparison, telemetry tables, and MCU memory budget.

## How to start

First time? Set up Python once using [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). Then, with your environment active, run `python -m pip install -r requirements.txt` from this folder.

**Run the webcam app** from this folder (`Lab6/`):

```bash
python -m http.server 8000
```

Then open **http://localhost:8000/web/** in Chrome or Edge and click **Start Camera**. Press **Ctrl+C** to stop the server.

**Run the notebook** in Google Colab (upload the `.ipynb` and the input files when prompted), or locally in VS Code/JupyterLab with your `gestureai` kernel. The upload cells use `google.colab`; when running locally, replace them with a path to your files.

## Mission and deliverable
Create **Stable action loop and defensible edge resource budget**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick win
See graph-optimization benchmark and live camera/classifier FPS telemetry within 20 minutes.

## Two-week student instructions
- W1: Measure classifier inference with ORT graph optimization disabled/basic/all; inspect operator changes where possible, note no speedup is an acceptable observation.
- W1: Separate detector time, classifier time, UI rendering FPS, and command response delay. Explain operator fusion, tensor reuse, layout and scheduling.
- W2: Implement frame skipping (every frame vs every 2/3 frames), temporal voting (3 vs 5 consistent outputs), and cooldown; compare false commands and reaction delay across 60-second scripted trials.
- W2: Estimate flash/SRAM needs of student classifier alone for hypothetical 256-KB Flash, 64-KB SRAM Cortex-M device; distinguish weight bytes, peak activations, stack/runtime/firmware.
- W2: Design realistic split placement: webcam + pretrained detector on browser/laptop; tiny gesture classifier MAY be MCU feasible with an appropriate runtime/transport, not proven without hardware profiling. Report energy only as assumption-based estimate E=P×t.

## Required individual modification
**Change inference sampling or voting threshold; measure response-time / false-command trade-off.** Running the instructor's code without modification cannot earn full credit.

## Submit
Scheduler comparison and telemetry tables, MCU memory budget sheet, system partition diagram, 1-page engineering memo.

## Explain it — answer in your own words
Which stage is the bottleneck? What causes missed commands with skipping? Why does ONNX success not prove MCU deployment?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. Submit a short (3–6 sentence) reflection showing one AI suggestion and how you **tested** it. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Recorded feature sequence and scripted predictions can replace camera for timing/action logic; estimated not measured energy.

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
