# LAB 02 — Data Acquisition, Sampling & Tiny AI Dataset

**ECE 6041: Embedded AI & Edge Computing**  
**Weeks:** 3–4 | **Total:** 70 points | **Alignment:** CLO 1, 4  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `web/` | Webcam app; use the **Capture dataset** panel to record gesture CSVs |
| `Lab02_Dataset_Inspection.ipynb` | Load your CSVs and inspect class balance and sessions |
| `classes.json` | Gesture class order: `open_palm`, `fist`, `peace`, `thumbs_up` |
| `requirements.txt` | Python packages |

**You need:** Nothing beyond your webcam.  
**You produce:** Two or more session CSVs (`gesture_<session>.csv`). **Keep them; Labs 3–5 use them.**

## How to start

First time? Set up Python once using [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). Then, with your environment active, run `python -m pip install -r requirements.txt` from this folder.

**Run the webcam app** from this folder (`Lab2/`):

```bash
python -m http.server 8000
```

Then open **http://localhost:8000/web/** in Chrome or Edge and click **Start Camera**. Press **Ctrl+C** to stop the server.

**Run the notebook** in Google Colab (upload the `.ipynb` and the input files when prompted), or locally in VS Code/JupyterLab with your `gestureai` kernel. The upload cells use `google.colab`; when running locally, replace them with a path to your files.

## Mission and deliverable
Create **Four-class gesture CSV with reproducible on-device feature extraction**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick win
Use page capture controls to save the first 10 landmark feature rows as CSV in 20 minutes.

## Two-week student instructions
- W1: Collect open_palm, fist, peace, thumbs_up using the same normalized 42-coordinate feature order as the web runtime.
- W1: Collect two sessions at different times; recommended minimum 100 good-quality examples per class in total, with coverage in both sessions. Do not save or submit raw video.
- W1: Plot class counts and one example landmark trace; check missing data and unknown labels.
- W2: Compare sample collection at full available rate and approximately every third frame. Calculate rows/minute and approximate CSV storage KB/minute; report correlation/duplication effects.
- W2: Create group-based train/test assignments by session/person; explain data leakage and why random frame split is misleading.

## Required individual modification
**Change sampling interval and measure data rate versus gesture coverage.** Running the instructor's code without modification cannot earn full credit.

## Submit
Two or more session CSVs, schema check, sampling-rate/size table, data card (consent, limitation, coverage).

## Explain it — answer in your own words
Why normalize wrist coordinates? Why split by session instead of neighboring frames? What does reduced sampling save?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. Submit a short (3–6 sentence) reflection showing one AI suggestion and how you **tested** it. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Use instructor-provided deidentified 42-feature CSVs for collection analysis; no camera required.

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
