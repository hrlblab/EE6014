# LAB 03 — Train & Compare Embedded-Size Models

**ECE 6041: Embedded AI & Edge Computing**  
**Weeks:** 5–6 | **Total:** 70 points | **Alignment:** CLO 1, 2, 3, 4  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `Lab03_Train_Classifier.ipynb` | Train the Small MLP with a session-based split; export PyTorch and ONNX |
| `Lab03_Embedded_Model_Design_Space.ipynb` | Compare Tiny / Small / Large models |
| `classes.json` | Gesture class order (must match the web app) |
| `requirements.txt` | Python packages |

**You need:** Your Lab 2 session CSVs.  
**You produce:** `gesture_model.pth`, `gesture_model.onnx`, `classes.json`. **Keep them; Labs 4–7 use them.**

## How to start

First time? Set up Python once using [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). Then, with your environment active, run `python -m pip install -r requirements.txt` from this folder.

**Run the notebooks** in Google Colab (upload the `.ipynb` and the input files when prompted), or locally in VS Code/JupyterLab with your `gestureai` kernel. The upload cells use `google.colab`; when running locally, replace them with a path to your files.

## Mission and deliverable
Create **Tiny/Small/Large trained PyTorch networks and measured design-space table**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick taste
Run training on reference CSV and see class predictions, loss and first accuracy chart in 20 minutes.

## Two-week student instructions
- W1: Reuse Lab 2 features, classes, and session split. Train Small MLP baseline in Colab CPU using a provided training loop.
- W1: Explain autograd, gradient descent, input/output dimensions, and how parameter count determines approximate weight memory.
- W2: Train Tiny 42→16→4, Small 42→64→32→4, Large 42→256→128→4 with identical split, epochs, and random seeds.
- W2: Measure held-out session accuracy, parameter count, FP32 bytes, model file size and CPU batch=1 median latency after warm-up.
- W2: Choose a candidate for the edge app, export checkpoint and ONNX, and document which architecture you chose; check class order and preprocessing compatibility.

## Required individual modification
**Compare at least two model capacities; submit a justified choice using both accuracy and resources.** Running the instructor's code without modification cannot earn full credit.

## Submit
Training plot, confusion matrix, three-model design table, selected checkpoint/ONNX/classes.json.

## Explain it — answer in your own words
Did the largest model generalize best? Why is weight memory not peak runtime memory? What would be different for a CNN?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. Submit a short (3–6 sentence) reflection showing one AI suggestion and how you **tested** it. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Use class-balanced reference CSV; GPU optional, CPU mandatory path.

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
