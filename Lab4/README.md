# LAB 04 — Portable Deployment: ONNX, CPU & AI Accelerators

**ECE 6041: Embedded AI & Edge Computing**  
**Total:** 70 points | **Alignment:** CLO 4, 5  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `Lab04_ONNX_Verify_Benchmark.ipynb` | PyTorch vs ONNX Runtime parity check and CPU benchmark |
| `requirements.txt` | Python packages |

**You need:** `gesture_model.pth` and `gesture_model.onnx` from Lab 3, plus one held-out session CSV from Lab 2.  
**You produce:** Parity results and a batch=1 / batch=32 latency table.

## How to start

First time? Set up Python once using [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). Then, with your environment active, run `python -m pip install -r requirements.txt` from this folder.

**Run the notebook** in Google Colab (upload the `.ipynb` and the input files when prompted), or locally in VS Code/JupyterLab with your `gestureai` kernel. The upload cells use `google.colab`; when running locally, replace them with a path to your files.

## Mission and deliverable
Create **Validated ONNX CPU deployment and engine comparison**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick taste
Run selected model in PyTorch and ONNX Runtime CPU and display predicted label parity in 20 minutes.

## Step-by-step instructions
Lab 4 runs in **Google Colab** (or locally). Benchmark numbers describe the computer that ran them, so always say where you ran.

**Step 1. Gather your files.** From your Lab 3 Step 11 download: `gesture_model.pth` and `gesture_model.onnx` (**the same download**). From Lab 2: the CSV of the session that Lab 3 used as **TEST** (for example `gesture_session2.csv`).

**Step 2. Run the notebook.** Go to https://colab.research.google.com and sign in. Click **File → Upload notebook** and choose `Lab04_ONNX_Verify_Benchmark.ipynb`. Click **Runtime → Run all**. When a **Choose Files** button appears, select the files listed here.
- First **Choose Files**: `gesture_model.onnx`. It runs the ONNX checker and prints the input shape `['batch', 42]`.
- Second **Choose Files**: `gesture_model.pth` **and** the test CSV.

**Step 3. Read the parity result.** Find **Max absolute difference** and **Same predicted class**.
✅ The difference is tiny (about 0.000001 = rounding) and **Same predicted class: 1.0**: PyTorch and ONNX Runtime give the same answers. If you get an `assert` error, the two model files come from different downloads.

**Step 4. Write your prediction first.** Will ONNX Runtime be faster than PyTorch at batch 1? Will 4 threads beat 1? How will time per call and samples per second change from batch 1 to batch 256?

**Step 5. Baseline benchmark.** The last cell starts with `THREADS=1` and `BATCHES=[1,32]`. It prints median and p95 times for both engines and downloads `lab4_benchmark_threads1.csv`.

**Step 6. Change threads and batch size (required modification).** In that cell change `BATCHES=[1,32]` to `BATCHES=[1,8,32,256]` and press **Shift+Enter**. Then change `THREADS=1` to `THREADS=2` and run it again, then `THREADS=4`. You get `lab4_benchmark_threads1.csv`, `…threads2.csv`, `…threads4.csv`.

**Step 7. Plot the results.** Add a cell at the end, paste this, and run it:

```python
import glob, pandas as pd, matplotlib.pyplot as plt
df = pd.concat([pd.read_csv(f) for f in glob.glob('lab4_benchmark_threads*.csv')])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for (engine, threads), g in df.groupby(['engine', 'threads']):
    g = g.sort_values('batch')
    ax[0].plot(g.batch, g.median_ms, 'o-', label=f'{engine}, {threads} thread(s)')
    ax[1].plot(g.batch, g.throughput_samples_per_s, 'o-', label=f'{engine}, {threads} thread(s)')
ax[0].set(xscale='log', yscale='log', xlabel='batch size', ylabel='median ms per call', title='Latency')
ax[1].set(xscale='log', yscale='log', xlabel='batch size', ylabel='samples per second', title='Throughput')
ax[1].legend(fontsize=7); plt.tight_layout(); plt.show()
```

Screenshot the plots. To save a notebook with its outputs: in Colab, **File → Download → Download .ipynb**.

**Step 8. Draw the runtime diagram.** Draw: `.pth` (PyTorch) → `torch.onnx.export` → `.onnx` → ONNX Runtime on CPU (here) → ONNX Runtime Web in the browser (Lab 7). Add where a GPU, an NPU, or TensorRT (NVIDIA) could run the model.

**Step 9. Project proposal.** Write a short table: target device and runtime; model (from Lab 3); privacy boundary (what stays on the device); metrics (accuracy, latency, size, FPS); risks.

**Step 10. Write the report and submit.** Write the parts listed in *What to submit* and export them as `LastName_VUID.pdf` (Word: **File → Save As → PDF**; Google Docs: **File → Download → PDF**). Make a folder named `LastName_VUID`, put the PDF and the files from *What to submit* in it, and zip it:
- **Windows:** right-click the folder → **Send to → Compressed (zipped) folder**.
- **macOS:** right-click the folder → **Compress**.
- **Linux:** right-click the folder → **Compress**, or run `zip -r LastName_VUID.zip LastName_VUID`.

Upload `LastName_VUID.zip` to the Lab 4 assignment on Brightspace.

## Required individual modification
**Change batch size or ORT CPU thread setting, rerun benchmarks, explain latency versus throughput.** Running the instructor's code without modification cannot earn full credit.

## What to submit (Brightspace)
Upload **one ZIP file** named **`LastName_VUID.zip`** (for example `Huo_huoy1.zip`) containing:

| File | Contents |
|---|---|
| `LastName_VUID.pdf` | Your report: the 7 parts below. Write it in Word, Google Docs, or anything else, then export/print to PDF. Put screenshots inside it. |
| `Lab04_ONNX_Verify_Benchmark.ipynb` | The notebook with its outputs. |
| `lab4_benchmark_threads1.csv`, `…threads2.csv`, `…threads4.csv` | Your benchmark results (Step 6). |

The PDF has 7 parts:

| # | Part | Include |
|---|---|---|
| 1 | Parity | Where you ran it (Colab or your computer); checker result; max absolute difference; same-predicted-class share; number of rows compared. |
| 2 | Benchmark | Your prediction (written first); a table of engine × threads × batch with median ms, p95 ms, and samples/s; the two plots. |
| 3 | Conclusion | Best setting for a live webcam app (batch 1) and for offline batch processing, with numbers; why more threads helped or hurt. |
| 4 | Runtime diagram | The Step 8 diagram with accelerators marked. |
| 5 | Project proposal | The Step 9 table. |
| 6 | Explain it | Your answers to the three questions below. |
| 7 | Completion milestone | "Before this lab I could not ___. Now I can ___." with proof. |

## Explain it — answer in your own words
Why does ONNX portability not guarantee execution on every accelerator? What is the difference between batch latency and throughput? Why is TensorRT not mandatory here?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
CPU-only execution earns full credit; no TensorRT install required.

## Grading (70 points)
| Item | Points | Full credit requires |
|---|---:|---|
| Files | 8 | Notebook with outputs; three benchmark CSVs |
| 1. Parity | 12 | Checker passes; max difference and class agreement reported correctly; platform stated |
| 2. Benchmark | 20 | Prediction first (4); three thread settings × four batch sizes (8); table matches the CSVs (4); plots (4) |
| 3. Conclusion | 6 | Latency vs throughput choice supported by your numbers |
| 4. Runtime diagram | 6 | Complete path, accelerators correctly placed |
| 5. Project proposal | 6 | All five items |
| 6. Explain it | 9 | Three answers, 3 points each, using your own numbers |
| 7. Completion milestone | 3 | Both sentences, with proof |
| **Total** | **70** | |

Deductions: wrong file name or format −2; late −5 points per 24 hours. Numbers that do not match your submitted files earn 0 for that item. Running the starter unchanged earns 0 for the modification item.

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
