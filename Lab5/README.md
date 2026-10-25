# LAB 05 — Precision, Quantization & Compression Trade-offs

**ECE 6041: Embedded AI & Edge Computing**  
**Total:** 70 points | **Alignment:** CLO 4, 6  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `Lab05_Quantization.ipynb` | Dynamic INT8 quantization; size, accuracy, and latency comparison |
| `requirements.txt` | Python packages |

**You need:** `gesture_model.onnx` from Lab 3 and a held-out session CSV from Lab 2.  
**You produce:** `gesture_model_int8.onnx` and the FP32 vs INT8 comparison.

## How to start

First time? Set up Python once using [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). Then, with your environment active, run `python -m pip install -r requirements.txt` from this folder.

**Run the notebook** in Google Colab (upload the `.ipynb` and the input files when prompted), or locally in VS Code/JupyterLab with your `gestureai` kernel. The upload cells use `google.colab`; when running locally, replace them with a path to your files.

## Mission and deliverable
Create **FP32/INT8 comparisons across different classifier capacities**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick taste
Run provided ONNX Runtime quantization for Small model and display before/after bytes in 20 minutes.

## Step-by-step instructions
Lab 5 runs in **Google Colab** (or locally). You need your Lab 3 models and the Lab 2 test-session CSV.

**Step 1. Toy INT8 by hand.** Take the values `[-1.0, -0.2, 0.0, 0.5, 2.0]`.
- **Asymmetric uint8:** scale = (max − min) / 255; zero point = round(−min / scale); q = round(x / scale) + zero point.
- **Symmetric int8:** scale = max|x| / 127; zero point = 0; q = round(x / scale).

Compute both by hand (scale, zero point, q, and back = (q − zero point) × scale), then check in a Colab cell:

```python
import numpy as np
vals = np.array([-1.0, -0.2, 0.0, 0.5, 2.0])
a_scale = (vals.max() - vals.min()) / 255; a_zp = int(round(-vals.min() / a_scale))
qa = np.clip(np.round(vals / a_scale) + a_zp, 0, 255)
print('asym scale', a_scale, 'zero point', a_zp, 'q', qa, 'back', (qa - a_zp) * a_scale)
s_scale = np.abs(vals).max() / 127; qs = np.clip(np.round(vals / s_scale), -127, 127)
print('sym scale', s_scale, 'q', qs, 'back', qs * s_scale)
```

**Step 2. Quantize the Small model.** Use a Small `gesture_model.onnx` from Lab 3. Go to https://colab.research.google.com and sign in. Click **File → Upload notebook** and choose `Lab05_Quantization.ipynb`. Click **Runtime → Run all**. When a **Choose Files** button appears, select the files listed here.
- First **Choose Files**: `gesture_model.onnx` (the name must be exactly this). The next cell prints **FP32 bytes** and **INT8 bytes**.
- Second **Choose Files**: your test-session CSV. It prints FP32 and INT8 accuracy and how often they agree.
- The last cell has `MODEL_NAME='Small'`; it saves `lab5_Small_threads1.csv` and downloads `gesture_model_int8.onnx`. Rename the INT8 file `Small_int8.onnx`.

**Step 3. See how the weights were stored.** Add a cell at the end and run:

```python
import onnx
from onnx import numpy_helper
fp_init = {t.name: numpy_helper.to_array(t) for t in onnx.load('gesture_model.onnx').graph.initializer}
q_init = {t.name: numpy_helper.to_array(t) for t in onnx.load('gesture_model_int8.onnx').graph.initializer}
wname = [n for n, a in fp_init.items() if a.ndim == 2][0]
w = fp_init[wname]
print('stored scale', q_init[wname + '_scale'], ' zero point', q_init[wname + '_zero_point'])
print('max|w|/127  ', np.abs(w).max() / 127, '   (max-min)/255', (w.max() - w.min()) / 255)
```

Which of your two Step 1 formulas did ONNX Runtime use? To save a notebook with its outputs: in Colab, **File → Download → Download .ipynb**.

**Step 4. Write your prediction first.** For Tiny, Small, and Large: will INT8 make each file 4× smaller? Faster? Less accurate?

**Step 5. Export Tiny and Large.** Open `Lab3/Lab03_Embedded_Model_Design_Space.ipynb` in Colab, **Runtime → Run all**, select your session CSVs. In the last cell set `selected='Tiny'` and run it; it downloads `gesture_model.onnx`. Put it in a folder `tiny`. Do the same with `selected='Large'` (folder `large`) and `selected='Small'` (folder `small`), so all three come from one run. Keep the file name `gesture_model.onnx`.

**Step 6. Quantize all three.** For each folder, open `Lab05_Quantization.ipynb` again (or **Runtime → Disconnect and delete runtime** to start clean), set `MODEL_NAME` to `'Tiny'`, `'Small'`, or `'Large'`, run all, and upload that folder's `gesture_model.onnx` and the **same** test CSV. You get `lab5_Tiny_threads1.csv`, `lab5_Small_threads1.csv`, `lab5_Large_threads1.csv` and three INT8 files (rename them `Tiny_int8.onnx`, `Small_int8.onnx`, `Large_int8.onnx`).

**Step 7. Make the comparison plots.** In a Colab cell, run this and select the three `lab5_*.csv` files:

```python
from google.colab import files
import pandas as pd, matplotlib.pyplot as plt
files.upload()
df = pd.concat([pd.read_csv(f'lab5_{m}_threads1.csv') for m in ['Tiny', 'Small', 'Large']])
b1 = df[df.batch == 1]
print(b1[['model', 'variant', 'file_bytes', 'test_accuracy', 'agreement_with_fp32', 'median_ms']])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for (model, variant), r in b1.groupby(['model', 'variant']):
    ax[0].scatter(r.file_bytes, r.test_accuracy, label=f'{model} {variant}')
    ax[1].scatter(r.median_ms, r.test_accuracy, label=f'{model} {variant}')
ax[0].set(xscale='log', xlabel='file size (bytes)', ylabel='held-out accuracy')
ax[1].set(xlabel='batch-1 median latency (ms)'); ax[1].legend(fontsize=7); plt.tight_layout(); plt.show()
```

**Step 8. Write the selection memo.** Which model and precision would you deploy in the browser (Lab 7), and which on a small microcontroller with 256 KB of memory (Lab 6)? Use your numbers.

**Step 9. Write the report and submit.** Write the parts listed in *What to submit* and export them as `LastName_VUID.pdf` (Word: **File → Save As → PDF**; Google Docs: **File → Download → PDF**). Make a folder named `LastName_VUID`, put the PDF and the files from *What to submit* in it, and zip it:
- **Windows:** right-click the folder → **Send to → Compressed (zipped) folder**.
- **macOS:** right-click the folder → **Compress**.
- **Linux:** right-click the folder → **Compress**, or run `zip -r LastName_VUID.zip LastName_VUID`.

Upload `LastName_VUID.zip` to the Lab 5 assignment on Brightspace.

## Required individual modification
**Change model capacity before quantization and compare all other conditions consistently.** Running the instructor's code without modification cannot earn full credit.

## What to submit (Brightspace)
Upload **one ZIP file** named **`LastName_VUID.zip`** (for example `Huo_huoy1.zip`) containing:

| File | Contents |
|---|---|
| `LastName_VUID.pdf` | Your report: the 6 parts below. Write it in Word, Google Docs, or anything else, then export/print to PDF. Put screenshots inside it. |
| `Lab05_Quantization.ipynb` | The notebook with outputs from your Small run (Steps 2–3). |
| `lab5_Tiny_threads1.csv`, `lab5_Small_threads1.csv`, `lab5_Large_threads1.csv` | Your comparison results (Step 6). |
| `Tiny_int8.onnx`, `Small_int8.onnx`, `Large_int8.onnx` | Your quantized models. |

The PDF has 6 parts:

| # | Part | Include |
|---|---|---|
| 1 | Toy INT8 | Your hand calculation table for both schemes, the code check, and which formula ONNX Runtime used (Step 3). |
| 2 | Small: FP32 vs INT8 | File sizes, held-out accuracy, agreement, and latency. |
| 3 | Model-size comparison | Your prediction (written first); a table for Tiny/Small/Large × FP32/INT8 (bytes, size ratio, accuracy, agreement, latency); the two plots; why each file shrank by the factor it did and why INT8 was or was not faster. |
| 4 | Selection memo | Browser choice and microcontroller choice, with numbers. |
| 5 | Explain it | Your answers to the three questions below. |
| 6 | Completion milestone | "Before this lab I could not ___. Now I can ___." with proof. |

## Explain it — answer in your own words
Why is INT8 not always faster? Why is quantized ONNX file size different from weight-byte estimates? Can the full hand-landmark stage run on an MCU?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
CPU execution only; if a quantized operator is unsupported, report the incompatibility and use teacher-provided benchmark artifact to analyze.

## Grading (70 points)
| Item | Points | Full credit requires |
|---|---:|---|
| Files | 8 | Notebook with outputs; three result CSVs; three INT8 models |
| 1. Toy INT8 | 10 | Hand table correct (6); stored-scale check and formula identified (4) |
| 2. Small: FP32 vs INT8 | 10 | Sizes, accuracy on the test session, agreement, latency |
| 3. Model-size comparison | 20 | Prediction first (4); all three models with the same CSV and settings (8); plots (4); explanation of size and speed (4) |
| 4. Selection memo | 8 | Both choices justified with your numbers |
| 5. Explain it | 9 | Three answers, 3 points each, using your own numbers |
| 6. Completion milestone | 5 | Both sentences, with proof |
| **Total** | **70** | |

Deductions: wrong file name or format −2; late −5 points per 24 hours. Numbers that do not match your submitted files earn 0 for that item. Running the starter unchanged earns 0 for the modification item.

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
