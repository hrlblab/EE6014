# LAB 03 — Train & Compare Embedded-Size Models

**ECE 6041: Embedded AI & Edge Computing**  
**Total:** 70 points | **Alignment:** CLO 1, 2, 3, 4  
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

## Step-by-step instructions
All of Lab 3 runs in **Google Colab** (no webcam needed). You need your Lab 2 session CSVs.

**Step 1. Gather your files.** Find `gesture_session1.csv`, `gesture_session2.csv` (and any other sessions) from Lab 2.

**Step 2. Run the training notebook.** Go to https://colab.research.google.com and sign in. Click **File → Upload notebook** and choose `Lab03_Train_Classifier.ipynb`. Click **Runtime → Run all**. When a **Choose Files** button appears, select the files listed here. Select **all** your session CSVs.

**Step 3. Check the split.** Find the printed lines `TRAIN SESSIONS` and `TEST SESSIONS`.
✅ They name **different** sessions: the model is tested on a session it never saw. If you see "Training split missing a gesture", a gesture is missing from a session; record it again (Lab 2).

**Step 4. Look at training.** Screenshot the training-loss plot (should go down) and the held-out accuracy plot, and note the printed **Held-out session accuracy**. Your number will differ from classmates' — any real number is fine.

**Step 5. Keep the model files.** The last cell prints a classification report and a confusion matrix, and downloads `gesture_model.pth`, `gesture_model.onnx`, and `classes.json`. Screenshot the confusion matrix. To save a notebook with its outputs: in Colab, **File → Download → Download .ipynb**.

**Step 6. Count the parameters by hand.** For 42 → 64 → 32 → 4, each layer has (inputs × outputs) weights plus (outputs) biases. Add them up, then multiply by 4 bytes (FP32) for the model size.

**Step 7. Write your predictions first.** (a) Will a random row split give higher accuracy than the session split, and why? (b) Will the Large model beat Small on the held-out session?

**Step 8. Leakage check.** In the training notebook click **+ Code** to add a cell at the end, paste this, and press **Shift+Enter**:

```python
from sklearn.model_selection import train_test_split
tr, te = train_test_split(np.arange(len(X)), test_size=0.25, random_state=7, stratify=y)
torch.manual_seed(7)
m = GestureMLP(hidden=64)
opt = torch.optim.Adam(m.parameters(), lr=0.001)
Xa, ya, Xb, yb = map(torch.from_numpy, (X[tr], y[tr], X[te], y[te]))
for epoch in range(40):
    m.train()
    for ix in torch.randperm(len(Xa)).split(64):
        loss = loss_fn(m(Xa[ix]), ya[ix]); opt.zero_grad(); loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    print('Random-split accuracy :', (m(Xb).argmax(1) == yb).float().mean().item())
print('Session-split accuracy:', accuracies[-1])
```

It trains the same model on a random row split. Write both numbers down.

**Step 9. Compare model sizes (required modification).** Go to https://colab.research.google.com and sign in. Click **File → Upload notebook** and choose `Lab03_Embedded_Model_Design_Space.ipynb`. Click **Runtime → Run all**. When a **Choose Files** button appears, select the files listed here. Select your session CSVs. It trains **Tiny** (42→16→4), **Small** (42→64→32→4), and **Large** (42→256→128→4) the same way and prints a table: held-out accuracy, parameters, FP32 KB, and CPU latency.
- Download the table: click the **folder icon** on the left → right-click `gesture_design_space.csv` → **Download**. Rename it `gesture_design_space_seed7.csv`.
- Small datasets give noisy results, so repeat with two more seeds: find `torch.manual_seed(7)` in the notebook, change `7` to `8`, **Runtime → Run all** again, download and rename; then `9`.

**Step 10. Change one more setting.** In `Lab03_Train_Classifier.ipynb` find `hidden=64` (marked TODO), change it to `hidden=32`, and **Runtime → Run all**. Note the held-out accuracy.

**Step 11. Choose your model.** Pick Tiny, Small, or Large using held-out accuracy **and** size/latency. In the design-space notebook's last cell, set `selected='Small'` to your choice (`'Tiny'`, `'Small'`, or `'Large'`) and run that cell. It downloads `gesture_model.pth`, `gesture_model.onnx`, `classes.json` for your choice. **Keep these three files together: Labs 4–7 use them.** To save a notebook with its outputs: in Colab, **File → Download → Download .ipynb**.

**Step 12. Find your worst gesture.** In the confusion matrix, which true gesture is most often predicted as another one? Think about how you performed those two gestures in each session (finger shape, thumb, rotation) and propose what to re-record.

**Step 13. Write the report and submit.** Write the parts listed in *What to submit* and export them as `LastName_VUID.pdf` (Word: **File → Save As → PDF**; Google Docs: **File → Download → PDF**). Make a folder named `LastName_VUID`, put the PDF and the files from *What to submit* in it, and zip it:
- **Windows:** right-click the folder → **Send to → Compressed (zipped) folder**.
- **macOS:** right-click the folder → **Compress**.
- **Linux:** right-click the folder → **Compress**, or run `zip -r LastName_VUID.zip LastName_VUID`.

Upload `LastName_VUID.zip` to the Lab 3 assignment on Brightspace.

## Required individual modification
**Compare at least two model capacities; submit a justified choice using both accuracy and resources.** Running the instructor's code without modification cannot earn full credit.

## What to submit (Brightspace)
Upload **one ZIP file** named **`LastName_VUID.zip`** (for example `Huo_huoy1.zip`) containing:

| File | Contents |
|---|---|
| `LastName_VUID.pdf` | Your report: the 6 parts below. Write it in Word, Google Docs, or anything else, then export/print to PDF. Put screenshots inside it. |
| `Lab03_Train_Classifier.ipynb`, `Lab03_Embedded_Model_Design_Space.ipynb` | Both notebooks with their outputs (Steps 2–10). |
| `gesture_design_space_seed7.csv`, `…seed8.csv`, `…seed9.csv` | The three design-space tables (Step 9). |
| `gesture_model.pth`, `gesture_model.onnx`, `classes.json` | Your chosen model, all three from the same run (Step 11). |

The PDF has 6 parts:

| # | Part | Include |
|---|---|---|
| 1 | Baseline | Train/test sessions, loss and accuracy plots, held-out session accuracy, confusion matrix, and your hand parameter count with FP32 size. |
| 2 | Leakage check | Your prediction (written first), random-split vs session-split accuracy, and why they differ. |
| 3 | Model-size comparison | Your prediction; a table for Tiny/Small/Large with accuracy for seeds 7, 8, 9 (and mean), parameters, FP32 KB, latency, and where it ran (Colab); the `hidden=32` result; which differences are real vs noise. |
| 4 | Model choice and worst gesture | Your chosen model with accuracy **and** resource reasons; the most confused gesture pair, the likely cause, and what you would re-record. |
| 5 | Explain it | Your answers to the three questions below. |
| 6 | Completion milestone | "Before this lab I could not ___. Now I can ___." with proof. |

## Explain it — answer in your own words
Did the largest model generalize best? Why is weight memory not peak runtime memory? What would be different for a CNN?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Use class-balanced reference CSV; GPU optional, CPU mandatory path.

## Grading (70 points)
| Item | Points | Full credit requires |
|---|---:|---|
| Files | 10 | Both notebooks with outputs; three design-space CSVs; model files from one run |
| 1. Baseline | 10 | Different train/test sessions; plots and confusion matrix; correct hand parameter count |
| 2. Leakage check | 10 | Prediction first (3); both accuracies (3); explanation using your data (4) |
| 3. Model-size comparison | 20 | Prediction first (4); Tiny/Small/Large × 3 seeds (8); `hidden=32` result (4); noise discussed and platform stated (4) |
| 4. Model choice and worst gesture | 8 | Choice justified by accuracy and resources (4); diagnosis with a concrete re-recording plan (4) |
| 5. Explain it | 9 | Three answers, 3 points each, using your own numbers |
| 6. Completion milestone | 3 | Both sentences, with proof |
| **Total** | **70** | |

Deductions: wrong file name or format −2; late −5 points per 24 hours. Numbers that do not match your submitted files earn 0 for that item. Running the starter unchanged earns 0 for the modification item.

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
