# LAB 02 — Data Acquisition, Sampling & Tiny AI Dataset

**ECE 6041: Embedded AI & Edge Computing**  
**Total:** 70 points | **Alignment:** CLO 1, 4  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `web/` | Webcam app; use the **Capture dataset** panel to record gesture CSVs |
| `Lab02_Dataset_Inspection.ipynb` | Load your CSVs and inspect class balance and sessions |
| `classes.json` | Gesture class order: `open_palm`, `fist`, `peace`, `thumbs_up` |
| `requirements.txt` | Python packages |

**You need:** Nothing beyond your webcam.  
**You produce:** Two or more session CSVs (`gesture_<session>.csv`), one CSV per sampling interval, and a PDF report. **Keep the session CSVs; Labs 3–5 use them.**

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

## First 20 minutes — quick taste
Use page capture controls to save the first 10 landmark feature rows as CSV in 20 minutes.

## Step-by-step instructions

**Step 1. Start the app.** Open a terminal in the `Lab2` folder:
- **Windows:** open the `Lab2` folder in File Explorer, click the address bar, type `cmd`, press Enter. (If you use Conda, open **Anaconda Prompt** instead and `cd` to the folder.)
- **macOS:** open Terminal, type `cd ` (with a space), drag the `Lab2` folder onto the window, press Enter.
- **Linux:** right-click inside the `Lab2` folder → **Open in Terminal**.

Then type these two commands, one at a time:

```bash
conda activate gestureai
python -m http.server 8000
```

Open **http://localhost:8000/web/** in Chrome or Edge and click **Start Camera**. Leave the terminal open while you work.
✅ You should see a green skeleton on your hand.

**Step 2. Learn the four gestures.** Use the **same hand shape** for a gesture every time you record it:

| Gesture | Hand shape |
|---|---|
| Open palm | All five fingers straight and spread, palm toward the camera |
| Fist | All fingers curled, thumb folded across them (not sticking out) |
| Peace | Index and middle finger up in a **V**, the other fingers folded |
| Thumbs up | Fist with the thumb pointing up |

**Step 3. Record session 1.**
1. In the **Capture dataset (Lab 2)** panel, type `session1` in **Session ID**.
2. Choose **Open palm** in **Gesture**, click **Start Capture**, and hold the gesture for about **30 seconds**. While holding, slowly tilt your hand a little left and right, and move it a little closer and farther. Keep the whole hand in view.
3. Click **Stop Capture**. Repeat 2–3 for **Fist**, **Peace**, and **Thumbs up**.
4. Click **Download CSV**. The file `gesture_session1.csv` is saved in your **Downloads** folder.
5. Click **Clear Samples**.

✅ The counter grows by about 160 rows per gesture. Rows are saved only while the skeleton is on your hand.

**Step 4. Record session 2 on another day** (or at least another time with different lighting). Reload the page, type `session2` as Session ID, and repeat Step 3. You get `gesture_session2.csv`.

**Step 5. Check your dataset in Google Colab** (report part 1).
1. Go to https://colab.research.google.com and sign in with a Google account.
2. Click **File → Upload notebook** and choose `Lab02_Dataset_Inspection.ipynb` from the `Lab2` folder.
3. Click **Runtime → Run all**. When a **Choose Files** button appears, select both session CSVs.
4. Screenshot the printed counts, the bar chart, and the session × gesture table.
5. Click **+ Code** to add a new cell at the end, paste the code below, and press **Shift+Enter**. It draws one recorded hand. Screenshot it.

```python
import matplotlib.pyplot as plt
row = raw[raw.label == 'peace'].iloc[0]
xs = row[[f'f{i}' for i in range(0, 42, 2)]].to_numpy(float)
ys = row[[f'f{i}' for i in range(1, 42, 2)]].to_numpy(float)
plt.scatter(xs, ys)
for i in range(21):
    plt.annotate(str(i), (xs[i], ys[i]))
plt.gca().invert_yaxis(); plt.gca().set_aspect('equal')
plt.title('One peace sample (wrist at 0,0)'); plt.show()
```

✅ The dots look like a peace sign. If not, you recorded a different shape: record that gesture again.

**Step 6. Write your prediction first** (report part 2). In two or three sentences: if the app saves every frame instead of one row every 180 ms, how will rows per minute and file size change? Will the extra rows show new hand poses?

**Step 7. Find your frame rate.** Start the app (Step 1) and read **Pipeline FPS** on the page. "Every third frame" is about `3000 ÷ FPS` milliseconds (for example 100 at 30 FPS).

**Step 8. Change the interval.**
1. Open `Lab2/web/app.js` in a code editor (VS Code is free for Windows, macOS, and Linux).
2. Press **Ctrl+F** (Windows/Linux) or **Cmd+F** (macOS) and search for `Date.now()-window.lastSaved>180`.
3. Change `180` to the interval you want to test, and save the file.
4. In the browser, press **Ctrl+Shift+R** (Windows/Linux) or **Cmd+Shift+R** (macOS) to reload, then click **Start Camera**.

**Step 9. Record one CSV per interval.** Test at least three values: `180` (starter), `0` (every frame), and your "every third frame" value. For each value:
1. Type `rate` plus the value as Session ID (for example `rate0`).
2. Choose **Open palm**, click **Start Capture**, hold your hand **still** for **30 seconds** (use a phone timer), click **Stop Capture**.
3. Click **Download CSV** (for example `gesture_rate0.csv`), then **Clear Samples**.

Keep everything else the same each time: gesture, position, lighting, 30 seconds. When you finish, set the value back to `180`.

**Step 10. Compute the numbers** (report part 2). In the same Colab notebook, add a new cell, paste this code, run it, and select your `gesture_rate*.csv` files:

```python
import os, numpy as np, pandas as pd
uploaded = files.upload()
F = [f'f{i}' for i in range(42)]
for name in sorted(n for n in uploaded if n.startswith('gesture_rate')):
    d = pd.read_csv(name).sort_values('timestamp')
    secs = (d.timestamp.iloc[-1] - d.timestamp.iloc[0]) / 1000
    step = np.linalg.norm(np.diff(d[F].to_numpy(), axis=0), axis=1)
    print(f"{name}: {len(d)} rows in {secs:.0f} s = {len(d) / secs * 60:.0f} rows/min, "
          f"{os.path.getsize(name) / 1024 / secs * 60:.0f} KB/min, near-duplicate rows {np.mean(step < 0.05):.0%}")
```

Copy the numbers into your table. "Near-duplicate rows" is the share of rows almost identical to the one before. A high value means the extra rows are mostly copies.

**Step 11. Write the report and submit.** Write the four parts listed in *What to submit* and export them as `LastName_VUID.pdf`. Make a folder named `LastName_VUID`, put the PDF and all your CSVs in it, and zip it:
- **Windows:** right-click the folder → **Send to → Compressed (zipped) folder**.
- **macOS:** right-click the folder → **Compress**.
- **Linux:** right-click the folder → **Compress**, or run `zip -r LastName_VUID.zip LastName_VUID`.

Upload `LastName_VUID.zip` to the Lab 2 assignment on Brightspace. **Keep your session CSVs: you need them in Labs 3–5.**

## Required individual modification
**Change sampling interval and measure data rate versus gesture coverage.** Running the instructor's code without modification cannot earn full credit.

Where: in `web/app.js`, `Date.now()-window.lastSaved>180` saves at most one row every 180 ms. Try at least three values, for example `180` (starter), `0` (every frame), and about `3000 ÷ FPS` (every third frame). For each value, hold one gesture still for 30 seconds with its own Session ID (`rate180`, `rate0`, …) and download a separate CSV.

## What to submit (Brightspace)
Upload **one ZIP file** named **`LastName_VUID.zip`** (for example `Huo_huoy1.zip`) containing:

| File | Contents |
|---|---|
| `LastName_VUID.pdf` | Your report: the four parts below. Write it in Word, Google Docs, or anything else, then export/print to PDF. Put screenshots inside it. |
| `gesture_session1.csv`, `gesture_session2.csv` (and more sessions if you have them) | Your dataset, exactly as downloaded from the app. You will reuse these in Labs 3–5. |
| `gesture_rate*.csv` | One CSV per sampling interval from the modification (for example `gesture_rate180.csv`, `gesture_rate0.csv`, `gesture_rate100.csv`). |

The PDF has four parts:

| # | Part | Include |
|---|---|---|
| 1 | Dataset | When and where each session was recorded (date, lighting); a table of rows per gesture per session; a screenshot of the notebook output showing class counts and no missing or unknown labels; one plotted example sample. |
| 2 | Sampling experiment | Your prediction, written before running. A table with one row per interval: interval (ms), rows, capture time (s), rows per minute, KB per minute (file size ÷ minutes). Two or three sentences: what more rows cost, and whether they added new hand poses or near-copies. |
| 3 | Explain it | Your answers to the three questions below, in your own words. |
| 4 | Completion milestone | "Before this lab I could not ___. Now I can ___." with one screenshot or number as proof. |

## Explain it — answer in your own words
Why normalize wrist coordinates? Why split by session instead of neighboring frames? What does reduced sampling save?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Use instructor-provided deidentified 42-feature CSVs for collection analysis; no camera required.

## Grading (70 points)
| Item | Points | Full credit requires |
|---|---:|---|
| Session CSVs | 15 | At least two sessions; all four gestures in every session; at least 100 rows per gesture in total |
| 1. Dataset | 15 | Count table and notebook screenshot match the CSVs; the example plot looks like the gesture |
| 2. Sampling experiment | 20 | Prediction written first (4); at least three intervals, each with its own `gesture_rate*.csv` (8); rows/min and KB/min correct for those files (4); conclusion supported by your numbers (4) |
| 3. Explain it | 15 | Three answers, 5 points each, using your own data |
| 4. Completion milestone | 5 | Both sentences completed, with proof |
| **Total** | **70** | |

Deductions: wrong file name or format −2; late −5 points per 24 hours. Numbers that do not match your submitted CSVs earn 0 for that item.

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
