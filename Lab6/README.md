# LAB 06 — Real-Time Scheduling, Memory & MCU Feasibility

**ECE 6041: Embedded AI & Edge Computing**  
**Total:** 70 points | **Alignment:** CLO 4, 7, 8  
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

## First 20 minutes — quick taste
See graph-optimization benchmark and live camera/classifier FPS telemetry within 20 minutes.

## Step-by-step instructions
The notebook runs in **Google Colab** (or locally); the scheduling experiments run **locally** in the webcam app. You need your `gesture_model.onnx` and `classes.json` from Lab 3.

**Step 1. Graph optimization.** Go to https://colab.research.google.com and sign in. Click **File → Upload notebook** and choose `Lab06_Optimization_and_MCU.ipynb`. Click **Runtime → Run all**. When a **Choose Files** button appears, select the files listed here. Select your `gesture_model.onnx`. One cell runs four optimization levels (disabled, basic, extended, all) and prints the time and the **operators** for each. Note which operators were merged (for example `Gemm` + `Relu` → `FusedGemm`). "No speedup" is an acceptable result. To save a notebook with its outputs: in Colab, **File → Download → Download .ipynb**.

**Step 2. Put your model in the app.** Copy `gesture_model.onnx` and `classes.json` into `Lab6/web/models/`. Open a terminal in the `Lab6` folder:
- **Windows:** open the `Lab6` folder in File Explorer, click the address bar, type `cmd`, press Enter. (If you use Conda, open **Anaconda Prompt** instead and `cd` to the folder.)
- **macOS:** open Terminal, type `cd ` (with a space), drag the `Lab6` folder onto the window, press Enter.
- **Linux:** right-click inside the `Lab6` folder → **Open in Terminal**.

Then type these two commands, one at a time, and open **http://localhost:8000/web/** in Chrome or Edge:

```bash
conda activate gestureai
python -m http.server 8000
```

Click **Load Classifier**, then **Start Camera**.
✅ The status lists your four classes and the page shows your gesture with a confidence.

**Step 3. Add logging to the app.** Open `Lab6/web/app.js` in a code editor (VS Code is free for Windows, macOS, and Linux). Press **Ctrl+F** (Windows/Linux) or **Cmd+F** (macOS) to find text. After saving, reload the browser with **Ctrl+Shift+R** (Windows/Linux) or **Cmd+Shift+R** (macOS).
1. Find `$('landmarkTime').textContent=(performance.now()-t0).toFixed(1)+' ms';` and paste right after it (same line): `console.log('landmarker_ms',(performance.now()-t0).toFixed(1));`
2. Find the decision line below and replace it with the second line. It works the same, but prints every command and its response time.

```
if(result.confidence>=.75){if(result.label===stable)streak++;else{stable=result.label;streak=1;}if(streak>=5 && Date.now()-lastActionAt>1600){act(result.label);lastActionAt=Date.now();}}
```

```
if(result.confidence>=.75){if(result.label===stable)streak++;else{stable=result.label;streak=1;window.streakStart=performance.now();}if(streak>=5 && Date.now()-lastActionAt>1600){act(result.label);lastActionAt=Date.now();console.log('COMMAND',result.label,'response_ms',Math.round(performance.now()-window.streakStart));}}
```

Open the browser developer tools with **F12** or **Ctrl+Shift+I** (Windows/Linux), **Cmd+Option+I** (macOS). Click the **Console** tab.
✅ You see `landmarker_ms` lines, and a `COMMAND … response_ms …` line each time a slide changes.

**Step 4. Measure each stage.** Fill a table: hand detector (median of 30 `landmarker_ms` lines), classifier (the **Classifier** box), Pipeline FPS, UI rendering FPS (developer tools → **⋮ → More tools → Rendering** → tick **Frame rendering stats**), and command response (`response_ms`). Which stage is the bottleneck?

**Step 5. Write your prediction first.** If the app processes only every 2nd or 3rd frame, or needs 3 or 8 agreeing frames instead of 5, how will false commands and response time change? What happens if you hold one gesture for 3 seconds?

**Step 6. The 60-second scripted trial.** Perform this script, holding each gesture about 3 s and dropping your hand about 1 s in between: **next, next, previous, confirm, next, pause, next (should be ignored), resume, previous, confirm, next, previous**. From the console, count: correct commands, extra (repeated) commands, wrong commands, missed commands, and the median `response_ms`. Do it first with the starter settings. To save the console: right-click in the Console → **Save as…**, one `.txt` file per setting.

No webcam? Use **Replay CSV** with `Lab7/sample_data/gesture_demo_stream.csv` (7 gestures: next, previous, confirm, pause, next-while-paused, resume, next).

**Step 7. Change the scheduling (required modification).**
1. **Frame skipping:** below the line `const CLASSES = [...]` add `const SKIP = 2;`. Then find `lastVideoTime=video.currentTime;const t0=performance.now();` and replace it with `lastVideoTime=video.currentTime;window.camFrame=(window.camFrame||0)+1;if(window.camFrame%SKIP!==0){requestAnimationFrame(loop);return;}const t0=performance.now();`. Pipeline FPS should drop to about half.
2. **Votes:** change `streak>=5` to `3` or `8`. **Cooldown:** change `>1600` (milliseconds).
3. Repeat the Step 6 trial for **at least four settings**, changing one thing at a time (for example: starter; SKIP 2; SKIP 3; votes 3; votes 8).

**Step 8. Memory budget for a microcontroller** (hypothetical 256 KB flash, 64 KB SRAM). Use the notebook's last cell. Fill a table: weights + biases (FP32: parameters × 4 bytes; INT8: weights × 1 byte + biases × 4 bytes), peak activations (largest layer input + output, × bytes), and **assumed** sizes for runtime code, firmware, and stack (write your source). Does the classifier fit? Does the hand detector (about 7.8 MB) fit?

**Step 9. Energy, partition, and memo.** Estimate energy per classification as E = P × t (measured time, **assumed** power; label every assumption). Draw which parts run where (camera, detector, classifier, decision). Write a one-page memo: problem, measurements, recommended setting, microcontroller feasibility, limitations.

**Step 10. Write the report and submit.** Write the parts listed in *What to submit* and export them as `LastName_VUID.pdf` (Word: **File → Save As → PDF**; Google Docs: **File → Download → PDF**). Make a folder named `LastName_VUID`, put the PDF and the files from *What to submit* in it, and zip it:
- **Windows:** right-click the folder → **Send to → Compressed (zipped) folder**.
- **macOS:** right-click the folder → **Compress**.
- **Linux:** right-click the folder → **Compress**, or run `zip -r LastName_VUID.zip LastName_VUID`.

Upload `LastName_VUID.zip` to the Lab 6 assignment on Brightspace.

## Required individual modification
**Change inference sampling or voting threshold; measure response-time / false-command trade-off.** Running the instructor's code without modification cannot earn full credit.

## What to submit (Brightspace)
Upload **one ZIP file** named **`LastName_VUID.zip`** (for example `Huo_huoy1.zip`) containing:

| File | Contents |
|---|---|
| `LastName_VUID.pdf` | Your report: the 7 parts below. Write it in Word, Google Docs, or anything else, then export/print to PDF. Put screenshots inside it. |
| `Lab06_Optimization_and_MCU.ipynb` | The notebook with its outputs. |
| `app.js` | Your modified `Lab6/web/app.js`. |
| `logs/` (`.txt` files) | Console logs from the scripted trials, one per setting (Step 6–7). |

The PDF has 7 parts:

| # | Part | Include |
|---|---|---|
| 1 | Graph optimization | A table of the four levels with time and operators; why there was or was not a speedup. |
| 2 | Stage timing | The Step 4 table and which stage is the bottleneck. |
| 3 | Scheduling experiment | Your prediction (written first); a table with one row per setting (frames processed, correct, repeats, wrong, missed, median response); your recommended setting; what you noticed while holding a gesture. |
| 4 | Memory budget | The Step 8 table with assumptions and sources; fits or not. |
| 5 | Energy, partition, memo | E = P × t with labeled assumptions; the diagram; the one-page memo. |
| 6 | Explain it | Your answers to the three questions below. |
| 7 | Completion milestone | "Before this lab I could not ___. Now I can ___." with proof. |

## Explain it — answer in your own words
Which stage is the bottleneck? What causes missed commands with skipping? Why does ONNX success not prove MCU deployment?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Recorded feature sequence and scripted predictions can replace camera for timing/action logic; estimated not measured energy.

## Grading (70 points)
| Item | Points | Full credit requires |
|---|---:|---|
| Files | 8 | Notebook with outputs; modified app.js; one console log per setting |
| 1. Graph optimization | 8 | All four levels with operators; explanation |
| 2. Stage timing | 8 | All five stages measured; bottleneck identified |
| 3. Scheduling experiment | 20 | Prediction first (3); frame skipping working, FPS halved (5); at least four settings with counts and response times from your logs (8); recommendation incl. the held-gesture behaviour (4) |
| 4. Memory budget | 8 | Correct weight and activation bytes; assumptions labeled with sources |
| 5. Energy, partition, memo | 6 | Assumptions labeled; diagram; memo with a recommendation |
| 6. Explain it | 9 | Three answers, 3 points each, using your own numbers |
| 7. Completion milestone | 3 | Both sentences, with proof |
| **Total** | **70** | |

Deductions: wrong file name or format −2; late −5 points per 24 hours. Numbers that do not match your submitted files earn 0 for that item. Running the starter unchanged earns 0 for the modification item.

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
