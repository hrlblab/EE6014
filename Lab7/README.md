# LAB 07 — Production-Like Edge AI Web App & Deployment Evidence

**ECE 6041: Embedded AI & Edge Computing**  
**Total:** 70 points | **Alignment:** CLO 4, 8, 9  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `web/` | The GestureAI app you finish and deploy |
| `web/models/` | Instructor reference model for the quick taste; replace it with your `gesture_model.onnx` and `classes.json` |
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

## First 20 minutes — quick taste
Open the starter UI, load supplied tested ONNX model and actuate one slide-control command within 20 minutes.

## Step-by-step instructions
Everything in Lab 7 runs **locally** in your browser. You need your verified `gesture_model.onnx` and `classes.json` (Labs 3–4) and your Lab 6 settings.

**Step 1. Try the app with the reference model.** Open a terminal in the `Lab7` folder:
- **Windows:** open the `Lab7` folder in File Explorer, click the address bar, type `cmd`, press Enter. (If you use Conda, open **Anaconda Prompt** instead and `cd` to the folder.)
- **macOS:** open Terminal, type `cd ` (with a space), drag the `Lab7` folder onto the window, press Enter.
- **Linux:** right-click inside the `Lab7` folder → **Open in Terminal**.

Then type these two commands, one at a time, and open **http://localhost:8000/web/** in Chrome or Edge:

```bash
conda activate gestureai
python -m http.server 8000
```

Click **Load Classifier** (the folder includes an instructor reference model), then **Start Camera**, and hold an open palm.
✅ The slide panel changes to **Slide 2**. No webcam: choose `Lab7/sample_data/gesture_demo_stream.csv` under *Replay a recorded feature CSV* and click **Replay CSV**.

**Step 2. Use your own model.** Copy **your** `gesture_model.onnx` and `classes.json` (from the same Lab 3 download) into `Lab7/web/models/`, replacing the reference files. Reload with **Ctrl+Shift+R** / **Cmd+Shift+R** and click **Load Classifier**.
✅ The status lists your classes. Test every gesture.

**Step 3. Bring in your Lab 6 changes.** Copy your Lab 6 edits (logging, `SKIP`, votes, cooldown) into `Lab7/web/app.js`. Keep a copy of the unchanged file for Step 8.

**Step 4. Complete the dashboard.** Open `Lab7/web/index.html` in a code editor (VS Code is free for Windows, macOS, and Linux). Press **Ctrl+F** (Windows/Linux) or **Cmd+F** (macOS) to find text. After saving, reload the browser with **Ctrl+Shift+R** (Windows/Linux) or **Cmd+Shift+R** (macOS). Find `<div>Confidence <strong id="confidence">—</strong></div>` and add after it `<div>Held-out accuracy <strong>NN% (Lab 3)</strong></div>` with your number. Command response time can come from the console (`response_ms`).

**Step 5. Profile the whole pipeline.** Fill a table: camera frame interval (1000 ÷ FPS), hand detector, classifier, decision-to-command (`response_ms`), UI FPS. Which stage limits the system?

**Step 6. Choose one product improvement (required modification)** and write your prediction first:
- **Debouncing / latch:** a held gesture fires only once until released.
- **UI feedback:** show progress while a gesture is being confirmed and when it fired.
- **Keyboard fallback:** arrow keys / P / Enter do the same as gestures.
- **Accessible controls:** screen-reader announcements and large focusable buttons.
- **Safe pause state:** a clear pause banner; ignored gestures say how to resume.

Implement it (an AI assistant may help, but you must be able to explain every line).

**Step 7. Usability test.** Use a fixed 20-command script. Condition A: without your improvement; condition B: with it. Do at least **10 trials per condition** (one trial = one command) and count correct, accidental, and missed commands, total time, and a 1–5 confidence rating. Alternate the order (A then B, then B then A).

**Step 8. Compare two settings.** Starter settings vs your final settings: false commands, repeats, and median response (from Lab 6 or rerun).

**Step 9. Find a failure case.** Try your hand half out of view, dim light, two hands, or fast changes between gestures. Document one: what you did, what happened (screenshot or console), why, and how to reduce it.

**Step 10. Check privacy and offline behaviour.** Open the browser developer tools with **F12** or **Ctrl+Shift+I** (Windows/Linux), **Cmd+Option+I** (macOS).
1. Click **Network**, start the app, wait until the camera runs, and click the **clear** button (🚫).
2. Use the app for 60 seconds. Count the new requests ✅ (should be **0**). Screenshot it.
3. Reload with the Network tab open and note which websites the page downloads from at startup.
4. Turn off Wi-Fi, reload with **Ctrl+Shift+R** / **Cmd+Shift+R**, and write down what happens.

**Step 11. Write the app README.** In `Lab7/README_APP.md`: what it does, how to run it, an architecture diagram, a table of model results (accuracy, size, latency from Labs 3–6), a privacy notice based on Step 10, and why you chose this model and these settings.

**Step 12. Record a 60–90 second demo video.** Show the skeleton, three or more gestures changing slides, the dashboard, your improvement, and the failure case. Hands only. **Windows:** Win+Alt+R (Xbox Game Bar) or Snipping Tool → Record. **macOS:** Cmd+Shift+5. **Linux:** Ctrl+Shift+Alt+R (GNOME) or OBS Studio.

**Optional: publish on HTTPS.** Create a public GitHub repository with your `Lab7` folder, then **Settings → Pages → Deploy from a branch** (`main`, `/ (root)`). Open `https://<your-user>.github.io/<repo>/web/`. Everything in it becomes public.

**Step 13. Write the report and submit.** Write the parts listed in *What to submit* and export them as `LastName_VUID.pdf` (Word: **File → Save As → PDF**; Google Docs: **File → Download → PDF**). Make a folder named `LastName_VUID`, put the PDF and the files from *What to submit* in it, and zip it:
- **Windows:** right-click the folder → **Send to → Compressed (zipped) folder**.
- **macOS:** right-click the folder → **Compress**.
- **Linux:** right-click the folder → **Compress**, or run `zip -r LastName_VUID.zip LastName_VUID`.

Upload `LastName_VUID.zip` to the Lab 7 assignment on Brightspace.

## Required individual modification
**Choose and implement one product-level improvement (debouncing, UI feedback, keyboard fallback, accessible controls, safe pause state) and measure/usability-test it.** Running the instructor's code without modification cannot earn full credit.

## What to submit (Brightspace)
Upload **one ZIP file** named **`LastName_VUID.zip`** (for example `Huo_huoy1.zip`) containing:

| File | Contents |
|---|---|
| `LastName_VUID.pdf` | Your report: the 8 parts below. Write it in Word, Google Docs, or anything else, then export/print to PDF. Put screenshots inside it. |
| `app/` | Your whole `Lab7` folder: `web/` with your model, `sample_data/`, and `README_APP.md` (Step 11). |
| `LastName_VUID_demo.mp4` | Your 60–90 s demo video (Step 12). |
| `logs/` (`.txt` files) | Console logs and your usability-test tally. |

The PDF has 8 parts:

| # | Part | Include |
|---|---|---|
| 1 | Links | Live URL, if you published the app (optional). |
| 2 | Integration | Which Lab 3 model you used, its class order, parity evidence from Lab 4, and a screenshot of your model loaded. |
| 3 | Dashboard and profile | A dashboard screenshot, the Step 5 table, and the limiting stage. |
| 4 | Improvement and usability test | Your prediction (written first), what you built, the A/B table, and what it shows (no difference is an acceptable result if explained). |
| 5 | Settings and failure case | The Step 8 comparison and the Step 9 failure case with evidence and a fix. |
| 6 | Privacy and offline | Network screenshots, request count after start, startup websites, offline result. |
| 7 | Explain it | Your answers to the three questions below. |
| 8 | Completion milestone | "Before this lab I could not ___. Now I can ___." with proof. |

## Explain it — answer in your own words
Where does computation run? What are the complete system bottlenecks? What would change for phone, MCU, FPGA or NPU?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
Simulated feature-stream mode must remain usable without webcam; publish only consented footage or hand-only recordings.

## Grading (70 points)
| Item | Points | Full credit requires |
|---|---:|---|
| App files | 8 | Your own model in `web/models/`; the app runs with at least three gesture controls |
| README_APP.md | 3 | Diagram, model results table, privacy notice, justification |
| Demo video | 4 | 60–90 s, shows the items in Step 12 |
| 1–2. Links and integration | 5 | Own model, class order, parity evidence (the live URL is optional) |
| 3. Dashboard and profile | 7 | Held-out accuracy shown; all stages measured; limiting stage named |
| 4. Improvement and usability test | 20 | Prediction first (3); improvement works and is explained (6); at least 10 trials per condition with counts (8); honest interpretation (3) |
| 5. Settings and failure case | 6 | Two settings compared with numbers; one failure case with evidence and a fix |
| 6. Privacy and offline | 6 | 0 requests after start shown; startup websites listed; offline result explained |
| 7. Explain it | 9 | Three answers, 3 points each, using your own numbers |
| 8. Completion milestone | 2 | Both sentences, with proof |
| **Total** | **70** | |

Deductions: wrong file name or format −2; late −5 points per 24 hours. Numbers that do not match your submitted files earn 0 for that item. Running the starter unchanged earns 0 for the modification item.

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
