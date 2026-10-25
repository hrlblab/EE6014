# LAB 01 — Sensor-to-Action Pipeline & PyTorch Foundations

**ECE 6041: Embedded AI & Edge Computing**  
**Total:** 70 points | **Alignment:** CLO 1, 4  
**Product:** GestureAI — Touchless Computer Control

## In this folder

| File | Purpose |
|---|---|
| `web/` | GestureAI webcam app (hand landmarks, timing telemetry) |
| `check_environment.py` | Verifies your Python packages and runs a tiny PyTorch inference |
| `requirements.txt` | Python packages for the course |

**You need:** Nothing; this is the first lab.  
**You produce:** One PDF report: setup, camera screenshot, pipeline diagram, answers to three questions, completion milestone.

## How to start

First time? Set up Python once using [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). Then, with your environment active, run `python -m pip install -r requirements.txt` from this folder.

**Check your environment:** `python check_environment.py`. Every line should print `PASS`.

**Run the webcam app** from this folder (`Lab1/`):

```bash
python -m http.server 8000
```

Then open **http://localhost:8000/web/** in Chrome or Edge and click **Start Camera**. Press **Ctrl+C** to stop the server.

## Mission and deliverable
Create **Live webcam landmarks and a tiny PyTorch inference demo**. Your product runs or is analyzed at the edge; hand landmark detection is pretrained, while the student-built gesture classifier is trained in PyTorch and exported to ONNX. A laptop/browser is a legitimate edge target, but is **not** equivalent to MCU deployment.

## First 20 minutes — quick taste
Open provided local webcam page, see 21 hand landmarks and raw pipeline timing within 20 minutes.

## Step-by-step instructions

**Step 1. Get the course code.** Open https://github.com/hrlblab/EE6014, click **Code → Download ZIP**, and unzip it somewhere you can find again (for example Documents). You get a folder with `Lab1` … `Lab7` inside. (If you know Git: `git clone https://github.com/hrlblab/EE6014.git`.)

**Step 2. Set up Python once.** Follow [`../SETUP_STUDENTS.md`](../SETUP_STUDENTS.md). This is a one-time setup for the whole course, not part of the Lab 1 grade.

**Step 3. Start the app.** Open a terminal in the `Lab1` folder:
- **Windows:** open the `Lab1` folder in File Explorer, click the address bar, type `cmd`, press Enter. (If you use Conda, open **Anaconda Prompt** instead and `cd` to the folder.)
- **macOS:** open Terminal, type `cd ` (with a space), drag the `Lab1` folder onto the window, press Enter.
- **Linux:** right-click inside the `Lab1` folder → **Open in Terminal**.

Then type these two commands, one at a time:

```bash
conda activate gestureai
python -m http.server 8000
```

Open **http://localhost:8000/web/** in Chrome or Edge, click **Start Camera**, and allow camera access. The first start downloads the hand model, so wait a few seconds. Leave the terminal open while you work; press **Ctrl+C** in it to stop the server when you are done.

✅ You should see a green skeleton with orange dots following your hand, and numbers next to **Landmarker** and **Pipeline FPS**.

If the camera does not start: allow the camera in the browser (click the lock icon left of the address), and in your system settings. **Windows:** Settings → Privacy & security → Camera. **macOS:** System Settings → Privacy & Security → Camera, then quit and reopen the browser. Close other apps that use the camera (Zoom, Teams, FaceTime).

**Step 4. Take the camera screenshot** (report part 2). Hold your hand in view so the skeleton, **Landmarker**, and **Pipeline FPS** are all visible. Your face does not need to be in the picture.
- **Windows:** Win+Shift+S. **macOS:** Cmd+Shift+4. **Linux:** Print Screen (or Shift+Print Screen to select an area).

**Step 5. Write down your setup** (report part 1).
- **Operating system and version.** Windows: Settings → System → About. macOS: Apple menu → About This Mac. Linux: Settings → About.
- **Browser and version:** type `chrome://version` (Chrome) or `edge://version` (Edge) in the address bar.
- **How you ran the app:** the commands from Step 3.

**Step 6. See what the page downloads.** Open the browser developer tools: **F12** or **Ctrl+Shift+I** (Windows/Linux), **Cmd+Option+I** (macOS). Click the **Network** tab. Reload the page with **Ctrl+Shift+R** (Windows/Linux) or **Cmd+Shift+R** (macOS) and click **Start Camera**. The list shows every file the page downloads and the website it comes from (for example `cdn.jsdelivr.net` and `storage.googleapis.com`). These files are downloaded at setup; after that, the work happens on your laptop.

**Step 7. Draw the pipeline diagram** (report part 3). Draw six boxes with arrows:

camera (sensing) → pre-processing → hand detector → gesture classifier → decision → action (change a slide)

Under each box, write **on your laptop** or **downloaded at setup** (or both), using what you saw in Step 6. The gesture classifier does not exist yet: you will train it in Lab 3, so mark it "not yet". Any tool is fine: paper and a phone photo, PowerPoint, Google Slides, or draw.io. Put the image in your report.

**Step 8. Observe the app for the three questions** (report part 4). With the app running, watch the **Landmarker** time (milliseconds per frame for the hand detector) and **Pipeline FPS** (frames finished per second). Note both numbers with your hand still, with your hand moving, and with no hand in view. Compare `1000 ÷ FPS` with the Landmarker time: they are related, but not the same thing. Use these observations in your answers to the questions in *Explain it* below.

**Step 9. Write the completion milestone** (report part 5): "Before this lab I could not ___. Now I can ___." Add a screenshot or a number that proves the second sentence.

**Step 10. Make the PDF and submit.** Write the five parts listed in *What to submit* in Word, Google Docs, or any editor. Save or export as PDF (Word: **File → Save As → PDF**; Google Docs: **File → Download → PDF**). Name it `LastName_VUID.pdf` (for example `Huo_huoy1.pdf`) and upload it to the Lab 1 assignment on Brightspace.

## What to submit (Brightspace)
Upload **one PDF file** named **`LastName_VUID.pdf`** (for example `Huo_huoy1.pdf`). Write it in Word, Google Docs, or anything else, then export/print to PDF. Put screenshots inside the PDF. No other files are needed.

The PDF has five parts:

| # | Part | Include |
|---|---|---|
| 1 | Setup | Your operating system, browser and version (`chrome://version` or `edge://version`), and how you ran the app (`python -m http.server 8000`). |
| 2 | Camera screenshot | The GestureAI page with the hand skeleton on your hand and the **Landmarker** and **Pipeline FPS** values visible (hand only; no face needed). |
| 3 | Pipeline diagram | Sensing → preprocessing → hand detector → classifier → decision → action, each box marked **on your laptop** or **downloaded at setup**. |
| 4 | Explain it | Your answers to the three questions below, in your own words. |
| 5 | Completion milestone | "Before this lab I could not ___. Now I can ___." with one screenshot or measurement as proof. |

## Explain it — answer in your own words
Why is landmark detection an inference task? Which parts run in the browser? What does FPS measure compared with one-stage latency?

## Generative AI policy
**Generative AI is encouraged.** Use an assistant to understand APIs, debug, propose experiments, or build front-end features. AI outputs and benchmark numbers are never evidence until you run/verify them. Do not upload identifiable webcam images without permission. Quizzes/discussions still follow the separate no-AI policy from the syllabus.

## Accessibility / hardware fallback
No camera: view provided recorded demo and analyze instructor-supplied stage timing CSV; explicitly label as reference data.

## Grading (70 points)
| Part | Points | Full credit requires |
|---|---:|---|
| 1. Setup | 10 | OS, browser and version, and how you ran the app |
| 2. Camera screenshot | 15 | Skeleton on your hand, Landmarker and Pipeline FPS visible, taken on your computer |
| 3. Pipeline diagram | 15 | All six stages in order, each correctly marked on-device or downloaded at setup |
| 4. Explain it | 24 | Three answers, 8 points each: correct, in your own words, and referring to what you saw in the app |
| 5. Completion milestone | 6 | Both sentences completed, with a screenshot or number as proof |
| **Total** | **70** | |

Deductions: wrong file name or not a PDF −2; late −5 points per 24 hours.

## Completion milestone
Before this lab I could not _____. Now I can _____. Include a screenshot or measurement that proves the second statement.
