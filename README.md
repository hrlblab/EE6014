# ECE 6041 — GestureAI: Laptop-First Embedded AI

Over seven labs you will build **GestureAI**, a touchless computer-control app that recognizes hand gestures from your webcam. You collect your own gesture dataset, train a small PyTorch classifier, export it to ONNX, compress and benchmark it, and deploy it in a real-time browser app.

Everything runs on an ordinary laptop. **No Jetson, NVIDIA GPU, Docker, or paid cloud is required.**

```
webcam → MediaPipe hand landmarks (pretrained) → 42 features → your classifier (PyTorch → ONNX) → gesture → action
```

## The seven labs

Each lab spans two weeks. Detailed instructions and rubrics are in [`handouts/`](handouts/).

| Lab | Weeks | Topic | What you do |
|---|---|---|---|
| [1](handouts/Lab_01_GestureAI.md) | 1–2 | Sensor-to-Action Pipeline & PyTorch Foundations | Run the webcam app, see hand landmarks, measure detector latency and FPS |
| [2](handouts/Lab_02_GestureAI.md) | 3–4 | Data Acquisition, Sampling & Tiny AI Dataset | Capture four gestures in two sessions; study sampling rate and dataset quality |
| [3](handouts/Lab_03_GestureAI.md) | 5–6 | Train & Compare Embedded-Size Models | Train Tiny/Small/Large classifiers; compare accuracy, size, and CPU latency |
| [4](handouts/Lab_04_GestureAI.md) | 7–8 | Portable Deployment: ONNX & CPU | Export to ONNX, verify parity, benchmark PyTorch vs ONNX Runtime |
| [5](handouts/Lab_05_GestureAI.md) | 9–10 | Precision, Quantization & Compression | Build an INT8 model; compare accuracy, size, and latency |
| [6](handouts/Lab_06_GestureAI.md) | 11–12 | Real-Time Scheduling, Memory & MCU Feasibility | Tune stability and false triggers; estimate an MCU SRAM/Flash budget |
| [7](handouts/Lab_07_GestureAI.md) | 13–14 | Production-Like Edge AI Web App | Ship your model in the browser app with full telemetry and a live demo |

The four gestures are `open_palm`, `fist`, `peace`, and `thumbs_up`.

## Getting started

### 1. Get the code

```bash
git clone https://github.com/hrlblab/EE6014.git
cd EE6014
```

No Git? Click **Code → Download ZIP** on this page and unzip it. When the course updates the starter code, run `git pull` inside the folder to get the latest version.

### 2. Set up Python (Conda)

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), then open **Anaconda/Miniconda Prompt** (Windows) or a terminal (macOS/Linux) in this folder:

```bash
conda create -n gestureai python=3.11 -y
conda activate gestureai
python -m pip install --upgrade pip
python -m pip install -r shared/requirements.txt
python shared/check_environment.py
```

Every line of the check should print `PASS`.

Prefer plain Python with `venv`, or using Windows? **[SETUP_STUDENTS.md](SETUP_STUDENTS.md)** has the full step-by-step guide for every OS, both environment options, Jupyter setup, troubleshooting, and the Lab 1 setup checkpoint.

### 3. Open the GestureAI web app

```bash
conda activate gestureai
python -m http.server 8000
```

Open **http://localhost:8000/web/** in Chrome or Edge, click **Start Camera**, and allow camera access. You should see a hand skeleton over your hand. Press **Ctrl+C** in the terminal to stop the server.

Do not open `web/index.html` by double-clicking it: the camera only works over `http://localhost` or HTTPS.

## Repository layout

```
handouts/    Lab 1–7 instructions and rubrics
notebooks/   Jupyter starters: dataset inspection, training, ONNX, quantization, optimization
web/         Browser app: camera, hand landmarks, dataset capture, gesture control
web/models/  Put your trained gesture_model.onnx and classes.json here (from Lab 3 on)
shared/      requirements.txt, environment check, class list
```

Notebooks can run locally (VS Code or JupyterLab with the `gestureai` kernel) or on free Google Colab CPU. The webcam app always runs on your own laptop.

## Good to know

- **Privacy:** webcam frames are processed in your browser and never uploaded. The app does download its libraries and the pretrained hand model from the internet when it starts, so it needs a network connection.
- **Measure on your own machine:** FPS and latency depend on your laptop. Compare results on the same machine using the same method; there is no fixed target number.
- **Generative AI is encouraged** for the labs. Each lab asks for a short reflection on one AI suggestion and how you verified it. Quizzes and discussions follow the separate no-AI policy in the syllabus.
- **Problems?** See the troubleshooting table in [SETUP_STUDENTS.md](SETUP_STUDENTS.md#7-common-problems). When asking for help, include your OS, browser, and the full error text.
