# ECE 6041 — GestureAI: Laptop-First Embedded AI

Over seven labs you will build **GestureAI**, a touchless computer-control app that recognizes hand gestures from your webcam. You collect your own gesture dataset, train a small PyTorch classifier, export it to ONNX, compress and benchmark it, and deploy it in a real-time browser app.

Everything runs on an ordinary laptop. **No Jetson, NVIDIA GPU, Docker, or paid cloud is required.**

```
webcam → MediaPipe hand landmarks (pretrained) → 42 features → your classifier (PyTorch → ONNX) → gesture → action
```

## The seven labs

Each lab spans two weeks and lives in its own self-contained folder: open `LabN/` and its `README.md` has the instructions, rubric, the files you need, and how to start.

| Lab | Weeks | Topic | What you do |
|---|---|---|---|
| [1](Lab1/) | 1–2 | Sensor-to-Action Pipeline & PyTorch Foundations | Run the webcam app, see hand landmarks, measure detector latency and FPS |
| [2](Lab2/) | 3–4 | Data Acquisition, Sampling & Tiny AI Dataset | Capture four gestures in two sessions; study sampling rate and dataset quality |
| [3](Lab3/) | 5–6 | Train & Compare Embedded-Size Models | Train Tiny/Small/Large classifiers; compare accuracy, size, and CPU latency |
| [4](Lab4/) | 7–8 | Portable Deployment: ONNX & CPU | Export to ONNX, verify parity, benchmark PyTorch vs ONNX Runtime |
| [5](Lab5/) | 9–10 | Precision, Quantization & Compression | Build an INT8 model; compare accuracy, size, and latency |
| [6](Lab6/) | 11–12 | Real-Time Scheduling, Memory & MCU Feasibility | Tune stability and false triggers; estimate an MCU SRAM/Flash budget |
| [7](Lab7/) | 13–14 | Production-Like Edge AI Web App | Ship your model in the browser app with full telemetry and a live demo |

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
python -m pip install -r Lab1/requirements.txt
python Lab1/check_environment.py
```

Every line of the check should print `PASS`.

Prefer plain Python with `venv`, or using Windows? **[SETUP_STUDENTS.md](SETUP_STUDENTS.md)** has the full step-by-step guide for every OS, both environment options, Jupyter setup, troubleshooting, and the Lab 1 setup checkpoint.

### 3. Open the GestureAI web app

```bash
conda activate gestureai
python -m http.server 8000 -d Lab1
```

Open **http://localhost:8000/web/** in Chrome or Edge, click **Start Camera**, and allow camera access. You should see a hand skeleton over your hand. For Labs 2, 6, and 7, serve that lab's folder instead (`-d Lab2`, and so on). Press **Ctrl+C** in the terminal to stop the server.

Do not open `web/index.html` by double-clicking it: the camera only works over `http://localhost` or HTTPS.

## Repository layout

```
README.md          This overview
SETUP_STUDENTS.md  One-time Python setup for every OS (Conda or venv) and troubleshooting
Lab1/  README.md, web/ (webcam app), check_environment.py, requirements.txt
Lab2/  README.md, web/ (dataset capture), Lab02_Dataset_Inspection.ipynb, classes.json
Lab3/  README.md, Lab03_Train_Classifier.ipynb, Lab03_Embedded_Model_Design_Space.ipynb, classes.json
Lab4/  README.md, Lab04_ONNX_Verify_Benchmark.ipynb
Lab5/  README.md, Lab05_Quantization.ipynb
Lab6/  README.md, Lab06_Optimization_and_MCU.ipynb, web/ (scheduling experiments)
Lab7/  README.md, web/ (final app; put your model in web/models/)
```

Every lab folder also has its own `requirements.txt`. You only need **one** Python environment for the whole course.

Labs build on each other: keep the CSVs from Lab 2 and the model files from Lab 3 (`gesture_model.pth`, `gesture_model.onnx`, `classes.json`). Each lab's README lists exactly which earlier files it needs.

Notebooks can run locally (VS Code or JupyterLab with the `gestureai` kernel) or on free Google Colab CPU. The webcam app always runs on your own laptop.

## Good to know

- **Privacy:** webcam frames are processed in your browser and never uploaded. The app does download its libraries and the pretrained hand model from the internet when it starts, so it needs a network connection.
- **Measure on your own machine:** FPS and latency depend on your laptop. Compare results on the same machine using the same method; there is no fixed target number.
- **Generative AI is encouraged** for the labs. Each lab asks for a short reflection on one AI suggestion and how you verified it. Quizzes and discussions follow the separate no-AI policy in the syllabus.
- **Problems?** See the troubleshooting table in [SETUP_STUDENTS.md](SETUP_STUDENTS.md#7-common-problems). When asking for help, include your OS, browser, and the full error text.
