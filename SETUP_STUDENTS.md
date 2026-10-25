# ECE 6041 — GestureAI Student Environment Setup

**Who:** Students using Windows 10/11, macOS, or Linux.  
**When:** Lab 1, Week 1, after the 20-minute browser quick taste.  
**Goal:** Open the GestureAI webcam app locally and run the course Python/ONNX notebooks. A Jetson, NVIDIA GPU, Docker, and paid Colab are **not required**.

## 0. Quick Taste — Open GestureAI first (before installing libraries)

1. Clone the course repository (`git clone https://github.com/hrlblab/EE6014.git`) or download it as a ZIP and unzip it, keeping its directory structure.
2. Open a terminal in the repository folder (the one containing `README.md`, `SETUP_STUDENTS.md`, and `Lab1/` … `Lab7/`).
3. If Python is already installed, run `python -m http.server 8000 -d Lab1`. On macOS/Linux, use `python3 -m http.server 8000 -d Lab1` if needed.
4. Open **http://localhost:8000/web/** in Chrome or Edge; click **Start Camera** and allow webcam access.
5. You should see hand landmarks when you put your hand in front of the camera. The app downloads pretrained browser models/scripts when online; webcam frames are processed locally in the browser.

**If Python is not installed:** ask the instructor for the hosted HTTPS preview link for the first 20-minute demo, or complete Step 1 below first. Do **not** open `Lab1/web/index.html` using a `file://` URL because module imports/camera access may fail.

> A working camera alone does not imply the student gesture classifier is already trained: it will be added in Labs 3–4.

## 1. Choose your Python environment

**Both methods are supported and equally accepted:**

- **Option A: Python 3.11 + `venv`** (Steps 1A, 2A). Best for students already using standard Python.
- **Option B: Miniconda/Anaconda + `conda`** (Steps 1B, 2B). Best for students who already use Conda. Miniconda is sufficient; a full Anaconda installation is not required.

**Choose ONE environment method. Do not install the course requirements into the Conda `base` environment or mix a `.venv` activation with a Conda environment.** Both methods use `Lab1/requirements.txt` and the same environment check. Set up **one** environment for the whole course and reuse it in every lab; each `LabN/requirements.txt` lists the same packages.

## 1A. Python 3.11 installation (venv route)

Install Python **3.11.x** from https://www.python.org/downloads/ (or use a version already approved by your instructor). Verify installation:

**Windows PowerShell:**
```powershell
py -3.11 --version
```

**macOS/Linux Terminal:**
```bash
python3.11 --version
```

On Windows, if the `py` launcher is missing, use `python --version` after a standard installation. Ensure that the output is Python 3.11.x.

## 2A. Create a virtual environment (venv route; one time)

From the repository root:

**Windows PowerShell:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**macOS/Linux:**
```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
```

Using the full virtual-environment Python path avoids PowerShell execution-policy/activation problems. You do not have to activate the environment.

## 1B–2B. Conda route (instead of Steps 1A–2A)

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or use an existing supported Conda installation. Open **Anaconda Prompt / Miniconda Prompt** on Windows, or a Conda-initialized terminal on macOS/Linux. From the repository root:

```bash
conda create -n gestureai python=3.11 -y
conda activate gestureai
python --version
python -m pip install --upgrade pip
```

If `conda activate` fails, use the Miniconda Prompt (Windows) or initialize the terminal according to the Conda installation instructions. The environment should show **Python 3.11.x**.

## 3. Install the packages (one time)

**Windows PowerShell:**
```powershell
.\.venv\Scripts\python.exe -m pip install -r Lab1/requirements.txt
```

**macOS/Linux using venv:**
```bash
./.venv/bin/python -m pip install -r Lab1/requirements.txt
```

**Any OS using Conda (with `gestureai` activated):**
```bash
python -m pip install -r Lab1/requirements.txt
```

Install all course dependencies into the selected environment. Use Conda to manage the Python environment, and the supplied requirements file with pip for consistent package installation; do not also install separate copies of the same dependencies into that environment unless instructed.

The course requirements include PyTorch, NumPy, pandas, scikit-learn, matplotlib, ONNX, and ONNX Runtime CPU. A compatible CPU version of PyTorch is sufficient; do not install CUDA just for this course.

## 4. Check the environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\python.exe Lab1/check_environment.py
```

**macOS/Linux using venv:**
```bash
./.venv/bin/python Lab1/check_environment.py
```

**Any OS using Conda:**
```bash
conda activate gestureai
python Lab1/check_environment.py
```

Successful output shows `PASS` for all required Python imports and displays the PyTorch CPU inference tensor shape.

## 5. Run the browser application

With the environment installed, run from the repository root. `-d LabN` selects which lab's app to serve; Lab 1 is shown here:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\python.exe -m http.server 8000 -d Lab1
```

**macOS/Linux using venv:**
```bash
./.venv/bin/python -m http.server 8000 -d Lab1
```

**Any OS using Conda:**
```bash
conda activate gestureai
python -m http.server 8000 -d Lab1
```

Open http://localhost:8000/web/ in Chrome or Edge. For a later lab, replace `Lab1` with `Lab2`, `Lab6`, or `Lab7`. Leave the terminal open while using the page. Use **Ctrl+C** to stop the server.

The app runs camera/landmark processing in the browser. The `http.server` command only serves files from your laptop; it is **not** an AI inference cloud server.

## 6. Working with notebooks (Labs 2–6)

Each lab folder contains its own notebook(s); see that folder's `README.md` for which input files it needs from earlier labs.

**Option A: Google Colab (recommended when local setup fails).** Upload the notebooks and input CSVs to Colab. In a setup cell, install the packages necessary for each notebook as directed. Colab is for Python training/analysis, **not** the real-time browser camera/FPS measurement.

**Option B: Run Python locally.** Open the notebook in VS Code with the Python/Jupyter extensions, or install Jupyter into `.venv` and run it locally:

**Windows:**
```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyter lab
```

**macOS/Linux:**
```bash
./.venv/bin/python -m pip install jupyterlab
./.venv/bin/python -m jupyter lab
```

**If using Conda**, install and launch Jupyter within the activated `gestureai` environment, if needed:

```bash
conda activate gestureai
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name gestureai --display-name "Python (gestureai)"
python -m jupyter lab
```

In VS Code or Jupyter, select the Python interpreter/kernel corresponding to your chosen `.venv` or `gestureai` environment. For benchmarking, run comparisons on the **same machine** with the same input and measurement method.

## 7. Common problems

| Problem | What to try |
|---|---|
| `conda` not recognized | Open the Miniconda/Anaconda Prompt, verify Conda installation, or reopen your terminal |
| `conda activate gestureai` fails | Verify `conda create` succeeded; use a Conda-initialized terminal |
| Wrong Python/kernel in VS Code | Select `gestureai` (Conda) or `.venv`, restart the notebook kernel and rerun environment check |
| `python`/`py` not recognized | Install Python 3.11; open a new terminal; use the OS-specific command above |
| PowerShell refuses activation | Use `.\\.venv\\Scripts\\python.exe` without activation |
| Port 8000 already in use | Stop another server or use port 8001 and open that URL |
| Camera permission denied | Allow camera for Chrome/Edge and check OS privacy settings |
| Blank camera from `file://` | Use `http://localhost:8000/web/` |
| Models/scripts cannot load | Confirm internet access, inspect browser console, and report the error; the starter uses external downloads |
| A Python package fails to install | Confirm Python version and OS/architecture; include full error text in support request |
| No webcam | In the web app, **Load Classifier**, then choose `Lab7/sample_data/gesture_demo_stream.csv` under *Replay a recorded feature CSV* and click **Replay CSV**; label reference measurements accurately |
| Notebook runs in Colab but webcam app does not | Browser and Python environments are separate: webcam app runs on your laptop |

## 8. Lab 1 environment checkpoint submission

Submit **one PDF page or screenshot set** showing:

- Python version (3.11.x) and selected environment (`venv` or Conda `gestureai`)
- `check_environment.py` output with all `PASS` lines
- Browser GestureAI page with hand landmarks visible (or documented fallback)
- Your operating system and browser version
- One sentence distinguishing **Python training**, **browser inference**, and **remote Colab compute**

Do not submit any personal/private webcam footage unless you choose to and have all necessary permissions. Generative AI help with troubleshooting is encouraged, but verify each suggested fix.
