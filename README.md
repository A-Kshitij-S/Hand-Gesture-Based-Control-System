```markdown
# SignBridge — Hand Gesture-Based System Control

A real-time hand-gesture control suite: ASL sign recognition with multi-language translation, gesture-based biometric login, gesture-driven OS control (mouse/volume/brightness), a gesture-controlled 3D voxel editor, a procedural 3D shape viewer, and a browser-based "headsetless VR" experiment — all powered by MediaPipe hand tracking and a custom-trained neural network.

---

## Features

### Core ASL Suite
*   **Sign Language Translator:** Real-time ASL letter recognition via webcam. Letters accumulate into words with hold-frame debouncing. Includes one-click translation into 15+ languages (Hindi, Spanish, French, Arabic, Chinese, Japanese, German, Portuguese, Russian, Korean, Italian, Bengali, Urdu, Tamil, Punjabi) with text-to-speech playback.
*   **Gesture Authentication:** Passwordless biometric login using cosine similarity over averaged MediaPipe landmark templates. Registration requires 10 sampled frames; login requires ≥90% similarity. No passwords or images are ever stored.
*   **Gesture to Word/Phrase:** Matches live gestures against a reference dataset (Hello, Thanks, Please, Yes, No, I Love You) and a 30+ phrase dictionary, featuring spoken output and session history.

### Gesture-Driven OS Control
*   **Gesture Mouse:** Index-fingertip cursor tracking, pinch-to-click (left/right), and finger-count scroll gestures, running as a standalone always-on-top window at ~30fps.
*   **Volume & Brightness Control:** Hand-distance-based system volume adjustment (via `pycaw`) and screen brightness adjustment (via `screen_brightness_control`).
*   **Gesture Whiteboard:** Draw and erase on a virtual canvas using hand tracking.

### 3D / Spatial Experiments
*   **Gesture Voxel Editor:** Place, erase, and rotate voxels in a 16x16x16 grid using pinch/fist/peace gestures. Rendered with a hand-built look-at camera projection (rotation matrices, perspective divide, depth-sorted face rendering) without an external 3D engine.
*   **Gesture 3D Shape Viewer:** OpenGL + GLSL shader-based viewer with 20+ procedural shapes (via `PyOpenGL` & `pyrr`). Rotate with your index finger, zoom with a pinch, and cycle shapes with thumb gestures.
*   **Headsetless VR (NXS-2087):** A browser-based WebXR-style experiment (`index.html` + local Python HTTP server) exploring headset-free spatial interaction via webcam.

---

## How It Works

**Inference Pipeline:**
```text
Webcam Frame
       ↓
MediaPipe Hands (21 landmarks × {x, y, z} = 63 features)
       ↓
Wrist-relative + scale-normalized transform (translation & scale invariance)
       ↓
Z-score normalization
       ↓
Dense Neural Network: 
Input(63) → Dense(256) → BatchNorm → Dropout → Dense(128) → BatchNorm → Dropout → Dense(64) → Dense(26, Softmax)
       ↓
Predicted Letter (A-Z) + Confidence → Word Builder / Translator / Auth / Phrase Engine

```

**Why landmarks instead of raw CNN on pixels?**

* **Efficiency:** ~2,400x smaller input (63 floats vs 150,528 pixels).
* **Robustness:** Invariant to lighting, background, skin tone, and hand position/scale.
* **Speed:** Trains in minutes on a standard CPU, achieving ≥93% validation accuracy across 26 classes.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| **Hand Detection** | MediaPipe Hands |
| **ML Model** | TensorFlow 2.15 / Keras |
| **Computer Vision** | OpenCV 4.9 |
| **Desktop UI** | Tkinter (native, no browser dependency) |
| **3D Rendering** | Custom NumPy projection pipeline (voxel editor) + PyOpenGL/GLSL (shape viewer) |
| **Translation** | deep-translator (Google) |
| **Authentication** | SQLite + cosine similarity |
| **Text-to-Speech** | pyttsx3 |
| **OS Control** | pyautogui, pycaw, screen_brightness_control, pynput |
| **Packaging** | PyInstaller |
| **Language** | Python 3.10+ |

---

## Project Structure

```text
Hand-Gesture-Based-System-Control/
├── codes/
│   ├── tkapp.py                          # Main desktop app - 13 pages, single entry point
│   ├── train_model.py                    # Model training (wrist-relative landmarks)
│   ├── merge_datasets.py                 # Merges multiple ASL datasets into one clean set
│   ├── build_app.py                      # PyInstaller packaging script
│   ├── gesture_mouse_standalone.py       # Mouse/volume/brightness control (launched as subprocess)
│   ├── gesture_3d_viewer_standalone.py   # OpenGL procedural 3D shape viewer
│   ├── voxel_editor_standalone.py        # Standalone voxel editor entry point
│   ├── voxel features/                   # hand_tracker, gesture_recognizer, voxel_engine, renderer
│   └── headsetless vr/                   # index.html + local server for the VR experiment
├── utils/
│   ├── model_utils.py                    # Model loading & real-time prediction
│   ├── auth_utils.py                     # Biometric authentication (SQLite + cosine similarity)
│   ├── translation_utils.py              # Multi-language translation
│   ├── phrase_utils.py                   # Phrase dictionary & word builder
│   ├── word_gesture_utils.py             # Gesture-to-word matching against reference images
│   ├── camera_utils.py                   # Shared webcam capture manager
│   └── shapes_3d.py                      # Procedural 3D shape generators
├── models/                               # Trained model + encoder + normalization params (generated)
├── dataset/                              # Reference gesture-to-word images
├── req/requirements.txt                  # Core Python dependencies
└── documentation/                        # PyInstaller .spec files

```

*(Note: `voxel_editor_standalone.py` may require path adjustments to correctly import from the `codes/voxel features/` directory depending on your execution environment).*

---

## Getting Started

### 1. Install Dependencies

Install the core requirements:

```bash
pip install -r req/requirements.txt

```

> **Platform Note (Windows OS Control & 3D Features):**
> * `gesture_mouse_standalone.py` (volume/brightness control) requires Windows-specific libraries: `pycaw`, `pynput`, `comtypes`, and `screen_brightness_control`.
> * The 3D shape viewer requires `PyOpenGL`, `PyOpenGL_accelerate`, and `pyrr`.
> * If utilizing these features, install them manually:
> `pip install pycaw pynput comtypes screen_brightness_control PyOpenGL PyOpenGL_accelerate pyrr`
> 
> 

### 2. Prepare the Dataset and Train the Model

Merge the raw datasets and train your local model:

```bash
python codes/merge_datasets.py   # Merges raw source datasets into merged_dataset/
python codes/train_model.py      # Trains the model (takes a few minutes on CPU)

```

*This generates `models/asl_model.h5`, `models/label_encoder.pkl`, `models/norm_mean.npy`, and `models/norm_std.npy`.*

### 3. Run the Application

Launch the main Tkinter interface:

```bash
python codes/tkapp.py

```

### 4. (Optional) Build a Standalone Executable

Package the application into a single-directory build:

```bash
python codes/build_app.py

```

*This packages `tkapp.py` via PyInstaller, bundling the `models/`, `utils/`, and `dataset/` directories.*

---

## Dataset

* **Classes:** 26 (A-Z)
* **Source:** Merged from multiple public ASL image datasets (raw, processed, dynamic, alpha variants), deduplicated, and capped per class for balance.
* **Processing:** The dataset trains solely on MediaPipe landmark extractions, **not** raw pixels, ensuring high efficiency and privacy.

---

## Security & Privacy

* **No webcam footage is stored to disk.**
* **No passwords are used or stored.**
* Only anonymous 63-float landmark vectors are stored in a local SQLite database for authentication matching.
* **Fully local execution** — no cloud API is required for core recognition. (Translation features require internet access but degrade gracefully when offline).

```

```
