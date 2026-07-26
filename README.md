Here's the README with the NSUT/ECE/BTP references and team section removed, and no emojis — purely technical.

```markdown
# SignBridge — Hand Gesture-Based System Control

A real-time hand-gesture control suite: ASL sign recognition with multi-language translation, gesture-based biometric login, gesture-driven OS control (mouse/volume/brightness), a gesture-controlled 3D voxel editor, a procedural 3D shape viewer, and a browser-based "headsetless VR" experiment — all powered by MediaPipe hand tracking and a custom-trained neural network.

---

## Features

### Core ASL Suite
- **Sign Language Translator** — real-time ASL letter recognition via webcam; letters accumulate into words with hold-frame debouncing; one-click translation into 15+ languages (Hindi, Spanish, French, Arabic, Chinese, Japanese, German, Portuguese, Russian, Korean, Italian, Bengali, Urdu, Tamil, Punjabi) with text-to-speech playback.
- **Gesture Authentication** — passwordless biometric login using cosine similarity over averaged MediaPipe landmark templates (register with 10 sampled frames, login requires ≥90% similarity). No passwords or images are ever stored.
- **Gesture to Word/Phrase** — matches live gestures against a reference dataset (Hello, Thanks, Please, Yes, No, I Love You) and a 30+ phrase dictionary, with spoken output and session history.

### Gesture-Driven OS Control
- **Gesture Mouse** — index-fingertip cursor tracking, pinch-to-click (left/right), and finger-count scroll gestures, running as a standalone always-on-top window at ~30fps.
- **Volume & Brightness Control** — hand-distance-based system volume (via `pycaw`) and screen brightness (via `screen_brightness_control`) adjustment.
- **Gesture Whiteboard** — draw/erase on a virtual canvas using hand tracking.

### 3D / Spatial Experiments
- **Gesture Voxel Editor** — place, erase, and rotate voxels in a 16x16x16 grid using pinch/fist/peace gestures, rendered with a hand-built look-at camera projection (rotation matrices, perspective divide, depth-sorted face rendering) — no external 3D engine.
- **Gesture 3D Shape Viewer** — OpenGL + GLSL shader-based viewer with 20+ procedural shapes (via `PyOpenGL`/`pyrr`); rotate with your index finger, zoom with a pinch, cycle shapes with thumb gestures.
- **Headsetless VR (NXS-2087)** — a browser-based WebXR-style experiment (`index.html` + local Python HTTP server) exploring headset-free spatial interaction via webcam.

---

## How It Works

```
Webcam Frame
     |
     v
MediaPipe Hands  (21 landmarks x {x, y, z} = 63 features)
     |
     v
Wrist-relative + scale-normalized transform  (translation + scale invariance)
     |
     v
Z-score normalization
     |
     v
Dense Neural Network
Input(63) -> Dense(256) -> BatchNorm -> Dropout -> Dense(128) -> BatchNorm -> Dropout -> Dense(64) -> Dense(26, Softmax)
     |
     v
Predicted Letter (A-Z) + Confidence  ->  Word Builder / Translator / Auth / Phrase Engine
```

**Why landmarks instead of raw CNN on pixels?**
- ~2,400x smaller input (63 floats vs 150,528 pixels)
- Invariant to lighting, background, skin tone, hand position/scale
- Trains in minutes on CPU, >=93% validation accuracy across 26 classes

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand Detection | MediaPipe Hands |
| ML Model | TensorFlow 2.15 / Keras |
| Computer Vision | OpenCV 4.9 |
| Desktop UI | Tkinter (native, no browser dependency) |
| 3D Rendering | Custom NumPy projection pipeline (voxel editor) + PyOpenGL/GLSL (shape viewer) |
| Translation | deep-translator (Google) |
| Authentication | SQLite + cosine similarity |
| Text-to-Speech | pyttsx3 |
| OS Control | pyautogui, pycaw, screen_brightness_control, pynput |
| Packaging | PyInstaller |
| Language | Python 3.10+ |

---

## Project Structure

```
Hand-Gesture-Based-System-Control/
├── codes/
│   ├── tkapp.py                     # Main desktop app - 13 pages, single entry point
│   ├── train_model.py               # Model training (wrist-relative landmarks)
│   ├── merge_datasets.py            # Merges multiple ASL datasets into one clean set
│   ├── build_app.py                 # PyInstaller packaging script
│   ├── gesture_mouse_standalone.py  # Mouse/volume/brightness control (launched as subprocess)
│   ├── gesture_3d_viewer_standalone.py  # OpenGL procedural 3D shape viewer
│   ├── voxel_editor_standalone.py   # Standalone voxel editor entry point
│   ├── voxel features/              # hand_tracker, gesture_recognizer, voxel_engine, renderer
│   └── headsetless vr/              # index.html + local server for the VR experiment
├── utils/
│   ├── model_utils.py               # Model loading & real-time prediction
│   ├── auth_utils.py                # Biometric authentication (SQLite + cosine similarity)
│   ├── translation_utils.py         # Multi-language translation
│   ├── phrase_utils.py              # Phrase dictionary & word builder
│   ├── word_gesture_utils.py        # Gesture-to-word matching against reference images
│   ├── camera_utils.py              # Shared webcam capture manager
│   └── shapes_3d.py                 # Procedural 3D shape generators
├── models/                          # Trained model + encoder + normalization params (generated)
├── dataset/                         # Reference gesture-to-word images
├── req/requirements.txt             # Python dependencies
└── documentation/                   # PyInstaller .spec files
```

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r req/requirements.txt
```

Platform note: `gesture_mouse_standalone.py` (volume/brightness control) additionally requires `pycaw`, `pynput`, `comtypes`, and `screen_brightness_control` — these are Windows-only and not pinned in `requirements.txt`. The 3D shape viewer requires `PyOpenGL`, `PyOpenGL_accelerate`, and `pyrr`. Install these separately if you plan to use those features.

### 2. Prepare the dataset and train the model
```bash
python codes/merge_datasets.py   # merges raw source datasets into merged_dataset/
python codes/train_model.py      # trains the model (few minutes on CPU)
```
This generates `models/asl_model.h5`, `models/label_encoder.pkl`, `models/norm_mean.npy`, and `models/norm_std.npy`.

### 3. Run the app
```bash
python codes/tkapp.py
```

### 4. (Optional) Build a standalone executable
```bash
python codes/build_app.py
```
Packages `tkapp.py` into a onedir PyInstaller build, bundling `models/`, `utils/`, and `data/`.

---

## Dataset

- **Classes:** 26 (A-Z)
- **Source:** merged from multiple public ASL image datasets (raw, processed, dynamic, alpha variants), deduplicated and capped per class for balance
- **Processing:** MediaPipe landmark extraction, not raw pixels

---

## Security & Privacy

- No webcam footage stored to disk
- No passwords used or stored
- Only anonymous 63-float landmark vectors stored in local SQLite for authentication
- Fully local — no cloud API required for core recognition (translation requires internet, degrades gracefully offline)

---

*SignBridge — built with MediaPipe, TensorFlow, OpenCV and Tkinter*
```

Same two caveats from before still apply and are worth resolving in the actual code at some point:
- `voxel_editor_standalone.py` imports from a `voxel_editor` package, but the actual folder is `codes/voxel features/` — likely a broken import as-is.
- `req/requirements.txt` is missing `pycaw`, `pynput`, `comtypes`, `screen_brightness_control`, `PyOpenGL`, and `pyrr`, which are used by the gesture mouse/volume/brightness and 3D viewer features.

Let me know if you want this trimmed further (e.g., drop the 3D/VR experiments section) or written directly to a file — that would require switching to Agent mode.
