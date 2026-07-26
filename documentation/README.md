# SignBridge — ASL Hand Gesture Recognition System

> **NSUT BTP Project | ECE Branch | 8th Semester | Team of 4**

A real-time American Sign Language (ASL) recognition platform with multi-language translation, gesture-based authentication, and gesture-to-phrase conversion.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (one-time, ~5-10 minutes)
```bash
python train_model.py
```
This will create `models/asl_model.h5`, `models/label_encoder.pkl`, and normalization files.

### 3. Run the App
```bash
streamlit run app/main.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure
```
ASL_HG_36000/
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/
│   ├── asl_model.h5            # Trained Keras model (generated)
│   ├── label_encoder.pkl       # Class label encoder (generated)
│   ├── norm_mean.npy           # Normalization mean (generated)
│   ├── norm_std.npy            # Normalization std (generated)
│   └── gesture_auth.db         # SQLite auth database (generated)
├── utils/
│   ├── model_utils.py          # Model loading & prediction
│   ├── translation_utils.py    # Multi-language translation
│   ├── auth_utils.py           # Gesture authentication
│   └── phrase_utils.py         # Phrase dictionary & word builder
├── app/
│   ├── main.py                 # Streamlit home page
│   └── pages/
│       ├── 1_🤟_Sign_Translation.py
│       ├── 2_🔐_Gesture_Auth.py
│       ├── 3_💬_Gesture_To_Word.py
│       └── 4_📖_About.py
└── ASL_Raw_Images/
    └── asl_dataset/            # Dataset: 36 classes, ~36,000 images
        ├── A/, B/, ..., Z/
        └── 0/, 1/, ..., 9/
```

---

## ✨ Features

### 1. 🌐 Sign Language Translator
- Real-time ASL gesture recognition via webcam
- Letters are accumulated into words with hold-frame debouncing
- One-click translation to **15+ languages**: Hindi, Spanish, French, Arabic, Chinese, Japanese, German, Portuguese, Russian, Korean, Italian, Bengali, Urdu, Tamil, Punjabi
- Copy-to-clipboard support

### 2. 🔐 Hand Gesture Authentication
- **Passwordless biometric login**: no passwords, no images stored
- Based on cosine similarity of MediaPipe hand landmarks
- Register with 10 sampled frames → averaged biometric template stored in SQLite
- Login requires ≥90% cosine similarity match
- User management UI (list, delete users)

### 3. 💬 Gesture to Word / Phrase
- Spell words letter by letter — gestures accumulate into words
- **30+ phrase dictionary**: "HELLO", "GOOD MORNING", "THANK YOU", "I LOVE YOU", "HELP", etc.
- Text-to-speech: recognized words/phrases are spoken aloud via pyttsx3
- Session history panel showing all recognized words

---

## 🧠 How It Works

```
Webcam Frame
     │
     ▼
MediaPipe Hands
(21 landmarks × {x, y, z} = 63 features)
     │
     ▼
Normalize features (Z-score)
     │
     ▼
Dense Neural Network
Input(63) → Dense(512) → Dense(256) → Dense(128) → Dense(64) → Dense(36, Softmax)
     │
     ▼
Predicted Class (A-Z or 0-9) + Confidence
     │
     ▼
Word Builder / Translator / Auth Module
```

**Why landmarks instead of raw CNN on pixels?**
- 10,000× smaller input (63 vs 150,528 pixels)
- Invariant to lighting, background, skin tone
- Trains in ~5 minutes on CPU
- Achieves ≥93% validation accuracy

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Hand Detection | MediaPipe Hands |
| ML Model | TensorFlow 2.15 / Keras |
| Computer Vision | OpenCV 4.9 |
| Translation | deep-translator (Google) |
| Authentication | SQLite + cosine similarity |
| Text-to-Speech | pyttsx3 |
| Frontend | Streamlit 1.31 |
| Dataset | ASL-HG 36000 (36 classes) |
| Language | Python 3.10+ |

---

## 📊 Dataset

- **Name:** ASL-HG 36000
- **Classes:** 36 (A–Z + 0–9)
- **Images:** ~36,000 labeled JPG images
- **Source:** `ASL_Raw_Images/asl_dataset/`
- **Processing:** MediaPipe landmark extraction (not raw pixels)

---

## 🔒 Security & Privacy

- ✅ No webcam footage stored to disk
- ✅ No passwords used or stored  
- ✅ Only anonymous 63-float landmark vectors stored in local SQLite
- ✅ Fully local — no cloud API required for core recognition
- ✅ Translation uses Google API (requires internet) — gracefully degrades offline

---

## 📋 Requirements

- Python 3.10+
- Webcam (any standard USB or built-in camera)
- ~500MB RAM during inference
- No GPU required

---

## 👥 Team — NSUT ECE BTP 2025-26

| # | Name | Role |
|---|------|------|
| 1 | Member 1 | Model Training & ML Pipeline |
| 2 | Member 2 | Gesture Authentication & Security |
| 3 | Member 3 | Frontend & UI/UX Design |
| 4 | Member 4 | Translation & Phrase Engine |

---

*SignBridge · Built with MediaPipe, TensorFlow & Streamlit*
