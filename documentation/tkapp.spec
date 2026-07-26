# -*- mode: python ; coding: utf-8 -*-
# SignBridge — PyInstaller spec (updated build)

import sys
from pathlib import Path

BASE = Path(r'C:\Users\Asus\Downloads\ASL-HG American Sign Language Hand Gesture Image D\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000')

a = Analysis(
    [str(BASE / 'tkapp.py')],
    pathex=[str(BASE)],
    binaries=[],
    datas=[
        (str(BASE / 'models'),                  'models'),
        (str(BASE / 'utils'),                   'utils'),
        (str(BASE / 'app'),                     'app'),
        (str(BASE / 'phrase_mapping.py'),       '.'),
        (str(BASE / 'gesture_mouse_standalone.py'), '.'),
    ],
    hiddenimports=[
        # GUI / system
        'tkinter', 'tkinter.ttk',
        # Vision
        'cv2', 'mediapipe', 'PIL', 'PIL.Image', 'PIL.ImageTk',
        # ML
        'tensorflow', 'sklearn', 'sklearn.preprocessing',
        # Audio / TTS
        'pyttsx3', 'pyttsx3.drivers', 'pyttsx3.drivers.sapi5',
        'gtts', 'pygame', 'pygame.mixer',
        # Translation
        'deep_translator', 'deep_translator.google',
        # Gesture controls
        'pyautogui', 'pynput', 'pynput.mouse', 'pynput.keyboard',
        'screen_brightness_control',
        'pycaw', 'comtypes', 'comtypes.client',
        # Networking (used by gTTS + deep_translator)
        'requests', 'urllib3', 'certifi',
        # Other
        'numpy', 'threading', 'queue',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude training-only stuff to keep size down
        'matplotlib', 'IPython', 'jupyter', 'notebook',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SignBridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # No black console window on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SignBridge',
)
