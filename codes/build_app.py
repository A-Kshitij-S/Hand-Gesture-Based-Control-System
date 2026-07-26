import PyInstaller.__main__
import sys
import os

# Get absolute path for reliability
base_path = os.path.abspath(os.getcwd())

# Define the data additions
# Syntax: (source, destination)
data_files = [
    (os.path.join(base_path, 'data'), 'data'),
    (os.path.join(base_path, 'models'), 'models'),
    (os.path.join(base_path, 'utils'), 'utils'),
    (os.path.join(base_path, 'app'), 'app'),
    ('gesture_mouse_standalone.py', '.')
]

# Flatten for PyInstaller command line
data_args = []
for src, dst in data_files:
    if os.path.exists(src):
        data_args.extend(['--add-data', f'{src}{os.pathsep}{dst}'])

# Combine all arguments
args = [
    'tkapp.py',
    '--noconfirm',
    '--onedir',
    '--clean',
    '--hidden-import', 'pyautogui',
    '--hidden-import', 'mediapipe',
    '--hidden-import', 'screen_brightness_control',
    '--hidden-import', 'pycaw',
    '--hidden-import', 'pynput',
    '--hidden-import', 'comtypes'
] + data_args

print(f"Starting PyInstaller build...")
PyInstaller.__main__.run(args)
