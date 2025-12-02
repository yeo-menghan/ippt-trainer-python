# Exercise Pose Detector 💪

A real-time push-up and sit-up detection system using computer vision and pose estimation. This application uses MediaPipe and OpenCV to track your exercise form, count repetitions, and provide instant feedback.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

- **Real-time Pose Detection**: Leverages MediaPipe's pose estimation for accurate body tracking
- **Automatic Rep Counting**: Intelligently counts push-ups and sit-ups based on body angles
- **Form Feedback**: Provides real-time feedback on exercise form and body alignment
- **Multi-Exercise Support**: Switch between push-ups and sit-ups on the fly
- **Visual Feedback**: Displays skeleton overlay with joint connections
- **Angle Tracking**: Shows real-time joint angles for detailed analysis

## 📋 Requirements

- Python 3.7 or higher
- Webcam or camera device
- Operating System: Windows, macOS, or Linux

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yeo-menghan/ippt-trainer-python.git
cd ippt-trainer-python
```

### 2. Sync Virtual Environment
```bash
pip install uv # if first time install
uv sync
source .venv/bin/activate
```

## 💻 Usage
Basic Usage
Run the script with default settings (push-up mode):

| Key    | Action | Description |
| -------- | ------- | ------- |
| P  | Push-Up Mode    | Switch to push-up detection and counting |
| S  | Sit-Up Mode     | Switch to sit-up detection and counting  |
| R  | Reset Counter   | Reset the current exercise counter to zero |
| Q  | Quit            | Close the application and release camera |

### Camera Setup Guide
General Requirements
- Distance: 1-1.5 metres from camera
- Lighting: Well-lit room (avoid backlighting)
- Background: Clear, uncluttered background
- Stability: Laptop should be stable (placed on the ground or on an elevated platform)
- Full Body Visibility: Entire body from head to feet should be in frame from the sideview (left / right)

