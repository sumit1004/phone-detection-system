AI-Based Mobile Phone Detection System with Audio Alert

An AI-powered real-time system that keeps the camera active, detects a mobile phone using computer vision, and plays a continuous warning sound until the phone is removed from the camera frame.

Project Overview

This project uses Artificial Intelligence (YOLO Object Detection) to monitor a live camera feed.
Whenever a mobile phone appears in front of the camera:

The phone is detected in real-time
A warning sound starts playing continuously
The sound stops immediately once the phone is removed

-The camera captures live video frames
-Each frame is sent to a YOLOv8 AI model
-The model checks if a mobile phone is present
-If detected → warning audio plays
-If removed → audio stops instantly

Technologies & Libraries Used
Technology	Purpose
-Python 3	Main programming language
-OpenCV (cv2)	Access webcam & display video
-YOLOv8 (Ultralytics)	AI object detection
-PyTorch	Runs the YOLO model
-pygame	Audio playback
-COCO Dataset	Pre-trained object classes (cell phone = class 67)

📁 Project Structure
```
Phone-Detection-System/
│
├── main.py              # Main Python application
├── warning.mp3          # Warning audio file
├── requirements.txt     # Required Python libraries
└── README.md            # Project documentation
```

Requirements
Before running this project, make sure you have:
Python 3.8 or higher
A working webcam
Internet (only for first-time model download)

Installation Guide
Clone the Repository
```
git clone https://github.com/your-username/phone-detection-system.git
cd phone-detection-system
```

Create Virtual Environment (Recommended)
```
python -m venv venv
```

Activate it:

Windows
```
venv\Scripts\activate
```

Mac / Linux
```
source venv/bin/activate
```

Install Required Libraries
```
pip install -r requirements.txt
```

How to Run the Project

Make sure:
warning.mp3 is present in the project folder
Camera is connected

Run the command:
```
python main.py
```
Controls
Key	Action
Q	Exit the application
📊 Detection Logic (In Simple Terms)
```
IF mobile phone detected:
    Play warning sound (loop)
ELSE:
    Stop warning sound
```

To avoid repeated sound restarts:
The system remembers the previous detection state
Audio plays only when state changes

Key Features
-Real-time phone detection
-Continuous audio alert
-Uses pre-trained AI (no training needed)
-Works on CPU (no GPU required)
-Beginner-friendly & well-structured
-Clean exit & resource management

Limitations
Accuracy depends on lighting condition
Very small or covered phones may not be detected
Webcam quality affects performance

Conclusion

This project demonstrates how AI + Computer Vision can be used for real-world monitoring systems.
It is ideal for college projects, hackathons, and AI demos.
