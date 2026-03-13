# posture-assessment
A computer vision project to detect sitting posture using keypoint detection.
Overview
This project develops a real-time posture monitoring system using computer vision and machine learning. The system detects a user's sitting posture using a webcam and classifies it as correct or incorrect. It helps users improve their sitting habits and avoid health issues caused by poor posture.
Technologies Used:
Python,
MediaPipe (pose estimation),
OpenCV (video processing),
Random Forest (machine learning model),
Scikit-learn, NumPy, Pandas.
How It Works:
Webcam captures live video.
MediaPipe extracts skeletal body landmarks.
Landmark coordinates are converted into features.
Random Forest classifier predicts posture type.
The system shows real-time posture feedback.
Features:Real-time posture detection
Works with a standard webcam,
Lightweight and CPU-friendly.
Provides instant visual feedback.
Performance:
The model achieved 96.9% accuracy in posture classification.
Applications:
Offices and workstations,
Educational environments,
Home study setups,
Ergonomic health monitoring.
