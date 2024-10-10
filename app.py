import cv2
import mediapipe as mp
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify
from models.DDNet_Custom import DDNet_Waving  # Import your custom model architecture
from dataloader.custom_loader import WavingConfig, get_CG  # Import utilities for pose processing
from collections import deque
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
# Enable CORS
CORS(app)

# Initialize MediaPipe Pose for pose detection
mp_drawing = mp.solutions.drawing_utils  # Drawing utility to render the skeleton
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = WavingConfig()
model = DDNet_Waving(config.frame_l, config.joint_n, config.joint_d, config.feat_d, config.filters, config.clc_num)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()
model.to(device)

# Initialize a deque to store the last `window_size` frames of poses
window_size = 32  # The same number of frames as used during training
pose_window = deque(maxlen=window_size)  # Sliding window to hold pose data

# Global variable for current wave status
current_wave_status = "Not Waving"

# Preprocess pose data
def preprocess_pose_data(pose_landmarks, config):
    key_joints = [
        pose_landmarks.landmark[i] for i in [
            0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30
        ]
    ]
    pose = np.array([[joint.x, joint.y] for joint in key_joints])
    hip_center = (pose[7] + pose[8]) / 2
    normalized_pose = pose - hip_center
    scale = np.linalg.norm(pose[1] - pose[7])
    normalized_pose = normalized_pose / scale
    return normalized_pose

# Classify wave
def classify_wave(pose_landmarks):
    global pose_window
    global current_wave_status

    normalized_pose = preprocess_pose_data(pose_landmarks, config)
    pose_window.append(normalized_pose)

    if len(pose_window) < window_size:
        padding_needed = window_size - len(pose_window)
        padding_poses = [np.zeros((15, 2)) for _ in range(padding_needed)]
        full_pose_stack = list(padding_poses) + list(pose_window)
    else:
        full_pose_stack = list(pose_window)

    pose_stack = np.stack(full_pose_stack, axis=0)
    M = get_CG(pose_stack, config)
    M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0).to(device)
    P_tensor = torch.tensor(pose_stack, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(M_tensor, P_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    current_wave_status = "Waving" if pred_class == 1 else "Not Waving"
    return current_wave_status

# Access webcam and start streaming to Flask
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            classify_wave(results.pose_landmarks)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame to stream in browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask route to render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask API endpoint to get the current wave status
@app.route('/wave_status')
def wave_status():
    print(current_wave_status)
    return jsonify({'status': current_wave_status})
    # return current_wave_status

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
