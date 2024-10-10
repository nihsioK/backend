import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.nn import functional as F
from models.DDNet_Custom import DDNet_Waving  # Import your custom model architecture
from dataloader.custom_loader import WavingConfig, get_CG  # Import utilities for pose processing
import logging
from collections import deque

# Set up #logging
# #logging.basicConfig(level=#logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def preprocess_pose_data(pose_landmarks, config):
    """
    Preprocess the pose data to extract the joints and generate the required input for the model.
    """
    # Extract the x and y coordinates of the 15 key joints
    key_joints = [
        pose_landmarks.landmark[i] for i in [
            0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30
        ]
    ]

    # Create a pose array with the x, y coordinates of the 15 key joints
    pose = np.array([[joint.x, joint.y] for joint in key_joints])

    # Normalize the pose relative to the hip center and scale (same as in training)
    hip_center = (pose[7] + pose[8]) / 2  # Midpoint between hips (23 and 24)
    normalized_pose = pose - hip_center
    scale = np.linalg.norm(pose[1] - pose[7])  # Distance between shoulders and hips
    normalized_pose = normalized_pose / scale

    # Return the normalized pose (no padding here yet, just return the single frame)
    return normalized_pose


def classify_wave(pose_landmarks):
    """
    Classify if a person is waving or not based on pose landmarks.
    """
    global pose_window

    # Preprocess the current frame
    normalized_pose = preprocess_pose_data(pose_landmarks, config)

    # Add the current frame to the pose window
    pose_window.append(normalized_pose)

    # If we don't have enough frames, pad with zeros to fill the window
    if len(pose_window) < window_size:
        # Pad with zeros to create a full 32 frames
        padding_needed = window_size - len(pose_window)
        padding_poses = [np.zeros((15, 2)) for _ in range(padding_needed)]
        full_pose_stack = list(padding_poses) + list(pose_window)
    else:
        # We have enough frames, use the full pose window
        full_pose_stack = list(pose_window)

    # Convert the pose stack into a NumPy array for further processing
    pose_stack = np.stack(full_pose_stack, axis=0)  # Shape: (32, 15, 2)

    # Compute the Joint-Centered Distance (JCD) matrix for the stacked poses
    M = get_CG(pose_stack, config)

    # Convert to tensors and move to the device (GPU/CPU)
    M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 32, 105)
    P_tensor = torch.tensor(pose_stack, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 32, 15, 2)

    # Log the shape of tensors being passed into the model
    #logging.info(f"Shape of M tensor: {M_tensor.shape}")
    #logging.info(f"Shape of P tensor: {P_tensor.shape}")

    # Perform inference
    output = model(M_tensor, P_tensor)

    # Log the model's output before classification
    #logging.info(f"Model output: {output}")

    # Get the predicted class (waving or not waving)
    pred_class = torch.argmax(output, dim=1).item()

    #logging.info(f"Predicted class: {pred_class}")

    return "Waving" if pred_class == 1 else "Not Waving"