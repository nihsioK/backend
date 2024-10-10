#! /usr/bin/env python
#! coding:utf-8

from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
from utils import get_CG, zoom  # Updated to include `get_CG` and `zoom`

# Load the waving/non-waving dataset
def load_waving_data(train_path="data/Custom/GT_train_1.pkl", test_path="data/Custom/GT_test_1.pkl"):
    """
    Load training and test data from pickle files and fit a LabelEncoder on the combined labels.
    """
    # Load the training and testing data from the pickle files
    with open(train_path, 'rb') as f:
        Train = pickle.load(f)
    with open(test_path, 'rb') as f:
        Test = pickle.load(f)

    # Label encoder: combine both training and test labels to fit
    combined_labels = Train['label'] + Test['label']  # Combine both lists
    le = preprocessing.LabelEncoder()
    le.fit(combined_labels)  # Fit on both training and test labels

    print("Loading Waving/Non-Waving Dataset")
    return Train, Test, le

# Configuration for the waving/non-waving dataset
class WavingConfig():
    def __init__(self):
        self.frame_l = 32  # Fixed length of frames to standardize input size
        self.joint_n = 15  # Number of joints (15 in our dataset)
        self.joint_d = 2   # Dimension of joints (x, y)
        self.clc_num = 2   # Binary classification (waving or non-waving)
        self.feat_d = 105  # Feature dimension (upper triangle distance matrix of 15 joints)
        self.filters = 64  # Number of filters for the network

# Generate dataset for the waving/non-waving action
def WavingDataGenerator(T, C, le):
    """
    Generates training data for the waving action dataset.
    T: dataset (Train/Test)
    C: configuration object
    le: LabelEncoder
    """
    X_0 = []  # Stores JCD features (upper triangle matrix of distances)
    X_1 = []  # Stores pose keypoints
    Y = []    # Stores labels (waving/non-waving)
    
    # Convert the labels using the LabelEncoder
    labels = le.transform(T['label'])

    # Iterate over all the pose data in the dataset
    for i in tqdm(range(len(T['pose']))):
        p = np.copy(T['pose'][i])  # Get the pose data for each frame (frame, joints, coords)
        
        # Standardize the pose data to have a fixed frame length (32 frames)
        p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
        
        # Compute the JCD (Joint-Centered Distance) matrix for the pose
        M = get_CG(p, C)  # (target_frame, (joint_num - 1) * joint_num / 2)

        # Append the pose and JCD features to their respective lists
        X_0.append(M)  # JCD features
        X_1.append(p)  # Pose keypoints
        Y.append(labels[i])  # Corresponding label

    # Convert the lists to numpy arrays
    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)

    return X_0, X_1, Y  # Return the generated dataset
