import cv2
import mediapipe as mp
import numpy as np
import joblib as jb
import tensorflow as tf
from collections import deque
from utils import prepare_single_feature_vector
from tcn import TCN

# Load the TCN model and label encoder
model = tf.keras.models.load_model('../models/model_fine_tuned_model_2.keras', custom_objects={'TCN': TCN})
encoder = jb.load('../models/model_label_encoder.joblib')

# Feature order
feature_order = [
    'x0_scaled', 'y0_scaled', 'z0_scaled', 'x1_scaled', 'y1_scaled', 'z1_scaled', 'x2_scaled', 'y2_scaled', 'z2_scaled', 'x3_scaled',
    'y3_scaled', 'z3_scaled', 'x4_scaled', 'y4_scaled', 'z4_scaled', 'x5_scaled', 'y5_scaled', 'z5_scaled', 'x6_scaled', 'y6_scaled',
    'z6_scaled', 'x7_scaled', 'y7_scaled', 'z7_scaled', 'x8_scaled', 'y8_scaled', 'z8_scaled', 'x9_scaled', 'y9_scaled', 'z9_scaled',
    'x10_scaled', 'y10_scaled', 'z10_scaled', 'x11_scaled', 'y11_scaled', 'z11_scaled', 'x12_scaled', 'y12_scaled', 'z12_scaled', 'x13_scaled',
    'y13_scaled', 'z13_scaled', 'x14_scaled', 'y14_scaled', 'z14_scaled', 'x15_scaled', 'y15_scaled', 'z15_scaled', 'x16_scaled', 'y16_scaled',
    'z16_scaled', 'x17_scaled', 'y17_scaled', 'z17_scaled', 'x18_scaled', 'y18_scaled', 'z18_scaled', 'x19_scaled', 'y19_scaled', 'z19_scaled',
    'x20_scaled', 'y20_scaled', 'z20_scaled', 'x21_scaled', 'y21_scaled', 'z21_scaled', 'x22_scaled', 'y22_scaled', 'z22_scaled', 'x23_scaled',
    'y23_scaled', 'z23_scaled', 'x24_scaled', 'y24_scaled', 'z24_scaled', 'x25_scaled', 'y25_scaled', 'z25_scaled', 'x26_scaled', 'y26_scaled',
    'z26_scaled', 'x27_scaled', 'y27_scaled', 'z27_scaled', 'x28_scaled', 'y28_scaled', 'z28_scaled', 'x29_scaled', 'y29_scaled', 'z29_scaled',
    'x30_scaled', 'y30_scaled', 'z30_scaled', 'x31_scaled', 'y31_scaled', 'z31_scaled', 'x32_scaled', 'y32_scaled', 'z32_scaled',
    'shoulder_width', 'hip_width', 'upper_arm_left', 'upper_arm_right', 'lower_arm_left', 'lower_arm_right', 'upper_leg_left', 'upper_leg_right',
    'lower_leg_left', 'lower_leg_right', 'symmetry_upper_arm', 'symmetry_lower_arm', 'symmetry_upper_leg', 'symmetry_lower_leg',
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 'left_knee', 'right_knee', 'left_hip', 'right_hip',
    'vel_norm_0', 'vel_norm_1', 'vel_norm_2', 'vel_norm_3', 'vel_norm_4', 'vel_norm_5', 'vel_norm_6', 'vel_norm_7', 'vel_norm_8', 'vel_norm_9',
    'vel_norm_10', 'vel_norm_11', 'vel_norm_12', 'vel_norm_13', 'vel_norm_14', 'vel_norm_15', 'vel_norm_16', 'vel_norm_17', 'vel_norm_18', 'vel_norm_19',
    'vel_norm_20', 'vel_norm_21', 'vel_norm_22', 'vel_norm_23', 'vel_norm_24', 'vel_norm_25', 'vel_norm_26', 'vel_norm_27', 'vel_norm_28', 'vel_norm_29',
    'vel_norm_30', 'vel_norm_31', 'vel_norm_32', 'vel_mean_norm', 'acc_norm_0', 'acc_norm_1', 'acc_norm_2', 'acc_norm_3', 'acc_norm_4', 'acc_norm_5',
    'acc_norm_6', 'acc_norm_7', 'acc_norm_8', 'acc_norm_9', 'acc_norm_10', 'acc_norm_11', 'acc_norm_12', 'acc_norm_13', 'acc_norm_14', 'acc_norm_15',
    'acc_norm_16', 'acc_norm_17', 'acc_norm_18', 'acc_norm_19', 'acc_norm_20', 'acc_norm_21', 'acc_norm_22', 'acc_norm_23', 'acc_norm_24', 'acc_norm_25',
    'acc_norm_26', 'acc_norm_27', 'acc_norm_28', 'acc_norm_29', 'acc_norm_30', 'acc_norm_31', 'acc_norm_32', 'acc_mean_norm'
]

# MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buffers
previous_scaled = deque(maxlen=2)
previous_velocity = deque(maxlen=2)
sequence_buffer = deque(maxlen=30)

cap = cv2.VideoCapture(0)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        # Extract feature vector for current frame
        feature_vector, prev_scaled, prev_velocity = prepare_single_feature_vector(
            current_keypoints=keypoints,
            previous_scaled=list(previous_scaled),  # Convert deque to list
            previous_velocity=list(previous_velocity),  # Convert deque to list
            feature_order=feature_order,
            video_name="webcam",
            frame_id=frame_id
        )
        previous_scaled = deque(prev_scaled, maxlen=2)
        previous_velocity = deque(prev_velocity, maxlen=2)

        if feature_vector is not None:
            sequence_buffer.append(feature_vector)

        if len(sequence_buffer) == 30:
            input_sequence = np.array(list(sequence_buffer)).reshape(1, 30, -1)
            prediction = model.predict(input_sequence)
            predicted_class = encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction)

            cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Exercise Recognition - Deployment Mode', frame)
    frame_id += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()