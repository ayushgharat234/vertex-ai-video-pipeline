# Real-Time Exercise Recognition Using Pose Estimation

## Project Overview
This project focuses on real-time exercise recognition from video data using webcam input. The primary goal is to identify and fine-tune the best-performing AI model capable of recognizing physical exercises from 3D human pose landmarks extracted with Google's MediaPipe. The complete pipeline—from data preprocessing to model training and deployment—was executed on Google Cloud Platform due to limited compute and storage 
requirements, availability on local hardware.

---

## Google Cloud Integration
Due to the privacy-sensitive nature of the dataset and the demand for high-performance compute, all heavy workloads were offloaded to Google Cloud. A GCS (Google Cloud Storage) bucket with public access prevention was used to store intermediate and final datasets:

Processed Dataset Paths:
```arduino
gs://exercise-recognition-dataset/processed/final_pose_features_dataset.csv
gs://exercise-recognition-dataset/processed/full_pose_dataset.csv
gs://exercise-recognition-dataset/processed/normalized_pose3d_dataset.csv
gs://exercise-recognition-dataset/processed/pose3d_acceleration_features.csv
gs://exercise-recognition-dataset/processed/pose3d_distance_features.csv
gs://exercise-recognition-dataset/processed/pose3d_symmetry_features.csv
gs://exercise-recognition-dataset/processed/pose3d_velocity_features.csv
gs://exercise-recognition-dataset/processed/pose_joint_angles_dataset.csv
```

---

## Jupyter Workflow
All preprocessing, experimentation, and model development were conducted in the following notebooks:

1. `01_Data_Understanding_and_Preprocessing.ipynb`

2. `02_Feature_Engineering.ipynb`

3. `03_Preparing_the_Training_Dataset_1.ipynb`

4. `03_Preparing_the_Training_Dataset_2.ipynb`

5. `04_Deployment.ipynb`

6. `05_Fine_Tuning.ipynb`

---

## Preprocessing & Feature Engineering
MediaPipe Pose Estimation was used to extract 33 pose landmarks as 3D coordinates (x, y, z). Additional metadata like video_id and frame_id was also recorded. The dataset was enriched through:

- Normalization and scaling of coordinates.

- Joint angle computation using compute_angle_3d() for:

    - Elbows: (11, 13, 15), (12, 14, 16)

    - Shoulders: (13, 11, 23), (14, 12, 24)

    - Hips: (11, 23, 25), (12, 24, 26)

    - Knees: (23, 25, 27), (24, 26, 28)

- Symmetry features (e.g., upper/lower limb symmetry)

- Distance features (e.g., shoulder width, hip width)

- Temporal dynamics: compute_temporal_diff() for velocity and acceleration

---
## Model Architecture
The deep learning model used for classification was based on a Temporal Convolutional Network (TCN) followed by dense layers. Here's a summary of the architecture:
```scss
Layer (type)         Output Shape     Param #
-------------------------------------------------
Masking              (None, 34, 189)        0
TCN                  (None, 64)        298,688
Dropout              (None, 64)              0
Dense (128 units)    (None, 128)         8,320
Dropout              (None, 128)             0
Dense (8 units)      (None, 8)           1,032
-------------------------------------------------
```

``Training Split``: 80% Train, 10% Validation, 10% Test

---

## Model Performance
The model was trained and iteratively fine-tuned. The key metrics recorded are:

**1. Initial Training:**

  * Accuracy: 0.4268
  * Loss: 2.6957
  * Validation Accuracy: 0.4472

**2. Fine-Tuning Phase 1:**

  * Test Accuracy: 0.4600
  * Loss: 6.0527

**3. Fine-Tuning Phase 2:**

  * Test Accuracy: 0.5000
  * Loss: 8.6931

---

## Pose Landmarks Reference
MediaPipe Pose model returns 33 landmark points. These represent major joints and keypoints. Example indices:

11: LEFT_SHOULDER, 13: LEFT_ELBOW, 15: LEFT_WRIST

23: LEFT_HIP, 25: LEFT_KNEE, 27: LEFT_ANKLE

Refer to the full index list ![here](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

---

## Deployment
The final model was deployed as a real-time webcam inference system. The architecture allows live video feeds to be processed through MediaPipe in-browser or on edge devices, feeding extracted features into the model for activity prediction.

---

## Future Scope
- Implement model quantization for faster inference.

- Integrate TensorFlow Lite for mobile deployment.

- Incorporate multi-person recognition.

- Explore Transformer-based temporal models.

---

## Conclusion

The project successfully demonstrated the feasibility of using real-time pose estimation for exercise recognition. By leveraging Google Cloud's infrastructure, we efficiently handled the extensive computational requirements for processing and modeling. The integration of MediaPipe for pose extraction provided a robust foundation for feature engineering, leading to improved model performance through fine-tuning. Future enhancements, such as model quantization and mobile deployment, offer promising avenues for expanding the system's capabilities and accessibility. Overall, this work highlights the potential of AI-driven solutions in enhancing fitness and rehabilitation applications.

