import pandas as pd
import numpy as np

# Load utility functions (I'm including them here for completeness - you said you have these)
def compute_angle_3d(a, b, c):
    """
    Computes the angle between three 3D points (a, b, c).
    """
    vec1 = a - b
    vec2 = c - b
    if np.allclose(vec1, np.zeros(3)) or np.allclose(vec2, np.zeros(3)):
        return 0.0
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos_theta = dot / (norm1 * norm2 + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)

def compute_pairwise_distance(a, b):
    """
    Computes the 3D Euclidean distance between two 3D points.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def normalize_and_scale_pose_data_realtime(keypoints):
    """
    Normalizes and scales pose data in real-time based on hip center and shoulder width,
    and computes joint angles.
    """
    if not keypoints or len(keypoints) != 33:
        print("Error: Invalid keypoints. Expected 33 keypoints.")
        return np.array([]), {}

    keypoints_np = np.array(keypoints)
    x_coords = keypoints_np[:, 0]
    y_coords = keypoints_np[:, 1]
    z_coords = keypoints_np[:, 2]

    hip_center_x = (x_coords[23] + x_coords[24]) / 2
    hip_center_y = (y_coords[23] + y_coords[24]) / 2
    hip_center_z = (z_coords[23] + z_coords[24]) / 2

    x_norm = x_coords - hip_center_x
    y_norm = y_coords - hip_center_y
    z_norm = z_coords - hip_center_z

    shoulder_width = np.sqrt(
        (x_coords[11] - x_coords[12]) ** 2 +
        (y_coords[11] - y_coords[12]) ** 2 +
        (z_coords[11] - z_coords[12]) ** 2
    )

    if shoulder_width == 0:
        print("Error: Shoulder width is zero. Cannot scale.")
        return np.array([]), {}

    x_scaled = x_norm / shoulder_width
    y_scaled = y_norm / shoulder_width
    z_scaled = z_norm / shoulder_width

    # Stack to preserve 33x3 shape instead of flattening
    scaled_pose_data = np.stack([x_scaled, y_scaled, z_scaled], axis=1)

    angle_features = {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26)
    }

    angle_data = {}
    for angle_name, (a_idx, b_idx, c_idx) in angle_features.items():
        a = scaled_pose_data[a_idx]
        b = scaled_pose_data[b_idx]
        c = scaled_pose_data[c_idx]
        angle = compute_angle_3d(a, b, c)
        angle_data[angle_name] = angle

    return scaled_pose_data, angle_data

def calculate_angles_and_distances_from_scaled_keypoints(scaled_keypoints):
    """
    Calculates joint angles and pairwise distances from scaled 3D keypoint coordinates.
    """
    if not isinstance(scaled_keypoints, np.ndarray) or scaled_keypoints.size != 99:
        print("Error: Invalid input. Expected a 1D numpy array of size 99.")
        return {}, {}
    keypoints_3d = scaled_keypoints.reshape(33, 3)
    angle_features = {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26)
    }
    angle_data = {}
    for angle_name, (a_idx, b_idx, c_idx) in angle_features.items():
        a = keypoints_3d[a_idx]
        b = keypoints_3d[b_idx]
        c = keypoints_3d[c_idx]
        angle = compute_angle_3d(a, b, c)
        angle_data[angle_name] = angle

    distance_features = {
        "shoulder_width": (11, 12),
        "hip_width": (23, 24),
        "upper_arm_left": (11, 13),
        "upper_arm_right": (12, 14),
        "lower_arm_left": (13, 15),
        "lower_arm_right": (14, 16),
        "upper_leg_left": (23, 25),
        "upper_leg_right": (24, 26),
        "lower_leg_left": (25, 27),
        "lower_leg_right": (26, 28),
    }
    distance_data = {}
    for distance_name, (i_idx, j_idx) in distance_features.items():
        a = keypoints_3d[i_idx]
        b = keypoints_3d[j_idx]
        distance = compute_pairwise_distance(a, b)
        distance_data[distance_name] = distance

    return angle_data, distance_data

def calculate_symmetry_features(df, distance_data):
    """
    Calculates symmetry features based on pairwise distances.
    """
    symmetry_features = pd.DataFrame()
    symmetry_features['frame_id'] = df['frame_id']
    symmetry_features['video_name'] = df['video_name']

    if not distance_data:
        print("Warning: Distance data is empty.  Returning empty DataFrame.")
        return symmetry_features

    required_keys = [
        'upper_arm_left', 'upper_arm_right', 'lower_arm_left', 'lower_arm_right',
        'upper_leg_left', 'upper_leg_right', 'lower_leg_left', 'lower_leg_right'
    ]
    if not all(key in distance_data for key in required_keys):
        print("Warning: Distance data is missing some required keys. Returning DataFrame with limited features.")
        for key in required_keys:
            if key in distance_data:
                if 'upper_arm_left' in distance_data and 'upper_arm_right' in distance_data:
                    symmetry_features['symmetry_upper_arm'] = np.abs(distance_data['upper_arm_left'] - distance_data['upper_arm_right'])
                if 'lower_arm_left' in distance_data and 'lower_arm_right' in distance_data:
                    symmetry_features['symmetry_lower_arm'] = np.abs(distance_data['lower_arm_left'] - distance_data['lower_arm_right'])
                if 'upper_leg_left' in distance_data and 'upper_leg_right' in distance_data:
                    symmetry_features['symmetry_upper_leg'] = np.abs(distance_data['upper_leg_left'] - distance_data['upper_leg_right'])
                if 'lower_leg_left' in distance_data and 'lower_leg_right' in distance_data:
                    symmetry_features['symmetry_lower_leg'] = np.abs(distance_data['lower_leg_left'] - distance_data['lower_leg_right'])
        return symmetry_features

    symmetry_features['symmetry_upper_arm'] = np.abs(distance_data['upper_arm_left'] - distance_data['upper_arm_right'])
    symmetry_features['symmetry_lower_arm'] = np.abs(distance_data['lower_arm_left'] - distance_data['lower_arm_right'])
    symmetry_features['symmetry_upper_leg'] = np.abs(distance_data['upper_leg_left'] - distance_data['upper_leg_right'])
    symmetry_features['symmetry_lower_leg'] = np.abs(distance_data['lower_leg_left'] - distance_data['lower_leg_right'])

    return symmetry_features

def compute_temporal_diff(dataframe, group_keys, order_col, step=1, prefix='vel'):
    """
    Computes temporal differences (velocity or acceleration) of pose keypoint coordinates.
    """
    temporal_df = dataframe.copy()
    new_cols = []
    for axis in ['x', 'y', 'z']:
        for i in range(33):
            col_name = f'{axis}{i}_scaled'
            if col_name in dataframe.columns:
                new_col = dataframe.groupby(group_keys)[col_name].diff(periods=step)
                new_col.name = f'{prefix}_{axis}{i}'
                new_cols.append(new_col)
            else:
                print(f"Warning: Column {col_name} not found in DataFrame.")
    temporal_df = pd.concat([temporal_df] + new_cols, axis=1)
    return temporal_df

def summarize_motion_features(df, prefix):
    """
    Computes summary statistics (norm, mean) of motion features (velocity or acceleration).
    """
    summary = pd.DataFrame()
    summary['frame_id'] = df['frame_id']
    summary['video_name'] = df['video_name']
    for i in range(33):
        v_x_col = f'{prefix}_x{i}'
        v_y_col = f'{prefix}_y{i}'
        v_z_col = f'{prefix}_z{i}'
        if v_x_col in df.columns and v_y_col in df.columns and v_z_col in df.columns:
            v_x = df[v_x_col]
            v_y = df[v_y_col]
            v_z = df[v_z_col]
            summary[f'{prefix}_norm_{i}'] = np.sqrt(v_x**2 + v_y**2 + v_z**2)
        else:
             print(f"Warning: Columns {v_x_col}, {v_y_col}, and {v_z_col} not found in DataFrame.")

    norm_cols = [col for col in summary.columns if f'{prefix}_norm_' in col]
    if norm_cols:
        summary[f'{prefix}_mean_norm'] = summary[norm_cols].mean(axis=1)
    else:
        print(f"Warning: No norm columns found for prefix '{prefix}'.")
    return summary

def prepare_single_feature_vector(
    current_keypoints: list,
    previous_scaled: list,
    previous_velocity: list,
    feature_order: list,
    video_name="live_feed",
    frame_id=0
) -> tuple:
    # 1. Normalize and scale current keypoints
    # scaled_vector is (33,3)
    scaled_vector, _ = normalize_and_scale_pose_data_realtime(current_keypoints)
    if scaled_vector.size == 0:
        # Return original history lists if keypoints are invalid to avoid issues with deque
        return None, previous_scaled, previous_velocity

    scaled_vector_flat = scaled_vector.flatten() # Shape (99,)

    # 2. Calculate angles and distances from the scaled keypoints
    angles, distances = calculate_angles_and_distances_from_scaled_keypoints(scaled_vector_flat)

    # 3. Calculate symmetry features from distances
    symmetry_df = calculate_symmetry_features(
        pd.DataFrame({'frame_id': [frame_id], 'video_name': [video_name]}),
        distances
    )
    # Ensure symmetry_df has expected columns, even if empty, to avoid errors
    expected_symmetry_cols = ['symmetry_upper_arm', 'symmetry_lower_arm', 'symmetry_upper_leg', 'symmetry_lower_leg']
    for col in expected_symmetry_cols:
        if col not in symmetry_df.columns:
            symmetry_df[col] = 0.0

    # 4. Compute velocity components
    all_scaled_frames_list = previous_scaled + [scaled_vector_flat]
    df_scaled_history = pd.DataFrame(all_scaled_frames_list, columns=[f'{axis}{i}_scaled' for axis in 'xyz' for i in range(33)])

    if len(df_scaled_history) > 1:
        current_velocity_components_df = df_scaled_history.diff().iloc[-1].to_frame().T
    else: # First frame, velocity is zero
        current_velocity_components_df = pd.DataFrame(np.zeros((1, 99)), columns=df_scaled_history.columns)
    current_velocity_components_df = current_velocity_components_df.fillna(0)

    # 5. Compute acceleration components
    all_velocity_frames_list = previous_velocity + [current_velocity_components_df]
    
    if all_velocity_frames_list:
        df_velocity_history = pd.concat(all_velocity_frames_list, ignore_index=True)
        if len(df_velocity_history) > 1:
            current_acceleration_components_df = df_velocity_history.diff().iloc[-1].to_frame().T
        else: # First velocity frame, acceleration is zero
            current_acceleration_components_df = pd.DataFrame(np.zeros((1, 99)), columns=df_velocity_history.columns)
    else: # Should not be reached if current_velocity_components_df is always added
        current_acceleration_components_df = pd.DataFrame(np.zeros((1, 99)), columns=df_scaled_history.columns)
    current_acceleration_components_df = current_acceleration_components_df.fillna(0)

    # 6. Assemble the feature vector according to feature_order
    all_computed_features = {}

    # Add scaled keypoints (x0_scaled, y0_scaled, z0_scaled, ...)
    for i in range(33):
        all_computed_features[f'x{i}_scaled'] = scaled_vector[i, 0]
        all_computed_features[f'y{i}_scaled'] = scaled_vector[i, 1]
        all_computed_features[f'z{i}_scaled'] = scaled_vector[i, 2]

    # Add distances (shoulder_width, hip_width, ...)
    for name, value in distances.items():
        all_computed_features[name] = value

    # Add symmetry features (symmetry_upper_arm, ...)
    symmetry_values_dict = symmetry_df.iloc[0].to_dict()
    for name, value in symmetry_values_dict.items():
        if name in feature_order: # Only add if it's an expected feature
            all_computed_features[name] = value

    # Add angles (left_elbow, right_elbow, ...)
    for name, value in angles.items():
        all_computed_features[name] = value

    # Calculate and add velocity norms (vel_norm_0, ..., vel_mean_norm)
    vel_components_flat = current_velocity_components_df.values.flatten()
    if vel_components_flat.size == 99:
        vel_xyz = vel_components_flat.reshape(33, 3)
        vel_norms = np.linalg.norm(vel_xyz, axis=1)
        for i in range(33):
            all_computed_features[f'vel_norm_{i}'] = vel_norms[i]
        all_computed_features['vel_mean_norm'] = np.mean(vel_norms) if vel_norms.size > 0 else 0.0
    else: # Fallback if dimensions mismatch
        for i in range(33): all_computed_features[f'vel_norm_{i}'] = 0.0
        all_computed_features['vel_mean_norm'] = 0.0

    # Calculate and add acceleration norms (acc_norm_0, ..., acc_mean_norm)
    acc_components_flat = current_acceleration_components_df.values.flatten()
    if acc_components_flat.size == 99:
        acc_xyz = acc_components_flat.reshape(33, 3)
        acc_norms = np.linalg.norm(acc_xyz, axis=1)
        for i in range(33):
            all_computed_features[f'acc_norm_{i}'] = acc_norms[i]
        all_computed_features['acc_mean_norm'] = np.mean(acc_norms) if acc_norms.size > 0 else 0.0
    else: # Fallback
        for i in range(33): all_computed_features[f'acc_norm_{i}'] = 0.0
        all_computed_features['acc_mean_norm'] = 0.0

    # Construct the final feature vector in the specified order
    final_feature_vector = np.array([all_computed_features.get(f_name, 0.0) for f_name in feature_order])

    if final_feature_vector.shape[0] != len(feature_order):
        print(f"Critical Error: Final feature vector length {final_feature_vector.shape[0]} "
              f"does not match feature_order length {len(feature_order)}.")
        # Potentially return None or raise error, as this will break model prediction
        return None, all_scaled_frames_list, all_velocity_frames_list

    # 7. Return the final feature vector and the updated history lists for deque
    return final_feature_vector, all_scaled_frames_list, all_velocity_frames_list