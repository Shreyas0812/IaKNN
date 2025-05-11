import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from functools import partial
import os


class NGSIMDataProcessor:
    def __init__(self, data_path, obs_horizon=2, pred_horizon=5, dt=0.1, location=None):
        """
        Initialize NGSIM data processor

        Args:
            data_path: Path to NGSIM dataset CSV file
            obs_horizon: Observation horizon in seconds
            pred_horizon: Prediction horizon in seconds
            dt: Time step in seconds
            location: Specific location to filter (e.g., 'us-101', 'i-80')
        """
        self.data_path = data_path
        self.obs_frames = int(obs_horizon / dt)
        self.pred_frames = int(pred_horizon / dt)
        self.dt = dt
        self.location = location

    def load_data(self):
        """Load NGSIM data from CSV file"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)

        # Filter by location if specified
        if self.location is not None:
            self.data = self.data[self.data["Location"] == self.location]

        self.data = self.data.drop_duplicates(
            subset=["Vehicle_ID", "Frame_ID"], keep="first"
        )

        # Sort by Frame_ID and Vehicle_ID
        self.data = self.data.sort_values(["Frame_ID", "Vehicle_ID"])

        # Convert scalar velocity and acceleration to vector components
        # if "Direction" in self.data.columns:
        #     direction_rad = (
        #         self.data["Direction"] * np.pi / 180
        #         if self.data["Direction"].max() > 6.28
        #         else self.data["Direction"]
        #     )

        #     # Calculate x and y components
        #     self.data["v_x"] = self.data["v_Vel"] * np.cos(direction_rad)
        #     self.data["v_y"] = self.data["v_Vel"] * np.sin(direction_rad)
        #     self.data["a_x"] = self.data["v_Acc"] * np.cos(direction_rad)
        #     self.data["a_y"] = self.data["v_Acc"] * np.sin(direction_rad)
        # else:
            # If Direction is not available, use Movement or assume forward direction
            # self.data["v_x"] = self.data["v_Vel"]
            # self.data["v_y"] = 0.0
            # self.data["a_x"] = self.data["v_Acc"]
            # self.data["a_y"] = 0.0

        # for each frame, select n vehicles which are in a periphery of 8 meters of host vehicle and select the host vehicle randomly
        host_vehicles = self.data.groupby("Frame_ID")["Vehicle_ID"].apply(
            lambda x: x.sample(n=1, random_state=42)
        )
        host_vehicles = host_vehicles.reset_index()
        vehicles_in_periphery = []
        for _, row in host_vehicles.iterrows():
            frame_data = self.data[self.data["Frame_ID"] == row["Frame_ID"]]
            host_vehicle = frame_data[frame_data["Vehicle_ID"] == row["Vehicle_ID"]]
            if not host_vehicle.empty:
                host_x, host_y = (
                    host_vehicle["Local_X"].values[0],
                    host_vehicle["Local_Y"].values[0],
                )
                surrounding_vehicles = frame_data[
                    (frame_data["Local_X"] - host_x) ** 2
                    + (frame_data["Local_Y"] - host_y) ** 2
                    <= 8**2
                ]
                vehicles_in_periphery.append(surrounding_vehicles)
        


        vehicles_per_frame = self.data.groupby("Frame_ID")["Vehicle_ID"].nunique()
        avg_vehicles_per_frame = vehicles_per_frame.mean()
        print(f"Average vehicles per frame: {avg_vehicles_per_frame:.2f}")

    def _validate_scene_data(self, scene_data):
        """Validate that extracted scene data is complete and valid"""
        if scene_data is None:
            return False

        observations = scene_data["observations"]
        ground_truth = scene_data["ground_truth"]
        initial_state = scene_data["initial_state"]

        # Check shapes
        if len(observations) != self.obs_frames:
            return False
        if len(ground_truth) != self.pred_frames:
            return False
        if initial_state is None or len(initial_state) != 4:
            return False

        return True

    def _process_frame_batch(
        self, frame_batch, frame_groups, unique_frames, num_surrounding=5
    ):
        """
        Process a batch of start frames in parallel

        Args:
            frame_batch: List of start frame indices
            frame_groups: Grouped data by Frame_ID
            unique_frames: Sorted list of unique frame IDs
            num_surrounding: Number of surrounding vehicles to consider

        Returns:
            List of extracted scenes
        """
        batch_scenes = []

        for start_idx in frame_batch:
            if start_idx + self.obs_frames + self.pred_frames > len(unique_frames):
                continue

            scene_frames = unique_frames[
                start_idx : start_idx + self.obs_frames + self.pred_frames
            ]

            # Check if all frames exist and are consecutive
            frame_gaps = np.diff(scene_frames)
            if not all(gap == 1 for gap in frame_gaps):  # Assuming frame step is 1
                continue

            # Extract all vehicles present in observation frames
            vehicles_present = set()
            for frame in scene_frames[: self.obs_frames]:
                frame_data = frame_groups.get_group(frame)
                vehicles_present.update(frame_data["Vehicle_ID"].unique())

            # For each vehicle as potential host
            for vehicle_id in vehicles_present:
                # Check if vehicle is present in all frames (observation + prediction)
                vehicle_present_in_all = all(
                    vehicle_id in frame_groups.get_group(frame)["Vehicle_ID"].values
                    for frame in scene_frames
                )

                if not vehicle_present_in_all:
                    continue

                # Extract scene data
                scene_data = self.extract_vehicle_scene(
                    scene_frames, frame_groups, vehicle_id, num_surrounding
                )

                # Validate the extracted scene
                if self._validate_scene_data(scene_data):
                    batch_scenes.append(scene_data)

        return batch_scenes

    def extract_scenes(self, max_scenes=None, num_workers=None):
        """Extract traffic scenes for training in parallel"""
        if not hasattr(self, "data"):
            self.load_data()

        # Determine number of workers (default to CPU count)
        if num_workers is None:
            num_workers = mp.cpu_count()

        print(f"Extracting scenes using {num_workers} workers...")

        # Create frame groups (shared across workers)
        frame_groups = self.data.groupby("Frame_ID")
        unique_frames = sorted(self.data["Frame_ID"].unique())

        # Calculate total frames to process
        total_frames = len(unique_frames) - (self.obs_frames + self.pred_frames) + 1

        # If max_scenes specified, limit the number of frames to process
        if max_scenes is not None:
            # Rough estimate to ensure we get at least max_scenes
            est_scenes_per_frame = 2  # Estimated number of scenes per frame
            frames_to_process = min(
                total_frames, max_scenes // est_scenes_per_frame + 100
            )
        else:
            frames_to_process = total_frames

        # Create batches of frames for parallel processing
        batch_size = max(1, frames_to_process // (num_workers * 4))
        frame_batches = [
            list(range(i, min(i + batch_size, frames_to_process)))
            for i in range(0, frames_to_process, batch_size)
        ]

        # Process batches in parallel
        all_scenes = []

        # Use multiprocessing Pool only if more than one worker
        if num_workers > 1:
            process_batch_func = partial(
                self._process_frame_batch,
                frame_groups=frame_groups,
                unique_frames=unique_frames,
            )

            with mp.Pool(processes=num_workers) as pool:
                batch_results = pool.map(process_batch_func, frame_batches)

                # Flatten results
                for batch_result in batch_results:
                    all_scenes.extend(batch_result)

                    # Break early if max_scenes reached
                    if max_scenes is not None and len(all_scenes) >= max_scenes:
                        all_scenes = all_scenes[:max_scenes]
                        break
        else:
            # Single-threaded processing
            for batch in frame_batches:
                batch_scenes = self._process_frame_batch(
                    batch, frame_groups, unique_frames
                )
                all_scenes.extend(batch_scenes)

                # Break early if max_scenes reached
                if max_scenes is not None and len(all_scenes) >= max_scenes:
                    all_scenes = all_scenes[:max_scenes]
                    break

        print(f"Extracted {len(all_scenes)} scenes in total")
        return all_scenes

    def extract_vehicle_scene(self, frames, frame_groups, host_id, num_surrounding=5):
        """Extract scene data for a host vehicle and surrounding vehicles"""
        observations = []
        ground_truth = []
        initial_state = None

        for t, frame in enumerate(frames):
            frame_data = frame_groups.get_group(frame)

            # Get host vehicle data
            host_data = frame_data[frame_data["Vehicle_ID"] == host_id]
            if host_data.empty:
                return None

            host_pos = host_data[["Local_X", "Local_Y"]].values[0]
            host_vel = host_data[["v_x", "v_y"]].values[0]
            host_acc = host_data[["a_x", "a_y"]].values[0]
            host_dim = host_data[["v_Width", "v_length"]].values[0]

            # Store initial state (first observation frame)
            if t == 0:
                initial_state = np.concatenate([host_pos, host_vel])

            # Find surrounding vehicles
            surrounding_vehicles = []

            # First, check for preceding and following vehicles if available
            preceding_id = (
                host_data["Preceding"].values[0]
                if "Preceding" in host_data.columns
                else None
            )
            following_id = (
                host_data["Following"].values[0]
                if "Following" in host_data.columns
                else None
            )

            # Add preceding vehicle if it exists
            if (
                preceding_id is not None
                and preceding_id > 0
                and preceding_id in frame_data["Vehicle_ID"].values
            ):
                preceding_data = frame_data[frame_data["Vehicle_ID"] == preceding_id]
                surrounding_vehicles.append(preceding_data)

            # Add following vehicle if it exists
            if (
                following_id is not None
                and following_id > 0
                and following_id in frame_data["Vehicle_ID"].values
            ):
                following_data = frame_data[frame_data["Vehicle_ID"] == following_id]
                surrounding_vehicles.append(following_data)

            # Find other nearby vehicles based on distance
            other_vehicles = frame_data[frame_data["Vehicle_ID"] != host_id].copy()
            if len(other_vehicles) > 0:
                # Filter out vehicles already added (preceding and following)
                if preceding_id is not None and preceding_id > 0:
                    other_vehicles = other_vehicles[
                        other_vehicles["Vehicle_ID"] != preceding_id
                    ]
                if following_id is not None and following_id > 0:
                    other_vehicles = other_vehicles[
                        other_vehicles["Vehicle_ID"] != following_id
                    ]

                # Calculate distance
                other_vehicles["distance"] = np.sqrt(
                    (other_vehicles["Local_X"] - host_pos[0]) ** 2
                    + (other_vehicles["Local_Y"] - host_pos[1]) ** 2
                )

                # Add closest vehicles up to num_surrounding total
                remaining_slots = num_surrounding - len(surrounding_vehicles)
                if remaining_slots > 0 and len(other_vehicles) > 0:
                    closest = other_vehicles.nsmallest(remaining_slots, "distance")
                    surrounding_vehicles.extend(
                        [closest.iloc[[i]] for i in range(len(closest))]
                    )

            # Extract features
            features = self.extract_features(host_data, surrounding_vehicles)

            if t < self.obs_frames:
                observations.append(features)
            else:
                # Create ground truth data with positions and velocities
                gt_vehicle_data = []
                gt_vehicle_data.extend(
                    [host_pos[0], host_pos[1], host_vel[0], host_vel[1]]
                )

                for surr_vehicle in surrounding_vehicles:
                    if len(surr_vehicle) > 0:
                        veh_pos = surr_vehicle[["Local_X", "Local_Y"]].values[0]
                        veh_vel = surr_vehicle[["v_x", "v_y"]].values[0]
                        gt_vehicle_data.extend(
                            [veh_pos[0], veh_pos[1], veh_vel[0], veh_vel[1]]
                        )
                    else:
                        # Padding for missing vehicles
                        gt_vehicle_data.extend([0, 0, 0, 0])

                # Padding to ensure consistent size
                while len(gt_vehicle_data) < 4 * (num_surrounding + 1):
                    gt_vehicle_data.extend([0, 0, 0, 0])

                ground_truth.append(gt_vehicle_data)

        return {
            "observations": np.array(observations),
            "initial_state": initial_state,
            "ground_truth": np.array(ground_truth),
        }

    def extract_features(self, host, surrounding_vehicles):
        """Extract features for the interaction layer"""
        # Host features
        host_features = []
        host_features.extend(
            host[
                [
                    "Local_X",
                    "Local_Y",
                    "v_x",
                    "v_y",
                    "a_x",
                    "a_y",
                    "v_Width",
                    "v_length",
                ]
            ].values[0]
        )

        # Surrounding vehicles features
        surr_features = []
        for vehicle in surrounding_vehicles:
            if len(vehicle) > 0:
                # Get first row if multiple rows exist
                vehicle = (
                    vehicle.iloc[0] if isinstance(vehicle, pd.DataFrame) else vehicle
                )

                # Calculate relative positions and velocities
                rel_x = vehicle["Local_X"] - host["Local_X"].values[0]
                rel_y = vehicle["Local_Y"] - host["Local_Y"].values[0]
                rel_vx = vehicle["v_x"] - host["v_x"].values[0]
                rel_vy = vehicle["v_y"] - host["v_y"].values[0]

                # Calculate distance and repulsive force
                dist = np.sqrt(rel_x**2 + rel_y**2)
                rel_v = np.sqrt(rel_vx**2 + rel_vy**2)
                repulsive_force = np.exp((rel_v) * self.dt - dist)

                # Space headway if available
                space_headway = (
                    vehicle["Space_Headway"] if "Space_Headway" in vehicle else dist
                )

                # Time headway if available
                time_headway = (
                    vehicle["Time_Headway"]
                    if "Time_Headway" in vehicle
                    else (dist / max(rel_v, 0.1))
                )

                surr_features.append(
                    [
                        rel_x,
                        rel_y,
                        rel_vx,
                        rel_vy,
                        vehicle["a_x"],
                        vehicle["a_y"],
                        vehicle["v_Width"],
                        vehicle["v_length"],
                        dist,
                        repulsive_force,
                        space_headway,
                        time_headway,
                    ]
                )

        # Ensure fixed number of surrounding vehicles
        while len(surr_features) < 5:
            surr_features.append([0] * 12)  # Padding with zeros

        # Flatten and concatenate features
        return np.concatenate([host_features, np.array(surr_features).flatten()])


class NGSIMDataset(Dataset):
    def __init__(self, scenes):
        """Dataset wrapper for NGSIM scenes"""
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]

        # Convert to torch tensors
        observations = torch.FloatTensor(scene["observations"])
        initial_state = torch.FloatTensor(scene["initial_state"])
        ground_truth = torch.FloatTensor(scene["ground_truth"])

        return observations, initial_state, ground_truth


def create_dataloaders(
    data_path,
    batch_size=64,
    obs_horizon=2,
    pred_horizon=5,
    train_ratio=0.7,
    val_ratio=0.15,
    max_scenes=None,
    num_workers=None,
    location=None,
):
    """Create train, validation, and test data loaders"""
    # Process data
    processor = NGSIMDataProcessor(
        data_path, obs_horizon, pred_horizon, location=location
    )
    scenes = processor.extract_scenes(max_scenes, num_workers)

    # Split data
    num_scenes = len(scenes)
    num_train = int(num_scenes * train_ratio)
    num_val = int(num_scenes * val_ratio)

    # Shuffle scenes
    np.random.shuffle(scenes)

    train_scenes = scenes[:num_train]
    val_scenes = scenes[num_train : num_train + num_val]
    test_scenes = scenes[num_train + num_val :]

    # Create datasets
    train_dataset = NGSIMDataset(train_scenes)
    val_dataset = NGSIMDataset(val_scenes)
    test_dataset = NGSIMDataset(test_scenes)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
