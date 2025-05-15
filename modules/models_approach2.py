import torch
import torch.nn as nn
import torch.nn.functional as F
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import pandas as pd
import os

class InteractionLayer(nn.Module):
    def __init__(self, input_dim=480, hidden_dim=32, num_vehicles=5, pred_horizon=5):
        super(InteractionLayer, self).__init__()

        # Store parameters
        self.input_dim = input_dim  # 480 features per timestep
        self.hidden_dim = hidden_dim
        self.num_vehicles = num_vehicles  # 1 ego + 4 neighboring
        self.pred_horizon = pred_horizon  # 5 timesteps
        
        # Define grid dimensions for rectangular spatial representation
        self.grid_height = 80
        self.grid_width = 6
        
        # Calculate CNN input channels - using 1 channel initially
        self.cnn_input_channels = 1
        
        # CNN for social tensor extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(self.cnn_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate CNN output size after two MaxPool2d operations (each divides dimensions by 2)
        cnn_output_height = self.grid_height // 4  # 80/4 = 20
        cnn_output_width = self.grid_width // 4    # 6/4 = 1 (rounded down)
        # Handle the case where division by 4 makes width too small
        cnn_output_width = max(1, cnn_output_width)
        cnn_output_size = 32 * cnn_output_height * cnn_output_width
        # print(f"CNN output size: {cnn_output_size}")
        
        # FCN for mixing social features
        self.fcn = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # LSTM encoder-decoder
        self.encoder_lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer for accelerations (only for ego vehicle)
        self.output_layer = nn.Linear(hidden_dim, 2)  # Just ax, ay for ego vehicle
    
    def forward(self, observations):
        """
        Forward pass through the interaction layer
        
        Args:
            observations: Tensor of shape [batch_size, seq_len, 480]
        
        Returns:
            Tensor of shape [batch_size, pred_horizon, 2] representing 
            predicted accelerations for ego vehicle
        """
        batch_size = observations.size(0)
        seq_len = observations.size(1)
        
        # Process each time step through CNN and FCN
        features = []
        for t in range(seq_len):
            # Reshape to 2D grid for CNN
            # Reshape the 480 features into a rectangular grid of 80x6
            cnn_input = observations[:, t].view(
                batch_size, self.cnn_input_channels, self.grid_height, self.grid_width
            )
            
            # Process through CNN
            cnn_out = self.cnn(cnn_input)
            cnn_out = cnn_out.view(batch_size, -1)  # Flatten
            
            # Process through FCN
            fcn_out = self.fcn(cnn_out)
            features.append(fcn_out)
        
        # Stack features and process through LSTM encoder
        features = torch.stack(features, dim=1)  # [batch_size, seq_len, 64]
        encoder_output, (h_n, c_n) = self.encoder_lstm(features)
        
        # Initialize decoder for prediction
        decoder_input = encoder_output[:, -1].unsqueeze(1)
        hidden = (h_n, c_n)
        
        # Generate predictions for each future timestep
        predictions = []
        for _ in range(self.pred_horizon):
            decoder_output, hidden = self.decoder_lstm(decoder_input, hidden)
            acceleration = self.output_layer(decoder_output.squeeze(1))  # Only ego acceleration
            predictions.append(acceleration)
            decoder_input = decoder_output
        
        # Stack predictions along time dimension
        return torch.stack(predictions, dim=1)  # [batch, pred_horizon, 2]


class MotionLayer():
    def __init__(self, dt=0.1):
        """
        Initialize the Motion Layer
        
        Args:
            dt: Time step between predictions (in seconds)
        """
        super(MotionLayer, self).__init__()
        self.dt = dt  # Time step

    def forward(self, accelerations, initial_state):
        """
        Transform accelerations to trajectories using Euler integration
        
        Args:
            accelerations: Tensor of shape [batch_size, 5, 2] containing predicted accelerations (ax, ay)
                           from interaction layer for 5 future timesteps
            initial_state: Tensor of shape [batch_size, 1, 6] containing the ego vehicle's last state
                           where 6 is (x, y, vx, vy, ax, ay)
        
        Returns:
            Tensor of shape [batch_size, 5, 6] representing predicted trajectories
            where 6 is (x, y, vx, vy, ax, ay) for each timestep
        """
        batch_size = accelerations.size(0)
        pred_horizon = accelerations.size(1)  # Should be 5
        
        # Initialize trajectories tensor to store outputs
        trajectories = torch.zeros(
            batch_size, pred_horizon, 6, device=accelerations.device
        )
        
        # Extract initial state components
        # Remove the middle dimension (which is 1)
        initial_state = initial_state.squeeze(1)  # [batch_size, 6]
        
        # Extract position, velocity, and acceleration from initial state
        positions = initial_state[:, :2].clone()      # [batch_size, 2] - (x, y)
        velocities = initial_state[:, 2:4].clone()    # [batch_size, 2] - (vx, vy)
        
        # Apply kinematic equations for each time step
        for t in range(pred_horizon):
            # Get current relative acceleration for this time step
            current_accel = accelerations[:, t]  # [batch_size, 2]
            
            # Convert relative acceleration to absolute acceleration (if needed)
            # Note: since accelerations from interaction layer are relative to previous acceleration,
            # we add them to the last known absolute acceleration
            absolute_accel = initial_state[:, 4:6] + current_accel

            # Update velocity using acceleration: v = v0 + a*dt
            velocities = velocities + absolute_accel * self.dt 
            
            # Update position using velocity: p = p0 + v*dt
            positions = positions + velocities * self.dt + absolute_accel * 0.5 * self.dt**2
            
            # Store updated state in trajectories tensor
            trajectories[:, t, :2] = positions                 # x, y
            trajectories[:, t, 2:4] = velocities               # vx, vy
            trajectories[:, t, 4:6] = absolute_accel           # ax, ay
        
        return trajectories
    

class FilterLayer(nn.Module):
    def __init__(self, dt=0.1, state_dim=6):
        """
        Initialize the Filter Layer for ego vehicle tracking
        
        Args:
            dt: Time step between predictions (in seconds)
            state_dim: Dimension of state vector (x, y, vx, vy, ax, ay)
        """
        super(FilterLayer, self).__init__()
        self.dt = dt
        self.state_dim = state_dim
        
        # Fixed process and measurement noise covariances
        self.process_noise_std = 0.1    # Standard deviation for process noise
        self.measurement_noise_std = 0.1  # Standard deviation for measurement noise
        
    def create_transition_matrix(self):
        """Create state transition matrix F as defined in the paper"""
        # Create the F matrix for ego vehicle's state [x, y, vx, vy, ax, ay]
        F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F
        
    def create_Q_matrix(self):
        """Create process noise covariance matrix"""
        Q = np.eye(self.state_dim) * (self.process_noise_std ** 2)
        return Q
        
    def create_R_matrix(self):
        """Create measurement noise covariance matrix"""
        R = np.eye(self.state_dim) * (self.measurement_noise_std ** 2)
        return R
    
    def initialize_kf(self, initial_state):
        """
        Initialize a Kalman filter with the given initial state
        
        Args:
            initial_state: numpy array of shape [6] containing initial state
                          (x, y, vx, vy, ax, ay)
        
        Returns:
            Initialized KalmanFilter object
        """
        kf = KalmanFilter(dim_x=self.state_dim, dim_z=self.state_dim)
        
        # Set initial state
        kf.x = initial_state
        
        # Set state transition matrix
        kf.F = self.create_transition_matrix()
        
        # Set process noise covariance
        kf.Q = self.create_Q_matrix()
        
        # Set measurement noise covariance
        kf.R = self.create_R_matrix()
        
        # Set measurement function (identity in this case)
        kf.H = np.eye(self.state_dim)
        
        # Initialize state covariance matrix
        kf.P = np.eye(self.state_dim)
        
        return kf
        
    def forward(self, interaction_trajectories, past_states):
        """
        Apply Kalman filter to estimate ego vehicle trajectories
        
        Args:
            interaction_trajectories: Tensor of shape [batch_size, 5, 6] from the motion layer
                                     containing predicted trajectories (measurements)
            past_states: Tensor of shape [batch_size, 2, 6] containing ego vehicle's past 2 states
                        where each state is (x, y, vx, vy, ax, ay)
        
        Returns:
            Tensor of shape [batch_size, 5, 6] representing filtered trajectories for ego vehicle
        """
        batch_size = interaction_trajectories.size(0)
        pred_horizon = interaction_trajectories.size(1)
        
        # Convert tensors to numpy for filterpy
        interaction_trajectories_np = interaction_trajectories.detach().cpu().numpy()
        past_states_np = past_states.detach().cpu().numpy()
        
        # Output array for filtered trajectories
        filtered_trajectories_np = np.zeros((batch_size, pred_horizon, self.state_dim))
        
        # Process each batch item separately
        for b in range(batch_size):
            # Initialize Kalman filter with the last state from past_states
            initial_state = past_states_np[b, -1]
            kf = self.initialize_kf(initial_state)
            
            # Perform filtering for each time step
            for t in range(pred_horizon):
                # Prediction step (without control input)
                kf.predict()
                
                # Update step with measurement from interaction_trajectories
                measurement = interaction_trajectories_np[b, t]
                kf.update(measurement)
                
                # Store filtered state
                filtered_trajectories_np[b, t] = kf.x

                # update the initial state for the next iteration
                # initial_state = kf.x
                # kf.x = initial_state
                

        
        # Convert back to tensor
        filtered_trajectories = torch.tensor(
            filtered_trajectories_np, 
            dtype=interaction_trajectories.dtype,
            device=interaction_trajectories.device
        )
        
        return filtered_trajectories


class IaKNN(nn.Module):
    def __init__(
        self,
        input_dim=480, 
        state_dim=6, 
        hidden_dim=32, 
        pred_horizon=5, 
        dt=0.1,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=50
    ):
        """
        Initialize the IaKNN model
        
        Args:
            input_dim: Dimension of input features
            state_dim: Dimension of state vector (x, y, vx, vy, ax, ay)
            hidden_dim: Hidden dimension size for neural networks
            pred_horizon: Number of future timesteps to predict
            dt: Time step between predictions (in seconds)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
        """
        super(IaKNN, self).__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.dt = dt
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Define model components
        self.interaction_layer = InteractionLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pred_horizon=pred_horizon
        )
        
        self.motion_layer = MotionLayer(dt=dt)
        
        self.filter_layer = FilterLayer(
            dt=dt,
            state_dim=state_dim
        )
        
        # Define loss function
        self.loss_fn = self.loss_function

    def loss_function(self, filtered_trajectories, ground_truth):
        """
        Calculate the loss between filtered trajectories and ground truth
        
        Args:
            filtered_trajectories: Tensor of shape [batch_size, 5, 6] representing filtered trajectories
                                    from the filter layer for the ego vehicle
            ground_truth: Tensor of shape [batch_size, 5, 6] containing ground truth trajectories
                        for the ego vehicle
        
        Returns:
            Scalar loss value
        """
        batch_size = filtered_trajectories.size(0)
        pred_horizon = filtered_trajectories.size(1)
        
        # Calculate squared error for each element in the trajectory
        # This implements the formula from the paper:
        # L{W,b} := 1/((L0 + 1) * N) * sum_i=1^N sum_t=t0^(t0+L0) ||Sˆi_t - Gi_t||2
        # Where N=1 (only ego vehicle), and L0=4 (5 timesteps starting from 0)
        
        # Compute L2 norm squared of difference at each timestep
        point_errors = torch.sum(torch.pow(filtered_trajectories - ground_truth, 2), dim=2)
        
        # Sum over all timesteps and batch, then normalize
        total_error = torch.sum(point_errors)
        loss = total_error / (pred_horizon * batch_size)
        
        return loss
        
    def forward(self, observations, initial_state):
        """
        Forward pass through the IaKNN model
        
        Args:
            observations: Tensor of shape [batch_size, seq_len, input_dim] for Interaction Layer
            initial_state: Tensor of shape [batch_size, 1, 6] containing initial state for Motion Layer
        
        Returns:
            Tensor of shape [batch_size, pred_horizon, 6] representing filtered trajectories
        """
        # Extract interaction-aware accelerations from observations
        accel = self.interaction_layer(observations)
        
        # Transform accelerations to interaction-aware trajectories
        traj = self.motion_layer.forward(accel, initial_state)
        
        # Apply Kalman filter to estimate trajectories
        filtered_traj = self.filter_layer(traj, initial_state)
        
        return filtered_traj
    
    def train_model(self, data_loader, val_loader=None):
        """
        Train the IaKNN model
        
        Args:
            data_loader: DataLoader containing training data
            val_loader: Optional DataLoader containing validation data
        
        Returns:
            Lists of training and validation losses
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # Training
            self.train()
            total_train_loss = 0
            train_batches = 0
            
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            for batch in progress_bar:
                # Assuming batch is a tensor of shape [batch_size, 86, 6]
                batch = batch.to(device)
                print(f"Batch shape: {batch.shape}")
                
                # Split data:
                # - Initial state: First row
                # - Observations for Interaction Layer: Next 80 rows
                # - Ground truth: Last 5 rows
                initial_state = batch[:, :, 0:1, :]  # Shape: [batch_size, 1, 6]
                # Squeeze the initial state to remove the middle dimension
                initial_state = initial_state.squeeze(1)  # Shape: [batch_size, 6]
                observations = batch[:, :, 1:81, :]  # Shape: [batch_size, 80, input_dim/80]
                ground_truth = batch[:, :, -self.pred_horizon:, :]  # Shape: [batch_size, 5, 6]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                filtered_traj = self.forward(observations, initial_state)
                
                # Calculate loss
                loss = self.loss_fn(filtered_traj, ground_truth)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss
                total_train_loss += loss.item()
                train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'train_loss': total_train_loss / train_batches})
            
            # Record average training loss
            avg_train_loss = total_train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.eval()
                total_val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        
                        # Split data
                        initial_state = batch[:, 0:1, :]
                        # Squeeze the initial state to remove the middle dimension
                        initial_state = initial_state.squeeze(1)  # Shape: [batch_size, 6]
                        observations = batch[:, 1:81, :]
                        ground_truth = batch[:, -self.pred_horizon:, :]
                        
                        # Forward pass
                        filtered_traj = self.forward(observations, initial_state)
                        
                        # Calculate loss
                        loss = self.loss_fn(filtered_traj, ground_truth)
                        
                        # Track loss
                        total_val_loss += loss.item()
                        val_batches += 1
                
                # Record average validation loss
                avg_val_loss = total_val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            else:
                print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.6f}')
        
        return train_losses, val_losses
    
    def test_model(self, test_loader):
        """
        Test the IaKNN model
        
        Args:
            test_loader: DataLoader containing test data
        
        Returns:
            Average test loss and predicted trajectories
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        total_test_loss = 0
        test_batches = 0
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                batch = batch.to(device)
                
                # Split data
                initial_state = batch[:, 0:1, :]
                observations = batch[:, 1:81, :]
                ground_truth = batch[:, -self.pred_horizon:, :]
                
                # Forward pass
                filtered_traj = self(observations, initial_state)
                
                # Calculate loss
                loss = self.loss_fn(filtered_traj, ground_truth)
                
                # Track loss and predictions
                total_test_loss += loss.item()
                test_batches += 1
                all_predictions.append(filtered_traj.cpu())
                all_ground_truths.append(ground_truth.cpu())
        
        # Calculate average test loss
        avg_test_loss = total_test_loss / test_batches
        print(f'Test Loss: {avg_test_loss:.6f}')
        
        # Concatenate predictions and ground truths
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truths = torch.cat(all_ground_truths, dim=0)
        
        return avg_test_loss, all_predictions, all_ground_truths
    
    def prepare_data(self, dataset, train_ratio=0.8, val_ratio=0.1):
        """
        Prepare data for training, validation, and testing
        
        Args:
            dataset: Dataset containing trajectory data
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
        
        Returns:
            DataLoaders for training, validation, and testing
        """
        # Calculate dataset sizes
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Split dataset
        print(f"type(dataset): {type(dataset)}")
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        print(f"type(train_dataset): {type(train_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def run_training(self, dataset):
        """
        Run complete training and testing pipeline
        
        Args:
            dataset: Dataset containing trajectory data
        
        Returns:
            Training history and test results
        """
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(dataset)
        
        # Train model
        print("Starting training...")
        start_time = time.time()
        train_losses, val_losses = self.train_model(train_loader, val_loader)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Test model
        print("Starting testing...")
        test_loss, predictions, ground_truths = self.test_model(test_loader)
        
        # Return results
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_loss': test_loss,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'training_time': training_time
        }
        
        return results
    
    def trial_run_single_data_point(self, data_point):
        """
        Run a single data point through the model for debugging
        
        Args:
            data_point: Tensor of shape [1, 86, 6] containing a single trajectory data point
                        where 86 is the number of time steps and 6 is the state dimension
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Prepare data point
        initial_state = data_point[:, :, 0:1, :].to(device)
        # Squeeze the initial state to remove the middle dimension
        initial_state = initial_state.squeeze(1)  # Shape: [1, 6]
        observations = data_point[:, :, 1:81, :].to(device)  # Shape: [1, 80, input_dim/80]
        ground_truth = data_point[:, :, -self.pred_horizon:, :].to(device)  # Shape: [1, 5, 6]

        # Forward pass
        filtered_traj = self.forward(observations, initial_state)

        # Calculate loss
        loss = self.loss_fn(filtered_traj, ground_truth)
        print(f"Loss: {loss.item()}")
        return filtered_traj, loss.item()
    
    
    def visualize_results(self, results):
        """
        Visualize training history and test results
        
        Args:
            results: Dictionary containing training and test results
        """
        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(results['train_losses'], label='Training Loss')
        plt.plot(results['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()
        
        # Plot example predictions
        num_samples = min(5, results['predictions'].shape[0])
        
        for i in range(num_samples):
            plt.figure(figsize=(15, 10))
            
            # Position plot (x, y)
            plt.subplot(2, 2, 1)
            pred_positions = results['predictions'][i, :, :2].numpy()
            true_positions = results['ground_truths'][i, :, :2].numpy()
            
            plt.plot(pred_positions[:, 0], pred_positions[:, 1], 'b-', marker='o', label='Predicted')
            plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', marker='x', label='Ground Truth')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Position Trajectory (Sample {i+1})')
            plt.legend()
            plt.grid(True)
            
            # Velocity plot (vx, vy)
            plt.subplot(2, 2, 2)
            pred_velocities = results['predictions'][i, :, 2:4].numpy()
            true_velocities = results['ground_truths'][i, :, 2:4].numpy()
            
            plt.plot(range(self.pred_horizon), pred_velocities[:, 0], 'b-', marker='o', label='Predicted vx')
            plt.plot(range(self.pred_horizon), true_velocities[:, 0], 'g-', marker='x', label='Ground Truth vx')
            plt.plot(range(self.pred_horizon), pred_velocities[:, 1], 'r-', marker='o', label='Predicted vy')
            plt.plot(range(self.pred_horizon), true_velocities[:, 1], 'm-', marker='x', label='Ground Truth vy')
            plt.xlabel('Time Step')
            plt.ylabel('Velocity')
            plt.title(f'Velocity Profile (Sample {i+1})')
            plt.legend()
            plt.grid(True)
            
            # Acceleration plot (ax, ay)
            plt.subplot(2, 2, 3)
            pred_accels = results['predictions'][i, :, 4:6].numpy()
            true_accels = results['ground_truths'][i, :, 4:6].numpy()
            
            plt.plot(range(self.pred_horizon), pred_accels[:, 0], 'b-', marker='o', label='Predicted ax')
            plt.plot(range(self.pred_horizon), true_accels[:, 0], 'g-', marker='x', label='Ground Truth ax')
            plt.plot(range(self.pred_horizon), pred_accels[:, 1], 'r-', marker='o', label='Predicted ay')
            plt.plot(range(self.pred_horizon), true_accels[:, 1], 'm-', marker='x', label='Ground Truth ay')
            plt.xlabel('Time Step')
            plt.ylabel('Acceleration')
            plt.title(f'Acceleration Profile (Sample {i+1})')
            plt.legend()
            plt.grid(True)
            
            # Error plot
            plt.subplot(2, 2, 4)
            position_errors = np.linalg.norm(pred_positions - true_positions, axis=1)
            velocity_errors = np.linalg.norm(pred_velocities - true_velocities, axis=1)
            accel_errors = np.linalg.norm(pred_accels - true_accels, axis=1)
            
            plt.plot(range(self.pred_horizon), position_errors, 'b-', marker='o', label='Position Error')
            plt.plot(range(self.pred_horizon), velocity_errors, 'g-', marker='x', label='Velocity Error')
            plt.plot(range(self.pred_horizon), accel_errors, 'r-', marker='s', label='Acceleration Error')
            plt.xlabel('Time Step')
            plt.ylabel('Error (L2 Norm)')
            plt.title(f'Prediction Errors (Sample {i+1})')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'prediction_sample_{i+1}.png')
            plt.show()
        
        # Summary statistics
        mean_position_error = np.mean(np.linalg.norm(
            results['predictions'][:, :, :2].numpy() - results['ground_truths'][:, :, :2].numpy(),
            axis=2
        ))
        mean_velocity_error = np.mean(np.linalg.norm(
            results['predictions'][:, :, 2:4].numpy() - results['ground_truths'][:, :, 2:4].numpy(),
            axis=2
        ))
        mean_accel_error = np.mean(np.linalg.norm(
            results['predictions'][:, :, 4:6].numpy() - results['ground_truths'][:, :, 4:6].numpy(),
            axis=2
        ))
        
        print(f"Mean Position Error: {mean_position_error:.4f}")
        print(f"Mean Velocity Error: {mean_velocity_error:.4f}")
        print(f"Mean Acceleration Error: {mean_accel_error:.4f}")

class PickleDataset(Dataset):
    def __init__(self, pickle_dir):
        self.files = [os.path.join(pickle_dir, f) 
                      for f in os.listdir(pickle_dir) if f.endswith('.pk1')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_pickle(self.files[idx])
        # Convert the DataFrame to a torch tensor (float32)

        df = df[19:]

        tensor = torch.tensor(df.values, dtype=torch.float32, r
        tensor = tensor.unsqueeze(0)
        return tensor

if __name__ == "__main__":
    # Set the data path
    pickle_dir = '../data/ego_df/us-101'
    
    # Create the dataset
    dataset = PickleDataset(pickle_dir)
    
    # Initialize the model
    model = IaKNN(
        input_dim=480,  
        state_dim=6,
        hidden_dim=32,
        pred_horizon=5,
        dt=0.1,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=50
    )
    
    # Run the complete training pipeline
    results = model.run_training(dataset)
    
    # Visualize the results
    model.visualize_results(results)


# # Example usage
# if __name__ == "__main__":
#     # Run a trial with dummy datapoint
#     dummy_data_point = torch.randn(2, 1, 86, 6)  # Shape: [1, 86, 6]
#     model = IaKNN()
#     filtered_traj, loss = model.trial_run_single_data_point(dummy_data_point)
#     print(f"Filtered Trajectory: {filtered_traj}")
#     print(f"Loss: {loss}")
