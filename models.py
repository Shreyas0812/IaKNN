import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_vehicles=6, pred_horizon=5):
        super(InteractionLayer, self).__init__()

        # Calculate CNN output dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_vehicles = num_vehicles
        self.pred_horizon = pred_horizon

        # Reshape to 2D grid for CNN
        self.grid_size = 10
        self.cnn_input_channels = max(1, input_dim // (self.grid_size * self.grid_size))

        # CNN for social tensor extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(self.cnn_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate CNN output size
        cnn_output_width = self.grid_size // 4
        cnn_output_height = self.grid_size // 4
        cnn_output_size = 32 * cnn_output_width * cnn_output_height

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

        # Output layer for accelerations (x, y for each vehicle)
        self.output_layer = nn.Linear(hidden_dim, 2 * num_vehicles)

    def forward(self, observations):
        """Forward pass through the interaction layer"""
        batch_size = observations.size(0)
        seq_len = observations.size(1)

        # Process each time step through CNN and FCN
        features = []
        for t in range(seq_len):
            # Reshape to 2D grid for CNN
            cnn_input = observations[:, t].view(
                batch_size, self.cnn_input_channels, self.grid_size, self.grid_size
            )

            cnn_out = self.cnn(cnn_input)
            cnn_out = cnn_out.view(batch_size, -1)  # Flatten
            fcn_out = self.fcn(cnn_out)
            features.append(fcn_out)

        # Stack features and process through LSTM encoder
        features = torch.stack(features, dim=1)
        encoder_output, (h_n, c_n) = self.encoder_lstm(features)

        # Initialize decoder for prediction
        decoder_input = encoder_output[:, -1].unsqueeze(1)
        hidden = (h_n, c_n)

        # Generate predictions for each future timestep
        predictions = []
        for _ in range(self.pred_horizon):
            decoder_output, hidden = self.decoder_lstm(decoder_input, hidden)
            acceleration = self.output_layer(decoder_output.squeeze(1))
            predictions.append(acceleration)
            decoder_input = decoder_output

        # Stack predictions along time dimension
        return torch.stack(predictions, dim=1)  # [batch, time, vehicles*2]


class MotionLayer(nn.Module):
    def __init__(self, num_vehicles=6, dt=0.1):
        super(MotionLayer, self).__init__()
        self.num_vehicles = num_vehicles
        self.dt = dt

    def forward(self, accelerations, initial_states):
        """Transform accelerations to trajectories using kinematics"""
        batch_size = accelerations.size(0)
        pred_horizon = accelerations.size(1)

        # Reshape accelerations to [batch, time, vehicles, 2]
        accelerations = accelerations.view(
            batch_size, pred_horizon, self.num_vehicles, 2
        )

        # Initialize trajectories tensor - for each vehicle: [x, y, vx, vy]
        trajectories = torch.zeros(
            batch_size, pred_horizon, self.num_vehicles, 4, device=accelerations.device
        )

        # Set initial conditions for the host vehicle
        positions = torch.zeros(
            batch_size, self.num_vehicles, 2, device=accelerations.device
        )
        velocities = torch.zeros(
            batch_size, self.num_vehicles, 2, device=accelerations.device
        )

        # Set the host vehicle's initial state (first vehicle is the host)
        positions[:, 0, :] = initial_states[:, :2]
        velocities[:, 0, :] = initial_states[:, 2:4]

        # Apply kinematic equations for each time step
        for t in range(pred_horizon):
            # Update velocities using acceleration (v = v0 + a*t)
            velocities = velocities + accelerations[:, t] * self.dt

            # Update positions (p = p0 + v*t + 0.5*a*t^2)
            positions = (
                positions
                + velocities * self.dt
                + 0.5 * accelerations[:, t] * self.dt**2
            )

            # Store in trajectories tensor
            trajectories[:, t, :, :2] = positions
            trajectories[:, t, :, 2:] = velocities

        # Reshape to [batch, time, vehicles*4]
        return trajectories.reshape(batch_size, pred_horizon, self.num_vehicles * 4)


class FilterLayer(nn.Module):
    def __init__(self, num_vehicles=6, state_dim=4, hidden_dim=32):
        super(FilterLayer, self).__init__()

        self.num_vehicles = num_vehicles
        self.state_dim = state_dim  # State dimension per vehicle (x, y, vx, vy)
        self.total_state_dim = num_vehicles * state_dim

        # LSTM for learning process noise covariance
        self.process_noise_lstm = nn.LSTM(
            self.total_state_dim, hidden_dim, batch_first=True
        )
        self.process_noise_fc = nn.Linear(hidden_dim, self.total_state_dim)

        # LSTM for learning measurement noise covariance
        self.measurement_noise_lstm = nn.LSTM(
            self.total_state_dim, hidden_dim, batch_first=True
        )
        self.measurement_noise_fc = nn.Linear(hidden_dim, self.total_state_dim)

    def forward(self, interaction_traj, dynamic_traj=None, initial_state=None):
        """Apply Kalman filter to estimate trajectories"""
        batch_size = interaction_traj.size(0)
        seq_len = interaction_traj.size(1)
        device = interaction_traj.device

        # If dynamic_traj is not provided, use interaction_traj
        if dynamic_traj is None:
            dynamic_traj = interaction_traj.clone()

        # Learn process noise covariance (diagonal)
        q_out, _ = self.process_noise_lstm(dynamic_traj)
        q_vec = torch.abs(self.process_noise_fc(q_out[:, -1]))
        Q = torch.diag_embed(q_vec)  # Make diagonal matrix

        # Learn measurement noise covariance (diagonal)
        r_out, _ = self.measurement_noise_lstm(interaction_traj)
        r_vec = torch.abs(self.measurement_noise_fc(r_out[:, -1]))
        R = torch.diag_embed(r_vec)  # Make diagonal matrix

        # Initialize state transition matrix
        F = self.create_transition_matrix(batch_size, device)

        # Initialize filtered states
        filtered_states = torch.zeros_like(interaction_traj)

        # Initial state
        if initial_state is None:
            s_hat = interaction_traj[:, 0]
        else:
            # Expand initial state to all vehicles if it's just for the host
            if initial_state.size(1) == 4:  # Just host vehicle
                s_hat = torch.zeros(batch_size, self.total_state_dim, device=device)
                s_hat[:, :4] = initial_state  # Set host vehicle state
            else:
                s_hat = initial_state

        # Initial covariance
        P_hat = (
            torch.eye(self.total_state_dim, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Run Kalman filter
        for t in range(seq_len):
            # Prediction step
            s_minus = torch.bmm(F, s_hat.unsqueeze(2)).squeeze(2)
            P_minus = torch.bmm(torch.bmm(F, P_hat), F.transpose(1, 2)) + Q

            # Update step
            z = interaction_traj[:, t]  # Measurement
            y = z - s_minus  # Innovation
            S = P_minus + R  # Innovation covariance

            # Kalman gain
            K_transpose = torch.linalg.solve(S, P_minus.transpose(1, 2))
            K = K_transpose.transpose(1, 2)

            # Update state
            s_hat = s_minus + torch.bmm(K, y.unsqueeze(2)).squeeze(2)

            # Update covariance
            I = (
                torch.eye(self.total_state_dim, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            P_hat = torch.bmm(I - torch.bmm(K, I), P_minus)

            # Store filtered state
            filtered_states[:, t] = s_hat

        return filtered_states

    def create_transition_matrix(self, batch_size, device):
        """Create state transition matrix F"""
        dt = 0.1  # Time step

        # [x, y, vx, vy] transition
        F_block = torch.tensor(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )

        # Create block diagonal matrix for all vehicles
        F = torch.zeros(self.total_state_dim, self.total_state_dim, device=device)

        for i in range(self.num_vehicles):
            start_idx = i * self.state_dim
            end_idx = (i + 1) * self.state_dim
            F[start_idx:end_idx, start_idx:end_idx] = F_block

        # Repeat for batch
        return F.unsqueeze(0).repeat(batch_size, 1, 1)


class IaKNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_vehicles=6,
        state_dim=4,
        hidden_dim=32,
        pred_horizon=5,
        dt=0.1,
    ):
        super(IaKNN, self).__init__()

        self.input_dim = input_dim
        self.num_vehicles = num_vehicles
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.dt = dt

        # Define model components
        self.interaction_layer = InteractionLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_vehicles=num_vehicles,
            pred_horizon=pred_horizon,
        )

        self.motion_layer = MotionLayer(num_vehicles=num_vehicles, dt=dt)

        self.filter_layer = FilterLayer(
            num_vehicles=num_vehicles, state_dim=state_dim, hidden_dim=hidden_dim
        )

    def forward(self, observations, initial_state=None):
        """Forward pass through the IaKNN model"""
        # 1. Extract interaction-aware accelerations from observations
        accel = self.interaction_layer(observations)

        # 2. Transform accelerations to interaction-aware trajectories
        if initial_state is None:
            # Extract initial state from observations
            initial_state = observations[:, -1, :4]

        traj = self.motion_layer(accel, initial_state)

        # 3. Apply Kalman filter to estimate trajectories
        filtered_traj = self.filter_layer(traj, initial_state=initial_state)

        return filtered_traj
