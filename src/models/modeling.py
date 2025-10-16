import numpy as np
from scipy.spatial import cKDTree
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


class WeidmannModel:
    def weidmann_speed(self, sk, v0, t, l_size):
        """
        Calculates pedestrian speed based on the Weidmann model.

        Parameters:
        -----------
        sk : float
            Mean spacing of the pedestrian.
        v0 : float
            Desired speed in m/s.
        t : float
            Relaxation time in seconds.
        l_size : float
            Characteristic length in meters.

        Returns:
        --------
        float
            Predicted pedestrian speed.
        """
        return v0 * (1 - np.exp((l_size - sk) / (v0 * t)))

    def calculate_weidmann_mse(self, mean_spacings, actual_speeds, v0, t, l_size):
        """
        Predicts speeds using the Weidmann model and calculates the Mean Squared Error (MSE).

        Parameters:
        -----------
        mean_spacings : dict
            Nested dictionary containing mean spacing values for each agent and time step.
        actual_speeds : dict
            Nested dictionary containing actual speed values for each agent and time step.
        v0 : float, optional
            Desired speed in m/s (default is 1.58).
        t : float, optional
            Relaxation time in seconds (default is 0.48).
        l_size : float, optional
            Characteristic length in meters (default is 0.61).

        Returns:
        --------
        mse : float
            The Mean Squared Error between predicted and actual speeds.
        predicted_speeds : dict
            Nested dictionary of predicted speeds for each agent and time step.
        """
        predicted_speeds = {}
        errors = []

        for agent in mean_spacings:
            predicted_speeds[agent] = {}
            for time_step in mean_spacings[agent]:
                sk = mean_spacings[agent][time_step]
                predicted_speed = self.weidmann_speed(sk, v0, t, l_size)
                predicted_speeds[agent][time_step] = predicted_speed
                
                # Collect error for MSE calculation
                if agent in actual_speeds and time_step in actual_speeds[agent]:
                    error = (actual_speeds[agent][time_step] - predicted_speed) ** 2
                    errors.append(error)

        # Calculate Mean Squared Error
        mse = np.mean(errors)

        return mse, predicted_speeds

    def calculate_mean_spacing(self, data, k=5, three_d=False):
        """
        Calculate the mean spacing for each agent based on k nearest neighbors.

        Parameters:
        - data (np.ndarray): Input array with columns [agent_no, time_frame_no, xpos, ypos, zpos].
        - k (int): Number of nearest neighbors to consider.

        Returns:
        - mean_spacings (dict): Dictionary with nested structure: {agent_id: {time_step: mean_spacing}}.
        - neighbors (dict): Dictionary with nested structure: {agent_id: {time_step: neighbors_positions}}.
        """
        # Extract unique time frames
        time_frames = np.unique(data[:, 1])

        mean_spacings = {}
        neighbors = {}
        current_data = []

        for t in time_frames:
            # Filter data for the current time frame
            time_frame_data = data[data[:, 1] == t]

            # Extract positions (x, y, z) and agent IDs
            if three_d:
                positions = time_frame_data[:, 2:5]
            else:
                positions = time_frame_data[:, 2:4]
            agents = time_frame_data[:, 0]

            # Build a KD-Tree for efficient neighbor search
            tree = cKDTree(positions)

            # Query k nearest neighbors (k+1 because the first neighbor is the point itself)
            distances, indices = tree.query(positions, k=k + 1)

            # Compute mean spacing (ignore the first distance which is zero)
            for i, agent in enumerate(agents):
                mean_spacing = np.mean(distances[i, 1:]) / 100  # Mean of k nearest neighbors, converting to meters
                if mean_spacing == np.inf:
                    continue

                # Nested dictionary structure
                if agent not in mean_spacings:
                    mean_spacings[agent] = {}
                mean_spacings[agent][int(t)] = mean_spacing

                if agent not in neighbors:
                    neighbors[agent] = {}
                neighbors[agent][int(t)] = positions[indices[i, 1:]]

                if three_d:
                    current_data.append(np.array([int(agent), int(t), positions[i, 0], positions[i, 1], positions[i, 2]]))
                else:
                    current_data.append(np.array([int(agent), int(t), positions[i, 0], positions[i, 1]]))

        return mean_spacings, np.array(current_data), neighbors

    def compute_agent_speeds(self, trajectory_data, frame_rate, is_3d=False):
        """
        Computes agent speeds from trajectory data.

        Parameters:
        -----------
        trajectory_data : np.ndarray
            Array with columns: [Agent ID, Time step, X, Y, (optional Z)].
        frame_rate : float
            Frame rate to compute time differences.
        is_3d : bool, optional
            If True, computes speeds in 3D; otherwise, in 2D.

        Returns:
        --------
        dict
            Speeds per agent and time step as {"agentID": {timeStep: speed (m/s)}}.
        """
        agent_speeds = {}

        for agent_id in np.unique(trajectory_data[:, 0]):
            agent_data = trajectory_data[trajectory_data[:, 0] == agent_id]

            for i in range(1, len(agent_data)):
                time_diff = (agent_data[i, 1] - agent_data[i - 1, 1]) * frame_rate
                if is_3d:
                    distance = np.linalg.norm(agent_data[i, 2:5] - agent_data[i - 1, 2:5]) / 100
                else:
                    distance = np.linalg.norm(agent_data[i, 2:4] - agent_data[i - 1, 2:4]) / 100

                speed = distance / time_diff

                # Nested dictionary structure
                if agent_id not in agent_speeds:
                    agent_speeds[agent_id] = {}
                agent_speeds[agent_id][int(agent_data[i, 1])] = speed

        return agent_speeds



class NeuralNetworkModel:
    def prepare_training_data(self, agent_speeds, mean_spacings, neighbors):
        """
        Prepares input and output data for neural network training.
        
        Parameters:
            agent_speeds (dict): Dictionary containing agent speeds.
            mean_spacings (dict): Dictionary containing mean spacings per agent and time step.
            neighbors (dict): Dictionary containing neighbor information.
        
        Returns:
            tuple: (input_data, output_data) as NumPy arrays
        """
        input_data = []
        output_data = []

        for agent in agent_speeds:
            for time_step in agent_speeds[agent]:
                if (
                    agent in mean_spacings and time_step in mean_spacings[agent]
                    and agent in neighbors and time_step in neighbors[agent]
                ):
                    neighbors_flattened = neighbors[agent][time_step].flatten()
                    mean_spacing = mean_spacings[agent][time_step]
                    speed = agent_speeds[agent][time_step]

                    input_data.append(np.append(neighbors_flattened, mean_spacing))
                    output_data.append(speed)

        return np.array(input_data), np.array(output_data)


    def create_nn_model(self, input_dim, output_dim, hidden_layers, optimizer='adam', loss='mse'):
        """
        Creates a neural network model for pedestrian speed prediction.

        Parameters:
        -----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output dimensions.
        hidden_layers : list[int]
            Number of neurons in each hidden layer.
        optimizer : str
            Optimizer for training.
        loss : str
            Loss function for training.

        Returns:
        --------
        Sequential
            Compiled Keras model.
        """
        model = Sequential([Input(shape=(input_dim,))])
        for units in hidden_layers:
            model.add(Dense(units, activation='sigmoid'))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(optimizer=optimizer, loss=loss)
        return model


    def perform_cross_validation(self, model, data, targets, test_data, test_targets, k_folds, epochs, batch_size, dropout=-1):
        """
        Performs K-fold cross-validation for a given neural network model.

        Parameters:
        -----------
        model : keras.Model
            The neural network model to train.
        data : np.ndarray
            Input data for training and validation.
        targets : np.ndarray
            Target labels for training and validation.
        test_data : np.ndarray
            Data for testing the model after validation.
        test_targets : np.ndarray
            Target labels for testing data.
        k_folds : int
            Number of folds for cross-validation.
        epochs : int
            Number of epochs for training.
        batch_size : int
            Batch size for training.
        dropout : float, optional
            Dropout rate to apply in the model (if applicable).

        Returns:
        --------
        tuple
            Average training, validation, and testing losses over K folds.
        """
        # Shuffle and split the data
        data, targets = shuffle(data, targets)
        data_folds = np.array_split(data, k_folds)
        target_folds = np.array_split(targets, k_folds)
        
        train_losses, validation_losses, test_losses = [], [], []
        early_stopping = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
        
        for fold in range(k_folds):
            # Create training and validation sets
            validation_data, validation_targets = data_folds[fold], target_folds[fold]
            train_data = np.concatenate([data_folds[i] for i in range(k_folds) if i != fold])
            train_targets = np.concatenate([target_folds[i] for i in range(k_folds) if i != fold])
            
            # Compile and train the model
            model.compile(optimizer='adam', loss='mse')
            history = model.fit(
                train_data, train_targets,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(validation_data, validation_targets),
                callbacks=[early_stopping]
            )
            
            # Record losses
            train_losses.append(history.history['loss'][-1])
            validation_losses.append(history.history['val_loss'][-1])
            
            # Evaluate on the test set
            test_predictions = model.predict(test_data, batch_size=batch_size, verbose=0)
            test_loss = np.mean([
                np.mean((test_targets[i:i+1000] - test_predictions[i:i+1000]) ** 2)
                for i in range(0, len(test_targets), 1000)
            ])
            test_losses.append(test_loss)
        
        return np.mean(train_losses), np.mean(validation_losses), np.mean(test_losses)


    def perform_bootstrapping(self, model, data, targets, test_data, test_targets, 
                            bootstrap_samples, bootstrap_size, 
                            k_folds, epochs, batch_size, dropout=-1):
        """
        Performs bootstrapping and K-fold cross-validation.

        Parameters:
        -----------
        model : keras.Model
            The neural network model to train.
        data : np.ndarray
            Input data for training and validation.
        targets : np.ndarray
            Target labels for training and validation.
        test_data : np.ndarray
            Data for testing the model after validation.
        test_targets : np.ndarray
            Target labels for testing data.
        bootstrap_samples : int
            Number of bootstrap samples to generate.
        bootstrap_size : int
            Size of each bootstrap sample.
        k_folds : int
            Number of folds for cross-validation.
        epochs : int
            Number of epochs for training.
        batch_size : int
            Batch size for training.
        dropout : float, optional
            Dropout rate to apply in the model (if applicable).

        Returns:
        --------
        tuple
            Average and standard deviation of training, validation, and testing losses.
        """
        train_means, validation_means, test_means = [], [], []
        
        for bootstrap_idx in range(bootstrap_samples):
            # Generate a bootstrap sample
            bootstrap_indices = np.random.choice(len(data), size=bootstrap_size, replace=True)
            bootstrap_data = data[bootstrap_indices]
            bootstrap_targets = targets[bootstrap_indices]
            
            # Perform K-fold cross-validation
            train_mean, val_mean, test_mean = self.perform_cross_validation(
                model=model,
                data=bootstrap_data,
                targets=bootstrap_targets,
                test_data=test_data,
                test_targets=test_targets,
                k_folds=k_folds,
                epochs=epochs,
                batch_size=batch_size,
                dropout=dropout
            )
            
            # Store the results
            train_means.append(train_mean)
            validation_means.append(val_mean)
            test_means.append(test_mean)
        
        # Compute mean and standard deviation of losses
        train_summary = (np.mean(train_means), np.std(train_means))
        validation_summary = (np.mean(validation_means), np.std(validation_means))
        test_summary = (np.mean(test_means), np.std(test_means))
        
        return train_summary, validation_summary, test_summary
