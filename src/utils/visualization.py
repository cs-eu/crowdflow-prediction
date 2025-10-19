import matplotlib.pyplot as plt
import numpy as np

def plot_bottlenecks(bottle_neck, bottle_neck_keys):
    """
    Visualizes bottleneck data across 4 subplots.

    Parameters:
    -----------
    bottle_neck : dict
        Dictionary containing bottleneck data for different keys.
    bottle_neck_keys : list
        List of keys corresponding to bottleneck data.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Bottlenecks')
    for i in range(2):
        for j in range(2):
            curr_data = bottle_neck[bottle_neck_keys[i * 2 + j]]
            axs[i, j].scatter(curr_data[:, 2], curr_data[:, 3], s=0.25)
            axs[i, j].set_title(bottle_neck_keys[i * 2 + j])
            axs[i, j].set_xlabel("X")
            axs[i, j].set_ylabel("Y")
    plt.tight_layout()
    plt.show()


def plot_corridors(corridor, corridor_keys):
    """
    Visualizes corridor data across 8 subplots.

    Parameters:
    -----------
    corridor : dict
        Dictionary containing corridor data for different keys.
    corridor_keys : list
        List of keys corresponding to corridor data.
    """
    fig, axs = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle('Corridors')
    for i in range(4):
        for j in range(2):
            curr_data = corridor[corridor_keys[i * 2 + j]]
            axs[i, j].scatter(curr_data[:, 2], curr_data[:, 3], s=0.25)
            axs[i, j].set_title(corridor_keys[i * 2 + j])
            axs[i, j].set_xlabel("X")
            axs[i, j].set_ylabel("Y")
    plt.tight_layout()
    plt.show()


def plot_paths_bottleneck(data, bottle_neck_keys, k=30, situation='Bottleneck_Data'):
    """
    Plots paths of the first k pedestrians in the bottleneck situation.

    Parameters:
    -----------
    data : dict
        Data dictionary containing bottleneck pedestrian data.
    bottle_neck_keys : list
        List of keys for the bottleneck situation.
    k : int
        Number of pedestrians to plot.
    situation : str
        Name of the bottleneck situation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(1, k + 1):
        pedestrian_data = data[situation][bottle_neck_keys[0]][data[situation][bottle_neck_keys[0]][:, 0] == i]
        ax.plot(pedestrian_data[:, 2], pedestrian_data[:, 3], label=f'Pedestrian {i}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Paths of First {k} Pedestrians in {situation}')
    plt.show()


def plot_paths_corridor(data, corridor_keys, k=30, situation='Corridor_Data'):
    """
    Plots paths of the first k pedestrians in the corridor situation.

    Parameters:
    -----------
    data : dict
        Data dictionary containing corridor pedestrian data.
    corridor_keys : list
        List of keys for the corridor situation.
    k : int
        Number of pedestrians to plot.
    situation : str
        Name of the corridor situation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(1, k + 1):
        pedestrian_data = data[situation][corridor_keys[0]][data[situation][corridor_keys[0]][:, 0] == i]
        ax.plot(pedestrian_data[:, 2], pedestrian_data[:, 3], label=f'Pedestrian {i}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Paths of First {k} Pedestrians in {situation}')
    plt.show()


def plot_mean_spacing_vs_speed(mean_spacings, actual_speeds):
    """
    Plots the relationship between mean spacing and actual speed.

    Parameters:
    -----------
    mean_spacings : dict
        Dictionary containing mean spacing values for each agent and time step.
    actual_speeds : dict
        Dictionary containing actual speed values for each agent and time step.
    """
    # Prepare lists for plotting
    mean_spacings_values = []
    actual_speeds_values = []

    # Iterate over all agents and time steps
    for agent in actual_speeds:
        for time_step in actual_speeds[agent]:
            # Ensure the agent and time_step exist in both mean_spacings and actual_speeds
            if agent in mean_spacings and time_step in mean_spacings[agent]:
                mean_spacings_values.append(mean_spacings[agent][time_step])
                actual_speeds_values.append(actual_speeds[agent][time_step])

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(mean_spacings_values, actual_speeds_values, s=1)
    plt.xlabel("Mean Spacing")
    plt.ylabel("Actual Speed")
    plt.title("Mean Spacing vs. Actual Speed")
    plt.show()


def plot_ETH_walking_pedestrian_data(data):
    """
    Visualizes walking pedestrian data in a single scatter plot (without speed).

    Parameters:
    -----------
    data : np.ndarray
        Numpy array containing walking pedestrian data with at least 3 columns 
        (frame, id, x, y and z).
    """
    _, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[:, 1], data[:, 2], s=0.5, alpha=0.5)

    ax.set_title("Walking Pedestrian Data Visualization")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    plt.show()


def plot_walking_pedestrian(data, walking_pedestrian_data_keys, k=30, situation='Walking_Pedestrian_Data'):
    """
    Plots paths of the first k pedestrians in the walking pedestrian situation.

    Parameters:
    -----------
    data : dict
        Data dictionary containing walking pedestrian data.
    walking_pedestrian_data_keys : list
        List of keys for the walking pedestrian situation.
    k : int
        Number of pedestrians to plot (default is 10).
    situation : str
        Name of the walking pedestrian situation.
    """
    _, ax = plt.subplots(figsize=(10, 6))

    # Extract pedestrian data
    pedestrian_data = data[situation][walking_pedestrian_data_keys[0]]

    # Get all unique pedestrian IDs (assuming the first column is pedestrian ID)
    pedestrian_ids = np.unique(pedestrian_data[:, 0])  # Extract unique pedestrian IDs

    # Select the first k unique pedestrian IDs (sorted order)
    selected_ids = pedestrian_ids[:min(k, len(pedestrian_ids))]
    # Plot paths for the first k unique pedestrians
    for ped_id in selected_ids:
        ped_data = pedestrian_data[pedestrian_data[:, 0] == ped_id]  # Filter data for this pedestrian
        ax.plot(ped_data[:, 1], ped_data[:, 2], label=f'Pedestrian {int(ped_id)}')  # X vs Y

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Paths of First {len(selected_ids)} Pedestrians')
    plt.show()
