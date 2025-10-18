import os
import numpy as np

class DataLoader:
    """
    A utility class for loading and managing trajectory data.

    Attributes:
    -----------
    dataset_path : str
        The root path of the dataset directory.
    frame_rate : float
        The frame rate of the dataset.
    data : dict
        Loaded data organized by situations and file names.
    """

    def __init__(self, dataset_path, frame_rate=1):
        """
        Initializes the DataLoader.

        Parameters:
        -----------
        dataset_path : str
            The root path of the dataset directory.
        frame_rate : float, optional
            Frame rate of the dataset (default is 1).
        """
        self.dataset_path = dataset_path
        self.frame_rate = frame_rate
        self.data = {}

    def load_situations(self, situations):
        """
        Loads data for specified situations into memory.

        Parameters:
        -----------
        situations : list of str
            List of folder names representing different situations.
        """
        for situation in situations:
            situation_path = os.path.join(self.dataset_path, situation)
            if not os.path.exists(situation_path):
                raise FileNotFoundError(f"Situation path '{situation_path}' does not exist.")
            
            self.data[situation] = self._load_files_from_folder(situation_path, situation)

    def _load_files_from_folder(self, folder_path, situation):
        """
        Loads all text files in a folder and returns them as a dictionary.

        Parameters:
        -----------
        folder_path : str
            Path to the folder containing the data files.

        Returns:
        --------
        dict
            A dictionary where keys are file names (without extensions) 
            and values are the loaded NumPy arrays.
        """
        files = os.listdir(folder_path)
        situation_data = {}
        for file_name in files:
            if file_name.endswith(".txt"):  # Ensure only text files are loaded
                file_path = os.path.join(folder_path, file_name)

                if situation == "Walking_Pedestrian_Data":  # extract only needed data
                    situation_data[file_name[:-4]] = np.loadtxt(file_path, usecols=(0, 1, 2, 3, 4))
                    situation_data[file_name[:-4]][:, [3, 4]] = situation_data[file_name[:-4]][:, [4, 3]] # Swap y,z columns to be in the right order
                else:
                    situation_data[file_name[:-4]] = np.loadtxt(file_path)
        return situation_data
    
    def get_scenario_data(self, scenario):
        """
        Retrieves the data for a scenario.

        Parameters:
        -----------
        situation : str
            The name of the scenario to retrieve.

        Returns:
        --------
        dict
            A dictionary where keys are file names and values are NumPy arrays.
        """
        if scenario not in self.data:
            raise ValueError(f"Scenario '{scenario}' is not loaded.")
        return self.data[scenario]

    def get_situation_data(self, scenario, situation):
        """
        Retrieves the data for a specific situation.

        Parameters:
        -----------
        scenario : str
            The name of the scenario to retrieve.
        situation: str
            The name of the situation to retrieve.

        Returns:
        --------
        dict
            A dictionary where keys are file names and values are NumPy arrays.
        """
        if scenario not in self.data or situation not in self.data[scenario]:
            raise ValueError(f"Scenario '{scenario} {situation}' is not loaded.")
        return self.data[scenario][situation]

    def get_frame_rate(self):
        """
        Returns the frame rate of the dataset.

        Returns:
        --------
        float
            The frame rate.
        """
        return self.frame_rate
