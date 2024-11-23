import os
import pandas as pd

class FileManager:

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads the csv in the specified location
        Returns a DataFrame of the csv
        """
        if (self.exists_file(file_path)):
            df = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"File does not exist at location: {file_path}!")
        return df

    def save_csv(self, df, file_path: str) -> None:
        """
        Saves a DataFrame in the specified location as a csv
        """
        dirs = file_path.split('/')
        dirs.pop()
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
        df.to_csv(file_path, index=False)

    def load_all_csvs_in_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Combines all csvs in the specified directory
        Returns a DataFrame
        """
        file_paths = os.listdir(directory_path)

        combined_df = pd.DataFrame()
        for file_path in file_paths:
            csv = self.load_csv(os.path.join(directory_path, file_path))
            combined_df = pd.concat([combined_df, csv], axis=0, ignore_index=True)

        return combined_df

    def exists_file(self, file_path: str) -> bool:
        """
        Checks if the specified file exists
        Returns true if it does, otherwise false
        """
        return os.path.isfile(file_path)