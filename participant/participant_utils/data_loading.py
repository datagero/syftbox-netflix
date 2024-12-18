import os
import sys
import json
import numpy as np
import csv
from typing import Tuple
from datetime import datetime
import subprocess
from loaders.netflix_loader import download_daily_data, get_latest_file

def load_tv_vocabulary(vocabulary_path):
    """
    Load the TV series vocabulary from the specified JSON file.
    """
    with open(vocabulary_path, "r") as f:
        return json.load(f)

def load_participant_ratings(private_folder):
    """
    Load participant's ratings from the private folder.
    """
    ratings_path = os.path.join(private_folder, "ratings.npy")
    return np.load(ratings_path, allow_pickle=True).item()

def load_global_item_factors(save_path):
    """
    Load the global item factors matrix (V).
    """
    global_V_path = os.path.join(save_path, "global_V.npy")
    return np.load(global_V_path)

def load_or_initialize_user_matrix(user_id, latent_dim, save_path="mock_dataset_location/tmp_model_parms"):

    user_matrix_path = os.path.join(save_path, f"{user_id}_U.npy")
    if os.path.exists(user_matrix_path):
        U_u = np.load(user_matrix_path)
        print(f"Loaded existing user matrix for {user_id}.")
    else:
        U_u = initialize_user_matrix(user_id, latent_dim, save_path)
    return U_u

def initialize_user_matrix(user_id, latent_dim, save_path="mock_dataset_location/tmp_model_parms"):

    # Create save directory if not exists
    os.makedirs(save_path, exist_ok=True)

    # Initialize user matrix
    U_u = np.random.normal(scale=0.01, size=(latent_dim,))

    # Save user matrix
    user_matrix_path = os.path.join(save_path, f"{user_id}_U.npy")
    np.save(user_matrix_path, U_u)
    print(f"Initialized and saved user matrix for {user_id}.")
    return U_u

def load_csv_to_numpy(file_path: str) -> np.ndarray:
    """
    Load a CSV file into a NumPy array, handling quoted fields.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        np.ndarray: A 2D NumPy array containing the data from the CSV.
    """
    cleaned_data = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            cleaned_data.append(row)

    return np.array(cleaned_data)


def get_or_download_latest_data(output_dir, csv_name, profile:str=None) -> Tuple[str, np.ndarray]:
    """
    Ensure the latest Netflix data exists or download it if missing.
    After retrieval, load the data into a NumPy array for further processing.

    Returns:
        np.ndarray: The latest Netflix viewing history as a structured array.
    """
    # Construct paths and file names
    # datapath = os.path.expanduser(output_dir) # removed to work properly on macOS
    datapath = os.path.join(output_dir, profile) if profile else output_dir
    os.makedirs(datapath, exist_ok=True)

    today_date = datetime.now().strftime("%Y-%m-%d")
    netflix_csv_prefix = os.path.splitext(csv_name)[0]
    
    filename = f"{netflix_csv_prefix}_{today_date}.csv"
    file_path = os.path.join(datapath, filename)
    file_path_static = os.path.join(datapath, netflix_csv_prefix + ".csv")

    static_file = None
    try:
        # Try to download the file using Chromedriver
        try:
            chromedriver_path = subprocess.check_output(['which', 'chromedriver'], text=True).strip()
            os.environ['CHROMEDRIVER_PATH'] = chromedriver_path
            if not os.path.exists(file_path):
                print(f"Data file not found. Downloading to {file_path}...")
                download_daily_data(datapath, filename, profile)
                print(f"Successfully downloaded Netflix data to {file_path}.")
            static_file = False
            
        except subprocess.CalledProcessError:
            print(f">> ChromeDriver not found. Unable to retrieve from Netflix via download.")
            print(f"Checking for a locally available static file: {file_path_static}...")
            static_file = os.path.exists(file_path_static)
            
        except Exception as e:
            print(f"{e}")

            # Try to use the static file if downloading failed
            if os.path.exists(file_path_static):
                print(f"Using static viewing history (manually downloaded from Netflix): {file_path_static}...")
                static_file = True
            else:
                print((
                    f">> Neither ChromeDriver is available for download nor the static file exists. "
                    f"Please retrieve the file manually from Netflix and make it available here: \n\t\t {datapath}"
                ))
            
                print(f">> Copying dummy file (data/dummy.csv) to {file_path_static}.")
                try:
                    with open('data/dummy.csv', 'rb') as src_file:
                        with open(file_path_static, 'wb') as dest_file:
                            dest_file.write(src_file.read())
                    print(f">> Copied dummy file (data/dummy.csv) to {file_path_static}. For test purpose only!")
                    static_file = True
                except Exception as e: 
                    print(f"[!] Error copying dummy file (data/dummy.csv) to {file_path_static}: {e}")
                    sys.exit(1)

    except Exception as e:
        print(f"Error retrieving Netflix data: {e}")
        raise

    if static_file is None:
        print("[!] Critical error: static_file is undefined!")
        sys.exit(1)

    if static_file:
        latest_data_file = file_path_static
    else:
        latest_data_file = get_latest_file(datapath, csv_name)

    # Load the CSV into a NumPy array
    print(f"Loading data from {latest_data_file}...")
    return latest_data_file, load_csv_to_numpy(latest_data_file)