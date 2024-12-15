import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(plt.style.available)

import pickle

def save_obj(obj, name):
    os.makedirs('demo/user_data', exist_ok=True)
    with open('demo/user_data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open('demo/user_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
# Step 1: Initialize the movie matrix (IMDB Ratings)
def initialize_movie_matrix(num_movies):
    """
    Generate a movie matrix with random IMDB ratings between 1 and 5.

    Args:
        num_movies (int): Number of movies.

    Returns:
        pd.DataFrame: Movie matrix with 'Movie_ID' and 'IMDB_Rating'.
    """
    movie_ids = [f"Movie_{i+1}" for i in range(num_movies)]
    imdb_ratings = np.random.uniform(1, 5, num_movies)  # Ratings between 1 and 5
    return pd.DataFrame({"Movie_ID": movie_ids, "IMDB_Rating": imdb_ratings})


# Step 2: Simulate user ratings and calculate deltas
def simulate_user_ratings(num_users, movie_matrix, min_rated, max_rated):
    """
    Simulate individual user ratings and compute deltas from the average IMDB ratings.

    Args:
        num_users (int): Number of users.
        movie_matrix (pd.DataFrame): DataFrame with movie IDs and IMDB ratings.
        min_rated (int): Minimum number of movies rated by a user.
        max_rated (int): Maximum number of movies rated by a user.

    Returns:
        tuple: (list of user ratings, list of user deltas)
    """
    user_ratings = []
    user_deltas = []
    for _ in range(num_users):
        # Select a random subset of movies rated by the user
        rated_movies = np.random.choice(
            movie_matrix["Movie_ID"], size=np.random.randint(min_rated, max_rated), replace=False
        )
        # Generate user ratings and calculate deltas
        ratings = {movie: np.random.uniform(1, 5) for movie in rated_movies}
        deltas = {
            movie: ratings[movie] - movie_matrix[movie_matrix["Movie_ID"] == movie]["IMDB_Rating"].values[0]
            for movie in ratings
        }
        user_ratings.append(ratings)
        user_deltas.append(deltas)
    return user_ratings, user_deltas

def simulate_user_ratings_with_bias(num_users, movie_matrix, min_rated, max_rated):
    """
    Simulate individual user ratings and compute deltas with a bias toward specific movies.
    Users independently rate each biased movie with an 80% probability.

    Args:
        num_users (int): Number of users.
        movie_matrix (pd.DataFrame): DataFrame with movie IDs and IMDB ratings.
        min_rated (int): Minimum number of movies rated by a user.
        max_rated (int): Maximum number of movies rated by a user.

    Returns:
        tuple: (list of user ratings, list of user deltas)
    """
    user_ratings = []
    user_deltas = []
    high_bias_movie = "Movie_55"  # Movie with biased high ratings
    low_bias_movie = "Movie_75"  # Movie with biased low ratings

    for _ in range(num_users):
        # Determine the number of movies rated by the user
        num_rated = np.random.randint(min_rated, max_rated)

        # Initialize rated movies list
        rated_movies = set()

        # Include biased movies with 80% probability
        if np.random.rand() < 0.8:
            rated_movies.add(high_bias_movie)
        if np.random.rand() < 0.8:
            rated_movies.add(low_bias_movie)

        # Select other random movies
        while len(rated_movies) < num_rated:
            random_movie = np.random.choice(movie_matrix["Movie_ID"])
            rated_movies.add(random_movie)

        # Generate user ratings and calculate deltas
        ratings = {
            movie: (np.random.uniform(4, 5) if movie == high_bias_movie else
                    np.random.uniform(1, 2) if movie == low_bias_movie else
                    np.random.uniform(1, 5))
            for movie in rated_movies
        }
        deltas = {
            movie: ratings[movie] - movie_matrix[movie_matrix["Movie_ID"] == movie]["IMDB_Rating"].values[0]
            for movie in ratings
        }
        user_ratings.append(ratings)
        user_deltas.append(deltas)

    return user_ratings, user_deltas

def simulate_user_ratings_with_bias_and_multiples(num_users, movie_matrix, min_rated, max_rated):
    """
    Simulate individual user ratings and compute deltas with a bias toward specific movies.
    User 0 rates all movies that are multiples of 10 highly, while others rate movies randomly with biases.

    Args:
        num_users (int): Number of users.
        movie_matrix (pd.DataFrame): DataFrame with movie IDs and IMDB ratings.
        min_rated (int): Minimum number of movies rated by a user.
        max_rated (int): Maximum number of movies rated by a user.

    Returns:
        tuple: (list of user ratings, list of user deltas)
    """
    user_ratings = []
    user_deltas = []
    high_bias_movie = "Movie_55"  # Movie with biased high ratings
    low_bias_movie = "Movie_75"  # Movie with biased low ratings

    for user_id in range(num_users):
        # Determine the number of movies rated by the user
        num_rated = np.random.randint(min_rated, max_rated)

        # Initialize rated movies list
        rated_movies = set()

        if user_id == 0:
            # User 0 rates all movies that are multiples of 10
            rated_movies = {f"Movie_{i}" for i in range(10, len(movie_matrix) + 1, 10)}
        else:
            # Include biased movies with 80% probability for other users
            if np.random.rand() < 0.8:
                rated_movies.add(high_bias_movie)
            if np.random.rand() < 0.8:
                rated_movies.add(low_bias_movie)

            # Select other random movies
            while len(rated_movies) < num_rated:
                random_movie = np.random.choice(movie_matrix["Movie_ID"])
                rated_movies.add(random_movie)

        # Generate user ratings and calculate deltas
        ratings = {
            movie: (5 if movie == high_bias_movie else
                    np.random.uniform(1, 2) if movie == low_bias_movie else
                    5 if user_id == 0 and int(movie.split("_")[1]) % 10 == 0 else
                    np.random.uniform(1, 5))
            for movie in rated_movies
        }
        deltas = {
            movie: ratings[movie] - movie_matrix[movie_matrix["Movie_ID"] == movie]["IMDB_Rating"].values[0]
            for movie in ratings
        }
        user_ratings.append(ratings)
        user_deltas.append(deltas)

    return user_ratings, user_deltas

def add_dp_noise_and_save(user_deltas, output_dir, num_items, epsilon, sensitivity, noise_type="laplace"):
    """
    Add noise for differential privacy to user updates and save the results, scaling noise based on the number of rated movies.

    Args:
        user_deltas (list): List of user deltas.
        output_dir (str): Directory to save non-DP and DP outputs.
        num_items (int): Total number of items (movies).
        epsilon (float): Privacy budget.
        sensitivity (float): Sensitivity of the aggregation function.
        noise_type (str): Type of noise to add ("laplace" or "gaussian").
    """
    os.makedirs(output_dir, exist_ok=True)
    for user_id, delta in enumerate(user_deltas):
        # Initialize full deltas (all items as zero)
        delta_full = {item_id: 0.0 for item_id in range(num_items)}
        delta_full.update({int(movie.split("_")[1]) - 1: value for movie, value in delta.items()})

        # Calculate the number of rated movies (k)
        k = len(delta)

        # Add noise (scale depends on the noise type and k)
        delta_with_noise = {}
        for item_id, value in delta_full.items():
            if noise_type == "laplace":
                noise_scale = (sensitivity / epsilon) # * k  # Scale for Laplace noise
                noise = np.random.laplace(loc=0, scale=noise_scale)
            elif noise_type == "gaussian":
                delta = 1e-5  # Typical value for δ
                # noise_scale = (sensitivity / epsilon) * np.sqrt(k) * np.sqrt(2 * np.log(1.25 / delta))
                noise_scale = (sensitivity / epsilon) * np.sqrt(k)  # Scale for Gaussian noise
                noise = np.random.normal(loc=0, scale=noise_scale)
            else:
                raise ValueError("Invalid noise type. Choose 'laplace' or 'gaussian'.")
            delta_with_noise[item_id] = value + noise

        # Save non-DP and DP results
        non_dp_path = os.path.join(output_dir, f"user_{user_id}_non_dp.npy")
        dp_path = os.path.join(output_dir, f"user_{user_id}_dp.npy")
        np.save(non_dp_path, delta_full)
        np.save(dp_path, delta_with_noise)

# Combined visualization function calling both row functions
def visualize_combined(output_dir, demo_users, movie_matrix):
    """
    Generate a combined figure with two rows:
    - Row 1: Line plots for all items.
    - Row 2: Bar plots for rated movies only.

    Args:
        output_dir (str): Directory containing the saved user updates.
        demo_users (list): List of user IDs to visualize.
        movie_matrix (pd.DataFrame): Matrix of movies with IMDB ratings.
    """
    # Set the overall style
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, axes = plt.subplots(2, len(demo_users), figsize=(24, 14), sharey=False)

    # Row 1: Line plots for all items
    for i, user_id in enumerate(demo_users):
        # Load data
        non_dp_path = os.path.join(output_dir, f"user_{user_id}_non_dp.npy")
        dp_path = os.path.join(output_dir, f"user_{user_id}_dp.npy")
        delta_non_dp = np.load(non_dp_path, allow_pickle=True).item()
        delta_dp = np.load(dp_path, allow_pickle=True).item()

        # Extract data for all items
        item_ids = list(delta_non_dp.keys())
        non_dp_magnitudes = [delta_non_dp[item_id] for item_id in item_ids]
        dp_magnitudes = [delta_dp[item_id] for item_id in item_ids]

        # Line plot for all items with narrower lines
        axes[0, i].plot(item_ids, non_dp_magnitudes, color="blue", lw=1, markersize=4, label="Non-DP Output")
        axes[0, i].plot(item_ids, dp_magnitudes, color="orange", lw=1, markersize=4, label="DP Output")
        axes[0, i].set_title(f"User {user_id} (All Items)", fontsize=12, weight='bold')
        axes[0, i].tick_params(axis='x', rotation=45, labelsize=8)
        axes[0, i].grid(alpha=0.4)

        if i == 0:
            axes[0, i].set_ylabel("Update Magnitude (All Items)", fontsize=12)
        axes[0, i].legend(fontsize=10)

    # Row 2: Bar plots for rated movies only
    for i, user_id in enumerate(demo_users):
        # Load data
        non_dp_path = os.path.join(output_dir, f"user_{user_id}_non_dp.npy")
        dp_path = os.path.join(output_dir, f"user_{user_id}_dp.npy")
        delta_non_dp = np.load(non_dp_path, allow_pickle=True).item()
        delta_dp = np.load(dp_path, allow_pickle=True).item()

        # Extract rated movies and their deltas
        rated_movies = [
            movie_matrix.iloc[item_id]["Movie_ID"] for item_id in delta_non_dp if delta_non_dp[item_id] != 0
        ]
        non_dp_magnitudes = [delta_non_dp[item_id] for item_id in delta_non_dp if delta_non_dp[item_id] != 0]
        dp_magnitudes = [delta_dp[item_id] for item_id in delta_non_dp if delta_non_dp[item_id] != 0]

        # Bar plot for rated movies
        x = np.arange(len(rated_movies))
        axes[1, i].bar(x - 0.2, non_dp_magnitudes, width=0.4, color="blue", label="Non-DP Output", alpha=0.7)
        axes[1, i].bar(x + 0.2, dp_magnitudes, width=0.4, color="orange", label="DP Output", alpha=0.7)

        # Formatting
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(rated_movies, rotation=45, ha="right", fontsize=8)
        axes[1, i].set_title(f"User {user_id} (Rated Movies Only)", fontsize=12, weight='bold')
        if i == 0:
            axes[1, i].set_ylabel("Update Magnitude (Rated Movies)", fontsize=12)
        axes[1, i].legend(fontsize=10)
        axes[1, i].grid(alpha=0.4)

    # Add a global title and adjust layout
    fig.suptitle("Combined Visualization: Non-DP vs DP Outputs for Selected Users", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = f"demo/figures/epsilon={epsilon}"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/demo_movies_individual_dp.png", dpi=300)
    plt.show()

# Step 5: Recompute optimal clipping threshold based on update magnitudes
def calculate_clipping_threshold(user_deltas, scaling_factor):
    """
    Calculate optimal clipping threshold based on the mean update magnitude.

    Args:
        user_deltas (list): List of user deltas.
        scaling_factor (float): Factor to scale the mean magnitude.

    Returns:
        float: Optimal clipping threshold.
    """
    magnitudes = []
    for delta in user_deltas:
        magnitudes.extend([abs(value) for value in delta.values()])
    mean_magnitude = np.mean(magnitudes)
    return scaling_factor * mean_magnitude


def visualize_comparison_across_epsilons(output_dir_epsilon1, output_dir_epsilon2, movie_matrix):
    """
    Compare visualizations for User 0 across two different epsilon values.

    Args:
        output_dir_epsilon1 (str): Directory containing user updates for epsilon 1.
        output_dir_epsilon2 (str): Directory containing user updates for epsilon 2.
        movie_matrix (pd.DataFrame): Matrix of movies with IMDB ratings.

    Returns:
        None
    """
    # Define epsilon values for titles
    epsilon1 = output_dir_epsilon1.split('epsilon=')[-1]
    epsilon2 = output_dir_epsilon2.split('epsilon=')[-1]

    # Load data for user 0
    user_id = 0
    non_dp_path_eps1 = os.path.join(output_dir_epsilon1, f"user_{user_id}_non_dp.npy")
    dp_path_eps1 = os.path.join(output_dir_epsilon1, f"user_{user_id}_dp.npy")

    non_dp_path_eps2 = os.path.join(output_dir_epsilon2, f"user_{user_id}_non_dp.npy")
    dp_path_eps2 = os.path.join(output_dir_epsilon2, f"user_{user_id}_dp.npy")

    delta_non_dp_eps1 = np.load(non_dp_path_eps1, allow_pickle=True).item()
    delta_dp_eps1 = np.load(dp_path_eps1, allow_pickle=True).item()

    delta_non_dp_eps2 = np.load(non_dp_path_eps2, allow_pickle=True).item()
    delta_dp_eps2 = np.load(dp_path_eps2, allow_pickle=True).item()

    # Extract data
    item_ids_eps1 = list(delta_non_dp_eps1.keys())
    non_dp_magnitudes_eps1 = [delta_non_dp_eps1[item_id] for item_id in item_ids_eps1]
    dp_magnitudes_eps1 = [delta_dp_eps1[item_id] for item_id in item_ids_eps1]

    item_ids_eps2 = list(delta_non_dp_eps2.keys())
    non_dp_magnitudes_eps2 = [delta_non_dp_eps2[item_id] for item_id in item_ids_eps2]
    dp_magnitudes_eps2 = [delta_dp_eps2[item_id] for item_id in item_ids_eps2]

    # Create subplots for side-by-side comparison
    fig, axes = plt.subplots(2, 2, figsize=(24, 12), sharey=False)

    # Row 1: Line plots for epsilon 1 and epsilon 2
    axes[0, 0].plot(item_ids_eps1, non_dp_magnitudes_eps1, color="blue", lw=1, label="Non-DP Output")
    axes[0, 0].plot(item_ids_eps1, dp_magnitudes_eps1, color="orange", lw=1, label="DP Output")
    axes[0, 0].set_title(f"Epsilon={epsilon1}: Line Plot (All Items)", fontsize=14, weight='bold')
    axes[0, 0].set_ylabel("Update Magnitude", fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.4)

    axes[0, 1].plot(item_ids_eps2, non_dp_magnitudes_eps2, color="blue", lw=1, label="Non-DP Output")
    axes[0, 1].plot(item_ids_eps2, dp_magnitudes_eps2, color="orange", lw=1, label="DP Output")
    axes[0, 1].set_title(f"Epsilon={epsilon2}: Line Plot (All Items)", fontsize=14, weight='bold')
    axes[0, 1].grid(alpha=0.4)

    # Row 2: Bar plots for rated movies
    rated_movies_eps1 = [
        movie_matrix.iloc[item_id]["Movie_ID"] for item_id in delta_non_dp_eps1 if delta_non_dp_eps1[item_id] != 0
    ]
    non_dp_rated_eps1 = [delta_non_dp_eps1[item_id] for item_id in delta_non_dp_eps1 if delta_non_dp_eps1[item_id] != 0]
    dp_rated_eps1 = [delta_dp_eps1[item_id] for item_id in delta_non_dp_eps1 if delta_non_dp_eps1[item_id] != 0]

    rated_movies_eps2 = [
        movie_matrix.iloc[item_id]["Movie_ID"] for item_id in delta_non_dp_eps2 if delta_non_dp_eps2[item_id] != 0
    ]
    non_dp_rated_eps2 = [delta_non_dp_eps2[item_id] for item_id in delta_non_dp_eps2 if delta_non_dp_eps2[item_id] != 0]
    dp_rated_eps2 = [delta_dp_eps2[item_id] for item_id in delta_non_dp_eps2 if delta_non_dp_eps2[item_id] != 0]

    x_eps1 = np.arange(len(rated_movies_eps1))
    axes[1, 0].bar(x_eps1 - 0.2, non_dp_rated_eps1, width=0.4, color="blue", label="Non-DP Output", alpha=0.7)
    axes[1, 0].bar(x_eps1 + 0.2, dp_rated_eps1, width=0.4, color="orange", label="DP Output", alpha=0.7)
    axes[1, 0].set_title(f"Epsilon={epsilon1}: Bar Plot (Rated Movies)", fontsize=14, weight='bold')
    axes[1, 0].set_xticks(x_eps1)
    axes[1, 0].set_xticklabels(rated_movies_eps1, rotation=45, fontsize=8)
    axes[1, 0].set_ylabel("Update Magnitude", fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.4)

    x_eps2 = np.arange(len(rated_movies_eps2))
    axes[1, 1].bar(x_eps2 - 0.2, non_dp_rated_eps2, width=0.4, color="blue", label="Non-DP Output", alpha=0.7)
    axes[1, 1].bar(x_eps2 + 0.2, dp_rated_eps2, width=0.4, color="orange", label="DP Output", alpha=0.7)
    axes[1, 1].set_title(f"Epsilon={epsilon2}: Bar Plot (Rated Movies)", fontsize=14, weight='bold')
    axes[1, 1].set_xticks(x_eps2)
    axes[1, 1].set_xticklabels(rated_movies_eps2, rotation=45, fontsize=8)
    axes[1, 1].grid(alpha=0.4)

    # Add global title and adjust layout
    fig.suptitle("Comparison of Non-DP vs DP Outputs for User 0 Across Epsilon Values", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("demo/figures/comparison_across_epsilons.png", dpi=300)
    plt.show()


def generate_visualization(output_dir):
    movie_matrix = load_obj('movie_matrix')

    # Visualization for selected users
    demo_users = [0, 1, 2, 3, 4]

    # Call the combined visualization function
    visualize_combined(output_dir, demo_users, movie_matrix)

if __name__ == "__main__":

    # Execution Parameters
    num_movies = 100
    num_users = 100
    min_rated = int(0.04 * num_movies)
    max_rated = int(0.08 * num_movies)

    scaling_factor = 1.5

    # Step 6: Execution
    movie_matrix = initialize_movie_matrix(num_movies)
    # user_ratings, user_deltas = simulate_user_ratings_with_bias(num_users, movie_matrix, min_rated, max_rated)
    user_ratings, user_deltas = simulate_user_ratings_with_bias_and_multiples(num_users, movie_matrix, min_rated, max_rated)

    save_obj(movie_matrix, 'movie_matrix')
    save_obj(user_deltas, 'user_deltas')
    save_obj(user_ratings, 'user_ratings')

    optimal_clipping_threshold = calculate_clipping_threshold(user_deltas, scaling_factor)
    print("Optimal Clipping Threshold:", optimal_clipping_threshold)

    epsilon = 1.0 # Privacy budget
    output_dir_epsilon1 = f"demo/dp_ratings_results/epsilon={epsilon}"

    add_dp_noise_and_save(user_deltas, output_dir_epsilon1, num_movies, epsilon, optimal_clipping_threshold, noise_type="laplace")
    generate_visualization(output_dir_epsilon1)

    # epsilon = 10.0 # Privacy budget
    # output_dir_epsilon2 = f"demo/dp_ratings_results/epsilon={epsilon}"

    # add_dp_noise_and_save(user_deltas, output_dir_epsilon2, num_movies, epsilon, optimal_clipping_threshold, noise_type="laplace")
    # generate_visualization(output_dir_epsilon2)

    # visualize_comparison_across_epsilons(output_dir_epsilon1, output_dir_epsilon2, movie_matrix)