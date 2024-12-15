import numpy as np
import os
import matplotlib.pyplot as plt

# Parameters
num_items = 100  # Number of items
vector_dim = 10  # Dimensionality of each item's vector
num_participants = 100  # Number of participants
learning_rate = 1.0  # Learning rate for weighted aggregation
weights = np.ones(num_participants) / num_participants  # Equal weights for simplicity

# Directory containing user updates
input_dir = "demo/dp_ratings_results/epsilon=1.0"
import matplotlib.pyplot as plt

# Parameters for aggregation
learning_rate = 1.0  # Learning rate for weighted aggregation
participant_counts = [10, 20, 50, 100, 200, 500, 1000]  # Different participant counts to test
available_participants = num_participants  # Matches the generated data

# Ensure participant counts do not exceed available data
participant_counts = [p for p in participant_counts if p <= available_participants]

# Experiment: Aggregate updates and observe noise cancellation
aggregated_noise_magnitudes_by_participants = []

for num_participants in participant_counts:
    # Initialize aggregated updates
    aggregated_non_dp = np.zeros((num_items, vector_dim))
    aggregated_dp = np.zeros((num_items, vector_dim))

    # Aggregate updates for the current number of participants
    for participant_id in range(num_participants):
        # Load non-DP and DP updates
        non_dp_path = os.path.join(input_dir, f"user_{participant_id}_non_dp.npy")
        dp_path = os.path.join(input_dir, f"user_{participant_id}_dp.npy")
        delta_V = np.load(non_dp_path, allow_pickle=True).item()
        delta_V_with_noise = np.load(dp_path, allow_pickle=True).item()

        # Weight and aggregate updates
        weight = 1.0 / num_participants  # Equal weights for simplicity
        for item_id in delta_V:
            aggregated_non_dp[item_id] += weight * delta_V[item_id]
            aggregated_dp[item_id] += weight * delta_V_with_noise[item_id]

    # Calculate noise in the aggregate and its magnitude
    aggregated_noise = aggregated_dp - aggregated_non_dp
    aggregated_noise_magnitude = np.linalg.norm(aggregated_noise, axis=1).mean()  # Average magnitude across items
    aggregated_noise_magnitudes_by_participants.append(aggregated_noise_magnitude)

# Plot noise magnitudes as a function of the number of participants
plt.figure(figsize=(10, 6))
plt.plot(participant_counts, aggregated_noise_magnitudes_by_participants, marker="o", color="red")
plt.title("Noise Cancellation Effect as Number of Participants Increases", fontsize=16)
plt.xlabel("Number of Participants", fontsize=14)
plt.ylabel("Average Noise Magnitude in Aggregate", fontsize=14)
plt.grid(alpha=0.3)
plt.show()