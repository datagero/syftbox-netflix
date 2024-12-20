import os
import copy
import numpy as np
from participant.participant_utils.data_loading import (
    load_tv_vocabulary, 
    load_global_item_factors, 
    load_participant_ratings,
    load_or_initialize_user_matrix)
from participant.federated_learning.svd_dp import (
    plot_delta_distributions,
    plot_ratings_norm,
    clip_deltas,
    apply_differential_privacy
)

def save_training_results(user_id, base_path, V, delta_V, U_u):
    """
    Save the updated V matrix, delta_V, and user matrix to disk.

    Parameters:
        user_id (str): Identifier for the user.
        base_path (str): Base directory for saving files.
        V (np.ndarray): Updated item factors matrix.
        delta_V (dict): Dictionary of delta updates for item factors.
        U_u (np.ndarray): Updated user matrix.
    """
    user_path = os.path.join(base_path, user_id)
    os.makedirs(user_path, exist_ok=True)  # Ensure the user directory exists

    # Save updated V
    participant_v_save_path = os.path.join(user_path, f"{user_id}_updated_V.npy")
    participant_deltav_save_path = os.path.join(user_path, f"{user_id}_delta_V.npy")
    np.save(participant_v_save_path, V)
    np.save(participant_deltav_save_path, delta_V)

    # Save updated user matrix
    user_matrix_path = os.path.join(user_path, f"{user_id}_U.npy")
    np.save(user_matrix_path, U_u)

    print(f"Updated and saved training results for user: {user_id} at {user_path}.")

def prepare_training_data(user_id, tv_vocab, final_ratings):
    """
    Prepare training data for the participant.
    """
    item_ids = {title: tv_vocab[title] for title in final_ratings if title in tv_vocab}
    return [(user_id, item_ids[t], final_ratings[t]) for t in final_ratings if t in item_ids]

def perform_local_training(train_data, initial_V, initial_U_u, alpha=0.01, lambda_reg=0.1, iterations=10):
    """
    Perform local training for the participant.
    """
    V = copy.deepcopy(initial_V)
    U_u = copy.deepcopy(initial_U_u)
    for _ in range(iterations):
        for (_, item_id, r) in train_data:
            pred = U_u.dot(V[item_id])
            error = r - pred
            U_u_grad = error * V[item_id] - lambda_reg * U_u
            V_i_grad = error * U_u - lambda_reg * V[item_id]
            U_u += alpha * U_u_grad
            V[item_id] += alpha * V_i_grad
    return initial_V, V, U_u

def participant_fine_tuning(user_id, private_folder, epsilon=None, clipping_threshold=None, noise_type="gaussian", save_path="mock_dataset_location/tmp_model_parms", plot=False):
    """
    Orchestrator function for participant fine-tuning.
    """
    # Step 1: Load vocabulary
    vocabulary_path = "aggregator/data/tv-series_vocabulary.json"
    tv_vocab = load_tv_vocabulary(vocabulary_path)

    # Step 2: Load global item factors
    V = load_global_item_factors(save_path)

    # Step 3: Load participant's ratings
    final_ratings = load_participant_ratings(private_folder)

    # Step 4: Load or initialize user matrix
    U_u = load_or_initialize_user_matrix(user_id, V.shape[1], save_path=os.path.join(save_path, user_id))

    # Step 5: Prepare training data
    train_data = prepare_training_data(user_id, tv_vocab, final_ratings)

    # Step 6: Perform local training
    initial_V, updated_V, updated_U_u = perform_local_training(train_data, V, U_u)

    # Step 7: Compute and privatize deltas
    ids_training = [item_id for (_, item_id, _) in train_data]
    delta_V = {item_id: updated_V[item_id] - initial_V[item_id] for (_, item_id, _) in train_data}
    # delta_V = {item_id: updated_V[item_id] - initial_V[item_id] for item_id, _ in enumerate(initial_V)}
    # delta_norms_before = [np.linalg.norm(v) for i, v in enumerate(delta_V.values()) if i in ids_training]
    delta_norms_before = [np.linalg.norm(v) for i, v in enumerate(delta_V.values())]

    # clipped_deltas, sensitivity = clip_deltas(delta_V, clipping_threshold)
    dp_deltas = apply_differential_privacy(delta_V, epsilon, 0.36, noise_type=noise_type)
    # dp_deltas = delta_V
    # delta_norms_after = [np.linalg.norm(v) for i, v in enumerate(dp_deltas.values()) if i in ids_training]
    delta_norms_after = [np.linalg.norm(v) for i, v in enumerate(dp_deltas.values())]

    # Step 8: Save results
    save_training_results(user_id, save_path, updated_V, dp_deltas, updated_U_u)

    if plot:
        # Step 9: Plot delta distributions
        # sorted_item_ids = sorted(delta_V.keys())
        sorted_item_ids = range(0, len(ids_training))
        # plot_delta_distributions(user_id, delta_norms_before, delta_norms_after)
        plot_ratings_norm(user_id, sorted_item_ids, delta_norms_before, delta_norms_after)

    print(f"Participant {user_id} finished training and updated item factors.")
    return delta_V
