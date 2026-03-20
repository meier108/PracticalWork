import numpy as np
from sklearn.isotonic import spearmanr
import tensorflow as tf
import umap
import pandas as pd
import matplotlib.pyplot as plt


from models.oracle import oracle_lookup_one_hot, oracle_exists_one_hot

def check_already_scored(mutation, scored_sequences):
    """
    Check if a mutated sequence has already been scored.
    
    Args:
        mutation: The mutated sequence to check
        scored_sequences: A set of already scored sequences
    Returns:
        True if the mutation has already been scored, False otherwise
    """
    return np.any(np.all(scored_sequences == mutation, axis=1))


def mutate_score_sequence_GB1(mutation_model, scorer, seed_sequence, score_seed=None, steps=100, num_mutations=50, 
                               top_k=5, acceptance_threshold=0.0, patience=10, use_exhaustive=True):
    """
    Generate mutated sequences with improved exploration.
    
    Args:
        mutation_model: Model to generate mutations (e.g., SMW)
        scorer: Model to score sequences
        seed_sequence: Initial sequence (list of indices)
        score_seed: Initial score. If None, will be computed from seed.
        steps: Number of mutation steps
        num_mutations: Number of mutations per step (ignored if use_exhaustive=True)
        top_k: Keep top-k candidates at each step for diversity
        acceptance_threshold: Accept mutations within this threshold of current best (allows slight regression)
        patience: Restart from best if no improvement for this many steps
        use_exhaustive: If True, use all single-point mutations (recommended for short sequences)
    """
    # Compute seed score if not provided
    if score_seed is None:
        seed_onehot = mutation_model.one_hot_encode(seed_sequence)
        score_seed = scorer.predict(seed_onehot)
        print(f"Computed seed score: {score_seed:.4f}")
    
    best_score = score_seed
    best_sequence = list(seed_sequence)
    current_score = score_seed
    current_sequence = list(seed_sequence)
    steps_without_improvement = 0
    
    values = {'mutation': [], 'predicted_score': []}
    
    # Record the seed as the first point
    seed_onehot = mutation_model.one_hot_encode(seed_sequence)
    values['mutation'].append(seed_onehot)
    values['predicted_score'].append(score_seed)
    
    for step in range(steps):
        # Generate mutations from current sequence
        if use_exhaustive and hasattr(mutation_model, 'mutate_all_positions'):
            mutated_sequences = mutation_model.mutate_all_positions(current_sequence)
        else:
            mutated_sequences = mutation_model.mutate(current_sequence, n_steps=num_mutations)
        
        # Score all mutations and rank them
        scored_mutations = []
        for mutated_seq in mutated_sequences:
            score = scorer.predict(mutated_seq)
            scored_mutations.append((mutated_seq, score))
        
        # Sort by score (descending) and take top-k
        scored_mutations.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_mutations[:top_k]
        
        # Check if any candidate improves over current
        found_improvement = False
        for mutated_seq, score in top_candidates:
            # Accept if better than current OR within acceptance threshold
            if score > current_score - acceptance_threshold:
                # Record if it's an actual improvement over current
                if score > current_score:
                    values['mutation'].append(mutated_seq)
                    values['predicted_score'].append(score)
                    print(f"Step {step+1}: Score {score:.4f} (improved from {current_score:.4f})")
                    found_improvement = True
                    steps_without_improvement = 0
                    
                    # Update global best
                    if score > best_score:
                        best_score = score
                        best_sequence = tf.argmax(tf.reshape(mutated_seq, (mutated_seq.shape[0] // 20, 20)), axis=-1).numpy().tolist()
                
                # Move to this candidate (even if slight regression for exploration)
                current_score = score
                current_sequence = tf.argmax(tf.reshape(mutated_seq, (mutated_seq.shape[0] // 20, 20)), axis=-1).numpy().tolist()
                break
        
        if not found_improvement:
            steps_without_improvement += 1
            # Restart from best if stuck
            if steps_without_improvement >= patience:
                print(f"Step {step+1}: Restarting from best (score={best_score:.4f})")
                current_sequence = best_sequence
                current_score = best_score
                steps_without_improvement = 0
    
    print(f"\nFinal: Found {len(values['mutation'])} trajectory points. Best score: {best_score:.4f}")
    return values



def mutate_score_sequences(mutation_model, scorer, oracle, seed_sequence, score_seed_sequence, scored_sequences, steps, num_mutations):
    """
    Generate mutated sequences and score them using the provided scorer.
    
    Args:
        mutation_model: Model to generate mutations (e.g., SMW)
        scorer: Model to score sequences (e.g., LSTMSequencePredictor or RandomForestPredictor)
        seed_sequence: Initial sequence to start mutations from
        score_seed_sequence: Initial score of the seed sequence
        steps: Number of mutation steps to perform
        num_mutations: Number of mutations to generate at each step
        scored_sequences: A set of already scored sequences
        oracle: A function that takes a sequence and returns its true binding score
        """
    
    old_score = score_seed_sequence
    values = {'mutation': [], 'predicted_score': []}

    for step in range(steps):
        mutated_sequences = mutation_model.mutate(seed_sequence, num_mutations)
        
        for idx, mutation in enumerate(mutated_sequences):
            already_scored = check_already_scored(mutation, scored_sequences)
            if not already_scored and  oracle_exists_one_hot(oracle, mutation):
                score = scorer.predict(mutation)
                if score > old_score:
                    # Update seed sequence and score if the new score is better
                    seed_sequence = tf.argmax(tf.reshape(mutation, (mutation.shape[0] // 4, 4)), axis=-1).numpy().tolist()
                    old_score = score
                    values['mutation'].append(mutation)
                    values['predicted_score'].append(score)
                    scored_sequences.append(tuple(mutation.flatten()))
                    print(f"Step {step+1}, Mutation {idx+1}: New score {score:.4f} (improved from {old_score:.4f})")
                else:
                    continue
                    #print(f"Step {step+1}, Mutation {idx+1}: Score {score:.4f} did not improve from {old_score:.4f}, skipping.")
            else:
                continue
                #print(f"Step {step+1}, Mutation {idx+1}: Already scored or not in oracle, skipping.")
                
    return values

def lookup_oracle(df_values, oracle):
    """
    Look up the true binding scores for a list of mutated sequences using the oracle.
    
    Args:
        df_values: A dictionary containing 'mutation' and 'predicted_score' lists
        oracle: A function that takes a sequence and returns its true binding score
    Returns:
        dataframe with mutations, predicted scores, true scores and error
        """
    df_values['real_score'] = df_values['mutation'].apply(lambda x: oracle_lookup_one_hot(oracle, x))
    df_values['error'] = df_values['real_score'] - df_values['predicted_score']
    return df_values

def plot_results_normalized(df_values):
    """
    Plot the predicted scores vs. true scores.
    Calculate spearman correlation between predicted and true scores.

    Args:
        df_values: A dataframe containing 'predicted_score', 'real_score', and 'error' columns
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    
    # Scatter plot of predicted vs real scores
    plt.figure(figsize=(14, 5)) 

    # Normalize both to 0-1 range for comparison
    pred_norm = (df_values['predicted_score'] - df_values['predicted_score'].min()) / (df_values['predicted_score'].max() - df_values['predicted_score'].min())
    real_norm = (df_values['real_score'] - df_values['real_score'].min()) / (df_values['real_score'].max() - df_values['real_score'].min())

    # Spearman Correlation between predicted and real scores
    spearman_corr, _ = spearmanr(df_values['predicted_score'], df_values['real_score'])
    print(f"Spearman Correlation: {spearman_corr:.4f}")

    plt.plot(pred_norm, label='Predicted (normalized)', marker='o', markersize=4, alpha=0.7)
    plt.plot(real_norm, label='Real (normalized)', marker='s', markersize=4, alpha=0.7)
    plt.xlabel('Optimization Step')
    plt.ylabel('Score (normalized)')
    plt.title('Are Predicted and Real Scores Growing Together?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_results(df_values, title="Predicted vs Real Scores"):
    """
    Plot the predicted scores vs. true scores.
    Calculate spearman correlation between predicted and true scores.

    Args:
        df_values: A dataframe containing 'predicted_score', 'real_score', and 'error' columns
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    
    # Scatter plot of predicted vs real scores
    plt.figure(figsize=(14, 5)) 

    # Normalize both to 0-1 range for comparison
    pred_norm = df_values['predicted_score']
    real_norm = df_values['real_score']

    # Spearman Correlation between predicted and real scores
    spearman_corr, _ = spearmanr(df_values['predicted_score'], df_values['real_score'])
    print(f"Spearman Correlation: {spearman_corr:.4f}")

    plt.plot(pred_norm, label='Predicted', marker='o', markersize=4, alpha=0.7)
    plt.plot(real_norm, label='Real', marker='s', markersize=4, alpha=0.7)
    plt.xlabel('Optimization Step', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_umap_embedding(df, testset : str, values : pd.DataFrame, SEED=42, vocab_size=20):
    # Prepare sequences for embedding
    test_df = df[df['split'] == testset]
    test_sequences_indexed = np.array(test_df['encoded_sequence'].tolist())
    X_test_onehot = tf.one_hot(test_sequences_indexed, depth=vocab_size).numpy().reshape(len(test_sequences_indexed), -1)
    X_trajectory = np.vstack(values['mutation'].values)
    test_scores = test_df['binding_scores'].values

    # Combine for consistent embedding
    X_combined = np.vstack([X_test_onehot, X_trajectory])

    # Fit UMAP
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_combined)

    # Split embeddings
    n_test = len(X_test_onehot)
    embedding_test = embedding[:n_test]
    embedding_trajectory = embedding[n_test:]
    trajectory_real_scores = values['predicted_score'].values

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background landscape colored by binding scores
    scatter = ax.scatter(
        embedding_test[:, 0], embedding_test[:, 1],
        c=test_scores, cmap='viridis', alpha=0.3, s=10
    )

    cbar = plt.colorbar(scatter)          
    cbar.set_label("Binding Score", fontsize=16)   # label text size
    cbar.ax.tick_params(labelsize=14)              # tick numbers size

    # Draw arrows between consecutive trajectory points
    for i in range(len(embedding_trajectory) - 1):
        ax.annotate(
            '',
            xy=(embedding_trajectory[i+1, 0], embedding_trajectory[i+1, 1]),
            xytext=(embedding_trajectory[i, 0], embedding_trajectory[i, 1]),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7)
        )

    # Trajectory points colored by score
    ax.scatter(
        embedding_trajectory[:, 0], embedding_trajectory[:, 1],
        c=trajectory_real_scores, cmap='viridis',
        edgecolors='red', linewidths=1.5, s=80, zorder=5
    )

    # Start and end markers
    ax.scatter(embedding_trajectory[0, 0], embedding_trajectory[0, 1],
            c='lime', s=250, marker='*', edgecolors='black', linewidths=2,
            label=f'Start (score={trajectory_real_scores[0]:.3f})', zorder=10)
    ax.scatter(embedding_trajectory[-1, 0], embedding_trajectory[-1, 1],
            c='red', s=250, marker='*', edgecolors='black', linewidths=2,
            label=f'End (score={trajectory_real_scores[-1]:.3f})', zorder=10)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    #ax.set_title('Optimization Trajectory on GB1 Fitness Landscape - Testset: ' + testset)
    ax.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.show()



def plot_umap_multi_trajectories(df_oracle, df_all_walks, SEED=42, title="Multi-Run Trajectories on Fitness Landscape"):
    """
    Plot multiple optimization trajectories on a shared UMAP embedding.
    
    Args:
        df_oracle: DataFrame with oracle data (must have 'one_hot_seq' and 'binding_scores')
        df_all_walks: DataFrame from run_multiple_walks with columns [run_id, step, predicted_score, real_score, mutation]
        SEED: Random seed for reproducibility
        title: Plot title
    """
    import matplotlib.cm as cm
    
    if df_all_walks.empty:
        print("Warning: df_all_walks is empty. No trajectories to plot.")
        return
    
    # Prepare oracle sequences for background landscape
    X_oracle = np.vstack(df_oracle['one_hot_seq'].values)
    oracle_scores = df_oracle['binding_scores'].values
    
    
    # Get all trajectory sequences
    X_trajectories = np.vstack(df_all_walks['mutation'].values)
    
    # Combine for consistent embedding
    X_combined = np.vstack([X_oracle, X_trajectories])
    
    # Fit UMAP on combined data
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_combined)
    
    # Split back
    n_oracle = len(X_oracle)
    embedding_oracle = embedding[:n_oracle]
    embedding_trajectories = embedding[n_oracle:]
    
    # Map trajectory embeddings back to runs
    df_all_walks = df_all_walks.copy()
    df_all_walks['umap_x'] = embedding_trajectories[:, 0]
    df_all_walks['umap_y'] = embedding_trajectories[:, 1]
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Background landscape
    scatter = ax.scatter(
        embedding_oracle[:, 0], embedding_oracle[:, 1],
        c=oracle_scores, cmap='viridis', alpha=0.2, s=10
    )
    plt.colorbar(scatter, label='Binding Score')
    
    # Get distinct colors for runs
    run_ids = df_all_walks['run_id'].unique()
    n_runs = len(run_ids)
    colors = cm.tab10(np.linspace(0, 1, min(n_runs, 10)))
    
    # Plot each trajectory
    for idx, run_id in enumerate(run_ids):
        run_data = df_all_walks[df_all_walks['run_id'] == run_id].sort_values('step')
        color = colors[idx % 10]
        
        # Draw arrows between consecutive points
        for i in range(len(run_data) - 1):
            ax.annotate(
                '',
                xy=(run_data.iloc[i+1]['umap_x'], run_data.iloc[i+1]['umap_y']),
                xytext=(run_data.iloc[i]['umap_x'], run_data.iloc[i]['umap_y']),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7)
            )
        
        # Trajectory points
        ax.scatter(
            run_data['umap_x'], run_data['umap_y'],
            c=[color], edgecolors='white', linewidths=0.5, s=60, zorder=5, alpha=0.8
        )
        
        # Start marker
        ax.scatter(
            run_data.iloc[0]['umap_x'], run_data.iloc[0]['umap_y'],
            c=[color], s=100, marker='o', edgecolors='black', linewidths=2,
            label=f'Run {run_id} (start={run_data.iloc[0]["real_score"]:.3f})', zorder=10
        )
        
        # End marker
        ax.scatter(
            run_data.iloc[-1]['umap_x'], run_data.iloc[-1]['umap_y'],
            c=[color], s=100, marker='*', edgecolors='black', linewidths=2, zorder=10
        )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
    
    return fig, ax