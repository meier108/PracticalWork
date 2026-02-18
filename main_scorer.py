import numpy as np
from sklearn.isotonic import spearmanr
import tensorflow as tf

from models.oracle import oracle_lookup, oracle_exists

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


def mutate_score_sequence_GB1(mutation_model, scorer, seed_sequence, score_seed, steps, num_mutations):
    old_score = score_seed
    values = {'mutation' : [], 'predicted_score' : []}
    for step in range(steps):
        mutated_sequences = mutation_model.mutate(seed_sequence, n_steps=num_mutations)
        for idx, mutated_seq in enumerate(mutated_sequences):
            score = scorer.predict(mutated_seq)
            if score > old_score:
                values['mutation'].append(mutated_seq)
                values['predicted_score'].append(score)
                old_score = score
                print(f"Step {step+1}, Mutation {idx+1}: Predicted score: {score:.4f}")
            else:
                continue
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
            if not already_scored and  oracle_exists(oracle, mutation):
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
            else:
                continue
                
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
    df_values['real_score'] = df_values['mutation'].apply(lambda x: oracle_lookup(oracle, x))
    df_values['error'] = df_values['real_score'] - df_values['predicted_score']
    return df_values

def plot_results(df_values):
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
