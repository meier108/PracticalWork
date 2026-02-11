import numpy as np
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
                    seed_sequence = tf.argmax(tf.reshape(mutation, (mutation.shape[1] // 4, 4)), axis=-1).numpy().tolist()
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
    Plot the predicted scores vs. true scores and the error distribution.
    Calculate spearman correlation between predicted and true scores.

    Args:
        df_values: A dataframe containing 'predicted_score', 'real_score', and 'error' columns
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Scatter plot of predicted vs real scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='real_score', y='predicted_score', data=df_values)
    plt.plot([df_values['real_score'].min(), df_values['real_score'].max()], 
             [df_values['real_score'].min(), df_values['real_score'].max()], 
             'r--')  # Line for perfect predictions
    plt.xlabel('True Binding Score')
    plt.ylabel('Predicted Binding Score')
    plt.title('Predicted vs True Binding Scores')
    
    # Distribution of errors
    plt.subplot(1, 2, 2)
    sns.histplot(df_values['error'], kde=True)
    plt.xlabel('Prediction Error (True - Predicted)')
    plt.title('Distribution of Prediction Errors')
    
    plt.tight_layout()
    plt.show()
