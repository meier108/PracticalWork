import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import copy
from sklearn.decomposition import PCA
#from create_split import create_split


def encode_sequences(sequences):
    """
    One-hot encode DNA sequences.

    Args:
        sequences: Series or list of DNA sequences (all same length)

    Returns:
        numpy array of shape (n_sequences, sequence_length * 4)
    """
    # Mapping from nucleotide to index
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    seq_length = len(sequences.iloc[0] if isinstance(sequences, pd.Series) else sequences[0])
    n_sequences = len(sequences)

    # Initialize one-hot encoded matrix
    encoded = np.zeros((n_sequences, seq_length * 4), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for pos, nuc in enumerate(seq):
            if nuc in nuc_to_idx:
                idx = pos * 4 + nuc_to_idx[nuc]
                encoded[i, idx] = 1.0

    return encoded


def compute_2d_projection(encoded_sequences, method='pca', random_state=42, cache_path=None):
    """
    Compute 2D projection of encoded sequences.

    Args:
        encoded_sequences: numpy array of encoded sequences
        method: 'pca', 'tsne', or 'umap'
        random_state: random seed
        cache_path: optional path to save cached projection. If None, saves to "{method}_n{size}_seed{seed}.pkl"

    Returns:
        numpy array of shape (n_sequences, 2)
    """
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        projection = reducer.fit_transform(encoded_sequences)
        explained_var = reducer.explained_variance_ratio_
        print(f"PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f} (total: {explained_var.sum():.3f})")

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000)
        projection = reducer.fit_transform(encoded_sequences)

    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=5, min_dist=0.3) # n_neighbors=15, min_dist=0.1)
        projection = reducer.fit_transform(encoded_sequences)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'")

    # Always save to cache
    if cache_path is None:
        n_sequences = len(encoded_sequences)
        cache_path = f"{method}_n{n_sequences}_seed{random_state}.pkl"

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(projection, f)
    print(f"Saved projection to {cache_path}")

    return projection


def plot_fitness_landscape(df, projection, save_path=None, plot_type='scatter'):
    """
    Plot 1: Fitness landscape colored by fitness scores.

    Args:
        df: DataFrame with 'binding_scores' column
        projection: 2D numpy array of projected coordinates
        save_path: Optional path to save figure
        plot_type: 'scatter', 'contour', or 'both'
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if plot_type in ['contour', 'both']:
        # Create smoothed contour plot
        from scipy.interpolate import griddata

        # Create grid for interpolation
        x = projection[:, 0]
        y = projection[:, 1]
        z = df['binding_scores'].values

        # Define grid
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate fitness values on grid
        zi = griddata((x, y), z, (xi_grid, yi_grid), method='linear')

        # Plot filled contours
        contourf = ax.contourf(xi_grid, yi_grid, zi, levels=10, cmap='viridis', alpha=0.8)

        # Add contour lines
        contour = ax.contour(xi_grid, yi_grid, zi, levels=10, colors='black', alpha=0.2, linewidths=0.5)

        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Binding Score (Fitness)', fontsize=12)

    if plot_type in ['scatter', 'both']:
        # Create scatter plot colored by fitness
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=df['binding_scores'].values,
            cmap='viridis',
            s=5 if plot_type == 'both' else 5,
            alpha=0.3 if plot_type == 'both' else 0.6,
            rasterized=True,
            edgecolors='none'
        )

        if plot_type == 'scatter':
            # Add colorbar only if not already added by contour
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Binding Score (Fitness)', fontsize=12)

    # Labels and title
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    title_suffix = {'scatter': 'Scatter', 'contour': 'Contour', 'both': 'Scatter + Contour'}
    ax.set_title(f'Fitness Landscape: 2D Projection ({title_suffix[plot_type]})', fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fitness landscape plot to {save_path}")

    plt.show()

    return fig, ax


def plot_split_validation(df_in, projection_in, save_path=None, subsampling_rate=0.5):
    """
    Plot 2: Split validation colored by split type.

    Args:
        df: DataFrame with 'split' column
        projection: 2D numpy array of projected coordinates
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Subsamplinf if needed
    df = copy.deepcopy(df_in)
    projection = copy.deepcopy(projection_in)
    if subsampling_rate < 1:
        n_samples = len(df)
        sampled_indices = np.random.choice(n_samples, size=int(n_samples * subsampling_rate), replace=False)
        df = df.iloc[sampled_indices].reset_index(drop=True)
        projection = projection[sampled_indices]

    # Define colors and order for splits
    split_colors = {
        'train': '#1f77b4',      # blue
        'test_a': '#ff7f0e',     # orange
        'test_b': '#2ca02c',     # green
        'test_c': '#d62728',     # red
        'unused': '#7f7f7f'      # gray
    }

    split_order = ['train', 'test_a', 'test_b', 'test_c', 'unused']
    split_labels = {
        'train': 'Train (low-fitness, within radius)',
        'test_a': 'Test A (any fitness, within radius)',
        'test_b': 'Test B (any fitness, beyond radius)',
        'test_c': 'Test C (any fitness, novel clusters)',
        'unused': 'Unused'
    }

    # Plot each split type
    for split_type in reversed(split_order):
        mask = df['split'] == split_type
        if mask.sum() > 0:
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=split_colors[split_type],
                label=split_labels[split_type],
                s=5,
                alpha=1,
                rasterized=True
            )

    # Labels and title
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title('Split Validation: Spatial Distribution of Train/Test Sets', fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='best', frameon=True, fontsize=10, markerscale=2)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved split validation plot to {save_path}")

    plt.show()

    return fig, ax


def plot_split_statistics(df):
    """
    Create supplementary plots showing split statistics.

    Args:
        df: DataFrame with 'split', 'binding_scores', 'distance_to_seed' columns
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fitness distribution by split
    split_order = ['train', 'test_a', 'test_b', 'test_c']
    split_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for split_type, color in zip(split_order, split_colors):
        mask = df['split'] == split_type
        if mask.sum() > 0:
            scores = df[mask]['binding_scores']
            axes[0].hist(scores, bins=50, alpha=0.6, label=split_type, color=color)

    axes[0].set_xlabel('Binding Score (Fitness)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Fitness Distribution by Split', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Distance distribution by split (excluding test_c)
    for split_type, color in zip(['train', 'test_a', 'test_b'], split_colors[:3]):
        mask = df['split'] == split_type
        if mask.sum() > 0:
            distances = df[mask]['distance_to_seed']
            axes[1].hist(distances, bins=20, alpha=0.6, label=split_type, color=color)

    axes[1].set_xlabel('Hamming Distance to Seed', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Distance Distribution by Split', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, axes


def visualize_fitness_landscape(df_with_splits,
                                projection_method='pca',
                                save_plots=True,
                                random_state=42,
                                cache_dir='assets'):
    """
    Visualize the fitness landscape for data that already has splits assigned.

    Generates three main plots:
    1. Fitness Landscape (Scatter)
    2. Fitness Landscape (Contour)
    3. Split Validation

    Args:
        df_with_splits: DataFrame with 'sequence', 'binding_scores', and 'split' columns
        projection_method: Method for 2D projection ('pca', 'tsne', or 'umap')
        save_plots: Whether to save plots to disk
        random_state: Random seed for reproducibility
        cache_dir: Directory to store cached projections

    Returns:
        tuple: (df_with_splits, projection) where projection is the 2D coordinates
    """
    print("="*60)
    print("Fitness Landscape Visualization")
    print("="*60)

    # Validate input
    required_columns = ['sequence', 'binding_scores', 'split']
    missing_columns = [col for col in required_columns if col not in df_with_splits.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    print(f"\n1. Dataset info: {len(df_with_splits)} sequences")

    split_counts = df_with_splits['split'].value_counts()
    print("   Split distribution:")
    for split_type in ['train', 'test_a', 'test_b', 'test_c', 'unused']:
        count = split_counts.get(split_type, 0)
        percentage = (count / len(df_with_splits)) * 100
        print(f"     {split_type:12s}: {count:6d} ({percentage:5.2f}%)")

    # Encode sequences
    print("\n2. Encoding sequences (one-hot)...")
    encoded = encode_sequences(df_with_splits['sequence'])
    print(f"   Encoded shape: {encoded.shape}")

    # Check for cached projection
    cache_filename = f"{projection_method}_n{len(df_with_splits)}_seed{random_state}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"\n3. Loading cached 2D projection from {cache_path}...")
        with open(cache_path, 'rb') as f:
            projection = pickle.load(f)
        print(f"   Loaded projection shape: {projection.shape}")
    else:
        # Compute 2D projection
        print(f"\n3. Computing 2D projection ({projection_method.upper()})...")
        projection = compute_2d_projection(encoded, method=projection_method,
                                          random_state=random_state, cache_path=cache_path)
        print(f"   Projection shape: {projection.shape}")

    # Generate plots
    print("\n4. Generating plots...")

    print("\n   Plot 1: Fitness Landscape (Scatter)")
    save_path_1 = 'fitness_landscape_scatter.png' if save_plots else None
    plot_fitness_landscape(df_with_splits, projection, save_path=save_path_1, plot_type='scatter')

    print("\n   Plot 2: Fitness Landscape (Contour)")
    save_path_2 = 'fitness_landscape_contour.png' if save_plots else None
    plot_fitness_landscape(df_with_splits, projection, save_path=save_path_2, plot_type='contour')

    print("\n   Plot 3: Split Validation")
    save_path_3 = 'split_validation.png' if save_plots else None
    plot_split_validation(df_with_splits, projection, save_path=save_path_3)

    print("\n   Plot 4: Split Statistics")
    plot_split_statistics(df_with_splits)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

    return df_with_splits, projection


if __name__ == "__main__":
    # Example usage: load data, create splits, then visualize
    data_path = '../tf_bind_8-SIX6_REF_R1/dataset.csv'
    df = pd.read_csv(data_path)
    #df_split = create_split(df, k_seeds=20, n_training_seeds=5, random_state=42)
    #df_split, projection = visualize_fitness_landscape(df_split,
    #                                                   projection_method='umap',
    #                                                   )
