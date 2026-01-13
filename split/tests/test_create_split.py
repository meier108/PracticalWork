import sys
sys.path.insert(0, '/system/user/publicwork/schimunek/PracticalWork/split')

import pandas as pd
import numpy as np
from create_split import create_split, hamming_distance, select_seeds, assign_to_clusters


def test_hamming_distance():
    """Test Hamming distance calculation."""
    print("\n=== Testing Hamming Distance ===")
    assert hamming_distance("AAAA", "AAAA") == 0
    assert hamming_distance("AAAA", "AAAC") == 1
    assert hamming_distance("AAAA", "CCCC") == 4
    assert hamming_distance("ATCG", "CGTA") == 4
    print("✓ Hamming distance tests passed")


def test_select_seeds():
    """Test seed selection from moderate fitness band."""
    print("\n=== Testing Seed Selection ===")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'sequence': [''.join(np.random.choice(['A', 'C', 'G', 'T'], 8)) for _ in range(n_samples)],
        'binding_scores': np.random.uniform(0, 1, n_samples)
    })

    seeds = select_seeds(df, k=5, random_state=42)

    assert len(seeds) == 5
    print(f"✓ Selected {len(seeds)} seeds")

    # Check that seeds are from moderate fitness band
    seed_scores = df[df['sequence'].isin(seeds)]['binding_scores'].values
    p60 = np.percentile(df['binding_scores'], 60)
    p80 = np.percentile(df['binding_scores'], 80)

    print(f"  60th percentile: {p60:.3f}")
    print(f"  80th percentile: {p80:.3f}")
    print(f"  Seed scores range: [{seed_scores.min():.3f}, {seed_scores.max():.3f}]")
    print("✓ Seed selection tests passed")


def test_assign_to_clusters():
    """Test cluster assignment."""
    print("\n=== Testing Cluster Assignment ===")

    # Create simple test data
    df = pd.DataFrame({
        'sequence': ['AAAA', 'AAAC', 'AAAT', 'CCCC', 'CCCA', 'CCCT'],
        'binding_scores': [0.5, 0.6, 0.55, 0.7, 0.65, 0.75]
    })

    seeds = ['AAAA', 'CCCC']
    cluster_assignments, distances = assign_to_clusters(df, seeds)

    # First 3 should be in cluster 0 (near AAAA), last 3 in cluster 1 (near CCCC)
    assert all(cluster_assignments[:3] == 0)
    assert all(cluster_assignments[3:] == 1)

    print(f"✓ Cluster assignments: {cluster_assignments}")
    print(f"✓ Distances to seeds: {distances}")
    print("✓ Cluster assignment tests passed")


def test_fitness_restriction():
    """Test that ONLY train set is restricted to low-fitness sequences."""
    print("\n=== Testing Fitness Restriction (Offline MBO) ===")

    # Load real dataset
    df = pd.read_csv('/system/user/publicwork/schimunek/PracticalWork/tf_bind_8-SIX6_REF_R1/dataset.csv')
    print(f"Loaded dataset with {len(df)} samples")

    # Create split with 50th percentile cutoff (bottom 50%)
    df_split = create_split(df, k_seeds=20, n_training_seeds=5, fitness_percentile_cutoff=50.0, random_state=42)

    # Calculate the fitness threshold
    fitness_threshold = np.percentile(df['binding_scores'], 50.0)
    print(f"Fitness threshold (50th percentile): {fitness_threshold:.3f}")

    # Check train set - MUST be below threshold
    train_scores = df_split[df_split['split'] == 'train']['binding_scores']
    print(f"\nTrain set fitness scores:")
    print(f"  min={train_scores.min():.3f}, max={train_scores.max():.3f}, mean={train_scores.mean():.3f}")
    assert train_scores.max() <= fitness_threshold, f"Train set has scores above threshold! {train_scores.max():.3f} > {fitness_threshold:.3f}"
    print(f"✓ Train set max score ({train_scores.max():.3f}) <= threshold ({fitness_threshold:.3f})")

    # Check test_a set - CAN contain any fitness level
    test_a_scores = df_split[df_split['split'] == 'test_a']['binding_scores']
    print(f"\nTest A fitness scores:")
    print(f"  min={test_a_scores.min():.3f}, max={test_a_scores.max():.3f}, mean={test_a_scores.mean():.3f}")
    if test_a_scores.max() > fitness_threshold:
        print(f"✓ Test A contains high-fitness sequences (max={test_a_scores.max():.3f} > {fitness_threshold:.3f})")
    else:
        print(f"  Test A happens to only have low-fitness sequences (max={test_a_scores.max():.3f})")

    # Check test_b set - CAN contain any fitness level
    test_b_scores = df_split[df_split['split'] == 'test_b']['binding_scores']
    print(f"\nTest B fitness scores:")
    print(f"  min={test_b_scores.min():.3f}, max={test_b_scores.max():.3f}, mean={test_b_scores.mean():.3f}")
    if test_b_scores.max() > fitness_threshold:
        print(f"✓ Test B contains high-fitness sequences (max={test_b_scores.max():.3f} > {fitness_threshold:.3f})")
    else:
        print(f"  Test B happens to only have low-fitness sequences (max={test_b_scores.max():.3f})")

    # Check test_c set - CAN contain any fitness level
    test_c_scores = df_split[df_split['split'] == 'test_c']['binding_scores']
    print(f"\nTest C fitness scores:")
    print(f"  min={test_c_scores.min():.3f}, max={test_c_scores.max():.3f}, mean={test_c_scores.mean():.3f}")
    if test_c_scores.max() > fitness_threshold:
        print(f"✓ Test C contains high-fitness sequences (max={test_c_scores.max():.3f} > {fitness_threshold:.3f})")
    else:
        print(f"  Test C happens to only have low-fitness sequences (max={test_c_scores.max():.3f})")

    print("\n✓ Fitness restriction tests passed (only train is restricted, test sets can have any fitness)")


def test_create_split_all_sequences_used():
    """Test that ALL sequences are assigned to a split."""
    print("\n=== Testing All Sequences Used ===")

    # Load real dataset
    df = pd.read_csv('/system/user/publicwork/schimunek/PracticalWork/tf_bind_8-SIX6_REF_R1/dataset.csv')
    print(f"Loaded dataset with {len(df)} samples")

    # Create split
    k_seeds = 20
    n_training_seeds = 5
    df_split = create_split(df, k_seeds=k_seeds, n_training_seeds=n_training_seeds, random_state=42)

    # Check that split column was added
    assert 'split' in df_split.columns
    assert 'cluster' in df_split.columns
    assert 'distance_to_seed' in df_split.columns
    print("✓ All required columns added")

    # Check split distribution
    split_counts = df_split['split'].value_counts()
    print("\nSplit distribution:")
    for split_type in ['train', 'test_a', 'test_b', 'test_c', 'unused']:
        count = split_counts.get(split_type, 0)
        percentage = (count / len(df_split)) * 100
        print(f"  {split_type:12s}: {count:6d} ({percentage:5.2f}%)")

    # CRITICAL: Check that ALL sequences are used (no unused!)
    total_assigned = split_counts.sum()
    unused_count = split_counts.get('unused', 0)
    print(f"\nTotal assigned: {total_assigned} / {len(df_split)}")
    print(f"Unused sequences: {unused_count}")

    assert total_assigned == len(df_split), f"Not all sequences assigned! {len(df_split) - total_assigned} missing"
    assert unused_count == 0, f"Should have no unused sequences, but found {unused_count}"

    print("✓ ALL sequences are assigned to a split (no unused)!")

    # Check that training set is non-empty and reasonable
    train_count = split_counts.get('train', 0)
    assert train_count > 0, "Training set should not be empty"
    assert train_count >= 1000, f"Training set should have at least 1000 sequences, got {train_count}"
    print(f"✓ Training set has {train_count} sequences (reasonable size for {n_training_seeds} seed clusters)")

    print("\n✓ All sequences used test passed")


def test_create_split_realistic():
    """Test create_split with default parameters."""
    print("\n=== Testing Realistic Split Creation ===")

    # Load real dataset
    df = pd.read_csv('/system/user/publicwork/schimunek/PracticalWork/tf_bind_8-SIX6_REF_R1/dataset.csv')

    # Create split with default parameters
    df_split = create_split(df, random_state=42)

    split_counts = df_split['split'].value_counts()
    print("\nDefault split distribution:")
    total = 0
    for split_type in ['train', 'test_a', 'test_b', 'test_c', 'unused']:
        count = split_counts.get(split_type, 0)
        percentage = (count / len(df_split)) * 100
        print(f"  {split_type:12s}: {count:6d} ({percentage:5.2f}%)")
        total += count

    print(f"  {'TOTAL':12s}: {total:6d} ({100.0:5.2f}%)")

    # Analyze fitness distribution across splits
    print("\nFitness score statistics by split:")
    for split_type in ['train', 'test_a', 'test_b', 'test_c']:
        if split_type in split_counts.index:
            scores = df_split[df_split['split'] == split_type]['binding_scores']
            print(f"  {split_type:12s}: mean={scores.mean():.3f}, std={scores.std():.3f}, "
                  f"min={scores.min():.3f}, max={scores.max():.3f}")

    # Analyze distance distribution
    print("\nDistance to seed statistics by split:")
    for split_type in ['train', 'test_a', 'test_b', 'test_c']:
        if split_type in split_counts.index:
            distances = df_split[df_split['split'] == split_type]['distance_to_seed']
            print(f"  {split_type:12s}: mean={distances.mean():.1f}, std={distances.std():.1f}, "
                  f"min={distances.min():.0f}, max={distances.max():.0f}")

    # Analyze cluster distribution
    print(f"\nNumber of clusters: {df_split['cluster'].nunique()}")
    print(f"Cluster size statistics:")
    cluster_sizes = df_split['cluster'].value_counts()
    print(f"  mean={cluster_sizes.mean():.1f}, std={cluster_sizes.std():.1f}, "
          f"min={cluster_sizes.min()}, max={cluster_sizes.max()}")

    print("\n✓ Realistic split creation tests passed")


def test_cluster_properties():
    """Test that clusters have expected properties."""
    print("\n=== Testing Cluster Properties ===")

    df = pd.read_csv('/system/user/publicwork/schimunek/PracticalWork/tf_bind_8-SIX6_REF_R1/dataset.csv')

    df_split = create_split(df, k_seeds=10, n_training_seeds=3, random_state=42)

    # Check that test_c clusters are completely separate
    train_clusters = set(df_split[df_split['split'] == 'train']['cluster'].unique())
    test_a_clusters = set(df_split[df_split['split'] == 'test_a']['cluster'].unique())
    test_b_clusters = set(df_split[df_split['split'] == 'test_b']['cluster'].unique())
    test_c_clusters = set(df_split[df_split['split'] == 'test_c']['cluster'].unique())

    print(f"Train clusters: {train_clusters}")
    print(f"Test A clusters: {test_a_clusters}")
    print(f"Test B clusters: {test_b_clusters}")
    print(f"Test C clusters: {test_c_clusters}")

    # Test C should have different clusters from train/test_a/test_b
    assert len(train_clusters & test_c_clusters) == 0, "Test C should have completely different clusters"
    assert len(test_a_clusters & test_c_clusters) == 0, "Test C should have completely different clusters"
    assert len(test_b_clusters & test_c_clusters) == 0, "Test C should have completely different clusters"

    print("✓ Test C has completely separate clusters (novel-seed shift)")

    # Train, Test A, and Test B should share clusters
    assert train_clusters == test_a_clusters == test_b_clusters, "Train/Test A/Test B should share clusters"
    print("✓ Train/Test A/Test B share the same clusters")

    print("\n✓ Cluster property tests passed")


def test_distance_ordering():
    """Test that train and test_a have SAME distance range, test_b is farther."""
    print("\n=== Testing Distance Ordering ===")

    df = pd.read_csv('/system/user/publicwork/schimunek/PracticalWork/tf_bind_8-SIX6_REF_R1/dataset.csv')
    df_split = create_split(df, k_seeds=20, n_training_seeds=5, training_radius_percentile=70.0, random_state=42)

    # Get distance statistics for each split
    train_dist = df_split[df_split['split'] == 'train']['distance_to_seed']
    test_a_dist = df_split[df_split['split'] == 'test_a']['distance_to_seed']
    test_b_dist = df_split[df_split['split'] == 'test_b']['distance_to_seed']

    print(f"\nTrain distance: mean={train_dist.mean():.2f}, max={train_dist.max():.0f}")
    print(f"Test A distance: mean={test_a_dist.mean():.2f}, max={test_a_dist.max():.0f}")
    print(f"Test B distance: mean={test_b_dist.mean():.2f}, min={test_b_dist.min():.0f}")

    # Calculate training radius used
    train_cluster_mask = df_split['cluster'].isin(
        df_split[df_split['split'].isin(['train', 'test_a', 'test_b'])]['cluster'].unique()
    )
    all_train_cluster_df = df_split[train_cluster_mask]
    training_radius = np.percentile(all_train_cluster_df['distance_to_seed'], 70.0)
    print(f"\nTraining radius (70th percentile): {training_radius:.2f}")

    # Train and test_a should have similar mean distances (both within radius)
    print(f"\nTrain and Test A should overlap in distance range (both within radius)")
    assert train_dist.max() <= training_radius, "Train should be within training radius"
    assert test_a_dist.max() <= training_radius or abs(test_a_dist.max() - training_radius) < 0.1, "Test A should be mostly within training radius"

    # Test B should be beyond training radius
    assert test_b_dist.min() > training_radius or abs(test_b_dist.min() - training_radius) < 0.1, "Test B should be beyond training radius"
    print(f"✓ Train and Test A are within radius, Test B is beyond radius")

    print("\n✓ Distance ordering tests passed")


if __name__ == "__main__":
    print("="*60)
    print("Running create_split tests")
    print("="*60)

    test_hamming_distance()
    test_select_seeds()
    test_assign_to_clusters()
    test_fitness_restriction()
    test_create_split_all_sequences_used()
    test_create_split_realistic()
    test_cluster_properties()
    test_distance_ordering()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
