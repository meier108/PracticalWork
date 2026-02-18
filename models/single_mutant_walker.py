"""Single Mutant Walker for sequence mutation."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import tensorflow as tf


class SMW:
    """Single Mutant Walker for generating sequence mutations."""

    def __init__(self, nucleotides: List[int]):
        """Initialize SMW with nucleotide vocabulary.

        Args:
            nucleotides: List of nucleotide encodings (e.g., [0, 1, 2, 3]).
        """
        self.nucleotides = nucleotides
        self.n = len(nucleotides)

    def one_hot_encode(self, sequence: List[int]) -> np.ndarray:
        """One-hot encode a sequence and flatten it.

        Args:
            sequence: List of nucleotide indices.

        Returns:
            Flattened one-hot encoded sequence of shape (1, seq_len * n).
        """
        sequence_tensor = tf.one_hot(sequence, depth=self.n)
        sequence_numpy = sequence_tensor.numpy()
        sequence_n_flat = sequence_numpy.flatten()
        #shape = sequence_numpy.shape[0] * sequence_numpy.shape[1]
        #sequence_n_flat = sequence_numpy.reshape(1, shape)
        return sequence_n_flat

    def mutate_sequence(self, sequence: List[int], pos: Optional[int] = None) -> List[int]:
        """Mutate a single position in the sequence.

        Args:
            sequence: List of nucleotide indices.
            pos: Position to mutate. If None, randomly choose a position.

        Returns:
            Mutated sequence.
        """
        sequence = list(sequence)
        if pos is None:
            pos = np.random.randint(0, len(sequence))
        # Get current nucleotide and possible mutations, make sure that nuc is different
        current_nucleotide = sequence[pos]
        possible_mutations = [nuc for nuc in self.nucleotides if nuc != current_nucleotide]
        # Replace nucleotide at position
        new_nucleotide = np.random.choice(possible_mutations)
        sequence[pos] = new_nucleotide
        return sequence

    def mutate(self, start_sequence: List[int], n_steps: int) -> List[np.ndarray]:
        """Generate a sequence of mutations starting from a seed sequence.

        Args:
            start_sequence: Initial sequence to start from.
            n_steps: Number of mutation steps to perform.

        Returns:
            List of one-hot encoded mutated sequences.
        """
        current_seq = start_sequence
        history = []
        for step in range(n_steps):
            mutant = self.mutate_sequence(current_seq, None)
            mutant_one_hot = self.one_hot_encode(mutant)
            current_seq = mutant
            history.append(mutant_one_hot)
        return history
