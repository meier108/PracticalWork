import numpy as np
import tensorflow as tf

class SMW:

    def __init__(self, nucleotides):
        self.nucleotides = nucleotides
        self.n = len(nucleotides)

    def one_hot_encode(self, sequence):
        sequence_tensor = tf.one_hot(
            sequence, depth=4
        )
        sequence_numpy = sequence_tensor.numpy() 
        shape = sequence_numpy.shape[0] * sequence_numpy.shape[1]
        sequence_n_flat = sequence_numpy.reshape(1, shape)
        return sequence_n_flat

    def mutate_sequence(self, sequence, pos):
        
        sequence = list(sequence)
        if pos is None:
            pos = np.random.randint(0, len(sequence))

        # Get current nucleotide and possible mutations, make sure that nuc is different
        current_nucleotide = sequence[pos]
        possible_mutations = [nuc for nuc in self.nucleotides if nuc !=
                                current_nucleotide]
        
        # Replace nucleotide at position 
        new_nucleotide = np.random.choice(possible_mutations)
        sequence[pos] = new_nucleotide
        return sequence
    
    def generate_all_mutations(self, sequence):
        mutants = []
        for pos in range(len(sequence)):
            current_nucleotide = sequence[pos]
            for nuc in self.nucleotides:
                if nuc != current_nucleotide:
                    mutated = sequence[:pos] + nuc + sequence[pos+1:]
                    mutants.append(mutated)
        return mutants
    
    def walk(self, start_sequence, n_steps, fitness_function):
        current_seq = start_sequence
        current_fitness = 0 #test just use fitness of 0 for first one 

        history = [(current_seq, current_fitness)]

        for step in range(n_steps):
            mutant = self.mutate_sequence(current_seq, None)
            mutant_one_hot = self.one_hot_encode(mutant)
            mutant_fitness = fitness_function(mutant_one_hot)

            # Check if mutant is better than current
            # Yes: move to mutant
            if mutant_fitness > current_fitness:
                current_seq = mutant
                current_fitness = mutant_fitness

            history.append((current_seq, current_fitness))

        return history
