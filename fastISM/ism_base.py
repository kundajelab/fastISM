import tensorflow as tf
import numpy as np

class ISMBase():
    def __init__(self, model, seq_input_idx=0, change_ranges=0, replace_with=0):
        # check if model is supported by current implementation
        pass

    def __call__(self, seq_batch):
        pass

class NaiveISM(ISMBase):
    def __init__(self, model, seqlen, num_chars=4,  seq_input_idx=0, change_ranges=None, replace_with=0):
        self.model = model
        self.seqlen = seqlen
        self.seq_input_idx = seq_input_idx  # TODO: ignored now, use for multi-input
        # TODO: ignored now, use for non-zero replacements
        self.replace_with = replace_with
        
        if change_ranges is None:
            # default would be mutations at each position, 1 bp wide
            change_ranges = [(i, i+1) for i in range(seqlen)]
        # unify "change_ranges", "affected_ranges", "perturned_ranges"
        self.change_ranges = change_ranges
        
        # only one input width allowed (currently)
        assert(len(set([x[1]-x[0] for x in change_ranges])) == 1)
        perturb_width = change_ranges[0][1] - change_ranges[0][0]

        # TODO: incorporate replace_with
        self.perturbation = tf.constant(
            np.zeros((1, perturb_width, num_chars)))
    
    def __call__(self, seq_batch):
        ism_outputs = []
        num_seqs = seq_batch.shape[0]
        perturbation = tf.tile(self.perturbation, [num_seqs, 1, 1])
        
        for i in range(len(self.change_ranges)):
            ism_input = tf.concat([
                seq_batch[:, :self.change_ranges[i][0]],
                perturbation,
                seq_batch[:, self.change_ranges[i][1]:],
            ], axis=1)
            ism_outputs.append(self.model(ism_input).numpy())
        
        # seq x num_mut x output_dim        
        ism_outputs = np.swapaxes(np.array(ism_outputs), 0, 1)
        return ism_outputs