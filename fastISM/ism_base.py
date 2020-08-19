import tensorflow as tf
import numpy as np

class ISMBase():
    def __init__(self, model, seq_input_idx=0, change_ranges=None, replace_with=0):
        # check if model is supported by current implementation
        self.model = model

        self.seq_input_idx = seq_input_idx  # TODO: use further for multi-input
        seq_input = self.model.inputs[seq_input_idx]
        self.seqlen = seq_input.shape[1]
        self.num_chars = seq_input.shape[2]

        # TODO: ignored now, use for non-zero replacements
        self.replace_with = replace_with

        if change_ranges is None:
            # default would be mutations at each position, 1 bp wide
            change_ranges = [(i, i+1) for i in range(self.seqlen)]
        # unify "change_ranges", "affected_ranges", "perturned_ranges"
        self.change_ranges = change_ranges

        # only one input width allowed (currently)
        assert(len(set([x[1]-x[0] for x in change_ranges])) == 1)
        perturb_width = change_ranges[0][1] - change_ranges[0][0]

        # TODO: incorporate replace_with
        self.perturbation = tf.constant(
            np.zeros((1, perturb_width, self.num_chars)))

    def __call__(self, seq_batch):
        pass

class NaiveISM(ISMBase):
    def __init__(self, model, seq_input_idx=0, change_ranges=None, replace_with=0):
        super().__init__(model, seq_input_idx, change_ranges, replace_with)
    
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