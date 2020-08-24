import tensorflow as tf
import numpy as np


class ISMBase():
    def __init__(self, model, seq_input_idx=0, change_ranges=None, replace_with=0):
        # check if model is supported by current implementation
        self.model = model
        self.num_outputs = len(model.outputs)
        self.num_inputs = len(model.inputs)

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

    def __call__(self, inp_batch):
        if self.num_inputs == 1:
            num_seqs = inp_batch.shape[0]
        else:
            num_seqs = inp_batch[self.seq_input_idx].shape[0]

        if self.num_outputs == 1:
            ism_outputs = []
        else:
            ism_outputs = [[] for _ in range(self.num_outputs)]

        self.pre_change_range_loop_prep(inp_batch, num_seqs)

        for i in range(len(self.change_ranges)):
            ism_output = self.get_ith_output(inp_batch, i)

            if self.num_outputs == 1:
                ism_outputs.append(ism_output.numpy())
            else:
                [ism_outputs[j].append(ism_output[j].numpy()) for
                    j in range(self.num_outputs)]

        if self.num_outputs == 1:
            ism_outputs = np.swapaxes(np.array(ism_outputs), 0, 1)
        else:
            ism_outputs = [np.swapaxes(np.array(x), 0, 1) for x in ism_outputs]

        return ism_outputs

    def pre_change_range_loop_prep(self, inp_batch, num_seqs):
        pass

    def get_ith_output(self, inp_batch, i):
        pass


class NaiveISM(ISMBase):
    def __init__(self, model, seq_input_idx=0, change_ranges=None, replace_with=0):
        super().__init__(model, seq_input_idx, change_ranges, replace_with)

    def pre_change_range_loop_prep(self, inp_batch, num_seqs):
        self.cur_perturbation = tf.tile(self.perturbation, [num_seqs, 1, 1])

    def get_ith_output(self, inp_batch, i):
        # prep input with ith change range mutation
        if self.num_inputs == 1:
            ism_input = tf.concat([
                inp_batch[:, :self.change_ranges[i][0]],
                self.cur_perturbation,
                inp_batch[:, self.change_ranges[i][1]:],
            ], axis=1)
        else:
            seq_input = tf.concat([
                inp_batch[self.seq_input_idx][:, :self.change_ranges[i][0]],
                self.cur_perturbation,
                inp_batch[self.seq_input_idx][:, self.change_ranges[i][1]:],
            ], axis=1)

            ism_input = inp_batch[:self.seq_input_idx] + \
                [seq_input] + inp_batch[self.seq_input_idx+1:]

        return self.model(ism_input, training=False)
