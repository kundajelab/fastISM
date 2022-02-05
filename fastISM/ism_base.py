import tensorflow as tf
import numpy as np


class ISMBase():
    def __init__(self, model, seq_input_idx=0, change_ranges=None):
        # check if model is supported by current implementation
        self.model = model
        self.num_outputs = len(model.outputs)
        self.num_inputs = len(model.inputs)

        self.seq_input_idx = seq_input_idx
        seq_input = self.model.inputs[seq_input_idx]
        self.seq_dtype = seq_input.dtype
        self.seqlen = seq_input.shape[1]
        self.num_chars = seq_input.shape[2]

        if change_ranges is None:
            # default would be mutations at each position, 1 bp wide
            change_ranges = [(i, i+1) for i in range(self.seqlen)]
        # TODO: unify nomenclature "change_ranges", "affected_ranges", "perturbed_ranges"
        self.change_ranges = change_ranges

        # only one input width allowed (currently)
        assert(len(set([x[1]-x[0] for x in change_ranges])) == 1)
        self.perturb_width = change_ranges[0][1] - change_ranges[0][0]

    def set_perturbation(self, replace_with):
        self.replace_with = replace_with
        if self.replace_with != 0:
            self.replace_with = np.array(replace_with)
            if self.replace_with.ndim == 1:
                self.replace_with = np.expand_dims(self.replace_with, 0)
            assert(self.replace_with.ndim == 2)

        if replace_with == 0:
            self.perturbation = tf.constant(
                np.zeros((1, self.perturb_width, self.num_chars)),
                dtype=self.seq_dtype)
        else:
            assert(self.replace_with.shape[0] == self.perturb_width)
            self.perturbation = tf.constant(np.expand_dims(self.replace_with, 0),
                                            dtype=self.seq_dtype)

    def __call__(self, inp_batch, replace_with=0):
        self.set_perturbation(replace_with)

        if self.num_inputs == 1:
            num_seqs = inp_batch.shape[0]
        else:
            num_seqs = inp_batch[self.seq_input_idx].shape[0]

        # setup bookeeping and return output on unperturbed input
        unperturbed_output = self.pre_change_range_loop_prep(
            inp_batch, num_seqs)

        # set up ism output tensors by intialising to unperturbed_output
        if self.num_outputs == 1:
            # take off GPU
            unperturbed_output = unperturbed_output.numpy()

            # batch_size x num_perturb x output_dim
            ism_outputs = np.repeat(np.expand_dims(unperturbed_output, 1),
                                    len(self.change_ranges), 1)
        else:
            unperturbed_output = [x.numpy() for x in unperturbed_output]

            ism_outputs = []
            for j in range(self.num_outputs):
                ism_outputs.append(np.repeat(np.expand_dims(unperturbed_output[j], 1),
                                             len(self.change_ranges), 1))

        for i, change_range in enumerate(self.change_ranges):
            # only run models on seqs that are being perturbed
            if self.num_inputs == 1:
                idxs_to_mutate = tf.squeeze(tf.where(tf.logical_not(tf.reduce_all(
                    inp_batch[:, change_range[0]:change_range[1]] == self.perturbation[0], axis=(1, 2)))),
                    axis=1)
            else:
                idxs_to_mutate = tf.squeeze(tf.where(tf.logical_not(tf.reduce_all(
                    inp_batch[self.seq_input_idx][:, change_range[0]:change_range[1]] == self.perturbation[0], axis=(1, 2)))),
                    axis=1)

            num_to_mutate = idxs_to_mutate.shape[0] 
            if num_to_mutate > 0:
                # output only on idxs_to_mutate
                ism_ith_output = self.get_ith_output(inp_batch, i, idxs_to_mutate)

                if self.num_outputs == 1:
                    ism_outputs[idxs_to_mutate, i] = ism_ith_output
                else:
                    for j in range(self.num_outputs):
                        ism_outputs[j][idxs_to_mutate,
                                    i] = ism_ith_output[j].numpy()

        # cleanup tensors that have been used
        self.cleanup()

        return ism_outputs

    def pre_change_range_loop_prep(self, inp_batch, num_seqs):
        pass

    def get_ith_output(self, inp_batch, i, idxs_to_mutate):
        pass


class NaiveISM(ISMBase):
    def __init__(self, model, seq_input_idx=0, change_ranges=None):
        super().__init__(model, seq_input_idx, change_ranges)

    def pre_change_range_loop_prep(self, inp_batch, num_seqs):
        self.cur_perturbation = tf.tile(self.perturbation, [num_seqs, 1, 1])

        return self.model(inp_batch, training=False)

    def run_model(self, x):
        return self.model(x, training=False)
 
    def get_ith_output(self, inp_batch, i, idxs_to_mutate):
        num_to_mutate = idxs_to_mutate.shape[0]

        # prep input with ith change range mutation
        if self.num_inputs == 1:
            ism_input = tf.concat([
                tf.gather(inp_batch[
                    :, :self.change_ranges[i][0]], idxs_to_mutate),
                self.cur_perturbation[:num_to_mutate],
                tf.gather(inp_batch[
                    :, self.change_ranges[i][1]:], idxs_to_mutate)
            ], axis=1)
        else:
            ism_input = []
            for j in range(self.num_inputs):
                if j == self.seq_input_idx:
                    ism_input.append(tf.concat([
                        tf.gather(inp_batch[self.seq_input_idx], idxs_to_mutate)[:,
                                                                                 :self.change_ranges[i][0]],
                        self.cur_perturbation[:num_to_mutate],
                        tf.gather(inp_batch[self.seq_input_idx], idxs_to_mutate)[:,
                                                                                 self.change_ranges[i][1]:],
                    ], axis=1))
                else:
                    ism_input.append(tf.gather(inp_batch[j], idxs_to_mutate))

        return self.run_model(ism_input)

    def cleanup(self):
        pass
