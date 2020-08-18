from ism_base import ISMBase
from fast_ism_utils import generate_models

import tensorflow as tf
import numpy as np


class FastISM(ISMBase):
    def __init__(self, model, seqlen, num_chars=4, seq_input_idx=0, change_ranges=None, replace_with=0):
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

        self.intermediate_output_model, self.intout_output_tensors, \
            self.fast_ism_model, self.input_specs = generate_models(
                model, seqlen, num_chars, seq_input_idx, change_ranges)
        self.intout_output_tensor_to_idx = {
            x: i for i, x in enumerate(self.intout_output_tensors)}

        if not self.verify_and_benchmark():
            # ??
            exit(1)

    def __call__(self, seq_batch):
        # run intermediate output on unperturbed sequence
        intout_output = self.intermediate_output_model(seq_batch)
        padded_inputs = self.prepare_intout_output(
            intout_output, seq_batch.shape[0])  # better name than padded?

        ism_outputs = []
        for i in range(len(self.change_ranges)):
            fast_ism_inputs = self.prepare_ith_input(padded_inputs, i)

            # TODO: what if multiple outputs
            ism_output = self.fast_ism_model(fast_ism_inputs).numpy()
            ism_outputs.append(ism_output)

        # seq x num_mut x output_dim
        ism_outputs = np.swapaxes(np.array(ism_outputs), 0, 1)
        return ism_outputs

    def prepare_intout_output(self, intout_output, num_seqs):
        inputs = []

        for input_spec in self.input_specs:
            if input_spec[0] == "SEQ_PERTURB":
                inputs.append(tf.tile(self.perturbation, [num_seqs, 1, 1]))
            elif input_spec[0] == "INTOUT":
                # pad the output if required
                to_pad = intout_output[self.intout_output_tensor_to_idx[input_spec[1]['node']]]
                num_filts = to_pad.shape[2]

                padded = tf.concat([
                    # left zeros
                    tf.zeros(
                        (num_seqs, input_spec[1]['padding'][0], num_filts)),
                    to_pad,
                    # right zeros
                    tf.zeros(
                        (num_seqs, input_spec[1]['padding'][1], num_filts)),
                ], axis=1)
                inputs.append(padded)

            elif input_spec[0] == "OFFSET":
                # nothing for now, add i specific offset later
                inputs.append(None)
            else:
                raise ValueError(
                    "{}: what is this input spec?".format(input_spec[0]))

        return inputs

    def prepare_ith_input(self, padded_inputs, i):
        inputs = []

        for input_idx, input_spec in enumerate(self.input_specs):
            if input_spec[0] == "SEQ_PERTURB":
                inputs.append(padded_inputs[input_idx])
            elif input_spec[0] == "INTOUT":
                # slice
                inputs.append(
                    padded_inputs[input_idx][:,
                                             input_spec[1]['slices'][i][0]: input_spec[1]['slices'][i][1]])
            elif input_spec[0] == "OFFSET":
                inputs.append(input_spec[1]['offsets'][i])
            else:
                raise ValueError(
                    "{}: what is this input spec?".format(input_spec[0]))

        return inputs

    def verify_and_benchmark(self):
        """
        Verify that outputs are correct by matching with Naive ISM. Also compare
        runtime against Naive ISM implementation.

        TODO: ensure generated data is on GPU already before calling either method
        """
        return True
