from math import ceil, floor


def get_int_if_tuple(param, idx=0):
    if isinstance(param, tuple):
        return param[idx]
    return param


def not_supported_error(message):
    raise NotImplementedError("""{} not supported yet, please post an Issue with 
                              your architecture and the authors will try their
                              best to help you!""".format(message))


class ChangeRangesBase():
    """
    Base class for layer-specific computations of which indices of the output
    are changed when list of input changed indices are specified. Conversely, given
    output ranges of indices that need to be produced by the layer, compute the input
    ranges that will be required for the same.

    In addition, given an input....

    TODO: document better and with examples!
    """

    def __init__(self, config):
        self.config = config
        self.validate_config()

    def validate_config(self):
        pass

    def forward(self, input_seqlen, input_change_ranges):
        """
        list of tuples. e.g. [(0,1), (1,2), (2,3)...] if single bp ISM
        """
        pass

    def backward(self, output_select_ranges):
        pass

    @staticmethod
    def forward_compose(change_ranges_objects_list, input_seqlen, input_change_ranges):
        # multiple ChangeRanges objects (in order), e.g. in a segment with conv->maxpool

        # base case which should generally only happen with input tensor
        if len(change_ranges_objects_list) == 0:
            return input_change_ranges, (0, 0), input_seqlen, input_change_ranges

        if len(change_ranges_objects_list) == 1:
            return change_ranges_objects_list[0].forward(input_seqlen, input_change_ranges)

        # chain forwards
        seqlen = input_seqlen
        affected_range = input_change_ranges
        for change_range_object in change_ranges_objects_list:
            input_range_corrected, _, seqlen, affected_range = change_range_object.forward(
                seqlen, affected_range)

        # chain backwards
        # this computes the input region to entire segment that is required
        # to obtain the final affected_range output. This ensures the segment
        # does not need any SliceAssign internally
        for change_range_object in reversed(change_ranges_objects_list[:-1]):
            input_range_corrected = change_range_object.backward(
                input_range_corrected)

        # compute new offsets
        # they will be relative to initial offsets
        initial_input_range_corrected, initial_padding, \
            _, _ = change_ranges_objects_list[0].forward(
                input_seqlen, input_change_ranges)

        # all other except the first must have 0 padding
        # as there is no provision to bad within a segment
        assert(all([x.forward(input_seqlen, input_change_ranges)[1] == (0, 0)
                    for x in change_ranges_objects_list[1:]]))

        # modified range_corrected should span at least the initial range
        # unless a Cropping1D layer is present, in which case this does not
        # need to hold true (e.g. at the edges the initial range can be cropped out)
        # this may also happen if MaxPooling1D is placed in a segment of its own
        # e.g. segment 0 before any other conv -- this is not handled for now
        if not any([isinstance(x, Cropping1DChangeRanges) for
                    x in change_ranges_objects_list]):            
            assert(all([x_new <= x_old for (x_new, y_new), (x_old, y_old) in zip(
                input_range_corrected, initial_input_range_corrected)]))

        return input_range_corrected, initial_padding, seqlen, affected_range


class Conv1DChangeRanges(ChangeRangesBase):
    def __init__(self, config):
        ChangeRangesBase.__init__(self, config)

        # in case dilation_rate > 1, compute effective kernel size
        self.effective_kernel_size = (get_int_if_tuple(
            config['kernel_size'])-1) * get_int_if_tuple(config['dilation_rate']) + 1

        # assuming "same" if not "valid" (checked in validate_config)
        # if valid and effective size is even then keras will pad with more zeros
        # on the right (used to be left before)
        self.padding_num = (0, 0) if config['padding'] == 'valid' else \
            (floor((self.effective_kernel_size-1)/2),
             ceil((self.effective_kernel_size-1)/2))

    def validate_config(self):
        if self.config['data_format'] != "channels_last":
            not_supported_error("data_format \"{}\"".format(
                self.config['data_format']))

        strides = get_int_if_tuple(self.config['strides'])
        if strides != 1:
            not_supported_error("Conv1D strides!=1")

        if self.config['groups'] > 1:
            not_supported_error("Groups > 1")

        if self.config['padding'] not in ['valid', 'same']:
            not_supported_error(
                "Padding \"{}\" for Conv1D".format(self.config['padding']))

    def forward(self, input_seqlen, input_change_ranges):
        # NB: returned input_range_corrected, offsets are wrt padded input
        # MAKE VERY CLEAR
        assert(all([(0 <= x < input_seqlen and 0 < y <= input_seqlen and y > x)
                    for x, y in input_change_ranges]))
        seqlen_with_padding = input_seqlen + sum(self.padding_num)

        # assuming input ranges have same width
        if (len(set([y-x for x, y in input_change_ranges])) != 1):
            not_supported_error("Input Change Ranges of different sizes")

        # required input range will involve regions around input_change_range
        input_change_range_with_filter = [
            (x-self.effective_kernel_size+1, y+self.effective_kernel_size-1) for
            x, y in input_change_ranges]

        # there will be self.padding_num[0] zeros in the beginning
        input_change_range_padded = [
            (x+self.padding_num[0], y+self.padding_num[0]) for
            x, y in input_change_range_with_filter]

        # account for edge effects
        input_range_corrected = []
        for x, y in input_change_range_padded:
            # this can happen e.g. in dilated convs where the effective
            # width gets as wide as input sequence
            if y-x > seqlen_with_padding:
                #import pdb;pdb.set_trace()
                x, y = 0, seqlen_with_padding
            if x < 0:
                x, y = 0, y-x
            elif y > seqlen_with_padding:
                x, y = x-(y-seqlen_with_padding), seqlen_with_padding

            input_range_corrected.append((x, y))

        # follows from requirement above
        assert(len(set([y-x for x, y in input_range_corrected])) == 1)

        # corrected change ranges must include input_change_ranges
        assert([x_c <= x and y_c >= y for (x, y), (x_c, y_c) in zip(
            input_change_ranges, input_range_corrected)])

        # output affected ranges
        output_affected_ranges = [(x, y-self.effective_kernel_size+1) for
                                  x, y in input_range_corrected]

        # output sequence length
        outseqlen = seqlen_with_padding - self.effective_kernel_size + 1

        return input_range_corrected, self.padding_num, outseqlen, output_affected_ranges

    def backward(self, output_select_ranges):
        assert(len(set([y-x for x, y in output_select_ranges])) == 1)

        ranges = [(x, y+self.effective_kernel_size-1)
                  for x, y in output_select_ranges]

        assert(all([(x >= 0 and y >= 0 and y > x)
                    for x, y in ranges]))

        return ranges


class MaxPooling1DChangeRanges(ChangeRangesBase):
    def __init__(self, config):
        ChangeRangesBase.__init__(self, config)
        self.pool_size = get_int_if_tuple(self.config['pool_size'])
        self.strides = get_int_if_tuple(self.config['strides'])

    def validate_config(self):
        if self.config['data_format'] != "channels_last":
            not_supported_error("data_format \"{}\"".format(
                self.config['data_format']))

        pool_size = get_int_if_tuple(self.config['pool_size'])
        strides = get_int_if_tuple(self.config['strides'])

        if pool_size != strides:
            not_supported_error("pool_size != strides")

        if self.config['padding'] != 'valid':
            not_supported_error(
                "Padding \"{}\" for Maxpooling1D".format(self.config['padding']))

    def forward(self, input_seqlen, input_change_ranges):
        # assuming input ranges have same width
        if (len(set([y-x for x, y in input_change_ranges])) != 1):
            not_supported_error("Input Change Ranges of different sizes")

        # shift to edges of nearest maxpool block
        input_change_range_shifted = [(self.pool_size*(x//self.pool_size),
                                       self.pool_size*ceil(y/self.pool_size)) for
                                      x, y in input_change_ranges]

        # sizes can change, calculate maxwidth and set all to same
        maxwidth = max([y-x for x, y in input_change_range_shifted])

        # set to same length
        input_range_corrected = [(x, x+maxwidth) if y <= input_seqlen
                                 else (y-maxwidth, y)
                                 for x, y in input_change_range_shifted]
        # NOTE: the below code ignores the last block when seqlen is not a multiple
        # of pool_size. This works only when padding == 'valid' and strides==pool_size.
        input_range_corrected = [(x, y) if y <= input_seqlen else
                                 (x-self.pool_size, y-self.pool_size) for x, y in input_range_corrected]
        assert([y <= input_seqlen for _, y in input_range_corrected])

        # corrected change ranges must include input_change_ranges
        assert([x_c <= x and y_c >= y for (x, y), (x_c, y_c) in zip(
            input_change_ranges, input_range_corrected)])

        output_affected_ranges = [(x//self.pool_size, y//self.pool_size) for
                                  (x, y) in input_range_corrected]

        # output sequence length (assumes "valid" paddng)
        assert(self.config["padding"] == "valid")
        outseqlen = input_seqlen // self.pool_size

        # (0,0) for no padding -- this would change if padding="same" is allowed
        return input_range_corrected, (0, 0), outseqlen, output_affected_ranges

    def backward(self, output_select_ranges):
        assert(len(set([y-x for x, y in output_select_ranges])) == 1)
        return [(x*self.pool_size, y*self.pool_size) for x, y in output_select_ranges]


class Cropping1DChangeRanges(ChangeRangesBase):
    def __init__(self, config):
        ChangeRangesBase.__init__(self, config)
        self.cropping = self.config['cropping']

    def validate_config(self):
        # all configs accepted
        return True

    def forward(self, input_seqlen, input_change_ranges):
        # assuming input ranges have same width
        if (len(set([y-x for x, y in input_change_ranges])) != 1):
            not_supported_error("Input Change Ranges of different sizes")

        # push right if within left cropping
        input_range_corrected = [(x, y) if x >= self.cropping[0] else
                                 (self.cropping[0], self.cropping[0] + (y-x))
                                 for x, y in input_change_ranges]

        # push left if within right cropping
        right_edge = (input_seqlen-self.cropping[1])
        input_range_corrected = [(x, y) if y < right_edge else
                                 (right_edge - (y-x), right_edge)
                                 for x, y in input_range_corrected]

        output_affected_ranges = [(x-self.cropping[0], y-self.cropping[0]) for
                                  (x, y) in input_range_corrected]

        outseqlen = input_seqlen - sum(self.cropping)

        assert(len(set([y-x for x, y in input_range_corrected])) == 1)

        # (0,0) for no padding
        return input_range_corrected, (0, 0), outseqlen, output_affected_ranges

    def backward(self, output_select_ranges):
        assert(len(set([y-x for x, y in output_select_ranges])) == 1)
        return [(x+self.cropping[0], y+self.cropping[0]) for x, y in output_select_ranges]
