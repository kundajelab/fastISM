import tensorflow as tf
from . import flatten_model
from collections import defaultdict
from copy import deepcopy

from .change_range import ChangeRangesBase, \
    Conv1DChangeRanges, \
    MaxPooling1DChangeRanges, \
    Cropping1DChangeRanges, \
    not_supported_error

# Union of layers below gives all supported layers.

# layers at which output at ith position depends on input at ith position only
SEE_THROUGH_LAYERS = {
    'InputLayer',  # TODO: should this even be here?
    'Activation',
    'Dropout',
    'ELU',
    'LeakyReLU',
    'PReLU',
    'ReLU',
    'BatchNormalization'
}

# layers which take in >1 inputs of the same shape and output has the same shape
# essentially similar SEE_THROUGH_LAYER but mentioned separately since there is
# additional bookkeeping when a layer has >1 inputs. Nonetheless this category is
# not used in any special way (yet)
AGGREGATE_LAYERS = {
    'Add',
    'Maximum',
    'Minumum',
    'Multiply',
    'Average',
    'Subtract'
}

# layers at which output at ith position depends on a window around the ith position
LOCAL_LAYERS = {
    'Conv1D',
    'MaxPooling1D',
    'Cropping1D'
}

# layers after which output at ith position depends on inputs at most or all positions
# however, this is not true for Flatten/Reshape, but it is assumed these are followed
# by Dense or similar
STOP_LAYERS = {
    'Flatten',
    'Reshape',
    'GlobalAveragePooling1D',
    'GlobalMaxPool1D',
    'Dense'
}


class SliceAssign(tf.keras.layers.Layer):
    # TODO: make it work along any axis
    def __init__(self, a_dim, b_dim):
        super(SliceAssign, self).__init__()

        self.a_dim = a_dim

        # after one slice assign, tf can't calculate dimension
        # since i is not known. So manually specify b_dim
        self.b_dim = b_dim

    def call(self, inputs):
        """
        GOAL: a[:,i:min(i+b.shape[1], a.shape[1])] = b
        clip b if i+b.shape[1] exceeds width of a, guarantee width of output
        is same as a. This could happen when a layer's output (b) feeds into
        multiple layers, but some layers don't need all positions of b
        (can happen near the edges).
        See test_skip_then_mxp of test/test_simple_skip_conn_architectures.py

        For Cropping1D layers, i can also be negative, which needs to be handled
        separately.

        :param inputs: [description]
        :type inputs: [type]
        :return: [description]
        :rtype: [type]
        """

        a, b, i = inputs

        # i<0
        def case1():
            return tf.concat([b[:, -i[0]:],
                              a[:, tf.math.maximum(self.b_dim+i[0], 0):]],
                             axis=1)

        # i>=0, b fits within a or goes past it and is cut
        def case2():
            return tf.concat([a[:, :i[0]],
                              b[:, :tf.math.maximum(self.a_dim-i[0], 0)],
                              a[:, i[0]+self.b_dim:]],
                             axis=1)

        # output will lose shape info (dim 1 will be set to None)
        return tf.cond(i[0] < 0, case1, case2)


class GraphSegment():
    # TODO: explain all the variables
    def __init__(self, start_node, input_seqlen, input_perturbed_ranges):
        self.start_node = start_node
        self.input_seqlen = input_seqlen
        self.input_perturbed_ranges = input_perturbed_ranges

        self.input_unperturbed_slices = None
        self.input_unperturbed_padding = None
        self.num_out_filters = None
        self.output_seqlen = None
        self.output_perturbed_ranges = None

    def update_forward_output(self, input_unperturbed_slices,
                              input_unperturbed_padding, output_seqlen,
                              output_perturbed_ranges):
        self.input_unperturbed_slices = input_unperturbed_slices
        self.input_unperturbed_padding = input_unperturbed_padding
        self.output_seqlen = output_seqlen
        self.output_perturbed_ranges = output_perturbed_ranges

    def update_num_filters(self, num_out_filters):
        self.num_out_filters = num_out_filters

    def input_unperturbed_width(self):
        # all should be same, return first
        return self.input_unperturbed_slices[0][1] - self.input_unperturbed_slices[0][0]

    def output_perturbed_width(self):
        # all should be same, return first
        return self.output_perturbed_ranges[0][1] - self.output_perturbed_ranges[0][0]

    def __repr__(self):
        return str(self.__dict__)


# perhaps convert these to a class so that nodes, edges don't have to be fed as input always
def segment_model(model, nodes, edges, inbound_edges, seq_input_idx, early_stop_layers):
    # segment model into groups that can be run as a unit. Intermediate outputs after each group
    # should be captured for unperturbed inputs. Segmenting thus helps minimise number of intermediate
    # outputs that need to be captured

    # starting with single sequence-only input models
    # will relax later

    input_layer = "LAYER/{}".format(model.input_names[seq_input_idx])
    assert(input_layer in nodes)

    assert(len(edges[input_layer]) == 1)
    input_tensor = edges[input_layer][0]

    # process starting from sequence input
    node_to_segment, stop_segment_idxs, segment_idx = segment_subgraph(
        input_tensor, nodes, edges, inbound_edges, dict(), set(), 0, 0)

    # in cases with a custom early stop layer/s, add downstream segments to stop_segments
    if early_stop_layers is not None:
        if not isinstance(early_stop_layers, list):
            early_stop_layers = [early_stop_layers]

        for early_stop_layer in early_stop_layers:
            stop_segment_idxs = update_stop_segments(
                "LAYER/{}".format(early_stop_layer), nodes, edges, node_to_segment, stop_segment_idxs)

    # process alternate inputs if any
    alternate_input_segment_idxs = set()
    for i, alternate_input in enumerate(model.input_names):
        if i != seq_input_idx:
            alternate_input_layer = "LAYER/{}".format(alternate_input)
            assert(alternate_input_layer in nodes)

            assert(len(edges[alternate_input_layer]) == 1)
            alternate_input_tensor = edges[alternate_input_layer][0]

            alternate_input_segment_idxs.add(segment_idx)

            node_to_segment = label_alternate_input_segment_idxs(
                alternate_input_tensor, nodes, edges, node_to_segment,
                stop_segment_idxs, alternate_input_segment_idxs, segment_idx)

            segment_idx += 1

    return node_to_segment, stop_segment_idxs, alternate_input_segment_idxs


def segment_subgraph(current_node, nodes, edges, inbound_edges,
                     node_to_segment, stop_segment_idxs, segment_idx,
                     num_convs_in_cur_segment):
    # segment_idx is the current segment_idx
    # node_to_segment is dict from node->segment

    # already segmented
    if current_node in node_to_segment:
        return node_to_segment, stop_segment_idxs, segment_idx+1

    if flatten_model.node_is_layer(current_node):
        layer_class = nodes[current_node].__class__.__name__

        # layer should have at least one out-edge
        assert(len(edges[current_node]) > 0)

        if len(edges[current_node]) > 1:
            raise NotImplementedError(
                "Layer with multiple outputs, what to do?")

        elif layer_class in STOP_LAYERS:
            # mark end of current segment
            segment_idx += 1

            # add to set of stop segments
            stop_segment_idxs.add(segment_idx)

            # recursively label all descendants (no more further segments)
            return label_stop_descendants(current_node, nodes, edges, node_to_segment, segment_idx), stop_segment_idxs, segment_idx+1

        elif (layer_class == 'MaxPooling1D' or layer_class == 'Cropping1D') \
                and segment_idx == 0:
            # special case for when a Cropping or MaxPooling1D layer is right
            # after input sequence before first Conv1D
            segment_idx += 1
            node_to_segment[current_node] = segment_idx

            assert(num_convs_in_cur_segment == 0)
            return segment_subgraph(edges[current_node][0], nodes, edges,
                                    inbound_edges, node_to_segment,
                                    stop_segment_idxs, segment_idx,
                                    num_convs_in_cur_segment)

        elif layer_class == 'Conv1D':
            # enforce that if a segment has a conv layer, it is always at the beginning
            # by doing this, pre-conv intermediate outputs will always be captured and
            # padding them would become a one-time operation
            segment_idx += 1

            node_to_segment[current_node] = segment_idx
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, stop_segment_idxs, segment_idx, 1)

        elif len(inbound_edges[current_node]) > 1:
            segment_idx += 1
            node_to_segment[current_node] = segment_idx
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, stop_segment_idxs, segment_idx, 0)

        else:
            # single-input, single-output layer -> propagate further
            node_to_segment[current_node] = segment_idx
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, stop_segment_idxs, segment_idx, num_convs_in_cur_segment)

    # it's a tensor
    else:
        # tensors can't have > 1 in-degree, but can have > 1 out-degree
        assert(len(inbound_edges[current_node]) == 1)

        node_to_segment[current_node] = segment_idx

        # terminal tensor, done
        if len(edges[current_node]) == 0:
            return node_to_segment, stop_segment_idxs, segment_idx+1

        # single edge out, propogate
        elif len(edges[current_node]) == 1:
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, stop_segment_idxs, segment_idx, num_convs_in_cur_segment)

        # multi edge out => multiple layers use this tensor
        # e.g. resnet layers
        else:
            segment_idx += 1  # increment segment idx
            for next_node in edges[current_node]:
                node_to_segment, stop_segment_idxs, segment_idx = segment_subgraph(
                    next_node, nodes, edges, inbound_edges, node_to_segment, stop_segment_idxs, segment_idx, 0)
            return node_to_segment, stop_segment_idxs, segment_idx


def label_stop_descendants(current_node, nodes, edges, node_to_segment, segment_idx):
    # label nodes downstream of STOP_LAYERS
    node_to_segment[current_node] = segment_idx

    for node in edges[current_node]:
        node_to_segment = label_stop_descendants(
            node, nodes, edges, node_to_segment, segment_idx)

    return node_to_segment


def update_stop_segments(current_node, nodes, edges, node_to_segment, stop_segment_idxs):
    # add all segments of nodes including and downstream of current_node
    # to stop_segment_idxs
    stop_segment_idxs.add(node_to_segment[current_node])

    for node in edges[current_node]:
        stop_segment_idxs = update_stop_segments(
            node, nodes, edges, node_to_segment, stop_segment_idxs)

    return stop_segment_idxs


def label_alternate_input_segment_idxs(current_node, nodes, edges, node_to_segment,
                                       stop_segment_idxs, alternate_input_segment_idxs,
                                       segment_idx):
    # label segments that start from alternate inputs

    if current_node in node_to_segment:
        if (node_to_segment[current_node] not in stop_segment_idxs) and \
                (node_to_segment[current_node] not in alternate_input_segment_idxs):
            raise not_supported_error(
                "Non-sequence input connects directly with sequence input before a STOP_LAYER--")
        else:
            return node_to_segment

    node_to_segment[current_node] = segment_idx

    for node in edges[current_node]:
        node_to_segment = label_alternate_input_segment_idxs(node, nodes, edges,
                                                             node_to_segment,
                                                             stop_segment_idxs,
                                                             alternate_input_segment_idxs,
                                                             segment_idx)

    return node_to_segment


def compute_segment_change_ranges(model, nodes, edges, inbound_edges,
                                  node_to_segment, stop_segment_idxs,
                                  input_seqlen, input_filters,
                                  input_change_ranges, seq_input_idx):
    """
    for each segment, given input change range compute 
    (ChangeRangesBase.forward_compose):
        - input range of intermediate output required
        - offsets for input tensor wrt intermediate output
        - output seqlen
        - output change range
        - number of filters in output.

    Starts only from sequence input that is changed. Does not deal with alternate
    inputs.

    Forward propagation through network one segment at a time till a segment in
    stop_segments_idxs is hit. Computes the change ranges for each segment and 
    propagates to the next segment.
    """

    # starting sequence input and compute change ranges
    input_layer = "LAYER/{}".format(model.input_names[seq_input_idx])
    assert(input_layer in nodes)

    assert(len(edges[input_layer]) == 1)
    input_tensor = edges[input_layer][0]

    # this will store outputs of ChangeRangesBase.forward_compose
    # as well as number of filters in output
    # it's a map from segment -> output
    segments = dict()

    # a segment can have multiple input_seqlens if it's a layer with multiple
    # in-bound edges
    segments_to_process = []
    segments_to_process_input_seqlens = defaultdict(list)
    segments_to_process_input_filters = defaultdict(list)
    segments_to_process_input_change_ranges = defaultdict(list)

    # initialise with input tensor, which has segment idx 0
    segments_to_process.append((0, input_tensor))
    segments_to_process_input_seqlens[0] = [input_seqlen]
    segments_to_process_input_filters[0] = [input_filters]
    segments_to_process_input_change_ranges[0] = [input_change_ranges]

    while segments_to_process:
        cur_segment_to_process, cur_segment_tensor = segments_to_process.pop(0)

        # inbound nodes that do not belong to stop_segments
        # this is because stop segments are not processed further
        non_stop_segment_inbound = [edge for edge in
                                    inbound_edges[cur_segment_tensor] if
                                    edge not in node_to_segment or  # for InputLayers
                                    node_to_segment[edge]
                                    not in stop_segment_idxs]

        # only a node in stop_segment_idxs can have an inbound node that belongs
        # to stop_segment_idxs
        if len(non_stop_segment_inbound) != len(inbound_edges[cur_segment_tensor]):
            assert(node_to_segment[cur_segment_tensor] in stop_segment_idxs)

        if len(segments_to_process_input_seqlens[cur_segment_to_process]) != \
                len(non_stop_segment_inbound):
            # should not be greater in any case
            assert(len(segments_to_process_input_seqlens[cur_segment_to_process]) <
                   len(non_stop_segment_inbound))
            # hold off and wait till other input segments are populated
            assert(len(segments_to_process) > 0)

        else:
            # resolve multiple input_change_ranges
            if len(set(segments_to_process_input_seqlens[cur_segment_to_process])) != 1:
                not_supported_error(
                    "This multi-input layer takes in inputs of different length")
            if len(set(segments_to_process_input_filters[cur_segment_to_process])) != 1:
                not_supported_error("This multi-input layer takes in \
                                        inputs of different filters")

            # if node marks beginning of dense/flatten/reshape layers, say
            # or belongs to segment in stop_segment_idxs
            if node_to_segment[cur_segment_tensor] in stop_segment_idxs:
                cur_input_seqlen = segments_to_process_input_seqlens[cur_segment_to_process][0]
                cur_input_change_ranges = segments_to_process_input_change_ranges[
                    cur_segment_to_process][0]

                segment = GraphSegment(cur_segment_tensor, cur_input_seqlen,
                                       cur_input_change_ranges)
                segment.update_forward_output([(0, cur_input_seqlen)] *
                                              # entire input range
                                              len(cur_input_change_ranges),
                                              (0, 0),  # no padding
                                              None,  # output seqlen NA
                                              None)  # affected range is the whole thing, NA
                segment.update_num_filters(None)  # output filters NA
                segments[cur_segment_to_process] = segment

            # process current segment
            else:
                change_range_objects = []

                # resolve multiple input_change_ranges
                cur_input_seqlen = segments_to_process_input_seqlens[cur_segment_to_process][0]
                segment_filters = segments_to_process_input_filters[cur_segment_to_process][0]

                # for change ranges, take the largest range over all input ranges
                # e.g. [ [(1,3), (4,6)], [(2,4), (4,5)] ] -> [(1,4), (3,6)]
                # all should have the same length
                cur_input_change_ranges = resolve_multi_input_change_ranges(
                    segments_to_process_input_change_ranges[cur_segment_to_process])

                segment = GraphSegment(cur_segment_tensor, cur_input_seqlen,
                                       cur_input_change_ranges)

                while True:
                    assert(node_to_segment[cur_segment_tensor]
                           == cur_segment_to_process)

                    if flatten_model.node_is_layer(cur_segment_tensor):
                        layer_name = nodes[cur_segment_tensor].__class__.__name__

                        if layer_name in LOCAL_LAYERS:
                            if layer_name == 'Conv1D':
                                change_range_objects.append(Conv1DChangeRanges(
                                    nodes[cur_segment_tensor].get_config()))

                                # number of filters updated by Conv1D layer
                                segment_filters = nodes[cur_segment_tensor].get_config()[
                                    'filters']

                            elif layer_name == 'MaxPooling1D':
                                change_range_objects.append(MaxPooling1DChangeRanges(
                                    nodes[cur_segment_tensor].get_config()))

                            elif layer_name == 'Cropping1D':
                                change_range_objects.append(Cropping1DChangeRanges(
                                    nodes[cur_segment_tensor].get_config()))

                        elif (layer_name not in SEE_THROUGH_LAYERS) and \
                                (layer_name not in AGGREGATE_LAYERS):
                            raise not_supported_error(
                                "Layer \"{}\"".format(layer_name))

                    if len(edges[cur_segment_tensor]) != 1 or \
                            node_to_segment[edges[cur_segment_tensor][0]] != cur_segment_to_process:
                        # if 0 or >1 out-edges, implies end of this segment
                        # >1 implies start of another segment,
                        # or if next segment does not belong to this segment

                        # process this segment
                        segment.update_forward_output(*ChangeRangesBase.forward_compose(
                            change_range_objects, cur_input_seqlen, cur_input_change_ranges))
                        segment.update_num_filters(segment_filters)
                        segments[cur_segment_to_process] = segment

                        # add next segments to list
                        for node in edges[cur_segment_tensor]:
                            # add start nodes to segments_to_process
                            next_segment = node_to_segment[node]

                            # do not add if already present
                            if (next_segment, node) not in segments_to_process:
                                segments_to_process.append(
                                    (next_segment, node))

                            # but add this info
                            segments_to_process_input_seqlens[next_segment].append(
                                segment.output_seqlen)
                            segments_to_process_input_filters[next_segment].append(
                                segment_filters)
                            segments_to_process_input_change_ranges[next_segment].append(
                                segment.output_perturbed_ranges)

                        # break from while True loop
                        break

                    else:
                        cur_segment_tensor = edges[cur_segment_tensor][0]

    return segments


def resolve_multi_input_change_ranges(input_change_ranges_list):
    """For AGGREGATE_LAYERS such as Add, the different inputs have different
    change ranges. For the change ranges, take the largest range over all
    input ranges:

    e.g. [ [(1,3), (4,6)], [(2,4), (4,5)] ] -> [(1,4), (3,6)]
           input1 -^         input2 -^

    :param input_change_ranges_list: list of list of tuples. Inner lists must
    have same length, where each ith tuple corresponds to ith mutation in the
    input (ith input change range).
    :type input_change_ranges_list: list[list[tuple]]
    :return: Resolved input change ranges. All ranges must have the same width.
    :rtype: list[tuple]
    """

    # change range lists should have the same length
    assert(len(set([len(x) for x in input_change_ranges_list])) == 1)

    # get maximal interval
    # [ [(1,3), (4,6)], [(2,4), (4,5)] ] -> [(1,4), (4,6)]
    input_change_ranges = [(min([x[0] for x in ranges]),
                            max([x[1] for x in ranges])) for ranges
                           in zip(*input_change_ranges_list)]

    # adjust intervals to have same width
    # [(1,4), (4,6)] -> [(1,4), (3,6)]
    max_end = max([y for _, y in input_change_ranges])
    max_width = max([y-x for x, y in input_change_ranges])

    input_change_ranges = [(x, x+max_width) if x+max_width <= max_end else
                           (max_end-max_width, max_end) for x, y
                           in input_change_ranges]

    return input_change_ranges


def generate_intermediate_output_model(model, nodes, edges, inbound_edges,
                                       outputs, node_to_segment,
                                       stop_segment_idxs):
    inputs = ["TENSOR/{}".format(i.name) for i in model.inputs]
    assert(all([i in nodes for i in inputs]))

    # passed as input as for nested nodes it can contain subgraph names
    # so flatten_model.get_flattened_graph returns output names after
    # stripping subgraph names
    assert(all([o in nodes for o in outputs]))

    node_to_tensor = dict()
    # this will include not just output but also intermediate outputs
    output_tensor_names = []
    for output_node in outputs:
        # reverse graph traversal
        node_to_tensor, output_tensor_names = \
            generate_intermediate_output_subgraph(output_node,
                                                  node_to_tensor,
                                                  output_tensor_names,
                                                  nodes, edges,
                                                  inbound_edges,
                                                  node_to_segment,
                                                  stop_segment_idxs)

        if output_node not in output_tensor_names:
            output_tensor_names.append(output_node)

    intermediate_output_model = tf.keras.Model(inputs=[node_to_tensor[i] for i in inputs],
                                               outputs=[node_to_tensor[o]
                                                        for o in output_tensor_names],
                                               name='intermediate_output_model')

    return intermediate_output_model, output_tensor_names


def generate_intermediate_output_subgraph(current_node, node_to_tensor,
                                          output_tensor_names, nodes, edges,
                                          inbound_edges, node_to_segment,
                                          stop_segment_idxs):
    # nodes: mapping from Node name -> layer object of model if layer else None
    # weights are copied within this

    # function traces back from current node
    # recursively adds all "upstream" tensors to node_to_tensor
    # decides if tensor itself should be in output_tensor_names

    # INVARIANT: should only run on tensors, not layers
    assert(not flatten_model.node_is_layer(current_node))

    # if exists already, don't recompute
    if current_node in node_to_tensor:
        return node_to_tensor, output_tensor_names

    # tensor should only have one parent layer
    assert(len(inbound_edges[current_node]) == 1)

    parent_layer = inbound_edges[current_node][0]

    # bipartite-ness => parent of node must be layer, not another tensor
    assert(flatten_model.node_is_layer(parent_layer))

    if len(edges[parent_layer]) > 1:
        raise NotImplementedError(
            "Layer with multiple outputs, what to do?")

    config = deepcopy(nodes[parent_layer].get_config())
    config['name'] = "IntOut_{}".format(config['name'])

    if len(inbound_edges[parent_layer]) == 0:
        # must be input layer
        assert(isinstance(nodes[parent_layer], tf.keras.layers.InputLayer))

        # make an input layer with same dimensions
        node_to_tensor[current_node] = tf.keras.layers.Input(**config)

    else:
        for parent_layer_input in inbound_edges[parent_layer]:
            node_to_tensor, output_tensor_names = \
                generate_intermediate_output_subgraph(parent_layer_input,
                                                      node_to_tensor,
                                                      output_tensor_names,
                                                      nodes, edges,
                                                      inbound_edges,
                                                      node_to_segment,
                                                      stop_segment_idxs)

        # make the layer
        layer = nodes[parent_layer].__class__(**config)

        if len(inbound_edges[parent_layer]) == 1:
            node_to_tensor[current_node] = layer(
                node_to_tensor[inbound_edges[parent_layer][0]])
        else:
            node_to_tensor[current_node] = layer(
                [node_to_tensor[n] for n in inbound_edges[parent_layer]])

        # set weights
        layer.set_weights(nodes[parent_layer].get_weights())

    # determine if output edges of node have different segment
    # and both are not in stop_segment_idxs (no need to cache output at junction
    # of 2 segments both of which belong to stop_segment_idxs)
    # if so, then add to output_tensor_names
    # this is trivially true if > 1 outbound edges (since segment_subgraph
    # would change segment_idx for tensor with multiple edges), but nonetheless
    # this is not explicitly assumed
    for n in edges[current_node]:
        if node_to_segment[current_node] != node_to_segment[n] and \
            (node_to_segment[current_node] not in stop_segment_idxs or
             node_to_segment[n] not in stop_segment_idxs) and \
                current_node not in output_tensor_names:
            output_tensor_names.append(current_node)
            break

    return node_to_tensor, output_tensor_names


def generate_fast_ism_model(model, nodes, edges, inbound_edges, outputs,
                            node_to_segment, stop_segment_idxs,
                            alternate_input_segment_idxs, segments):
    inputs = ["TENSOR/{}".format(i.name) for i in model.inputs]
    assert(all([i in nodes for i in inputs]))

    # passed as input as for nested nodes it can contain subgraph names
    # so flatten_model.get_flattened_graph returns output names after
    # stripping subgraph names
    assert(all([o in nodes for o in outputs]))

    # tensor for each node edge, if a node has multiple edges
    # then it is a point for segment change, and thus will have
    # different slice assignment outputs for each edge
    node_edge_to_tensor = dict()
    input_tensors = []
    input_specs = []
    # this will include all inputs including intermediate outputs from
    # unperturbed sequence
    for output_node in outputs:
        # reverse graph traversal
        node_edge_to_tensor, input_tensors, input_specs = \
            generate_fast_ism_subgraph(output_node, node_edge_to_tensor,
                                       input_tensors, input_specs, nodes,
                                       edges, inbound_edges, node_to_segment,
                                       stop_segment_idxs,
                                       alternate_input_segment_idxs, segments)

    fast_ism_model = tf.keras.Model(inputs=input_tensors,
                                    outputs=[node_edge_to_tensor[(inbound_edges[o][0], o)]
                                             for o in outputs],
                                    name='fast_ism_model')

    return fast_ism_model, input_specs


def generate_fast_ism_subgraph(current_node, node_edge_to_tensor, input_tensors,
                               input_specs, nodes, edges, inbound_edges,
                               node_to_segment, stop_segment_idxs,
                               alternate_input_segment_idxs, segments):
    # nodes: mapping from Node name -> layer object of model if layer else None
    # weights are copied within this

    # function traces back from current node (reverse graph traversal)
    # recursively adds all "upstream" tensors to node_edge_to_tensor

    # INVARIANT: should only run on tensors, not layers
    assert(not flatten_model.node_is_layer(current_node))

    # tensor should only have one parent layer
    assert(len(inbound_edges[current_node]) == 1)

    parent_layer = inbound_edges[current_node][0]

    # bipartite-ness => parent of node must be layer, not another tensor
    assert(flatten_model.node_is_layer(parent_layer))

    # if exists already, don't recompute
    # TODO: re-check this code
    if (parent_layer, current_node) in node_edge_to_tensor:
        return node_edge_to_tensor, input_tensors, input_specs

    if len(edges[parent_layer]) > 1:
        raise NotImplementedError(
            "Layer with multiple outputs, what to do?")

    if node_to_segment[current_node] in alternate_input_segment_idxs:
        return process_alternate_input_node(current_node, node_edge_to_tensor,
                                            input_tensors, input_specs, nodes,
                                            edges, inbound_edges,
                                            node_to_segment,
                                            alternate_input_segment_idxs)

    config = deepcopy(nodes[parent_layer].get_config())
    config['name'] = "FastISM_{}".format(config['name'])

    if len(inbound_edges[parent_layer]) == 0:
        # must be input layer
        assert(isinstance(nodes[parent_layer], tf.keras.layers.InputLayer))

        # make an input layer
        input_segment = segments[node_to_segment[current_node]]
        input_width = input_segment.input_unperturbed_width()

        if len(config['batch_input_shape']) != 3:
            raise ValueError(
                "Currently sequence inputs should be of dim (None, seqlen, num_chars)")

        config['batch_input_shape'] = (
            None, input_width, config['batch_input_shape'][2])

        # TODO: special treatment for non-seq input
        config['name'] = "FastISM_input_perturbation"
        node_edge_to_tensor[(parent_layer, current_node)
                            ] = tf.keras.layers.Input(**config)

        input_tensors.append(node_edge_to_tensor[(parent_layer, current_node)])
        input_specs.append(("SEQ_PERTURB",))

    else:
        for parent_layer_input in inbound_edges[parent_layer]:
            node_edge_to_tensor, input_tensors, input_specs = \
                generate_fast_ism_subgraph(parent_layer_input,
                                           node_edge_to_tensor, input_tensors,
                                           input_specs, nodes, edges,
                                           inbound_edges, node_to_segment,
                                           stop_segment_idxs,
                                           alternate_input_segment_idxs,
                                           segments)

        # make the layer
        layer_name = nodes[parent_layer].__class__.__name__

        if layer_name == 'Cropping1D' and node_to_segment[parent_layer] not in stop_segment_idxs:
            # do nothing, forward tensor
            node_edge_to_tensor[(parent_layer, current_node)] = node_edge_to_tensor[(
                inbound_edges[parent_layer][0], parent_layer)]

        else:
            if layer_name == 'Conv1D':
                # Padding will be added externally (unless within stop_segment_idx)
                if node_to_segment[parent_layer] not in stop_segment_idxs:
                    config['padding'] = 'valid'

                layer = nodes[parent_layer].__class__(**config)

            elif layer_name == 'Flatten':
                # this is necessary as SliceAssign loses dimension data
                # and Flatten then shows shape as (None, None)
                # which wreaks havoc downstream
                layer = tf.keras.layers.Reshape(
                    nodes[parent_layer].output_shape[1:])
            else:
                layer = nodes[parent_layer].__class__(**config)

            # call layer
            if len(inbound_edges[parent_layer]) == 1:
                node_edge_to_tensor[(parent_layer, current_node)] = layer(
                    node_edge_to_tensor[(inbound_edges[parent_layer][0], parent_layer)])
            else:
                node_edge_to_tensor[(parent_layer, current_node)] = layer(
                    [node_edge_to_tensor[(n, parent_layer)] for n in inbound_edges[parent_layer]])

            # set weights
            layer.set_weights(nodes[parent_layer].get_weights())

    # if output edges of node have different segment
    # and both segments are not stop segments
    # perform slice assignment and store relevant tensor
    for next_layer_node in edges[current_node]:
        if (node_to_segment[current_node] != node_to_segment[next_layer_node]) and \
            ((node_to_segment[current_node] not in stop_segment_idxs) or
                (node_to_segment[next_layer_node] not in stop_segment_idxs)):

            cur_segment = segments[node_to_segment[current_node]]

            next_segment = segments[node_to_segment[next_layer_node]]

            next_segment_in_width = next_segment.input_unperturbed_width()
            cur_segment_out_filters = cur_segment.num_out_filters
            cur_segment_out_width = cur_segment.output_perturbed_width()

            # 2 additional inputs:
            # - slice of intermediate output from unperturbed input
            # - scalar offset for perturbed wrt above
            name = 'FastISM_{}_{}_slice'.format(
                current_node, next_layer_node).replace(":", "/")
            unperturbed_intout_input = tf.keras.Input(
                shape=(next_segment_in_width, cur_segment_out_filters),
                name=name)

            input_tensors.append(unperturbed_intout_input)
            # add info sufficient for preparing inputs
            input_specs.append(("INTOUT_SEQ", {
                "node": current_node,
                "slices": next_segment.input_unperturbed_slices,
                "padding": next_segment.input_unperturbed_padding
            }))

            name = 'FastISM_{}_{}_offset'.format(
                current_node, next_layer_node).replace(":", "/")
            offset_input = tf.keras.Input(batch_size=1, shape=(), dtype='int32',
                                          name=name)

            # compute offset by which the output of this layer would
            # be placed on the unperturbed intermediate output
            # need to adjust for padding as next_segment.input_unperturbed_slices
            # are after padding, whereas cur_segment.output_perturbed_ranges
            # are before padding
            left_pad = next_segment.input_unperturbed_padding[0]
            offsets = [x+left_pad-x_u for (x, _), (x_u, _) in
                       zip(cur_segment.output_perturbed_ranges,
                           next_segment.input_unperturbed_slices)]

            input_tensors.append(offset_input)
            input_specs.append(("OFFSET", {
                "offsets": offsets
            }))

            # slice assignment layer
            layer = SliceAssign(next_segment_in_width, cur_segment_out_width)
            node_edge_to_tensor[(current_node,
                                 next_layer_node)] = layer([unperturbed_intout_input,
                                                            node_edge_to_tensor[(
                                                                parent_layer, current_node)],
                                                            offset_input])

        else:
            # carry forward same tensor
            node_edge_to_tensor[(current_node, next_layer_node)] = node_edge_to_tensor[(
                parent_layer, current_node)]

    assert(len(input_tensors) == len(input_specs))
    return node_edge_to_tensor, input_tensors, input_specs


def process_alternate_input_node(current_node, node_edge_to_tensor,
                                 input_tensors, input_specs, nodes, edges,
                                 inbound_edges, node_to_segment,
                                 alternate_input_segment_idxs):
    assert(node_to_segment[current_node] in alternate_input_segment_idxs)

    parent_layer = inbound_edges[current_node][0]

    name = 'FastISM_{}'.format(current_node).replace(":", "/")

    shape = nodes[parent_layer].output_shape
    if isinstance(shape, list):
        assert(len(shape) == 1)
        shape = shape[0]

    alt_input_intout_output = tf.keras.Input(
        shape=shape[1:],
        name=name)

    input_tensors.append(alt_input_intout_output)
    # add info sufficient for preparing inputs
    input_specs.append(("INTOUT_ALT", {
        "node": current_node
    }))

    node_edge_to_tensor[(parent_layer, current_node)] = alt_input_intout_output

    for next_layer_node in edges[current_node]:
        # choose edges that connect into stop layer/downstream of stop layer
        # not explicitly checking if downstream of stop layer since that is
        # enforced by segment_model
        if node_to_segment[next_layer_node] not in alternate_input_segment_idxs:
            node_edge_to_tensor[(current_node, next_layer_node)
                                ] = alt_input_intout_output

    return node_edge_to_tensor, input_tensors, input_specs


def generate_models(model, seqlen, num_chars, seq_input_idx, change_ranges, early_stop_layers=None):
    # generate 2 models: first returns intermediate outputs for unperturbed inputs,
    # second is the "FastISM" model that runs on perturbed inputs

    nodes, edges, inbound_edges, _, output_nodes = flatten_model.get_flattened_graph(
        model)

    # break model into segments
    node_to_segment, stop_segment_idxs, alternate_input_segment_idxs = segment_model(
        model, nodes, edges, inbound_edges, seq_input_idx, early_stop_layers)

    # stop_segment_idxs contains all the segments beyond which full computation
    # takes place, i.e. computations are equivalent to naive implementation.
    # By default segments including and downstream of those containing
    # STOP_LAYERS are contained in stop_segment_idxs. However, if custom
    # early_stop_layers are defined, segments including and downstream of those
    # are also added to stop_segment_idxs and are equivalent to naive
    # implementation. Intermediate output at junctions between two segments both
    # in stop_segment_idxs are not stored as they are not required

    # for each segment, compute metadata used for stitching together outputs
    # dict: segment_idx -> GraphSegment object
    segments = compute_segment_change_ranges(model, nodes, edges,
                                             inbound_edges,
                                             node_to_segment,
                                             stop_segment_idxs,
                                             seqlen, num_chars,
                                             change_ranges,
                                             seq_input_idx)
    # TODO: check if this makes sense
    # compute_segment_change_ranges does not process segments belonging to
    # alternate (non-sequence) inputs
    # assert(len(segments) == len(set(node_to_segment.values())) -
    #        len(alternate_input_segment_idxs))

    # weaker version. Would not have an entry for segments in stop_segment_idxs
    # that are only connected to other segments in stop_segment_idxs
    assert(len(segments) >= len(set(node_to_segment.values())) -
           len(alternate_input_segment_idxs) - len(stop_segment_idxs) + 1)

    # augment model to return a model that returns intermediate outputs
    # returns all tensors that occur at segment change-points
    intout_model, intout_output_tensors = generate_intermediate_output_model(
        model, nodes, edges, inbound_edges, output_nodes, node_to_segment,
        stop_segment_idxs)

    fast_ism_model, input_specs = generate_fast_ism_model(
        model, nodes, edges, inbound_edges, output_nodes, node_to_segment,
        stop_segment_idxs, alternate_input_segment_idxs, segments)

    return output_nodes, intout_model, intout_output_tensors, fast_ism_model, input_specs
