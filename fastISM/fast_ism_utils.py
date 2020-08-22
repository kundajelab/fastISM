import tensorflow as tf
from . import flatten_model
from collections import defaultdict
from copy import deepcopy

from .change_range import ChangeRangesBase, Conv1DChangeRanges, MaxPooling1DChangeRanges, not_supported_error

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
    # 'Subtract' TODO: bugs out since inbound_edges does not contain right order of nodes
}

# layers at which output at ith position depends on a window around the ith position
LOCAL_LAYERS = {
    'Conv1D',
    'MaxPooling1D'
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
        # GOAL: a[:,i:min(i+b.shape[1], a.shape[1])] = b
        # clip b if i+b.shape[1] exceeds width of a, guarantee width of output
        # is same as a. This could happen when a layer's output (b) feeds into
        # multiple layers, but some layers don't need all positions of b
        # (can happen near the edges).
        # See test_skip_then_mxp of test/test_simple_skip_conn_architectures.py

        a, b, i = inputs

        # output will lose shape info (dim 1 will be set to None)
        return tf.cond(i[0]+self.b_dim <= self.a_dim,
                       lambda: tf.concat(
                           [a[:, :i[0]], b, a[:, i[0]+self.b_dim:]], axis=1),
                       lambda: tf.concat([a[:, :i[0]], b[:, :self.a_dim-i[0]]], axis=1))


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
def segment_model(model, nodes, edges, inbound_edges):
    # segment model into groups that can be run as a unit. Intermediate outputs after each group
    # should be captured for unperturbed inputs. Segmenting thus helps minimise number of intermediate
    # outputs that need to be captured

    # starting with single sequence-only input models
    # will relax later
    assert(len(model.inputs) == 1)

    input_layer = "LAYER/{}".format(model.input_names[0])
    assert(input_layer in nodes)

    assert(len(edges[input_layer]) == 1)
    input_tensor = edges[input_layer][0]

    return segment_subgraph(input_tensor, nodes, edges, inbound_edges, dict(), 0, 0)[0]


def segment_subgraph(current_node, nodes, edges, inbound_edges, node_to_segment, segment_idx, num_convs_in_cur_segment):
    # segment_idx is the current segment_idx
    # node_to_segment is dict from node->segment

    # already segmented
    if current_node in node_to_segment:
        return node_to_segment, segment_idx

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
            # recursively label all descendants (no more further segments)
            return label_descendants(current_node, nodes, edges, node_to_segment, segment_idx), segment_idx+1

        elif layer_class == 'Conv1D':
            # enforce that if a segment has a conv layer, it is always at the beginning
            # by doing this, pre-conv intermediate outputs will always be captured and
            # padding them would become a one-time operation
            segment_idx += 1

            node_to_segment[current_node] = segment_idx
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, segment_idx, 1)

        elif len(inbound_edges[current_node]) > 1:
            segment_idx += 1
            node_to_segment[current_node] = segment_idx
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, segment_idx, 0)

        else:
            # single-input, single-output layer -> propagate further
            node_to_segment[current_node] = segment_idx
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, segment_idx, num_convs_in_cur_segment)

    # it's a tensor
    else:
        # tensors can't have > 1 in-degree, but can have > 1 out-degree
        assert(len(inbound_edges[current_node]) == 1)

        node_to_segment[current_node] = segment_idx

        # terminal tensor, done
        if len(edges[current_node]) == 0:
            return node_to_segment, segment_idx+1

        # single edge out, propogate
        elif len(edges[current_node]) == 1:
            return segment_subgraph(edges[current_node][0], nodes, edges, inbound_edges, node_to_segment, segment_idx, num_convs_in_cur_segment)

        # multi edge out => multiple layers use this tensor
        # e.g. resnet layers
        else:
            segment_idx += 1  # increment segment idx
            for next_node in edges[current_node]:
                node_to_segment, segment_idx = segment_subgraph(
                    next_node, nodes, edges, inbound_edges, node_to_segment, segment_idx, 0)
            return node_to_segment, segment_idx


def label_descendants(current_node, nodes, edges, node_to_segment, segment_idx):
    node_to_segment[current_node] = segment_idx

    for node in edges[current_node]:
        node_to_segment = label_descendants(
            node, nodes, edges, node_to_segment, segment_idx)

    return node_to_segment


def compute_segment_change_ranges(model, nodes, edges, inbound_edges,
                                  node_to_segment, input_seqlen, input_filters,
                                  input_change_ranges):
    """
    for each segment, given input change range compute (ChangeRangesBase.forward_compose):
        - input range of intermediate output required
        - offsets for input tensor wrt intermediate output
        - output seqlen
        - output change range
        - number of filters in output
    """

    # starting with single sequence-only input models
    # will relax later
    assert(len(model.inputs) == 1)

    input_layer = "LAYER/{}".format(model.input_names[0])
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
    # only sequence input tensor should be in segment 0
    assert(sum([node_to_segment[x] == 0 for x in node_to_segment]) == 1)
    segments_to_process.append((0, input_tensor))
    segments_to_process_input_seqlens[0] = [input_seqlen]
    segments_to_process_input_filters[0] = [input_filters]
    segments_to_process_input_change_ranges[0] = [input_change_ranges]

    while segments_to_process:
        cur_segment_to_process, cur_segment_tensor = segments_to_process.pop(0)

        if len(segments_to_process_input_seqlens[cur_segment_to_process]) != \
                len(inbound_edges[cur_segment_tensor]):
            # should not be greater in any case
            assert(len(segments_to_process_input_seqlens[cur_segment_to_process]) <
                   len(inbound_edges[cur_segment_tensor]))
            # hold off and wait till other input segments are populated
            assert(len(segments_to_process) > 0)

        # if node marks beginning of dense/flatten/reshape layers, say
        elif nodes[cur_segment_tensor].__class__.__name__ in STOP_LAYERS:
            if len(segments_to_process_input_seqlens[cur_segment_to_process]) > 1:
                raise NotImplementedError("This Block layer takes in multiple \
                     inputs which is not currently supported")
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
                                          None)  # affected range is the whole thing, NA)
            segment.update_num_filters(None)  # output filters NA
            segments[cur_segment_to_process] = segment

        # process current segment
        else:
            change_range_objects = []

            # resolve multiple input_change_ranges
            if len(set(segments_to_process_input_seqlens[cur_segment_to_process])) != 1:
                raise NotImplementedError(
                    "This multi-input layer takes in inputs of different length, \
                        currently not supported")
            if len(set(segments_to_process_input_filters[cur_segment_to_process])) != 1:
                raise NotImplementedError("This multi-input layer takes in \
                                          inputs of different filters, currently not supported")
            cur_input_seqlen = segments_to_process_input_seqlens[cur_segment_to_process][0]
            segment_filters = segments_to_process_input_filters[cur_segment_to_process][0]

            # for change ranges, take the largest range over all input ranges
            # e.g. [[(1,3), (4,6)], [(2,4), (1,5)] -> [(1,4), (1,6)]
            cur_input_change_ranges = [(min([x[0] for x in ranges]),
                                        max([x[1] for x in ranges])) for ranges
                                       in zip(*segments_to_process_input_change_ranges[cur_segment_to_process])]

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
                            segments_to_process.append((next_segment, node))

                        # but add this info
                        segments_to_process_input_seqlens[next_segment].append(
                            segment.output_seqlen)
                        segments_to_process_input_filters[next_segment].append(
                            segment_filters)
                        segments_to_process_input_change_ranges[next_segment].append(
                            segment.output_perturbed_ranges)

                    break

                else:
                    cur_segment_tensor = edges[cur_segment_tensor][0]

    # exclude and input_tensor segment
    # re-evaluate after fixing input_tensor special status
    # this is definitely not necessarily true
    # need to handle block layers properly
    # TODO
    assert(len(segments) ==
           len(set(node_to_segment.values())))

    return segments


def generate_intermediate_output_model(model, nodes, edges, inbound_edges,
                                       outputs, node_to_segment):
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
        node_to_tensor, output_tensor_names = generate_intermediate_output_subgraph(
            output_node, node_to_tensor, output_tensor_names, nodes, edges, inbound_edges, node_to_segment)

        if output_node not in output_tensor_names:
            output_tensor_names.append(output_node)

    intermediate_output_model = tf.keras.Model(inputs=[node_to_tensor[i] for i in inputs],
                                               outputs=[node_to_tensor[o]
                                                        for o in output_tensor_names],
                                               name='intermediate_output_model')

    return intermediate_output_model, output_tensor_names


def generate_intermediate_output_subgraph(current_node, node_to_tensor, output_tensor_names, nodes, edges, inbound_edges, node_to_segment):
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
            node_to_tensor, output_tensor_names = generate_intermediate_output_subgraph(
                parent_layer_input, node_to_tensor, output_tensor_names, nodes, edges, inbound_edges, node_to_segment)

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
    # if so, then add to output_tensor_names
    # this is trivially true if > 1 outbound edges (since segment_subgraph
    # would change segment_idx for tensor with multiple edges), but nonetheless
    # this is not explicitly assumed
    for n in edges[current_node]:
        if node_to_segment[current_node] != node_to_segment[n]:
            if current_node not in output_tensor_names:
                output_tensor_names.append(current_node)
                break

    return node_to_tensor, output_tensor_names


def generate_fast_ism_model(model, nodes, edges, inbound_edges, outputs, node_to_segment, segments):
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
        node_edge_to_tensor, input_tensors, input_specs = generate_fast_ism_subgraph(
            output_node, node_edge_to_tensor, input_tensors, input_specs, nodes, edges,
            inbound_edges, node_to_segment, segments)

    fast_ism_model = tf.keras.Model(inputs=input_tensors,
                                    outputs=[node_edge_to_tensor[(inbound_edges[o][0], o)]
                                             for o in outputs],
                                    name='fast_ism_model')

    return fast_ism_model, input_specs


def generate_fast_ism_subgraph(current_node, node_edge_to_tensor, input_tensors,
                               input_specs, nodes, edges, inbound_edges,
                               node_to_segment, segments):
    # nodes: mapping from Node name -> layer object of model if layer else None
    # weights are copied within this

    # function traces back from current node
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
                                           segments)

        # make the layer
        layer_name = nodes[parent_layer].__class__.__name__

        if layer_name == 'Conv1D':
            # Padding will be added externally
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

        if len(inbound_edges[parent_layer]) == 1:
            node_edge_to_tensor[(parent_layer, current_node)] = layer(
                node_edge_to_tensor[(inbound_edges[parent_layer][0], parent_layer)])
        else:
            node_edge_to_tensor[(parent_layer, current_node)] = layer(
                [node_edge_to_tensor[(n, parent_layer)] for n in inbound_edges[parent_layer]])

        # set weights
        layer.set_weights(nodes[parent_layer].get_weights())

    # if output edges of node have different segment
    # perform slice assignment and store relevant tensor
    for next_layer_node in edges[current_node]:
        if node_to_segment[current_node] != node_to_segment[next_layer_node]:
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
            input_specs.append(("INTOUT", {
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


def generate_models(model, seqlen, num_chars, seq_input_idx, change_ranges):
    # generate 2 models: first returns intermediate outputs for unperturbed inputs,
    # second is the "FastISM" model that runs on perturbed inputs

    nodes, edges, _, output_nodes = flatten_model.get_flattened_graph(model)
    inbound_edges = defaultdict(list)
    for x in edges:
        for y in edges[x]:
            inbound_edges[y].append(x)

    # break model into segments
    node_to_segment = segment_model(model, nodes, edges, inbound_edges)

    # for each segment, compute metadata used for stitching together outputs
    # dict: segment_idx -> GraphSegment object
    segments = compute_segment_change_ranges(model, nodes, edges, inbound_edges, node_to_segment,
                                             seqlen, num_chars, change_ranges)

    # augment model to return a model that returns intermediate outputs
    intout_model, intout_output_tensors = generate_intermediate_output_model(
        model, nodes, edges, inbound_edges, output_nodes, node_to_segment)

    fast_ism_model, input_specs = generate_fast_ism_model(
        model, nodes, edges, inbound_edges, output_nodes, node_to_segment,
        segments)

    return intout_model, intout_output_tensors, fast_ism_model, input_specs
