import tensorflow as tf
from tensorflow.python.keras.layers import wrappers
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import functional
from tensorflow.python.util import nest
from collections import defaultdict
import pydot


def is_input_layer(layer):
    """Checks if layer is an input layer

    :param layer: A Keras layer
    :type layer: tf.keras.layers
    :return: True if layer is input layer, else False
    :rtype: bool
    """
    return isinstance(layer, tf.keras.layers.InputLayer)


def strip_subgraph_names(name, subgraph_names):
    "subgraph_name1/subgraph_name2/layer/name -> layer/name"
    while name.split("/")[0] in subgraph_names:
        name = name[name.find("/")+1:]
    return name


def node_is_layer(node_name):
    # if False then Tensor
    return node_name.startswith("LAYER")


def list_replace(l, old, new):
    return [new if t == old else t for t in l]


def is_bipartite(edges):
    # only layer->tensor/tensor->layer connections allowed
    for node in edges:
        is_layer = node_is_layer(node)
        for ngb in edges[node]:
            if node_is_layer(ngb) == is_layer:
                return False
    return True


def is_consistent(edges, inbound_edges):
    inbound_edges_from_edges = defaultdict(set)
    for x in edges:
        for y in edges[x]:
            inbound_edges_from_edges[y].add(x)
    if set(inbound_edges_from_edges.keys()) != set(inbound_edges.keys()):
        return False

    for x in inbound_edges:
        if set(inbound_edges[x]) != inbound_edges_from_edges[x]:
            return False
    return True


def get_flattened_graph(model, is_subgraph=False):
    """[summary]

    :param model: [description]
    :type model: [type]
    :param is_subgraph: [description], defaults to False
    :type is_subgraph: bool, optional
    :return: [description]
    :rtype: [type]
    """
    # Inspired by: https://github.com/tensorflow/tensorflow/blob/b36436b/tensorflow/python/keras/utils/vis_utils.py#L70
    # Wrapper support like in model_to_dot??
    # MORE comments
    # gets rid of intermediate inputlayers and makes graph bipartite
    layers = model.layers
    nodes = dict()
    edges = defaultdict(list)
    inbound_edges = defaultdict(list)
    subgraph_names = set()

    if isinstance(model, tf.keras.Sequential):
        if not model.built:
            model.build()
        # same as in model_to_dot, without this the layers don't contain
        # the input layer for some reason
        layers = super(tf.keras.Sequential, model).layers

    for _, layer in enumerate(layers):
        layer_name = "LAYER/{}".format(layer.name)

        if isinstance(layer, tf.keras.Sequential) or isinstance(layer, functional.Functional):
            subgraph_nodes, subgraph_edges, subgraph_inbound_edges, \
                subsubgraph_names, _ = get_flattened_graph(
                    layer, is_subgraph=True)

            nodes.update(subgraph_nodes)
            edges.update(subgraph_edges)
            inbound_edges.update(subgraph_inbound_edges)
            subgraph_names.add(layer.name)
            subgraph_names.update(subsubgraph_names)

        else:
            for o in nest.flatten(layer.output):
                nodes["TENSOR/{}".format(o.name)] = None

                if not (is_subgraph and isinstance(layer, tf.keras.layers.InputLayer)):
                    # TBD if necessary
                    nodes[layer_name] = layer
                    # layer -> tensor edge (trivial)
                    edges[layer_name].append("TENSOR/{}".format(o.name))
                    inbound_edges["TENSOR/{}".format(o.name)
                                  ].append(layer_name)

    # tensor -> inputLayer tensor edges
    # tensor -> Layer edges
    for _, layer in enumerate(layers):
        # len(layer.inbound_nodes) is > 1 when models are nested
        # however, it seems like all different layer.inbound_nodes[i].input_tensors
        # point to the same tensors, through different scopes
        # using the 1st seems to work along with stripping subgraph names
        # assert(len(layer.inbound_nodes) == 1)

        layer_input_tensors = [x.name for x in nest.flatten(
            layer.inbound_nodes[0].input_tensors)]
        # if inbound node comes from a subgraph, it will start with "subgraph_name/"
        # if it comes from subgraph within subgraph, it will start with "subgraph_name1/subgraph_name2/"
        # but "nodes" do not have subgraph_name in them
        for i in range(len(layer_input_tensors)):
            layer_input_tensors[i] = strip_subgraph_names(
                layer_input_tensors[i], subgraph_names)

        layer_input_tensors = [
            "TENSOR/{}".format(x) for x in layer_input_tensors]

        assert(all([x in nodes for x in layer_input_tensors]))

        if isinstance(layer, tf.keras.Sequential) or isinstance(layer, functional.Functional):
            layer_inputlayer_names = [
                "TENSOR/{}".format(x.name) for x in layer.inputs]

            assert(all([x in nodes for x in layer_inputlayer_names]))
            assert(len(layer_input_tensors) == len(layer_inputlayer_names))

            # assuming order of inputs is preserved
            # inbound_edges should store inputs in correct order for multi
            # input layers
            for x, y in zip(layer_input_tensors, layer_inputlayer_names):
                # transfering edges of y to x and deleting y
                for e in edges[y]:
                    edges[x].append(e)
                    # replace y by x in inbound_edges
                    inbound_edges[e] = list_replace(inbound_edges[e], y, x)

                del edges[y]
                del nodes[y]

        elif not isinstance(layer, tf.keras.layers.InputLayer):
            layer_name = "LAYER/{}".format(layer.name)

            for x in layer_input_tensors:
                edges[x].append(layer_name)
                # this preserves order of inputs
                inbound_edges[layer_name].append(x)

    assert(is_bipartite(edges))

    # ensure edges and inbound_edges agree
    assert(is_consistent(edges, inbound_edges))

    # strip model output names
    output_nodes = ["TENSOR/{}".format(strip_subgraph_names(x.name, subgraph_names))
                    for x in model.outputs]
    assert(all([o in nodes for o in output_nodes]))

    return nodes, edges, inbound_edges, subgraph_names, output_nodes


def viz_graph(nodes, edges, outpath):
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    # dot.set('concentrate', True)
    # dot.set_node_defaults(shape='record')
    dot.set('dpi', 96)
    for x in nodes:
        dot.add_node(pydot.Node(x.replace(":", "/"),
                                label=x.replace(":", "/")))
    for x in edges:
        for y in edges[x]:
            dot.add_edge(pydot.Edge(x.replace(":", "/"), y.replace(":", "/")))
    dot.write(outpath, format='png')
