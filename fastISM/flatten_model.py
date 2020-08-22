import tensorflow as tf
from tensorflow.python.keras.layers import wrappers
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import functional
from tensorflow.python.util import nest
from collections import defaultdict
import pydot


def is_input_layer(layer):
    return isinstance(layer, tf.keras.layers.InputLayer)


def strip_subgraph_names(name, subgraph_names):
    "subgraph_name1/subgraph_name2/layer/name -> layer/name"
    while name.split("/")[0] in subgraph_names:
        name = name[name.find("/")+1:]
    return name


def node_is_layer(node_name):
    # if False then Tensor
    return "LAYER" in node_name


def is_bipartite(edges):
    # only layer->tensor/tensor->layer connections allowed
    for node in edges:
        is_layer = node_is_layer(node)
        for ngb in edges[node]:
            if node_is_layer(ngb) == is_layer:
                return False
    return True


def get_flattened_graph(model, is_subgraph=False):
    # Inspired by: https://github.com/tensorflow/tensorflow/blob/b36436b087bd8e8701ef51718179037cccdfc26e/tensorflow/python/keras/utils/vis_utils.py#L70
    # Wrapper support like in model_to_dot??
    # MORE comments
    # get rid of intermediate inputlayers, make graph bipartite
    layers = model.layers
    nodes = dict()
    edges = defaultdict(list)
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
            subgraph_nodes, subgraph_edges, subsubgraph_names, _ = get_flattened_graph(
                layer, is_subgraph=True)

            nodes.update(subgraph_nodes)
            edges.update(subgraph_edges)
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
                "TENSOR/{}".format(x.output.name) for x in layer.layers if is_input_layer(x)]

            assert(all([x in nodes for x in layer_inputlayer_names]))
            assert(len(layer_input_tensors) == len(layer_inputlayer_names))

            # assuming order of inputs is preserved
            # need a way to store order info for multi input layers!
            for x, y in zip(layer_input_tensors, layer_inputlayer_names):
                # transfering edges of y to x and deleting y
                for e in edges[y]:
                    edges[x].append(e)
                del edges[y]
                del nodes[y]

        elif not isinstance(layer, tf.keras.layers.InputLayer):
            layer_name = "LAYER/{}".format(layer.name)

            for x in layer_input_tensors:
                edges[x].append(layer_name)

    assert(is_bipartite(edges))

    # strip model output names
    output_nodes = ["TENSOR/{}".format(strip_subgraph_names(x.name, subgraph_names))
                    for x in model.outputs]
    assert(all([o in nodes for o in output_nodes]))

    return nodes, edges, subgraph_names, output_nodes


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
