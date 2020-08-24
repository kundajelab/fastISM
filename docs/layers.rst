Supported Layers
================
This sections covers the layers that are currently supported by fastISM. fastISM supports a subset of layers in ``tf.keras.layers`` that are most commonly used for sequence-based models. 

**NOTE**: Restrictions on layers apply only till :ref:`stop-layers`, beyond which all layers are allowed.

The layers below have been classified by which positions of the output are a function of the input at the ``i`` th position.

See Through Layers
------------------
See through layers are layers for which the output at the ``i`` th position depends on the input at the ``i`` th position only.

**Supported**:
|SEETHRU|

Aggregation Layers
------------------
Aggregation layers are also See Through Layers as the output at the ``i`` th position depends on the input at the ``i`` th position only. The main difference is that Aggregation layers take in multiple inputs, and thus their output at the ``i`` th position depends on the ``i`` th position of all their inputs.

**Supported**:
|AGG| 

Local Layers
------------
For Local layers, the input at the ``i`` th position affects a fixed interval of outputs around the ``i`` th position.

**Supported**:
|LOCAL|

.. _stop-layers:

Stop Layers
-----------
Layers after which output at ``i`` th position depends on inputs at most or all positions in the input. However, this is not true for Flatten/Reshape, but it is assumed these are followed by Dense or similar.

**Supported**:
|STOP|