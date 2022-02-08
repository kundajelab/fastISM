Supported Layers
================
This sections covers the layers that are currently supported by fastISM. fastISM supports a subset of layers in ``tf.keras.layers`` that are most commonly used for sequence-based models. 

**NOTE**: Restrictions on layers apply only till :ref:`stop-layers`, beyond which all layers are allowed.

The layers below have been classified by which positions of the output are a function of the input at the ``i`` th position.

Local Layers
------------
For Local layers, the input at the ``i`` th position affects a fixed interval of outputs around the ``i`` th position.

**Supported**:
|LOCAL|

Currently, custom Local Layers are not supported as they may require additional logic to be incorporated into the code. Please post an `Issue <https://github.com/kundajelab/fastISM/issues>`_ on GitHub to work out a solution.

See Through Layers
------------------
See through layers are layers for which the output at the ``i`` th position depends on the input at the ``i`` th position only.

**Supported**:
|SEETHRU|

To add a custom see-through layer:
``fastism.fast_ism_utils.SEE_THROUGH_LAYERS.add("YourLayer")``

Aggregation Layers
------------------
Aggregation layers are also See Through Layers as the output at the ``i`` th position depends on the input at the ``i`` th position only. The main difference is that Aggregation layers take in multiple inputs, and thus their output at the ``i`` th position depends on the ``i`` th position of all their inputs.

**Supported**:
|AGG| 

To add a custom aggregation layer:
``fastism.fast_ism_utils.AGGREGATE_LAYERS.add("YourLayer")``

.. _stop-layers:

Stop Layers
-----------
Layers after which output at ``i`` th position depends on inputs at most or all positions in the input. However, this is not strictly true for Flatten/Reshape, but it is assumed these are followed by Dense or similar.

**Supported**:
|STOP|

To add a custom stop layer:
``fastism.fast_ism_utils.STOP_LAYERS.add("YourLayer")``

Pooling Layers
--------------
Pooling layers are also Local Layers but are special since they are typically used to reduce the size of the input.

**Supported**:
|POOL|

To add a custom pooling layer:
``fastism.fast_ism_utils.POOLING_LAYERS.add("YourLayer")``

Custom pooling layers must have the class attributes ``pool_size``, ``strides`` (which must be equal to ``pool_size``),  ``padding`` (which must be ``valid``), ``data_format`` (which must be ``channels_last``). Here is an example of a custom pooling layer.

.. code-block:: python

    class AttentionPooling1D(tf.keras.layers.Layer):
    	# don't forget to add **kwargs
        def __init__(self, pool_size = 2, **kwargs):
            super().__init__()
            self.pool_size = pool_size
            
            # need for pooling layer
            self.strides = self.pool_size 
            self.padding = "valid" # ensure it behaves like MaxPooling1D with valid padding
            self.data_format = "channels_last"	        
    
        def build(self, input_shape):
            _, length, num_features = input_shape
            self.w = self.add_weight(
                shape=(num_features, num_features),
                initializer="random_normal",
                trainable=True,
            )
        
        # implement so that layer can be duplicated
        def get_config(self):
            config = super().get_config()
            config.update({
                "pool_size": self.pool_size,
                "data_format": self.data_format,
                "strides": self.strides,
                "padding": self.padding
            })
            return config
        
        def call(self, inputs):
            _, length, num_features = inputs.shape
            
            if length == None: # this can happen at when creating fast_ism_model
                return inputs  # don't do anything for now
                
            inputs = tf.reshape(
                inputs,
                (-1, length // self.pool_size, self.pool_size, num_features))
        
            return tf.reduce_sum(
                inputs * tf.nn.softmax(tf.matmul(inputs, self.w), axis=-2),
                axis=-2)


Code adapted from `Enformer <https://github.com/deepmind/deepmind-research/blob/master/enformer/enformer.py>`_. Note that pooling layers can contain weights.
