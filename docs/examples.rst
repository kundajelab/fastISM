Examples
========

This section covers some of the common use cases and functionalities of fastISM.

Alternate Mutation
------------------
By default, inputs at the ith position are set to zero. **TODO**

Alternate Range
----------------
Can also set range of inputs instead of single positions.

Multi-input models
------------------
fastISM supports models which have other inputs in addition to the sequence input that is perturbed. These alternate inputs are assumed to stay constant through different perturbations of the primary sequence input. Consider the model below in which an addition vector is concatenated with the flattened sequence output:

.. code-block:: python

   def get_model():
       rna = tf.keras.Input((100,)) # non-sequence input
       seq = tf.keras.Input((100,4))
     
       x = tf.keras.layers.Conv1D(20, 3)(seq)
       x = tf.keras.layers.Conv1D(20, 3)(x)
       x = tf.keras.layers.Flatten()(x)
 
       rna = tf.keras.layers.Dense(10)(rna)
 
       x = tf.keras.layers.Concatenate()([x, rna])
       x = tf.keras.layers.Dense(10)(x)
       x = tf.keras.layers.Dense(1)(x)
       model = tf.keras.Model(inputs=[rna,seq], outputs=x)

To inform fastISM that the second input is the primary sequence input that will be perturbed:

.. code-block:: python

   >>> model = get_model()
   >>> fast_ism_model = fastISM.FastISM(model, seq_input_idx=1) 

Then to obtain the outputs:

.. code-block:: python

   for rna_batch, seq_batch in data_batches:
      ism_batch = fast_ism_model([rna_batch, seq_batch]) 
 
   # or equivalently without splitting inputs
   for data_batch in data_batches
       ism_batch = fast_ism_model(data_batch)

**NOTE**: Currently, multi-input models in which descendants of alternate inputs interact directly with descendants of primary sequence input *before* a :ref:`Stop Layer <stop-layers>` are not supported, i.e. a descendant of an alternate input in general should only interact with a flattened version of primary input sequence.

Recursively Defined models
--------------------------
Keras allows defning models in a nested fashion. As such, recursively defined models should not pose an issue to fastISM. The example below works:

.. code-block:: python

   def res_block(input_shape):
       inp = tf.keras.Input(shape=input_shape)
       x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)    
       x = tf.keras.layers.Add()([inp, x])
       model = tf.keras.Model(inputs=inp, outputs=x)
       return model
  
   def fc_block(input_shape):
       inp = tf.keras.Input(shape=input_shape)
       x = tf.keras.layers.Dense(10)(inp)
       x = tf.keras.layers.Dense(1)(x)
     
       model = tf.keras.Model(inputs=inp, outputs=x)
       return model
 
   def get_model():
       res = res_block(input_shape=(108,20)))
       fcs = fc_block(input_shape=(36*20,))
 
       inp = tf.keras.Input((108, 4))
       x = tf.keras.layers.Conv1D(20, 3, padding='same')(inp)
       x = res(x)
       x = tf.keras.layers.MaxPooling1D(3)(x)
       x = tf.keras.layers.Flatten()(x)
       x = fcs(x)
 
       model = tf.keras.Model(inputs=inp, outputs=x)
     
       return model
 
   >>> model = get_model()
   >>> fast_ism_model = FastISM(model)