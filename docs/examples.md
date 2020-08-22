# Examples

This section covers some of the common use cases and functionalities of fastISM.

## Alternate Mutation
By default, inputs at the ith position are set to zero. **TODO**

## Alternate Range
Can also set range of inputs instead of single positions.

## Multi-input models
**TODO**

## Recursively Defined models
Keras allows defning models in a nested fashion. As such, recursively defined models should not pose an issue to fastISM. The example below works:

```python
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
```