import tensorflow as tf


def basset_model(seqlen=1000, numchars=4, num_outputs=1, name='basset_model'):
    inp = tf.keras.Input(shape=(seqlen, numchars))

    # conv mxp 1
    x = tf.keras.layers.Conv1D(
        300, 19, strides=1, padding='same', activation='relu', name='conv1')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(3)(x)

    # conv mxp 2
    x = tf.keras.layers.Conv1D(
        200, 11, strides=1, padding='same', activation='relu', name='conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(4)(x)

    # conv mxp 3
    x = tf.keras.layers.Conv1D(
        200, 7, strides=1, padding='same', activation='relu', name='conv3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(4)(x)

    # fc
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1000, activation='relu', name='fc1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1000, activation='relu', name='fc2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(num_outputs, name='fc3')(x)

    model = tf.keras.Model(inputs=inp, outputs=x, name=name)

    return model
