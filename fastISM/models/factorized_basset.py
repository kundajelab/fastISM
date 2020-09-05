import tensorflow as tf


def factorized_basset_model(seqlen=1000, numchars=4, num_outputs=1, name='factorized_basset_model'):
    inp = tf.keras.Input(shape=(seqlen, numchars))

    # conv mxp 1
    x = tf.keras.layers.Conv1D(48, 3, padding='same', name='conv1a')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', name='conv1b')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(100, 3, padding='same', name='conv1c')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(150, 7, padding='same', name='conv1d')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(300, 7, padding='same', name='conv1e')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool1D(3)(x)

    # conv mxp 2
    x = tf.keras.layers.Conv1D(200, 7, padding='same', name='conv2a')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(200, 3, padding='same', name='conv2b')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(200, 3, padding='same', name='conv2c')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool1D(4)(x)

    # conv mxp 3
    x = tf.keras.layers.Conv1D(200, 7, padding='same', name='conv3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

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
