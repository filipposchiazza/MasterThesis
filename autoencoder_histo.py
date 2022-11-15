import pickle
import numpy as np
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras import Model, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_FILENAME = 'CNN_classification_task/converted_labeled_data/training_dataset.pkl'

INPUT_SHAPE = (50, 50, 3)
CONV_FILTERS = [32, 64, 128]
CONV_KERNELS = [3, 3, 3]
CONV_STRIDES = [1, 1, 1]
LATENT_DIM = 7500

learning_rate = 0.001
batch_size = 50
num_epochs = 50


def load_data(filename, images_shape):
    # Read data as numpy array and apply conversion of labels from string to int
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    f.close()
    data = dataset["data"][:1000]
    labels = dataset["labels"][:1000]
    del dataset
    labels = np.asarray(labels)
    labels = labels.astype(int)
    data = np.asarray(data).reshape((-1, images_shape[0], images_shape[1], images_shape[2]))
    data = data.astype("float32") / 255
    return data, labels

# Load data
x_train, y_train = load_data(filename=DATA_FILENAME, images_shape=INPUT_SHAPE)




encoder_input = layers.Input(shape=(50, 50, 3))

# encoder
x = layers.Conv2D(filters=16, 
                  kernel_size=3, 
                  strides=1,
                  padding='same',
                  activation='relu')(encoder_input)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(filters=32, 
                  kernel_size=3, 
                  strides=1,
                  padding='same',
                  activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(filters=64, 
                  kernel_size=3, 
                  strides=1,
                  padding='same',
                  activation='relu')(x)
x = layers.MaxPooling2D((3,3))(x)

shape_before_flatten = K.int_shape(x)[1:]
x = layers.Flatten()(x)
x = layers.Dense(units=LATENT_DIM)(x)

# decoder
x = layers.Dense(units=np.prod(shape_before_flatten))(x)
x = layers.Reshape(target_shape=shape_before_flatten)(x)



x = layers.Conv2DTranspose(filters=64,
                           kernel_size=3,
                           padding='same',
                           activation='relu')(x)

x = layers.UpSampling2D((3,3))(x)
x = layers.Conv2DTranspose(filters=32,
                           kernel_size=3,
                           padding='same',
                           activation='relu')(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2DTranspose(filters=16,
                           kernel_size=3,
                           padding='same',
                           activation='relu')(x)

x = layers.UpSampling2D()(x)

x = layers.experimental.preprocessing.Resizing(50, 50)(x)

decoder_output = layers.Conv2DTranspose(filters=INPUT_SHAPE[-1],
                           kernel_size=3,
                           padding='same',
                           activation='sigmoid')(x)

autoencoder = Model(inputs=encoder_input, outputs=decoder_output)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mean_absolute_error')

early_stopping = EarlyStopping(monitor='loss',
                               patience=5,
                               restore_best_weights=True)

lr_schedule = ReduceLROnPlateau(monitor='loss',
                                patience=3, 
                                verbose=True)

autoencoder.fit(x_train, 
                x_train, 
                epochs=50, 
                batch_size=50, 
                callbacks=[early_stopping, lr_schedule])



############################################################################
# Test reconstruction

vae = models.load_model('model_ae_rec_histo')







