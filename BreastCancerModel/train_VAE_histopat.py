from vae import VAE
import pickle
import numpy as np


FILE_NAME = 'CNN_classification_task/converted_labeled_data/training_dataset.pkl'

FILTERS = (32, 64, 64, 128, 128)
KERNELS = (3, 3, 3, 3, 3)
STRIDES = (1, 1, 1, 1, 1)
LATENT_SPACE_DIM = 30

LEARNING_RATE = 0.00001
BATCH_SIZE = 15
EPOCHS = 10


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



data_train, labels_train = load_data(filename=FILE_NAME, images_shape=(50, 50, 3))

vae = VAE(input_shape = (50, 50, 3), 
          conv_filters = FILTERS, 
          conv_kernels = KERNELS, 
          conv_strides = STRIDES, 
          latent_space_dim = LATENT_SPACE_DIM, 
          num_channel = 3)

vae.compile_model(kl_weight = 1e-4, learning_rate = LEARNING_RATE)

divisor = int(len(data_train) * 0.80)

vae.train(x_train = data_train[:divisor], 
          x_validation = data_train[divisor:], 
          batch_size = BATCH_SIZE, 
          num_epochs = EPOCHS)

vae.save('model_histopat')





