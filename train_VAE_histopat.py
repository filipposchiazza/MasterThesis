from vae import VAE
import pickle
import numpy as np

LEARNING_RATE = 0.000001
BATCH_SIZE = 25
EPOCHS = 10


def load_dataset (file_name, shape):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    f.close()
    data_reshaped = np.array(data).reshape((-1, shape[0], shape[1], shape[2]))
    data_normalized = data_reshaped.astype('float32') / 255
    return data_normalized


if __name__ == '__main__':
    # load dataset
    data = load_dataset(file_name = './data/data_converted/medium_data_converted.pkl', 
                        shape = (50, 50, 3))

    vae = VAE(input_shape = (50, 50, 3),
              conv_filters = (32, 64, 64, 64, 32),
              conv_kernels = (5, 5, 5, 5, 5),
              conv_strides = (1, 1, 1, 1, 1), 
              latent_space_dim = 10,
              kl_weight = 0.001,
              num_channel = 3)

    vae.summary()
    vae.compile_model(learning_rate=LEARNING_RATE)
    vae.train(data[0:2000], BATCH_SIZE, EPOCHS)
    vae.save('model_histo_images_small')


