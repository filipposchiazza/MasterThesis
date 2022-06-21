from vae import VAE
from tensorflow.keras.datasets import mnist


LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]

N_EPOCHS = [30, 40, 50, 60]

LATENT_SPACE_DIM = [3, 5, 10, 20, 50]


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalization
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # add channel dimension (1 for gray scale images)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, y_train, x_test, y_test



def train(x_train, learning_rate, batch_size, epochs, latent_dim, kl_weight):
    
    vae = VAE(input_shape = (28, 28, 1),
                              conv_filters = (32, 64, 64, 64),
                              conv_kernels = (3, 3, 3, 3),
                              conv_strides = (1, 2, 2, 1), 
                              latent_space_dim = latent_dim,
                              kl_weight = kl_weight,
                              num_channel = 1)
    #vae.summary()
    vae.compile_model(learning_rate=learning_rate)
    vae.train(x_train, batch_size, epochs)
    return vae
    


if __name__ == '__main__':
    x_train, _, _, _ = load_mnist()
    
    # Train the model for different values of the kl_weight
    for kl in KL_WEIGHTS:
        vae = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS, latent_dim=2,  kl_weight=kl)
        vae.save("model/KL_impact/kl_weight=" + str(kl))
    
    # Train the model for different number of epochs
    for n_epochs in N_EPOCHS:
        vae = train(x_train, LEARNING_RATE, BATCH_SIZE, n_epochs, latent_dim=2, kl_weight=0.001)
        vae.save("model/N_epochs_impact/n_epochs=" + str(n_epochs))
        
    # Train the model for different values of the latent space dimension
    for latent_dim in LATENT_SPACE_DIM:
        vae = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS, latent_dim=latent_dim, kl_weight=0.001)
        vae.save("model/Latent_space_dim_impact/")
        
    
    

    

