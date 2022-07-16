import numpy as np
from vae import VAE
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split



LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 50

KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]

LATENT_SPACE_DIM = [3, 5, 8, 10, 15, 20, 30]


def load_mnist(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # concatenate together all data and targets
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    # normalization
    x = x.astype("float32") / 255
    # add channel dimension (1 for gray scale images)
    x = x.reshape(x.shape + (1,))
    # obtain train dataset
    x_train, x_remain, y_train, y_remain = train_test_split(x, y, train_size=train_ratio)
    # divide validation and test dataset
    x_val, x_test, y_val, y_test = train_test_split(x_remain, y_remain, train_size=(validation_ratio/(validation_ratio+test_ratio)))
    return x_train, y_train, x_val, y_val, x_test, y_test    




def train(x_train, x_validation, learning_rate, batch_size, epochs, latent_dim, kl_weight):
    
    vae = VAE(input_shape = (28, 28, 1),
                              conv_filters = (32, 64, 64, 64),
                              conv_kernels = (3, 3, 3, 3),
                              conv_strides = (1, 2, 2, 1), 
                              latent_space_dim = latent_dim,
                              kl_weight = kl_weight,
                              num_channel = 1)
    #vae.summary()
    vae.compile_model(learning_rate=learning_rate)
    vae.train(x_train = x_train, 
              x_validation = x_validation, 
              batch_size = batch_size,
              num_epochs = epochs)
    return vae
    


if __name__ == '__main__':
    x_train, _, x_val, _, _, _ = load_mnist(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
    
    # Train the model for different values of the kl_weight
    for kl in KL_WEIGHTS:
        
        vae = train(x_train = x_train, 
                    x_validation = x_val,
                    learning_rate = LEARNING_RATE, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    latent_dim = 2,  
                    kl_weight = kl)
                    
        vae.save("model/KL_impact/kl_weight=" + str(kl))
        
    # Train the model with a KL weight of 0.005
    vae = train(x_train = x_train, 
                x_validation = x_val,
                learning_rate = LEARNING_RATE, 
                batch_size = BATCH_SIZE, 
                epochs = 30, 
                latent_dim = 2,  
                kl_weight = 0.005)
    vae.save("model/KL_impact/kl_weight=0.005")
                    
        
    # Choose two best values of KL weight: 0.0001 and 0.001 
    # Study the effect of the latent space dimension
    
    # Train the model for different values of the latent space dimension (0.0001)
    for latent_dim in LATENT_SPACE_DIM:
        
        vae = train(x_train = x_train,
                    x_validation = x_val,
                    learning_rate = LEARNING_RATE,
                    batch_size = BATCH_SIZE, 
                    epochs = 35,
                    latent_dim = latent_dim,
                    kl_weight = 0.0001)
        
        vae.save("model/Latent_space_dim_impact/latent_space_dim=" + str(latent_dim))
        
    # Train the model for different values of the latent space dimension (0.001)   
    for latent_dim in LATENT_SPACE_DIM:
        
        vae = train(x_train = x_train,
                    x_validation = x_val,
                    learning_rate = LEARNING_RATE,
                    batch_size = BATCH_SIZE, 
                    epochs = 35,
                    latent_dim = latent_dim,
                    kl_weight = 0.001)
        
        vae.save("model/Latent_space_dim_impact/latent_space_dim=" + str(latent_dim))
        
    

    

