import numpy as np
from vae import VAE
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 35

KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]

LATENT_SPACE_DIM = [3, 5, 8, 10, 15, 20, 30] # previous value: 2

LAYERS_PARAM = [{"filters":(16, 32, 32, 64, 128), "kernels":(3, 3, 3, 3, 3), "strides":(1, 1, 2, 2, 1)},
               {"filters":(16, 32, 32, 64, 64, 128), "kernels":(3, 3, 3, 3, 3, 3), "strides":(1, 1, 1, 2, 2, 1)}]

LEARNING_RATES = [0.005, 0.00005]

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
    x_train, x_remain, y_train, y_remain = train_test_split(x, y, train_size=train_ratio, random_state=1)
    # divide validation and test dataset
    x_val, x_test, y_val, y_test = train_test_split(x_remain, y_remain, random_state=1, train_size=(validation_ratio/(validation_ratio+test_ratio)))
    return x_train, y_train, x_val, y_val, x_test, y_test    




def train(x_train, x_validation, learning_rate, batch_size, epochs, latent_dim, kl_weight,
          conv_filters = (32, 64, 64, 64),
          conv_kernels = (3, 3, 3, 3),
          conv_strides = (1, 2, 2, 1)):
    
    vae = VAE(input_shape = (28, 28, 1),
                              conv_filters = conv_filters,
                              conv_kernels = conv_kernels,
                              conv_strides = conv_strides, 
                              latent_space_dim = latent_dim,
                              num_channel = 1)
    #vae.summary()
    vae.compile_model(kl_weight=kl_weight, learning_rate=learning_rate)
    vae.train(x_train = x_train, 
              x_validation = x_validation, 
              batch_size = batch_size,
              num_epochs = epochs)
    return vae


def cyclical_annealing_train(x_train,
                             x_validation,
                             learning_rate,
                             batch_size,
                             epochs,
                             latent_dim,
                             num_cycles,
                             R = 0.5,
                             conv_filters = (32, 64, 64, 64),
                             conv_kernels = (3, 3, 3, 3),
                             conv_strides = (1, 2, 2, 1),
                             reco_weight = 1.0):
    
    vae = VAE(input_shape = (28, 28, 1),
                              conv_filters = conv_filters,
                              conv_kernels = conv_kernels,
                              conv_strides = conv_strides, 
                              latent_space_dim = latent_dim,
                              num_channel = 1)
    
    vae.cyclical_annealing(x_train = x_train,
                           x_validation = x_validation,
                           batch_size = batch_size,
                           num_epochs = epochs,
                           learning_rate = learning_rate,
                           num_cycles = num_cycles,
                           R = R,
                           reco_weight = reco_weight,
                           update_train = True,
                           weights_path = 'model/cyclical_annealing_schedule/weights.h5')  
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
        
    
    # In terms of latent space dimension, it emerges in both cases that
    # 15 is the best choise for the FID result
    # Fix it and move to study the deepness of the network
    
    # use 0.0001 as KL_weight (better in terms of FID index)
    for layers_param in LAYERS_PARAM:
        conv_filters = layers_param["filters"]
        conv_kernels = layers_param["kernels"]
        conv_strides = layers_param["strides"]
        
        vae = train(x_train = x_train, 
                    x_validation = x_val, 
                    learning_rate = LEARNING_RATE, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    latent_dim = 15, 
                    kl_weight = 0.0001,
                    conv_filters = conv_filters,
                    conv_kernels = conv_kernels,
                    conv_strides = conv_strides)
        
        vae.save("model/deepness_impact/filters=" + str(conv_filters))
        
    # use 0.001 as KL_weight (better in terms of visual appearence) 
    for layers_param in LAYERS_PARAM:
        conv_filters = layers_param["filters"]
        conv_kernels = layers_param["kernels"]
        conv_strides = layers_param["strides"]
        
        vae = train(x_train = x_train, 
                    x_validation = x_val, 
                    learning_rate = LEARNING_RATE, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    latent_dim = 15, 
                    kl_weight = 0.001,
                    conv_filters = conv_filters,
                    conv_kernels = conv_kernels,
                    conv_strides = conv_strides)
        
        vae.save("model/deepness_impact/KL=0.001_DIM=15/filters=" + str(conv_filters))
        
    
    # investigate the impact of the learning rate
    for learning_rate in LEARNING_RATES:
        vae = train(x_train = x_train, 
                    x_validation = x_val, 
                    learning_rate = learning_rate, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    latent_dim = 2, 
                    kl_weight = 0.001)
        
        vae.save("model/learning_rate_impact/learning_rate=" + str(learning_rate))
        
   
        

    # Perform Cyclical Annealing schedule
    vae = cyclical_annealing_train(x_train = x_train, 
                                   x_validation = x_val, 
                                   learning_rate = LEARNING_RATE, 
                                   batch_size = 25, 
                                   epochs = 20, 
                                   latent_dim = 2, 
                                   num_cycles = 2,
                                   reco_weight=1000,
                                   R = 0.5)
                                   
    vae.save("model/cyclical_annealing_schedule")
                             
    
    

