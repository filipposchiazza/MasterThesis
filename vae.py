import os
import pickle
import numpy as np
import tensorflow.keras as k
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping

tf.compat.v1.disable_eager_execution()
#######################################################################################################################

class CyclicalAnnealingCallback(Callback):
    
    def __init__(self, vae, num_epochs, num_cycles, R, start_weight, learning_rate=0.001):
        self.num_epochs = num_epochs
        self.num_cycles = num_cycles
        self.R = R
        self.weight = start_weight
        self.vae = vae
        self.learning_rate = learning_rate
    
    def on_epoch_end(self, epoch, logs={}):
        ratio = self.num_epochs / self.num_cycles
        tau = ((epoch) % ratio) / ratio
        new_weight = np.minimum(tau/self.R, 1)
        k.backend.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(k.backend.get_value(self.weight)))
        print ("Current epoch is " + str(epoch))
        optimizer = k.optimizers.Adam(learning_rate = self.learning_rate)
        self.vae.model.compile(optimizer=optimizer,
                           loss = self.vae.combined_loss,
                           metrics=[self.vae.recostruction_loss, self.vae.kl_loss])
          


#######################################################################################################################
class VAE:
    
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim, num_channel):
        """ Create VAE object.
        

        Parameters
        ----------
        input_shape : list
            Input dimension.
        conv_filters : list or tuple 
            Number of convolutional filters for each layer.
        conv_kernels : list or tuple
            Dimension of the convolutional kernels for each layer.
        conv_strides : list or tuple
            Strides for each layer.
        latent_space_dim : int
            Dimension of the latent space (embedding).
        kl_weight : float
            Weight of KL in the total loss function.
        num_channel : int
            1 for Gray-scale images and 3 for RGB images.

        Returns
        -------
        None.

        """
        self.input_shape = input_shape    # [28, 28, 1] image size
        self.conv_filters = conv_filters  # [3, 4, 6] 3 conv filters in first layer, 4 in the second and so on
        self.conv_kernels = conv_kernels  # [3, 5, 3] 3x3 kernel size in the first layer and so on
        self.conv_strides = conv_strides  # [1, 1, 1]
        self.latent_space_dim = latent_space_dim
        self.num_channel = num_channel
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self._num_conv_layers = len(conv_filters)
        self._model_input = None
        self._shape_before_bottleneck = None
        
        self._build() # when an object VAE is created, this method is called

    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_vae()
        
###########################################################################################################################
    #Build the encoder    
    
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        self._model_input = encoder_input
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = k.Model(encoder_input, bottleneck, name = "encoder")
        
        
    def _add_encoder_input(self):
        return k.layers.Input(shape=self.input_shape, name="encoder_input")
    
    
    def _add_conv_layers(self, encoder_input):
        "Create all convolutional blocks in the encoder"
        x = encoder_input
        for i in range(self._num_conv_layers):
            x = self._add_conv_layer(i, x)
        return x
    
    
    def _add_conv_layer(self, layer_index, x):
        "Add a convolutional layer consisting of: Conv2D + ReLU + batch normalization"
        
        layer_number = layer_index + 1 # I do not want to have layer 0
        
        conv_layer = k.layers.Conv2D(
            filters = self.conv_filters[layer_index],  # number of filters in this convolutional layer
            kernel_size = self.conv_kernels[layer_index],  # size of kernels in this convolutional layer
            strides = self.conv_strides[layer_index],  # stride for this convolutional layer
            padding = 'same',
            activation=k.activations.relu,
            name = "encoder_conv_layer_number_{}".format(layer_number))
        
        x = conv_layer(x)
        x = k.layers.BatchNormalization(name = "encoder_batch_norm_layer_number_{}".format(layer_number))(x)
        
        return x
    
    
    def _add_bottleneck(self, x):
        "Flatten data and add a bottleneck with Gaussian sampling (Dense layer)"
        self._shape_before_bottleneck = k.backend.int_shape(x)[1:] #store the shape before flatten, because it will be useful for building decoder
                                                                   #(first element is batch size, we are not interested in it)
        x = k.layers.Flatten()(x)
        self.mu = k.layers.Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = k.layers.Dense(self.latent_space_dim, name="log_variance")(x)
        
        def sample_point_from_normal_distribution(args):
            """
            In order to sample a point from a generic normal distribution with mean Mu and standard deviation Sigma,
            it is possible to sample a point epsilon from a standard normal distribution and than apply the trasformation
            Z = Mu + Sigma * epsilon
            """
            mu, log_variance = args
            epsilon = k.backend.random_normal(shape=k.backend.shape(self.mu), mean=0., stddev=1.)
            sampled_point = mu + k.backend.exp(log_variance / 2) * epsilon
            return sampled_point
        
        x = k.layers.Lambda(sample_point_from_normal_distribution,
                            name="encoder_output")([self.mu, self.log_variance])
        
        return x
    
##########################################################################################################################   
    #Build the decoder    
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshaped_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshaped_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = k.Model(decoder_input, decoder_output, name = "decoder")
      
        
    def _add_decoder_input(self):
        return k.layers.Input(shape=self.latent_space_dim, name="decoder_input")
    
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = k.layers.Dense(num_neurons, name="decoder_dense_layer")(decoder_input)
        return dense_layer 
    
    
    def _add_reshape_layer(self, dense_layer):
        return k.layers.Reshape(self._shape_before_bottleneck)(dense_layer)
    
    
    def _add_conv_transpose_layers(self, x):
        "loop through all the conv layers in reverse order and stop at the first layer"
        for i in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(i, x)
        return x
    
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index
        
        conv_transpose_layer = k.layers.Conv2DTranspose(
            filters = self.conv_filters[layer_index], 
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = 'same',
            activation = k.activations.relu, 
            name = "decoder_conv_transpose_layer_number_{}".format(layer_number))
        
        x = conv_transpose_layer(x)
        x = k.layers.BatchNormalization(name = "decoder_batch_norm_layer_number_{}".format(layer_number))(x)
        
        return x
    
    def _add_decoder_output(self, x):
        
        conv_transpose_layer = k.layers.Conv2DTranspose(
            filters = self.num_channel, # 1 for gray-scale image, 3 for RGB image 
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = 'same',
            activation = k.activations.sigmoid,
            name = "decoder_conv_transpose_layer_number_{}".format(self._num_conv_layers))
        
        output = conv_transpose_layer(x)
        return output
   
#########################################################################################################################    
    #Build the variational autoencoder

    def _build_vae(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = k.Model(model_input, model_output, name="Variational Autoencoder")
      
#########################################################################################################################
    #Define the recostruction loss function combined with the kl distance
    
    def reco_loss(self, reco_weight):
        def reconstruction_loss(y_target, y_prediction):
            error = y_target - y_prediction
            reconstruction_loss = k.backend.mean(k.backend.square(error), axis = [1, 2, 3])
            return reco_weight * reconstruction_loss
        return reconstruction_loss
    
    def kl_loss(self, y_target, y_prediction):
        "Difference from a standard multivariate normal distribution"
        kl_loss = -0.5 * k.backend.sum(1 + self.log_variance - k.backend.square(self.mu) - k.backend.exp(self.log_variance), axis=1)
        return kl_loss

    
    def combined_loss(self, kl_weight, reco_weight):
        "Wrap the loss as a function of weight"
        def loss(y_target, y_prediction):
            reconstruction_loss = self.reco_loss(reco_weight)(y_target, y_prediction)
            kl_loss = self.kl_loss(y_target, y_prediction)
            combined_loss = reconstruction_loss + kl_weight * kl_loss
            return combined_loss 
        return loss
            

#########################################################################################################################
    #Compile the variational autoencoder
        
    def compile_model(self, kl_weight, reco_weight=1.0, learning_rate=0.001):
        self.kl_weight = kl_weight
        optimizer = k.optimizers.Adam(learning_rate = learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss = self.combined_loss(kl_weight, reco_weight=reco_weight),
                           metrics=[self.reco_loss(reco_weight), self.kl_loss])   

        
#########################################################################################################################
    #Train the variational autoencoder
       
    def train(self, 
              x_train, 
              x_validation, 
              batch_size, 
              num_epochs):
        
        """ Training function.
        

        Parameters
        ----------
        x_train : numpy array
            Training dataset.
        x_validation : numpy array
            Validation dataset: the model does not use this dataset for training.
        batch_size : int
            Batch size.
        num_epochs : int
            Number of epochs.


        Returns
        -------
        None.

        """
        

        # create an early stopping criteria
        early_stopping = EarlyStopping(monitor = 'val_loss',
                                           patience = 5,
                                           verbose = 1,
                                           mode = 'min',
                                           restore_best_weights = True)
            
            
        self.model.fit(x_train, 
                       x_train,
                       batch_size = batch_size,
                       epochs = num_epochs,
                       shuffle = True,
                       validation_data=(x_validation, x_validation),
                       callbacks = [early_stopping])
        
        
##################################################################################        
     
    # Perform Cyclical annealing     
    def _update_kl_weight(self, current_epoch, num_epochs, num_cycles, R):
        ratio = num_epochs / num_cycles
        tau = ((current_epoch) % ratio) / ratio
        new_weight = np.minimum(tau/R, 1)
        return new_weight
        
    
    def cyclical_annealing(self, 
                           x_train, 
                           x_validation, 
                           batch_size, 
                           num_epochs,
                           learning_rate,
                           num_cycles,
                           R,
                           start_kl_weight = 1e-4,
                           reco_weight = 1.0,
                           update_train = False,
                           weights_path = ''):
        # Set initial value of kl_weight
        self.kl_weight = start_kl_weight
        
        # Define an optimizer
        optimizer = k.optimizers.Adam(learning_rate = learning_rate)
        
        if update_train == True:
            self.model.load_weights(weights_path)
            
        
        # Train the first epoch
        print("Start of epoch 1/" + str(num_epochs))
        print("Current kl_weight: " + str(self.kl_weight))
        self.model.compile(optimizer=optimizer,
                           loss = self.combined_loss(self.kl_weight, reco_weight),
                           metrics=[self.reco_loss(reco_weight), self.kl_loss])
        self.model.fit(x_train, 
                       x_train,
                       batch_size = batch_size,
                       epochs = 1,
                       shuffle = True,
                       validation_data=(x_validation, x_validation))
        
        # Store the weights in a variable
        weights = self.model.get_weights()
        
        # Save history for the first epoch
        self.history = self.model.history.history
                
        
        # Perform the cicles 
        for epoch in range(1, num_epochs):
            
            # Update the kl_weight
            self.kl_weight = self._update_kl_weight(current_epoch = epoch,
                                                    num_epochs = num_epochs,
                                                    num_cycles = num_cycles,
                                                    R = R)
            # Re-compile the model
            self.model.compile(optimizer=optimizer,
                               loss = self.combined_loss(self.kl_weight, reco_weight),
                               metrics=[self.reco_loss(reco_weight), self.kl_loss])
            
            # Set the weights of the network
            self.model.set_weights(weights)
            
            # Train
            print("Start of epoch " + str(epoch+1) + "/" + str(num_epochs))
            print("Current kl_weight: " + str(self.kl_weight))
            self.model.fit(x_train, 
                           x_train,
                           batch_size = batch_size,
                           epochs = 1,
                           shuffle = True,
                           validation_data=(x_validation, x_validation))
            
            # Save weights
            weights = self.model.get_weights()
            
            # Update history 
            for key in self.history.keys():
                self.history[key].append(self.model.history.history[key][0])
            

                
                


#########################################################################################################################
    #Reconstruction method
    
    def reconstruct(self, images):
        latent_representation = self.encoder.predict(images)
        reconstructed_image = self.decoder.predict(latent_representation)
        return reconstructed_image, latent_representation

#########################################################################################################################
    #Generating method
    
    def generate(self, n_sample_to_generate):
        data = np.random.normal(0, 1, (n_sample_to_generate, self.latent_space_dim))
        generated_samples = self.decoder.predict(data)
        return generated_samples
    
#########################################################################################################################
    #Saving method
    def save(self, save_folder = '.'):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
        self._save_history(save_folder)
    
    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim,
            self.kl_weight,
            self.num_channel
            ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
        f.close()
            
    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)
    
    def _save_history(self, save_folder):
        history = self.model.history.history
        save_path = os.path.join(save_folder, "history.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(history, f)
        f.close()
        
    
    
##########################################################################################################################
    #Loading method
    
    @classmethod
    def load (cls, save_folder = '.'):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        # remove the kl weight from parameters (index = 5)
        if len(parameters) == 7:
            parameters.pop(5)
        vae = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        vae._load_weights(weights_path)
        return vae
        
    def _load_weights (self, weights_path):
        self.model.load_weights(weights_path)
    
    @classmethod
    def load_history(cls, save_folder = '.'):
        history_path = os.path.join(save_folder, "history.pkl")
        with open(history_path, "rb") as f:
            history = pickle.load(f)
        return history
        
        
        
        
"""        
if __name__ == '__main__':
    vae = VAE(input_shape=(28, 28, 1),
              conv_filters=(32, 64, 64, 64), 
              conv_kernels=(3, 3, 3, 3),
              conv_strides=(1, 2, 2, 1), 
              latent_space_dim=2,
              kl_weight=0.001,
              num_channel=1)
    vae.summary()
"""