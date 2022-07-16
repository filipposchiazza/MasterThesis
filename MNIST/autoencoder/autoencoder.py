import os
import pickle
import numpy as np
import tensorflow.keras as k


class Autoencoder:
    
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):

        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self._num_conv_layers = len(conv_filters)
        self._model_input = None
        self._shape_before_bottleneck = None
        
        self._build()

    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
        
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
        x = encoder_input
        for i in range(self._num_conv_layers):
            x = self._add_conv_layer(i, x)
        return x
    
    
    def _add_conv_layer(self, layer_index, x):
        "Add a convolutional layer consisting of: Conv2D + ReLU + batch normalization"
        
        layer_number = layer_index + 1 # I do not want to have layer 0
        
        conv_layer = k.layers.Conv2D(
            filters = self.conv_filters[layer_index], 
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = 'same',
            activation=k.activations.relu,
            name = "encoder_conv_layer_number_{}".format(layer_number))
        
        x = conv_layer(x)
        x = k.layers.BatchNormalization(name = "encoder_batch_norm_layer_number_{}".format(layer_number))(x)
        
        return x
    
    
    def _add_bottleneck(self, x):
        "Flatten data and add a bottleneck (Dense layer)"
        self._shape_before_bottleneck = k.backend.int_shape(x)[1:] #store the shape before flatten, because it will be useful for building decoder
        x = k.layers.Flatten()(x)
        x = k.layers.Dense(self.latent_space_dim, name = "encoder_output")(x)
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
            filters = 1, # 1 for gray-scale image, 3 for RGB image 
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = 'same',
            activation = k.activations.sigmoid,
            name = "decoder_conv_transpose_layer_number_{}".format(self._num_conv_layers))
        
        output = conv_transpose_layer(x)
        return output
   
#########################################################################################################################    
    #Build the autoencoder

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = k.Model(model_input, model_output, name="Autoencoder")
      
#########################################################################################################################
    #Compile the autoencoder
    
    def compile_model(self, learning_rate = 0.001):
        optimizer = k.optimizers.Adam(learning_rate = learning_rate)
        loss_function = k.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss = loss_function)
        
#########################################################################################################################
    #Train the autoencoder
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, 
                       x_train,
                       batch_size = batch_size,
                       epochs = num_epochs,
                       shuffle = True)
        
        
#########################################################################################################################
    #Reconstruction method
    
    def reconstruct(self, images):
        latent_representation = self.encoder.predict(images)
        reconstructed_image = self.decoder.predict(latent_representation)
        return reconstructed_image, latent_representation

########################################################################################################################

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
    
    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
            ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
            
    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)
        
    
    
##########################################################################################################################
    #Loading method
    
    @classmethod
    def load (cls, save_folder = '.'):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder._load_weights(weights_path)
        return autoencoder
        
    def _load_weights (self, weights_path):
        self.model.load_weights(weights_path)
        
    
        
    

        