import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback


#tf.config.run_functions_eagerly(True)

class Callback_prova(Callback):
    
    def __init__(self, step):
        self.step = step
    
    def on_epoch_end(self, epoch, logs={}):
        keras.backend.set_value(self.model.kl_weight, value=self.step)




class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class VAE(keras.Model):
    def __init__(self, input_shape, conv_kernels, conv_filters, conv_strides, latent_dim, kl_weight, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.ishape= input_shape
        self.conv_kernels = conv_kernels
        self.conv_filters = conv_filters
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.num_channels = input_shape[-1]
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        #self.kl_weight = tf.Variable(kl_weight, trainable=False)
        self.kl_weight = keras.backend.variable(kl_weight)
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_absolute_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    
    def test_step(self, input_data):
        validation_data, _ = input_data # <-- Seperate X and y
        z_mean, z_log_var, z = self.encoder(validation_data)
        val_reconstruction = self.decoder(z)
        val_reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_absolute_error(validation_data, val_reconstruction), axis=(1, 2)
                )
            )
        val_kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        val_kl_loss = tf.reduce_mean(tf.reduce_sum(val_kl_loss, axis=1))
        val_total_loss = val_reconstruction_loss + val_kl_loss
        return {"total_loss": val_total_loss} # <-- modify the return value here

    
    
    
    
    def _build_encoder(self):
        encoder_inputs = keras.Input(shape=self.ishape, name='encoder_input')
        
        x = encoder_inputs
        for i, (filt, kernel, strides) in enumerate(zip(self.conv_filters, self.conv_kernels, self.conv_strides)):
            x = layers.Conv2D(filters=filt,
                              kernel_size=kernel,
                              strides=strides,
                              padding='same',
                              activation='relu',
                              name=f'encoder_conv_layer_{i}')(x)
        
        self.shape_before_flatten = K.int_shape(x)[1:]
        x = layers.Flatten(name='encoder_flatten')(x)
        
        z_mean = layers.Dense(units=self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(units=self.latent_dim, name="z_log_var")(x)
        
        z = Sampling()([z_mean, z_log_var])
        
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder
    
    
    def _build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        
        num_neurons = np.prod(self.shape_before_flatten)
        x = layers.Dense(units=num_neurons, name='decoder_dense_layer')(latent_inputs)
        
        x = layers.Reshape(target_shape=(self.shape_before_flatten), name='decoder_reshaper_layer')(x)
        
        for i in reversed(range(len(self.conv_kernels))):
            x = layers.Conv2DTranspose(filters=self.conv_filters[i],
                                       kernel_size=self.conv_kernels[i],
                                       strides=self.conv_strides,
                                       padding='same',
                                       activation='relu',
                                       name=f'decoder_conv_trans_{i}')(x)
            
        decoder_outputs = layers.Conv2DTranspose(filters=self.num_channels,
                                                 kernel_size=self.conv_kernels[-1],
                                                 padding='same',
                                                 activation='sigmoid',
                                                 name='decoder_output')(x)
        
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
x_val = np.expand_dims(x_test, -1).astype("float32") / 255

mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

#cb = Callback_prova(step=10)

vae = VAE(input_shape=(28, 28, 1), 
          conv_kernels=[3, 3], 
          conv_filters=[32, 64], 
          conv_strides=[2, 2], 
          latent_dim=2, 
          kl_weight = 1.0)

vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, 
        epochs=30, 
        batch_size=128,
        validation_data=(x_val, x_val))

###############################################################################
