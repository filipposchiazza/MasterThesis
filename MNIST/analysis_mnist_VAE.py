import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from vae import VAE
from train_VAE_mnist import load_mnist



def select_images(images, labels, num_images=8):
    "Select randomly a number of images (and corresponding labels) from images (and labels)." 
    index = np.random.choice(range(len(images)), num_images)
    sample_images = images[index]
    sample_labels = labels[index]
    return sample_images, sample_labels



def plot_reconstructed_images(images, reconstructed_images, kl_weight, save=False):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()   # remove the extra channel, that is the 1 in the size tuple
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    #fig.suptitle("Reconstruction: KL weight = " + str(kl_weight), fontsize=20)
    if save == True:
        filename = 'Images_to_show/recostruction_KL_weight=' + str(kl_weight) + '.png'
        plt.savefig(filename)
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels, kl_weight, save=False):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    #plt.title("KL weight = " + str(kl_weight), fontsize=20)
    if save == True:
        filename = 'Images_to_show/latent_space_KL_weight=' + str(kl_weight) + '.png'
        plt.savefig(filename)
    plt.show()
    
    
def generation(vae, num_images_to_generate, kl_weight, save=False):
    generated_images = vae.generate(num_images_to_generate)
    fig = plt.figure(figsize=(15, 3))
    for i, image in enumerate(generated_images):
        image = image.squeeze()
        ax = fig.add_subplot(1, num_images_to_generate, i+1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
    #plt.suptitle("Generation: KL weight = " + str(kl_weight), fontsize=20)
    if save == True:
        filename = 'Images_to_show/generation_KL_weight=' + str(kl_weight) + '.png'
        fig.savefig(filename)
    plt.show()
    
def plot_epochs_history(save_folder):
    # load the dictionary history
    history = VAE.load_history(save_folder)
    
    # obtain the number of epochs to generate the x-axis
    num_epochs = len(history["loss"])
    x = np.arange(1, num_epochs+1)
    
    # plot train vs validation loss
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x, history["loss"], color="black", label="Training loss")
    ax1.plot(x, history["val_loss"], color="green", label="Validation loss")
    
    plt.legend()
    plt.title("Training vs Validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.show()
    
    # plot train vs validation reconstruction loss
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, history["reconstruction_loss"], color="black", label="Training reconstruction loss")
    ax2.plot(x, history["val_reconstruction_loss"], color="green", label="Validation reconstruction loss")
    
    plt.legend()
    plt.title("Training vs Validation reconstruction loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Reconstruction loss value")
    plt.show()
    
    # plot train vs validation Kullback-Leibner loss
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(x, history["kl_loss"], color="black", label="Training KL loss")
    ax3.plot(x, history["val_kl_loss"], color="green", label="Validation KL loss")
    
    plt.legend()
    plt.title("Training vs Validation Kullback-Leibner loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("KL loss value")
    plt.show()
    
    # plot combined, reconstruction and KL loss for training dataset
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(x, history["loss"], color="black", label="Total combined loss")
    ax4.plot(x, history["reconstruction_loss"], color="green", label="Reconstruction loss")
    ax4.plot(x, 0.001*np.asarray(history["kl_loss"]), color="red", label="KL loss")
    
    plt.legend()
    plt.title("Training loss and its components")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.show()    
    
    # plot combined, reconstruction and KL loss for validation dataset
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(x, history["val_loss"], color="black", label="Total combined loss")
    ax5.plot(x, history["val_reconstruction_loss"], color="green", label="Reconstruction loss")
    ax5.plot(x, 0.001*np.asarray(history["val_kl_loss"]), color="red", label="KL loss")
    
    plt.legend()
    plt.title("Validation loss and its components")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.show()  
    
        
def plot_images_grid(images_set, grid_size, title='', xlabel='', ylabel=''):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111, nrows_ncols=grid_size, axes_pad=0.1)

    for ax, im in zip(grid, images_set):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()
    
    
def plot_latent_space(vae, n=30, figsize=15, title=''):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("$z_0$")
    plt.ylabel("$z_1$")
    plt.title(title)
    plt.imshow(figure, cmap="gray_r")
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    
    #######################################################################################
    # Analysis KL weight variation on the latent space, on the recostruction and on the generation
    KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]
    
    for kl in KL_WEIGHTS:
        vae = VAE.load("model/KL_impact/kl_weight=" + str(kl))
        
        plot_latent_space(vae, title="Generation with KL weight=" + str(kl))
        
        #plot original, reconstructed and generated images
        num_sample_images_to_show = 25
        sample_images, _ = select_images(x_val, y_val, num_sample_images_to_show)
        reconstructed_images, _ = vae.reconstruct(sample_images)
        generated_images = vae.generate(25)
        
        plot_images_grid(sample_images, (5, 5))
        plot_images_grid(reconstructed_images, (5, 5))
        plot_images_grid(generated_images, (5, 5))
    
        #plot the latent space representation
        _, latent_representations = vae.reconstruct(x_val)
        plot_images_encoded_in_latent_space(latent_representations, y_val, kl, save=False)
    
        
        # plot epochs history
        plot_epochs_history("model/KL_impact/kl_weight=" + str(kl))
        
        
    # take the two "best" values for the KL weight and fix them.
    # Now analyze the effect of the variation of the latent space dimension
    LATENT_SPACE_DIM = [2, 3, 5, 8, 10, 15, 20, 30]
    for dim in LATENT_SPACE_DIM:
        vae = VAE.load("model/KL_impact/kl_weight=0.001/Latent_space_dim_impact/latent_space_dim=" + str(dim))
        
        #plot some reconstructed images vs original ones
        num_sample_images_to_show = 8
        sample_images, _ = select_images(x_val, y_val, num_sample_images_to_show)
        reconstructed_images, _ = vae.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images, 0.001, save=False)
    
        # generate images
        generation(vae, 8, 0.001, save=False)
        
        # plot epochs history
        #plot_epochs_history("model/KL_impact/kl_weight=0.0001/Latent_space_dim_impact/latent_space_dim=" + str(dim))
        
        
    # Consider the best combination of values of KL_weight and latent space dim
    # Focus on the deepness of the CNN
    
    # KL weight = 0.0001
    FILTERS = [(16, 32, 32, 64, 128), (16, 32, 32, 64, 64, 128)]
    for filt in FILTERS:
        vae = VAE.load("model/deepness_impact/KL=0.0001_DIM=15/filters=" + str(filt))
        
        #plot some reconstructed images vs original ones
        num_sample_images_to_show = 8
        sample_images, _ = select_images(x_val, y_val, num_sample_images_to_show)
        reconstructed_images, _ = vae.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images, 0.0001, save=False)
    
        # generate images
        generation(vae, 8, 0.0001, save=False)
        
        # plot epochs history
        plot_epochs_history("model/deepness_impact/KL=0.0001_DIM=15/filters=" + str(filt))
        
    # KL weight = 0.001
    for filt in FILTERS:
        vae = VAE.load("model/deepness_impact/KL=0.001_DIM=15/filters=" + str(filt))
        
        #plot some reconstructed images vs original ones
        num_sample_images_to_show = 8
        sample_images, _ = select_images(x_val, y_val, num_sample_images_to_show)
        reconstructed_images, _ = vae.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images, 0.001, save=False)
    
        # generate images
        generation(vae, 8, 0.001, save=False)
        
        # plot epochs history
        plot_epochs_history("model/deepness_impact/KL=0.001_DIM=15/filters=" + str(filt))
    
    
    
    
    # analysis effects of learning rate
    LEARNING_RATES = [0.005, 0.00005]
    for learning_rate in LEARNING_RATES:
        vae = VAE.load("model/learning_rate_impact/learning_rate=" + str(learning_rate))
        
        #plot some reconstructed images vs original ones
        num_sample_images_to_show = 25
        sample_images, _ = select_images(x_val, y_val, num_sample_images_to_show)
        reconstructed_images, _ = vae.reconstruct(sample_images)
        #plot_reconstructed_images(sample_images, reconstructed_images, 0.001, save=False)
    
        #plot the latent space representation
        #_, latent_representations = vae.reconstruct(x_val)
        #plot_images_encoded_in_latent_space(latent_representations, y_val, 0.0001, save=False)
    
        # generate images
        generated_images = vae.generate(25)
        
        # plot epochs history
        plot_epochs_history("model/learning_rate_impact/learning_rate=" + str(learning_rate))
        
        plot_images_grid(sample_images, (5, 5))
        plot_images_grid(reconstructed_images, (5, 5))
        plot_images_grid(generated_images, (5, 5))  
        
    
    # Cyclical annealing
    vae = VAE.load("model/cyclical_annealing_schedule")
    num_sample_images_to_show = 25
    sample_images, _ = select_images(x_val, y_val, num_sample_images_to_show)
    reconstructed_images, _ = vae.reconstruct(sample_images)
    generated_images = vae.generate(25)
    
    plot_images_grid(sample_images, (5, 5))
    plot_images_grid(reconstructed_images, (5, 5))
    plot_images_grid(generated_images, (5, 5)) 
    
    _, latent_representations = vae.reconstruct(x_val)
    plot_images_encoded_in_latent_space(latent_representations, y_val, 0.0001, save=False)
    
    plot_epochs_history("model/cyclical_annealing_schedule")
    
        
    
    
    
    
    
    