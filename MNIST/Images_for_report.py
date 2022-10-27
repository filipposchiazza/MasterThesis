import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import ImageGrid
from vae import VAE
from train_VAE_mnist import load_mnist
import pickle




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
    fig.suptitle("Reconstruction process with $KL\:weight = " + str(kl_weight) + '$', fontsize=20)
    if save == True:
        filename = 'Images_to_show/recostruction_KL_weight=' + str(kl_weight) + '.png'
        plt.savefig(filename)
    plt.show()
    
    
def plot_images_encoded_in_latent_space(latent_representations, sample_labels, kl_weight=None, save=False):

    cmap = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)

    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=sample_labels, cmap=cmap, norm=norm, s=2, edgecolor='none')
    plt.colorbar(ticks=np.linspace(0, 9, 10))

    #plt.title("KL weight = " + str(kl_weight), fontsize=20)
    if save == True:
        filename = 'Images_to_show/latent_space_KL_weight=' + str(kl_weight) + '.png'
        plt.savefig(filename)    
        
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    if kl_weight != None:
        plt.title('MNIST latent space with $KL\:weight=' + str(kl_weight) + '$')
    else:
        plt.title('MNIST latent space for Cyclical Annealing of KL weight')
    plt.show()
    
    
def generation(vae, num_images_to_generate, kl_weight, save=False):
    generated_images = vae.generate(num_images_to_generate)
    fig = plt.figure(figsize=(15, 3))
    for i, image in enumerate(generated_images):
        image = image.squeeze()
        ax = fig.add_subplot(1, num_images_to_generate, i+1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
    plt.suptitle("Generation process with $KL\:weight = " + str(kl_weight) + '$', fontsize=20)
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
    ax4.plot(x, np.asarray(history["kl_loss"]), color="red", label="KL loss")
    
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
    ax5.plot(x, np.asarray(history["val_kl_loss"]), color="red", label="KL loss")
    
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
        ax.imshow(im, cmap="gray_r")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    fig.suptitle(title)
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
    sample_range_x = np.round(grid_x, 2)
    sample_range_y = np.round(grid_y, 2)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("$z_0$")
    plt.ylabel("$z_1$")
    plt.title(title)
    plt.imshow(figure, cmap="gray_r")
    plt.show()
    
    

def plot_FID_scores(vector_parameters,
                    fid_scores,
                    x_label = "parameter of interest",
                    y_label = "FID score",
                    title = "Frechet inception distance as a function of ..."):
    x = []
    for param in vector_parameters:
        x.append(str(param))
    plt.plot(x, fid_scores, c='r', marker='o', markeredgecolor='blue', markerfacecolor='blue', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.show()    


def cyclical_kl_weight(current_epoch, num_epochs, num_cycles, R):
    ratio = num_epochs / num_cycles
    tau = ((current_epoch-1) % ratio) / ratio
    new_weight = np.minimum(tau/R, 1)
    return new_weight
    

    
if __name__ == "__main__":
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    
    KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.005, 0.007, 0.01, 0.1, 1, 10]
    
    # Show what happen when kl_weight=0.0
    vae = VAE.load("model/KL_impact/kl_weight=0.0")
    _, latent_representations = vae.reconstruct(x_val)
    plot_images_encoded_in_latent_space(latent_representations, y_val, kl_weight=0.0, save=False)
    

    # Show what happen with KL_weight=10
    vae = VAE.load("model/KL_impact/kl_weight=10")
    _, latent_representations = vae.reconstruct(x_val)
    plot_images_encoded_in_latent_space(latent_representations, y_val, kl_weight=10, save=False)
    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_val, y_val, num_sample_images_to_show)
    reconstructed_images, _ = vae.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images, kl_weight=10)
    generation(vae, num_images_to_generate=8, kl_weight=10)
    
    
    # Evaluate FID scores for different KL weights
    with open("model/KL_impact/fid_scores.pkl", "rb") as f:
        fid_kl_weights = pickle.load(f)
    f.close()
    # plot the results (FID score vs KL_weights)
    plot_FID_scores(vector_parameters = KL_WEIGHTS, 
                    fid_scores = fid_kl_weights,
                    x_label = "KL weight",
                    y_label = "FID score",
                    title = "Frechet Inception Distance with respect to KL weight")
    
    
    # Evaluate two best results in terms of FID index
    vae1 = VAE.load("model/KL_impact/kl_weight=0.0001")
    vae2 = VAE.load("model/KL_impact/kl_weight=0.001")
    _, latent_representations1 = vae1.reconstruct(x_val)
    plot_images_encoded_in_latent_space(latent_representations1, y_val, kl_weight=0.0001, save=False)
    _, latent_representations2 = vae2.reconstruct(x_val)    
    plot_images_encoded_in_latent_space(latent_representations2, y_val, kl_weight=0.001, save=False)
    
    plot_latent_space(vae1, title='Generated MNIST data with $KL\:weight=0.0001$')
    plot_latent_space(vae2, title='Generated MNIST data with $KL\:weight=0.001$')
    
    plot_epochs_history('model/KL_impact/kl_weight=0.0001')
    plot_epochs_history('model/KL_impact/kl_weight=0.001')
    
    
    # Evaluate Cyclical Annealing schedule
    vae = VAE.load("model/cyclical_annealing_schedule")
    FID = 235.72502429437012
    
    _, latent_representations = vae.reconstruct(x_val)
    plot_images_encoded_in_latent_space(latent_representations, y_val, save=False)
    
    plot_latent_space(vae, title="Cyclical annealing schedule: generation")

    plot_epochs_history("model/cyclical_annealing_schedule")
    
    # Draw KL_weight(epoch)
    x = np.arange(1, 61, 1)
    y = cyclical_kl_weight(current_epoch=x, num_epochs=60, num_cycles=6, R=0.5)
    plt.plot(x, y, c='r', linewidth = 1.5, marker='o',markersize=3, markerfacecolor='black', markeredgecolor='black')
    plt.xlabel('epoch')
    plt.ylabel('KL weight')
    plt.title('KL weight Cyclical Annealing schedule')
    plt.grid()
    
    
    
    # LATENT SPACE DIMENSION ANALYSIS
    LATENT_SPACE_DIM = [2, 3, 5, 8, 10, 15, 20, 30] 
    # Consider KL_weight = 0.0001
    # Evaluate FID scores for different latent space dimensions
    with open("model/KL_impact/kl_weight=0.0001/Latent_space_dim_impact/fid_scores.pkl", "rb") as f:
        fid_latent_dim = pickle.load(f)
    f.close()
    # plot the results (FID score vs KL_weights)
    plot_FID_scores(vector_parameters = LATENT_SPACE_DIM, 
                    fid_scores = fid_latent_dim,
                    x_label = "Latent space dimension",
                    y_label = "FID score",
                    title = "FID with respect to latent space dimension (KL weight=0.0001)")
    vae = VAE.load("model/KL_impact/kl_weight=0.0001/Latent_space_dim_impact/latent_space_dim=15/")
    images_generated = vae.generate(25)
    plot_images_grid(images_generated, (5, 5), title='', xlabel='', ylabel='')
    
    
    # Consider KL_weight = 0.001
    # Evaluate FID scores for different latent space dimensions
    with open("model/KL_impact/kl_weight=0.001/Latent_space_dim_impact/fid_scores.pkl", "rb") as f:
        fid_latent_dim = pickle.load(f)
    f.close()
    # plot the results (FID score vs KL_weights)
    plot_FID_scores(vector_parameters = LATENT_SPACE_DIM, 
                    fid_scores = fid_latent_dim,
                    x_label = "Latent space dimension",
                    y_label = "FID score",
                    title = "FID with respect to latent space dimension (KL weight=0.001)")
    vae = VAE.load("model/KL_impact/kl_weight=0.001/Latent_space_dim_impact/latent_space_dim=15/")
    images_generated = vae.generate(25)
    plot_images_grid(images_generated, (5, 5), title='', xlabel='', ylabel='')
    
    
    # DEEPNESS IMPACT
    FILTERS = [(16, 32, 32, 64, 128), (16, 32, 32, 64, 64, 128)]
    for filt in FILTERS:
        vae = VAE.load("model/deepness_impact/KL=0.0001_DIM=15/filters=" + str(filt))
        
        # Generation
        images_generated = vae.generate(25)
        plot_images_grid(images_generated, (5, 5), title='', xlabel='', ylabel='')
        
        
        # plot epochs history
        plot_epochs_history("model/deepness_impact/KL=0.0001_DIM=15/filters=" + str(filt))
        
    # KL weight = 0.001
    for filt in FILTERS:
        vae = VAE.load("model/deepness_impact/KL=0.001_DIM=15/filters=" + str(filt))
        
        #generation
        images_generated = vae.generate(25)
        plot_images_grid(images_generated, (5, 5), title='', xlabel='', ylabel='')
        
        
        # plot epochs history
        plot_epochs_history("model/deepness_impact/KL=0.001_DIM=15/filters=" + str(filt))
    
    with open ("model/deepness_impact/KL=0.001_DIM=15/fid_scores.pkl", 'rb') as f:
        fid_deepness = pickle.load(f)
    f.close()
    

    
    

