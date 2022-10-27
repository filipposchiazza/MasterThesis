import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from autoencoder import * 
from train_autoencoder_mnist import load_mnist

def select_images(images, labels, num_images=8):
    index = np.random.choice(range(len(images)), num_images)
    sample_images = images[index]
    sample_labels = labels[index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()

    
def plot_images_encoded_in_latent_space(latent_representations, sample_labels):

    cmap = plt.cm.rainbow
    norm = colors.BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)

    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=sample_labels, cmap=cmap, norm=norm, s=2, edgecolor='none')
    plt.colorbar(ticks=np.linspace(0, 9, 10))

    plt.title('Autoencoder latent space of MNIST data', fontsize=15)    
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')

    plt.show()
    
    
def generation(ae, num_images_to_generate, save=False):
    generated_images = ae.generate(num_images_to_generate)
    fig = plt.figure(figsize=(15, 3))
    for i, image in enumerate(generated_images):
        image = image.squeeze()
        ax = fig.add_subplot(1, num_images_to_generate, i+1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
    plt.suptitle("Autoencoder generation", fontsize=20)
    #if save == True:
        #filename = 'Images_to_show/generation_KL_weight=' + str(kl_weight) + '.png'
        #fig.savefig(filename)
    plt.show()
    
    
if __name__ == "__main__":
    autoencoder = Autoencoder.load("model_upgrade")
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Reconstruction
    num_sample_images_to_show = 12
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)
    
    """
    num_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)
    """
    # Plot the latent space
    _, latent_representations = autoencoder.reconstruct(x_test)
    plot_images_encoded_in_latent_space(latent_representations, y_test)
    
    # Generation
    generation(autoencoder, 12)
