import numpy as np
import matplotlib.pyplot as plt
from vae import VAE
from train_VAE_mnist import load_mnist
import umap

def select_images(images, labels, num_images=8):
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
    fig.suptitle("Recostruction: KL weight = " + str(kl_weight), fontsize=20)
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
    plt.title("KL weight = " + str(kl_weight), fontsize=20)
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
    plt.suptitle("Generation: KL weight = " + str(kl_weight), fontsize=20)
    if save == True:
        filename = 'Images_to_show/generation_KL_weight=' + str(kl_weight) + '.png'
        fig.savefig(filename)
    plt.show()
        
        
def dim_reduction(vae, data):
    _, latent_representation = vae.reconstruct(data)
    reducer = umap.Umap()
    embedding = reducer.fit_transform(latent_representation)
    plt.scatter(embedding[:, 0], embedding[:, 1])
    
    
    
    
    
    
if __name__ == "__main__":
    
    x_train, y_train, x_test, y_test = load_mnist()
    
    #######################################################################################
    # Analysis KL weight variation on the latent space, on the recostruction and on the generation
    KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]
    
    for kl in KL_WEIGHTS:
        vae = VAE.load("model/KL_impact/kl_weight=" + str(kl))
        #plot some reconstructed images vs original ones
        num_sample_images_to_show = 8
        sample_images, _ = select_images(x_train, y_train, num_sample_images_to_show)
        reconstructed_images, _ = vae.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images, kl, save=True)
    
        #plot the latent space representation
        _, latent_representations = vae.reconstruct(x_test)
        plot_images_encoded_in_latent_space(latent_representations, y_test, kl, save=True)
    
        # generate images
        generation(vae, 8, kl, save=True)
    

    


