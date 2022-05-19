from vae import VAE
from train_VAE_histopat import load_dataset

import numpy as np
import matplotlib.pyplot as plt
import umap


def create_test_dataset(data, start, stop):
    return data[start:stop]

def select_random_images(images, num_images):
    indexes = np.random.choice(range(len(images)), num_images)
    images_sampled = images[indexes]
    return images_sampled

def plot_original_vs_reconstructed(images_sampled, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images_sampled)
    for i, (image, reconstructed_image) in enumerate(zip(images_sampled, reconstructed_images)):
       # image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image)
      #  reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image)
    plt.show()
    
    
def dim_reduction(vae, data, n_neighbors, min_dist):
    _, latent_representation = vae.reconstruct(data)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(latent_representation)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1)
    plt.show()
    


if __name__ == '__main__':
    
    data = load_dataset(file_name = './data/data_converted/medium_data_converted.pkl', 
                             shape = (50, 50, 3))
    data_test = create_test_dataset(data, 50000, 60000)
    
    vae = VAE.load('model_histo_images')
    
    images_sampled = select_random_images(images = data_test, num_images = 4)
    reconstructed_images = vae.reconstruct(images_sampled)
    plot_original_vs_reconstructed(images_sampled=images_sampled, reconstructed_images=reconstructed_images)
    
