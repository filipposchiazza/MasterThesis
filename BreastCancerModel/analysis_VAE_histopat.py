from vae import VAE
import pickle
import numpy as np
import matplotlib.pyplot as plt
import umap


FILE_NAME = 'CNN_classification_task/converted_labeled_data/training_dataset.pkl'

def load_data(filename, images_shape):
    # Read data as numpy array and apply conversion of labels from string to int
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    f.close()
    data = dataset["data"][:1000]
    labels = dataset["labels"][:1000]
    del dataset
    labels = np.asarray(labels)
    labels = labels.astype(int)
    data = np.asarray(data).reshape((-1, images_shape[0], images_shape[1], images_shape[2]))
    data = data.astype("float32") / 255
    return data, labels

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
    
    data_train, labels_train = load_data(filename=FILE_NAME, images_shape=(50, 50, 3))
    
    
    vae = VAE.load('model_histopat_rec')
    
    images_sampled = select_random_images(images = data_train, num_images = 4)
    reconstructed_images = vae.reconstruct(images_sampled)
    plot_original_vs_reconstructed(images_sampled=images_sampled, reconstructed_images=reconstructed_images)
    
