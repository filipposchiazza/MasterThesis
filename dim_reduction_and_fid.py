from vae import VAE
import umap

from tensorflow.keras.applications.inception_v3 import InceptionV3
from skimage.transform import resize
from scipy.linalg import sqrtm

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import pickle

MNIST = False
HYSTOPAT = True

DATA_FILENAME = '../CNN_classification_task/converted_labeled_data/training_dataset'

# Insert here the path to the models to be tested    
model_name_list = ['model/01_first_trial',
                   'model/02_deeper',
                   'model/03_focus_reconstruction',
                   'model/04_latent_dim_is_15']

EMBEDDING_SAVE_FILE = 'embeddings.pkl'
FID_SAVE_FILE = 'fid.pkl'
NUM_IMAGES_FOR_FID_CALCULATION = 10000

##############################################################################################
# Define functions used for FID evaluation

def select_images(images, num_images):
    "Select randomly a number of images from images." 
    index = np.random.choice(range(len(images)), num_images)
    sample_images = images[index]
    return sample_images

def resize_images(real_images, generated_images, size=(299, 299, 3)):
    real_images_resized = []
    generated_images_resized = []
    for real, gen in zip(real_images, generated_images):
        support = resize(real, size)    
        real_images_resized.append(support)
        support = resize(gen, size)
        generated_images_resized.append(support)
    real_images_resized = np.asarray(real_images_resized) 
    generated_images_resized = np.asarray(generated_images_resized) 
    return real_images_resized, generated_images_resized

def apply_InceptionV3(dataset1, dataset2):
    "Apply inceptionV3"
    act1 = inception_model.predict(dataset1)
    act2 = inception_model.predict(dataset2)
    return act1, act2

def calculate_fid(act1, act2):
    "Calculate the fid score"
    # calculate means and covariance matrices
    mu1 = np.mean(act1, axis=0)
    mu2 = np.mean(act2, axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    sigma2 = np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
    	covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

#############################################################################################   

# Define functions for loading the datasets

def load_data(filename, images_shape):
    # Read data as numpy array and apply conversion of labels from string to i$
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    f.close()
    data = dataset["data"]
    labels = dataset["labels"]
    del dataset
    labels = np.asarray(labels)
    labels = labels.astype(int)
    data = np.asarray(data).reshape((-1, images_shape[0], images_shape[1], images_shape[2]))
    data = data.astype("float32") / 255
    return data, labels
    
    
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

################################################################################################################

# MNIST dataset
if MNIST == True:
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    

# Hystopat dataset
if HYSTOPAT == True:
    data_train, labels_train = load_data(filename=DATA_FILENAME, images_shape=(50, 50, 3))
    divisor = int(len(data_train) * 0.80)
    x_val = data_train[divisor:]

umap_embeddings = []
fid_list = []

# Create inception model and select real images for FID evaluation
inception_model = InceptionV3(include_top=False,  pooling='avg', input_shape=(75, 75, 3))
real_images = select_images(x_val, NUM_IMAGES_FOR_FID_CALCULATION)


for model in model_name_list:
    
    vae = VAE.load(model)
    # umap focus
    _, latent_representation = vae.reconstruct(x_val)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent_representation)
    embedding = embedding.tolist()
    umap_embeddings.append(embedding)
    print('embedding finished')
    
    # fid focus
    generated_images = vae.generate(NUM_IMAGES_FOR_FID_CALCULATION)
    # resize real and generated images
    real_images_resized, generated_images_resized = resize_images(real_images, generated_images, size=(75, 75, 3))
    # apply InceptionV3
    act1, act2 = apply_InceptionV3(dataset1=real_images_resized, dataset2=generated_images_resized)
    # calculate fid
    fid = calculate_fid(act1, act2)
    fid_list.append(fid)
    print('fid finished')
    
with open (EMBEDDING_SAVE_FILE, 'wb') as f:
    pickle.dump(umap_embeddings, f)

with open (FID_SAVE_FILE, 'wb') as f:
    pickle.dump(fid_list, f)
    

