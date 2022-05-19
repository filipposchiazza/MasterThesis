import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from train_VAE_mnist import load_mnist
from analysis_mnist_VAE import select_images
from skimage.transform import resize
from vae import VAE
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt



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
    act1 = model.predict(dataset1)
    act2 = model.predict(dataset2)
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


KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 10]

if __name__ == "__main__":
        
    # load data
    x_train, y_train, x_test, y_test = load_mnist()
    
    # create the model InceptionV3
    model = InceptionV3(include_top=False,  pooling='avg', input_shape=(75, 75, 3))
    
    # take 10000 real images 
    real_images, _ = select_images(x_train, y_train, num_images = 10000)
    
    fid_scores = []
    counter = 1
    
    for kl in KL_WEIGHTS:
        # generate 10000 images
        vae = VAE.load("model/KL_impact/kl_weight=" + str(kl))
        generated_images = vae.generate(10000)
        
        # resize real and generated images
        real_images_resized, generated_images_resized = resize_images(real_images, generated_images, size=(75, 75, 3))
        
        # apply InceptionV3
        act1, act2 = apply_InceptionV3(dataset1=real_images_resized, dataset2=generated_images_resized)

        # calculate FID
        fid = calculate_fid(act1, act2)
        
        # append fid to fid_scores
        fid_scores.append(fid)
        
        print("Step: " + str(counter) + "/" + str(len(KL_WEIGHTS)))
        counter += 1
    
        
    x = []
    for kl in KL_WEIGHTS:
        x.append(str(kl))
    plt.plot(x, fid_scores, c='r', marker='o', markerfacecolor='blue', linestyle='--')
    plt.xlabel("KL weight")
    plt.ylabel("FID score")
    plt.title("Frechet inception distance as a function of the KL weight")
    
    




