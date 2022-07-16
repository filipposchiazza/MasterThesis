import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from train_VAE_mnist import load_mnist
from analysis_mnist_VAE import select_images
from skimage.transform import resize
from vae import VAE
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


###############################################################################
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


def calculate_FID_for_vector_parameters(kl_weights_vector,
                                   real_images,
                                   common_path = "model/KL_impact/kl_weight=",
                                   num_images_for_FID = 10000,
                                   verbose = True):
    """Evaluate the FID score for different values of the same parameter: for example 
    for many values of the KL weight or the latent space dimension"""
    
    fid_scores = []
    counter = 1
    
    for kl in kl_weights_vector:
        # generate num_images_for_FID images
        vae = VAE.load(common_path + str(kl))
        generated_images = vae.generate(num_images_for_FID)
        
        # resize real and generated images
        real_images_resized, generated_images_resized = resize_images(real_images, generated_images, size=(75, 75, 3))
        
        # apply InceptionV3
        act1, act2 = apply_InceptionV3(dataset1=real_images_resized, dataset2=generated_images_resized)

        # calculate FID
        fid = calculate_fid(act1, act2)
        
        # append fid to fid_scores
        fid_scores.append(fid)
        
        if verbose == True:
            print("Step: " + str(counter) + "/" + str(len(kl_weights_vector)))
        counter += 1
        
        return fid_scores
    

def plot_FID_scores(vector_parameters,
                    fid_scores,
                    x_label = "parameter of interest",
                    y_label = "FID score",
                    title = "Frechet inception distance as a function of ..."):
    x = []
    for param in vector_parameters:
        x.append(str(param))
    plt.plot(x, fid_scores, c='r', marker='o', markerfacecolor='blue', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    
  

KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.005, 0.007, 0.01, 0.1, 1, 10]
LATENT_SPACE_DIM = [2, 3, 5, 8, 10, 15, 20, 30]


########################################################################################

if __name__ == "__main__":
        
    # load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    
    # create the model InceptionV3
    model = InceptionV3(include_top=False,  pooling='avg', input_shape=(75, 75, 3))
    
    # take 10000 real images 
    real_images, _ = select_images(x_val, y_val, num_images = 10000)
    
    # calculate FID scores for all kl_weights
    fid_kl_weights = calculate_FID_for_vector_parameters(kl_weights_vector=KL_WEIGHTS, 
                                                         real_images=real_images)
    
    # plot the results (FID score vs KL_weights)
    plot_FID_scores(vector_parameters=KL_WEIGHTS, fid_scores=fid_kl_weights)
    
    
    
    
    
    
    # FID score for study the effect of latent space dimension
    
    fid_scores_dim_reduction_0001 = []
    fid_scores_dim_reduction_001 = []
    counter = 1
    
    for dim in LATENT_SPACE_DIM:
        # generate 10000 images
        vae_0001 = VAE.load("model/KL_impact/kl_weight=0.0001/Latent_space_dim_impact/latent_space_dim=" + str(dim))
       
        generated_images_0001 = vae_0001.generate(10000)

        
        # resize real and generated images
        real_images_resized_0001, generated_images_resized_0001 = resize_images(real_images, generated_images_0001, size=(75, 75, 3))
        
        
        # apply InceptionV3
        act1_0001, act2_0001 = apply_InceptionV3(dataset1=real_images_resized_0001, dataset2=generated_images_resized_0001)
       
        # calculate FID
        fid_0001 = calculate_fid(act1_0001, act2_0001)
  
        
        # append fid to fid_scores
        fid_scores_dim_reduction_0001.append(fid_0001)
        
                
        print("Step: " + str(counter) + "/" + str(len(LATENT_SPACE_DIM)))
        counter += 1
        
    for dim in LATENT_SPACE_DIM:
        # generate 10000 images
       
        vae_001 = VAE.load("model/KL_impact/kl_weight=0.001/Latent_space_dim_impact/latent_space_dim=" + str(dim))
       
        generated_images_001 = vae_001.generate(10000)
            
        # resize real and generated images
       
        real_images_resized_001, generated_images_resized_001 = resize_images(real_images, generated_images_001, size=(75, 75, 3))
            
        # apply InceptionV3
    
        act1_001, act2_001 = apply_InceptionV3(dataset1=real_images_resized_001, dataset2=generated_images_resized_001)

        # calculate FID
        
        fid_001 = calculate_fid(act1_001, act2_001)
            
        # append fid to fid_scores
        
        fid_scores_dim_reduction_001.append(fid_001)
            
                    
        print("Step: " + str(counter) + "/" + str(len(LATENT_SPACE_DIM)))
        counter += 1
    
    x = []
    for dim in LATENT_SPACE_DIM:
        x.append(str(dim))
        
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(x, fid_scores_dim_reduction_0001, color="black", label="KL=0.0001")
    #ax1.plot(x, fid_scores_dim_reduction_001, color="green", label="KL=0.001")
    plt.legend()
    plt.show()
                



