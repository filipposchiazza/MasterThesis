import numpy as np
import os
import pickle
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


def calculate_FID_for_vector_parameters(vector_parameter,
                                   real_images,
                                   common_path = "model/KL_impact/kl_weight=",
                                   num_images_for_FID = 10000,
                                   verbose = True):
    """Evaluate the FID score for different values of the same parameter: for example 
    for many values of the KL weight or the latent space dimension"""
    
    fid_scores = []
    counter = 1
    
    for kl in vector_parameter:
        # generate num_images_for_FID images
        vae = VAE.load(common_path + str(kl))
        generated_images = vae.generate(num_images_for_FID)
        
        # resize real and generated images
        real_images_resized, generated_images_resized = resize_images(real_images, generated_images, size=(75, 75, 3))
        
        # apply InceptionV3
        act1, act2 = apply_InceptionV3(dataset1=real_images_resized, dataset2=generated_images_resized)

        # calculate FID
        fid = calculate_fid(act1, act2)
        print(fid)
        
        # append fid to fid_scores
        fid_scores.append(fid)
        
        if verbose == True:
            print("Step: " + str(counter) + "/" + str(len(vector_parameter)))
        counter += 1
        
    return fid_scores
    

def save_FID_scores(save_folder, fid_scores, save=True):
    if save == True:
        save_path = os.path.join(save_folder, "fid_scores.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(fid_scores, f)
        


def plot_FID_scores(vector_parameters,
                    fid_scores,
                    x_label = "parameter of interest",
                    y_label = "FID score",
                    title = "Frechet inception distance as a function of ..."):
    x = []
    for param in vector_parameters:
        x.append(str(param))
    plt.plot(x, fid_scores, c='r', marker='o',markeredgecolor='blue', markerfacecolor='blue', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.show()
    
  

KL_WEIGHTS = [0., 1e-5, 1e-4, 0.001, 0.005, 0.007, 0.01, 0.1, 1, 10]
LATENT_SPACE_DIM = [2, 3, 5, 8, 10, 15, 20, 30]
FILTERS = [(16, 32, 32, 64, 128), (16, 32, 32, 64, 64, 128)]

########################################################################################

if __name__ == "__main__":
        
    # load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    
    # create the model InceptionV3
    model = InceptionV3(include_top=False,  pooling='avg', input_shape=(75, 75, 3))
    
    # take 10000 real images 
    real_images, _ = select_images(x_val, y_val, num_images = 10000)
    
    # calculate FID scores for all kl_weights
    fid_kl_weights = calculate_FID_for_vector_parameters(vector_parameter=KL_WEIGHTS, real_images=real_images)
    
    # save fid_scores
    save_FID_scores(save_folder="model/KL_impact", fid_scores=fid_kl_weights, save=True)
    
    # load FID scores
    with open("model/KL_impact/fid_scores.pkl", "rb") as f:
        fid_kl_weights = pickle.load(f)
    f.close()
    
    # plot the results (FID score vs KL_weights)
    plot_FID_scores(vector_parameters = KL_WEIGHTS, 
                    fid_scores = fid_kl_weights,
                    x_label = "KL weight",
                    y_label = "FID score",
                    title = "Frechet inception distance as a function of KL weight")
                    
    
    
    # Select two best value for KL weight:
    # 0.0001 because of FID score
    # 0.001 because of visual analysis
    # For each of these two values, evaluate different latent space dimensions
    
    fid_latent_dim_kl_0001 = calculate_FID_for_vector_parameters(vector_parameter = LATENT_SPACE_DIM, real_images = real_images, common_path = "model/KL_impact/kl_weight=0.0001/Latent_space_dim_impact/latent_space_dim=")
    
    save_FID_scores(save_folder="model/KL_impact/kl_weight=0.0001/Latent_space_dim_impact", fid_scores=fid_latent_dim_kl_0001)
    
    fid_latent_dim_kl_001 = calculate_FID_for_vector_parameters(vector_parameter = LATENT_SPACE_DIM, real_images = real_images, common_path = "model/KL_impact/kl_weight=0.001/Latent_space_dim_impact/latent_space_dim=")
 
    save_FID_scores(save_folder="model/KL_impact/kl_weight=0.001/Latent_space_dim_impact", fid_scores=fid_latent_dim_kl_001)
    
    with open("model/KL_impact/kl_weight=0.001/Latent_space_dim_impact/fid_scores.pkl", "rb") as f:
        fid_latent_dim_kl_001 = pickle.load(f)
    f.close()
    
    with open("model/KL_impact/kl_weight=0.0001/Latent_space_dim_impact/fid_scores.pkl", "rb") as f:
        fid_latent_dim_kl_0001 = pickle.load(f)
    f.close()
    
    # plot the results (FID score vs latent space dimension)
    
    plot_FID_scores(vector_parameters = LATENT_SPACE_DIM, 
                    fid_scores = fid_latent_dim_kl_0001,
                    x_label = "Latent space dimension",
                    y_label = "FID score",
                    title = "Frechet inception distance as a function of the latent space dimension")
    
    plot_FID_scores(vector_parameters = LATENT_SPACE_DIM, 
                    fid_scores = fid_latent_dim_kl_001,
                    x_label = "Latent space dimension",
                    y_label = "FID score",
                    title = "Frechet inception distance as a function of the latent space dimension")
    
    
    
    # FID for deepness
    # KL weight = 0.0001
    fid_deepness_kl_0001 = calculate_FID_for_vector_parameters(vector_parameter = FILTERS, real_images = real_images, common_path = "model/deepness_impact/KL=0.0001_DIM=15/filters=")
    save_FID_scores(save_folder="model/deepness_impact/KL=0.0001_DIM=15", fid_scores=fid_deepness_kl_0001)
    # KL weight = 0.001
    fid_deepness_kl_001 = calculate_FID_for_vector_parameters(vector_parameter = FILTERS, real_images = real_images, common_path = "model/deepness_impact/KL=0.001_DIM=15/filters=")
    save_FID_scores(save_folder="model/deepness_impact/KL=0.001_DIM=15", fid_scores=fid_deepness_kl_001)
    



    # Evaluate FID score for cyclical annealing (latent_dim = 2)
    vae = VAE.load("model/cyclical_annealing_schedule")
    generated_images = vae.generate(10000)
        
    # resize real and generated images
    real_images_resized, generated_images_resized = resize_images(real_images, generated_images, size=(75, 75, 3))
        
    # apply InceptionV3
    act1, act2 = apply_InceptionV3(dataset1=real_images_resized, dataset2=generated_images_resized)

    # calculate FID
    fid = calculate_fid(act1, act2)
    print(fid)

                



