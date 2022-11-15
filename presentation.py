from vae import VAE
import matplotlib.pyplot as plt
import pickle
import numpy as np

vae_model_list = ['BreastCancerModel/model/01_first_trial',
                  'BreastCancerModel/model/02_deeper',
                  'BreastCancerModel/model/03_focus_reconstruction',
                  'BreastCancerModel/model/04_latent_dim=15']

FILE_NAME = 'CNN_classification_task/converted_labeled_data/training_dataset.pkl'

def load_data_validation(filename, images_shape):
    # Read data as numpy array and apply conversion of labels from string to int
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    f.close()
    divisor = int(len(dataset["data"]) * 0.8)
    x_val = dataset["data"][divisor:]
    y_val = dataset["labels"][divisor:]
    del dataset
    y_val = np.asarray(y_val)
    y_val = y_val.astype(int)
    x_val = np.asarray(x_val).reshape((-1, images_shape[0], images_shape[1], images_shape[2]))
    x_val = x_val.astype("float32") / 255
    return x_val, y_val


    

if __name__ == '__main__':
    
    x_val, y_val = load_data_validation(filename=FILE_NAME, images_shape=(50, 50, 3))
    test_image = x_val[0]
    plt.imshow(test_image)
    test_image = test_image.reshape((1, 50, 50, 3))
    
    vae = VAE.load(vae_model_list[0])
    reconstructed_image,_ = vae.reconstruct(test_image)
    reconstructed_image = reconstructed_image.reshape((50, 50, 3))
    plt.imshow(reconstructed_image)
        
    generated_image = vae.generate(1)
    generated_image = generated_image.reshape((50, 50, 3))
    plt.imshow(generated_image)
    
    # FID indexes 
    with open('BreastCancerModel/fid.pkl', 'rb') as f:
        fid_list = pickle.load(f)
    print(fid_list)
    
    with open('BreastCancerModel/embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
        
        

    e = embeddings[0]
    e = np.asarray(e)
    plt.hexbin(e[:,0], e[:,1], gridsize=30)
            
    positive = []
    negative = []
    for i, label in enumerate(y_val):
        if label == 0:
            negative.append(e[i])
        else:
            positive.append(e[i])
        
    negative = np.asarray(negative)
    positive = np.asarray(positive)
            
    plt.hexbin(negative[:,0], negative[:,1], gridsize=30)
    plt.hexbin(positive[:,0], positive[:,1], gridsize=30)
 
    
 
    
 
###########################################################################################    
    for model_directory in vae_model_list:
        
        vae = VAE.load(model_directory)
        reconstructed_image,_ = vae.reconstruct(test_image)
        reconstructed_image = reconstructed_image.reshape((50, 50, 3))
        plt.imshow(reconstructed_image)
        
        generated_image = vae.generate(1)
        generated_image = generated_image.reshape((50, 50, 3))
        plt.imshow(generated_image)

        
    for i in range(len(embeddings)):
        e = embeddings[i]
        e = np.asarray(e)
        plt.hexbin(e[:,0], e[:,1], gridsize=30)
        
        positive = []
        negative = []
        for i, label in enumerate(y_val):
            if label == 0:
                negative.append(e[i])
            else:
                positive.append(e[i])
        
        negative = np.asarray(negative)
        positive = np.asarray(positive)
        
        plt.hexbin(negative[:,0], negative[:,1], gridsize=30)
        plt.hexbin(positive[:,0], positive[:,1], gridsize=30)


