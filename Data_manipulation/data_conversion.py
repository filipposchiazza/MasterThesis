from glob import glob
import random
import pickle
import cv2


def convert_dataset(path, directory, shape):
    image_paths = glob(path, recursive=True)
    len_dataset = len(image_paths)
    data = []
    labels = []
    counter = 0
    for filename in image_paths:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_sized = cv2.resize(img, shape, interpolation = cv2.INTER_LINEAR)
        data.append(img_sized)
        labels.append(filename[-5])  # 0 or 1
        counter += 1
        print (str(counter) + '/' + str(len_dataset))
        
    support = list(zip(data, labels))
    random.shuffle(support)
    data, labels = zip(*support)
    del support
    
    # Divide the dataset in 25% test and 75% train
    separator = int(0.75 * len(labels))
    # Separate train data
    train_data = data[:separator]
    train_labels = labels[:separator]
    train_dataset = {"data":train_data, "labels":train_labels}
    # Save training data
    with open(directory + 'training_dataset.pkl', "wb") as f:
        pickle.dump(train_dataset, f)
    f.close()
    del train_dataset
    del train_labels
    del train_data
    # Separate test data
    test_data = data[separator:]
    test_labels = labels[separator:]
    test_dataset = {"data":test_data, "labels":test_labels}
    # Save test dataset
    with open(directory + 'test_dataset.pkl', "wb") as f:
        pickle.dump(test_dataset, f)
    f.close()   

    
    
    
if __name__ == '__main__':
    convert_dataset(path='./data/IDC_regular_ps50_idx5/**/*.png',
                    directory = './data/converted_labeled_data/',
                    shape=(50,50))
    


