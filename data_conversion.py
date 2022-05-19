from glob import glob
import pickle
import cv2

def convert_dataset(path, file_name, shape):
    image_paths = glob(path, recursive=True)
    data = []
    counter = 0
    for filename in image_paths[:70000]:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_sized = cv2.resize(img, shape, interpolation = cv2.INTER_LINEAR)
        data.append(img_sized)
        counter += 1
        print (counter)
    
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    f.close()
    
    
    
if __name__ == '__main__':
    convert_dataset(path='./data/IDC_regular_ps50_idx5/**/*.png', 
                    file_name='medium_data_converted.pkl', 
                    shape=(50,50))
    


