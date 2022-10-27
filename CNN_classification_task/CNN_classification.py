import numpy as np
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import layers, models
import tensorflow as tf

def load_data(filename, images_shape):
    # Read data as numpy array and apply conversion of labels from string to int
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



TRAINING = True
TESTING = True

##################################################################################

if TRAINING == True:
    
    
    FILENAME = './converted_labeled_data/training_dataset.pkl'
    #SAVE_FOLDER = "./CNN_classification_task"

    data_train, labels_train = load_data(filename=FILENAME, images_shape=(50, 50, 3))
    
    model = models.Sequential()

    model.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (50, 50, 3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.15))

    model.add(layers.Conv2D(128, (2, 2), activation = 'relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.15))

    model.add(layers.Conv2D(256, (2, 2), activation = 'relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.15))

    model.add(layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(1,1))
    model.add(layers.Dropout(0.15))

    model.add(layers.Conv2D(128, (2, 2), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()
    
    model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])
    
    lr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=0, factor=0.5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    divisor = int(len(data_train) * 0.75)

    historys = model.fit(data_train[:divisor], labels_train[:divisor], epochs=50, callbacks=[lr, early_stopping], batch_size=25, validation_data=(data_train[divisor:], labels_train[divisor:]), shuffle=True)    
    
    #save_file = os.path.join(SAVE_FOLDER, "history.pkl")
    
    with open("history.pkl", "wb") as f:
        pickle.dump(model.history.history, f)
        
    #save_file = os.path.join(SAVE_FOLDER, "weights.h5")
    
    model.save('model')
    
###################################################################################

if TESTING == True:
    
    FILENAME = './converted_labeled_data/test_dataset.pkl'
    
    data_test, labels_test = load_data(filename=FILENAME, images_shape=(50, 50, 3))
    
    model = tf.keras.models.load_model('model')
    
    model.evaluate(data_test, labels_test)



