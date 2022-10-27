import pickle


def change_key(path, old_key, new_key):
    filename = path + '/history.pkl'
    with open(filename, "rb") as f:
        history = pickle.load(f)
    f.close()
    
    history[new_key] = history.pop(old_key)
    
    with open(filename, "wb") as f:
        pickle.dump(history, f)
    f.close()
    
    
if __name__ == '__main__':
    
    FILTERS = [(16, 32, 32, 64, 128), (16, 32, 32, 64, 64, 128)]
    for f in FILTERS:
        path = "model/deepness_impact/KL=0.0001_DIM=15/filters=" + str(f)
        change_key(path=path, old_key='_calculate_recostruction_loss', new_key='reconstruction_loss')
        change_key(path=path, old_key='_calculate_kl_loss', new_key='kl_loss')
        change_key(path=path, old_key='val__calculate_recostruction_loss', new_key='val_reconstruction_loss')
        change_key(path=path, old_key='val__calculate_kl_loss', new_key='val_kl_loss')
    





