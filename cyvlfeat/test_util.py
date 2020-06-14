import cyvlfeat

def lena():
    import pickle
    import os
    import numpy as np

    path = os.path.join(os.path.dirname(cyvlfeat.__file__), 'data', 'lena.dat')
    with open(path, 'rb') as f:
        lena = np.array(pickle.load(f))
    return lena
