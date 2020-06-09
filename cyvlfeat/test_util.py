def lena():
    import pickle
    import os
    import numpy as np

    git_repo_path = os.path.join(os.path.dirname(__file__),
                                 '..', 'conda')
    path = os.environ.get('RECIPE_DIR', git_repo_path)
    path = os.path.join(path, 'lena.dat')
    with open(path, 'rb') as f:
        lena = np.array(pickle.load(f))
    return lena
