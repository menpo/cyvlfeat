import numpy as np
from .cyfisher import cy_fisher


def fisher(X, MEANS, COVARIANCES, PRIORS,
           Normalized=True, SquareRoot=True,
           Improved=True, Fast=False, Verbose=False):
    '''
    Computes the Fisher vector encoding of the vectors x relative to
    the diagonal covariance Gaussian mixture model with means MEANS,
    covariances COVARIANCES, and prior mode probabilities PRIORS.

    X has one column per data vector (e.g. a SIFT descriptor), and
    MEANS and COVARIANCES one column per GMM component (covariance
    matrices are assumed diagonal, hence these are simply the variance
    of each data dimension). PRIORS has size equal to the number of
    GMM components. All data must be of the class np.float32
 
    ENC is a vector of the class np.float32 of size equal to the
    product of 2 * the data dimension * the number of components.
 
    By default, the standard Fisher vector is computed. FISHER()
    accepts the following options:
 
    Normalized::
      If specified, L2 normalize the Fisher vector.
 
    SquareRoot::
      If specified, the signed square root function is applied to
      ENC before normalization.
 
    Improved::
      If specified, compute the improved variant of the Fisher
      Vector. This is equivalent to specifying the Normalized and
      SquareRoot options.
 
    Fast::
      If specified, uses slightly less accurate computations but
      significantly increase the speed in some cases (particularly
      with a large number of Gaussian modes).
 
    Verbose::
      Increase the verbosity level.
      '''

    # check for None
    if X is None or MEANS is None or COVARIANCES is None or PRIORS is None:
        raise ValueError('a required input is None')

    # validate the gmm parameters
    D = MEANS.shape[0] # the feature dimensionality
    K = MEANS.shape[1] # the number of GMM modes
    N = X.shape[1] # the number of samples
    if COVARIANCES.shape[0]!=D:
        raise ValueError('COVARIANCES and MEANS do not have the same dimensionality')
    
    if PRIORS.ndim!=1:
        raise ValueError('PRIORS had unexpected shape')
    
    if COVARIANCES.shape[1]!=K or PRIORS.shape[0]!=K:
        raise ValueError('COVARIANCES or PRIORS does not have the correct number of modes')

    if X.shape[0]!=D:
        raise ValueError('X and MEANS do not have the same dimensionality')
    
    try:
        ENC = cy_fisher(X.astype(np.float32),
                        MEANS.astype(np.float32),
                        COVARIANCES.astype(np.float32),
                        PRIORS.astype(np.float32),
                        np.int32(Normalized),
                        np.int32(SquareRoot),
                        np.int32(Improved),
                        np.int32(Fast),
                        np.int32(Verbose))
    except Exception as e:
        print 'Runtime error: ', e
        raise Exception(e)
    return ENC
    