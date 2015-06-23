import numpy as np
import cyvlfeat.sift.sift as cyssift


def sift(image, octaves=-1, levels=3, first_octave=0, edge_thresh=-1, peak_thresh=-1,
	norm_thresh = -1, magnif = -1, window_size=-1, force_orientations=False,
	float_descriptors=False, verbose=False):
  
  int                nikeys = -1 ;
  
  

	#/* -----------------------------------------------------------------
	#*                                               Check the arguments
    #* -------------------------------------------------------------- */

	# Validate image size
	if image.ndim != 2:
		raise ValueError('Only 2D arrays are supported')
     
    if not isinstance(octaves, int) or octaves < 0:
		raise ValueError("'Octaves' must be a nonnegative integer.")
  
    if not isinstance(levels, int) or levels < 1:
        raise ValueError("'Levels' must be a positive integer.")
  
    if not isinstance(first_octave, int) :
		raise ValueError("'FirstOctave' must be an integer")
      #o_min = (int) *mxGetPr(optarg) ;

    if edge_thresh < 1:
        raise ValueError("'EdgeThresh' must be not smaller than 1.")

    if peak_thresh < 0:
		raise ValueError("'PeakThresh' must be a non-negative real.")
    
    if norm_thresh < 0:
		raise ValueError("'NormThresh' must be a non-negative real.")

    if magnif < 0:
        raise ValueError("'Magnif' must be a non-negative real.")
      
    if window_size < 0:
        raise ValueError("'WindowSize' must be a non-negative real.")

    frames=np.asarray(frames)
	if frames.ndim<>2 or frames.shape[0]<>4:
		raise ValueError("'Frames' must be a 4 x N matrix.")
    
      ikeys_array = mxDuplicateArray (optarg) ;
      nikeys      = mxGetN (optarg) ;
      ikeys       = mxGetPr (ikeys_array) ;
      if (! check_sorted (ikeys, nikeys)) {
        qsort (ikeys, nikeys, 4 * sizeof(double), korder) ;
      }
  

    geometry = geometry.astype(np.int32)
    step = step.astype(np.int32)
    size = size.astype(np.int32)
    bounds = bounds.astype(np.int32)
    image = np.require(image, dtype=np.float32, requirements='F')
	
    frames, descriptors = cysift.dsift(image, step, size, bounds, window_size,
                                       norm, fast, float_descriptors, geometry,
                                       verbose)
    return frames, np.ascontiguousarray(descriptors)
