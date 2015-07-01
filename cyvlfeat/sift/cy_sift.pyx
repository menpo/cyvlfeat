# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as cnp
cimport cython

from libc.stdio cimport printf
from libc.stdlib cimport qsort
# Import the header files
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.mathop cimport *
from cyvlfeat._vl.sift cimport *
# #include <vl/sift.h>


# ** @internal
# ** @file     sift.c
# ** @author   Andrea Vedaldi
# ** @brief    Scale Invariant Feature Transform (SIFT) - MEX
# **/


# #include <vl/mathop.h>

# #include <math.h>
# #include <assert.h>



# ** ------------------------------------------------------------------
# ** @internal
# ** @brief Transpose descriptor
# **
# ** @param dst destination buffer.
# ** @param src source buffer.
# **
# ** The function writes to @a dst the transpose of the SIFT descriptor
# ** @a src. The transpose is defined as the descriptor that one
# ** obtains from computing the normal descriptor on the transposed
# ** image.
# **/

cdef inline void transpose_descriptor (vl_sift_pix* dst, vl_sift_pix* src):
    cdef int BO = 8  # number of orientation bins
    cdef int BP = 4  # number of spatial bins
    cdef int i, j, t
    cdef int jp
    cdef int o  
    cdef int op 
    for j in range(BP):
        jp = BP - 1 - j
        for i in range(BP):
            o = BO * i + BP*BO * j 
            op = BO * i + BP*BO * jp
            dst [op] = src[o]
        for t in range(1,BO):
            dst [BO - t + op] = src[t + o]


#
# ** @internal
# ** @brief Ordering of tuples by increasing scale
# **
# ** @param a tuple.
# ** @param b tuple.
# **
# ** @return @c a[2] < b[2]
# **/


cdef int korder(void * a, void * b):
    cdef double x = (<double*> a)[2] - (<double*>b)[2]
    if (x < 0):
        return -1
    if (x > 0):
        return +1
    return 0


# ------------------------------------------------------------------
# ** @internal
# ** @brief Check for sorted keypoints
# **
# ** @param keys keypoint list to check
# ** @param nkeys size of the list.
# **
# ** @return 1 if the keypoints are storted.
# **/


cdef bint check_sorted (double * keys, vl_size nkeys):
    cdef vl_uindex k
    for k in range( nkeys-1):
        if korder(keys, keys + 4) > 0:
            return False
        keys += 4
  
    return True

cdef enum:
    descriptor_length = 128

@cython.boundscheck(False)
cpdef sift(
    cnp.ndarray[float, ndim=2, mode='fortran'] data,
    cnp.ndarray[double, ndim=2, mode='fortran'] ikeys_array,
    int octaves, int levels, int first_octave,
    double edge_thresh, double peak_thresh, double norm_thresh,
    double magnif, double window_size, bint force_orientations, 
    int verbose, bint returnDescriptors, bint floatDescriptors ):
	# what to do about missing
    
    cdef double  *ikeys = <double *>0
    cdef int nikeys = -1
    
    cdef VlSiftFilt *filt
    cdef vl_bool first
    cdef VlSiftKeypoint* frames_array
    cdef int nframes = 0, reserved = 0, i,j,q
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] frames = \
        cnp.empty((4, reserved), dtype=cnp.float64, order='F')
    cdef cnp.ndarray[float, ndim=2, mode='fortran'] descr_float = \
        cnp.empty((descriptor_length, reserved), dtype=cnp.float32, order='F')
                  
    cdef cnp.ndarray[vl_uint8, ndim=2, mode='fortran'] descr_int = \
        cnp.empty((descriptor_length, reserved), dtype=cnp.uint8, order='F')
        
    cdef int height = data.shape[0]
    cdef int width = data.shape[1]
    cdef int err
    cdef const VlSiftKeypoint *keys
    cdef int  nkeys
    cdef double angles[4]
    cdef int nangles
    cdef VlSiftKeypoint ik
    cdef const VlSiftKeypoint *k
    cdef vl_sift_pix  buf[descriptor_length]
    cdef vl_sift_pix rbuf[descriptor_length]
    
    # TODO deal with ikeys sort
    # ikeys copy
    if ikeys_array.size()>0:
        nikeys      = ikeys_array.shape[1]
        ikeys       = &ikeys_array[0, 0]
        
        if not check_sorted(ikeys, nikeys):
            ikeys_array = ikeys_array[:,ikeys_array[2,:].argsort()]
            #todo check definition
#            qsort(ikeys, nikeys, 4 * sizeof(double), korder)
    else:
        nikeys=-1
  
    
    #/* create a filter to process the image */
    filt = vl_sift_new (height, width, octaves, levels, first_octave)

    if (peak_thresh >= 0):
        vl_sift_set_peak_thresh(filt, peak_thresh)
    if (edge_thresh >= 0):
        vl_sift_set_edge_thresh(filt, edge_thresh)
    if (norm_thresh >= 0):
        vl_sift_set_norm_thresh(filt, norm_thresh)
    if (magnif >= 0):
        vl_sift_set_magnif(filt, magnif)
    if (window_size >= 0):
        vl_sift_set_window_size(filt, window_size) 

    if (verbose):
        printf("vl_sift: filter settings:\n")
        printf("vl_sift:   octaves      (O)      = %d\n",
                vl_sift_get_noctaves      (filt))
        printf("vl_sift:   levels       (S)      = %d\n",
                vl_sift_get_nlevels       (filt))
        printf("vl_sift:   first octave (first_octave)  = %d\n",
                vl_sift_get_octave_first  (filt))
        printf("vl_sift:   edge thresh           = %g\n",
                vl_sift_get_edge_thresh   (filt))
        printf("vl_sift:   peak thresh           = %g\n",
                vl_sift_get_peak_thresh   (filt))
        printf("vl_sift:   norm thresh           = %g\n",
                vl_sift_get_norm_thresh   (filt))
        printf("vl_sift:   window size           = %g\n",
                vl_sift_get_window_size   (filt))
        printf("vl_sift:   float descriptor      = %d\n",
                floatDescriptors)

        if nikeys >= 0:
            printf("vl_sift: will source frames? yes (%d read)\n", nikeys)
        else:
            printf("vl_sift: will source frames? no\n")
        printf("vl_sift: will force orientations? {}\n".format("yes" if force_orientations else "no"))
    
    #/* ...............................................................
    # *                                             Process each octave
    # * ............................................................ */
    i     = 0
    first = True
    
    
    while (True):
        
        keys  = <VlSiftKeypoint *>0
        nkeys = 0
        
        if (verbose):
            printf ("vl_sift: processing octave %d\n",
                   vl_sift_get_octave_index (filt))
      

        # Calculate the GSS for the next octave #.................... */
        if (first):
            err   = vl_sift_process_first_octave(filt, &data[0,0])
            first = False
        else:
            err = vl_sift_process_next_octave(filt)
      
        if (err):
            break

        if (verbose > 1):
            printf("vl_sift: GSS octave %d computed\n",
                vl_sift_get_octave_index(filt))
                
        #* Run detector ............................................. */
        if (nikeys < 0):
            vl_sift_detect(filt)
            keys  = vl_sift_get_keypoints(filt)
            nkeys = vl_sift_get_nkeypoints(filt)
            i     = 0

            if (verbose > 1):
                printf ("vl_sift: detected %d (unoriented) keypoints\n", nkeys)
        
        else:
            nkeys = nikeys
      
        

        # For each keypoint ........................................ */
        for i in range(nkeys):
            
            #/* Obtain keypoint orientations ........................... */
            if (nikeys >= 0):
                vl_sift_keypoint_init (filt, &ik,
                                       ikeys_array[1, i],
                                       ikeys_array[0, i],
                                       ikeys_array[2, i])

                if (ik.o != vl_sift_get_octave_index(filt)):
                    break

                k = &ik

                #/* optionally compute orientations too */
                if (force_orientations):
                      nangles = vl_sift_calc_keypoint_orientations(filt, angles, k)
                else:
                    angles [0] = VL_PI / 2 - ikeys [4 * i + 3]
                    nangles    = 1
              
            else:
                k = keys + i
                nangles = vl_sift_calc_keypoint_orientations(filt, angles, k)
            

            #/* For each orientation ................................... */
            for q in range( nangles):

                #/* compute descriptor (if necessary) */
                if returnDescriptors:
                    vl_sift_calc_keypoint_descriptor (filt, buf, k, angles [q])
                    transpose_descriptor (rbuf, buf)
              

                #/* make enough room for all these keypoints and more */
                if (reserved < nframes + 1):
                    reserved += 2 * nkeys
                    frames = frames.resize((4, reserved))
#                    frames = realloc(frames, 4 * sizeof(double) * reserved)
                    # TODO don't understand realloc
                    if returnDescriptors:
                        if floatDescriptors:
                            descr_float = descr_float.resize(descriptor_length, reserved)
#                            descr_float = realloc(descr_float, descriptor_length*sizeof(double)*reserved)
                        else:
                            descr_int = descr_int.resize(descriptor_length, reserved)
#                            descr_int = realloc(descr_int, descriptor_length*sizeof(cnp.uint8_t)*reserved)

                # TODO changed/*  Save back with MATLAB conventions. 
                # Notice that the input
                #   * image was the transpose of the actual image. */
                frames[0, nframes] = k.x
                frames[1, nframes] = k.y
                frames[2, nframes] = k.sigma
                frames[3, nframes] = VL_PI / 2 - angles [q]
#                frames[4 * nframes + 0] = k.x
#                frames[4 * nframes + 1] = k.y
#                frames[4 * nframes + 2] = k.sigma
#                frames[4 * nframes + 3] = VL_PI / 2 - angles [q]

                if returnDescriptors:
                    if (floatDescriptors):
                        for j in range(descriptor_length):
                            descr_float [j, nframes] = 512.0 * rbuf [j]
#                            descr_float [descriptor_length * nframes + j] = 512.0 * rbuf [j]
                    else:
                        for j in range(descriptor_length):
                            descr_int[j, nframes] = <cnp.uint8_t>min(512.0 * rbuf [j],255.0)
#                            descr_int[descriptor_length * nframes + j] = <cnp.uint8_t>min(512.0 * rbuf [j],255.0)
                nframes +=1
            #/* next orientation */
        #/* next keypoint */
    #/* next octave */

    if (verbose):
        printf ("vl_sift: found %d keypoints\n", nframes)
    
    #/* cleanup */
    vl_sift_delete(filt)
    
    # TODO note that frames actually allocated more memory
    
        
    #TODO ikeys array would be freed if was created?
    
    #/* ...............................................................
    # *                                                       Save back 
    # * ............................................................ */
    if returnDescriptors:
        if floatDescriptors:
            return frames, descr_float
        else:
            return frames, descr_int
    else:
        return frames
#/* end: do job */

