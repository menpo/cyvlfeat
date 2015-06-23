# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
from cyvlfeat._vl.host cimport vl_size

cdef extern from "vl/sift.h":
    # @brief SIFT filter pixel type */
    ctypedef float vl_sift_pix

    # ------------------------------------------------------------------
    # ** @brief SIFT filter keypoint
    # **
    # ** This structure represent a keypoint as extracted by the SIFT
    # ** filter ::VlSiftFilt.
    # **/

    ctypedef struct VlSiftKeypoint:

        int o           #< o coordinate (octave). */

        int i_x "ix"         #< Integer unnormalized x coordinate. */
        int i_y "iy"         #< Integer unnormalized y coordinate. */
        int i_s "is"          #< Integer s coordinate. */ conflicts with is keyword
        float x     #< x coordinate. */
        float y     #< y coordinate. */
        float s     #< s coordinate. */
        float sigma #< scale. */


# ------------------------------------------------------------------
# ** @brief SIFT filter
# **
# ** This filter implements the SIFT detector and descriptor.
# **/

    ctypedef struct VlSiftFilt:

        double sigman       #< nominal image smoothing. */
        double sigma0       #< smoothing of pyramid base. */
        double sigmak       #< k-smoothing */
        double dsigma0      #< delta-smoothing. */

        int width           #< image width. */
        int height          #< image height. */
        int O               #< number of octaves. */
        int S               #< number of levels per octave. */
        int o_min           #< minimum octave index. */
        int s_min           #< minimum level index. */
        int s_max           #< maximum level index. */
        int o_cur           #< current octave. */

        vl_sift_pix *temp   #< temporary pixel buffer. */
        vl_sift_pix *octave #< current GSS data. */
        vl_sift_pix *dog    #< current DoG data. */
        int octave_width    #< current octave width. */
        int octave_height   #< current octave height. */

        vl_sift_pix *gaussFilter  #< current Gaussian filter */
        double gaussFilterSigma   #< current Gaussian filter std */
        vl_size gaussFilterWidth  #< current Gaussian filter width */

        VlSiftKeypoint* keys#< detected keypoints. */
        int nkeys           #< number of detected keypoints. */
        int keys_res        #< size of the keys buffer. */

        double peak_thresh  #< peak threshold. */
        double edge_thresh  #< edge threshold. */
        double norm_thresh  #< norm threshold. */
        double magnif       #< magnification factor. */
        double windowSize   #< size of Gaussian window (in spatial bins) */

        vl_sift_pix *grad   #< GSS gradient data. */
        int grad_o          #< GSS gradient data octave. */



    # @name Create and destroy
    # ** @{
    # **/

    VlSiftFilt* vl_sift_new(int width, int height, int noctaves, int nlevels, int o_min)
    void vl_sift_delete(VlSiftFilt *f)
    # @} */

    # @name Process data
    # ** @{
    int vl_sift_process_first_octave(VlSiftFilt *f, vl_sift_pix *im)
    int vl_sift_process_next_octave(VlSiftFilt *f)
    void vl_sift_detect(VlSiftFilt *f)
    int  vl_sift_calc_keypoint_orientations(VlSiftFilt *f, double angles [4],
                                              VlSiftKeypoint *k)
    void vl_sift_calc_keypoint_descriptor(VlSiftFilt *f,
                                              vl_sift_pix *descr,
                                              VlSiftKeypoint * k,
                                              double angle)

    void  vl_sift_calc_raw_descriptor(VlSiftFilt *f, vl_sift_pix * image,
                                              vl_sift_pix *descr,
                                              int widht, int height,
                                              double x, double y,
                                              double s, double angle0)


    void  vl_sift_keypoint_init              (VlSiftFilt *f,
                                              VlSiftKeypoint *k,
                                              double x,
                                              double y,
                                              double sigma)
    # @} */

    # @name Retrieve data and parameters
    # ** @{
    # **/
     # ------------------------------------------------------------------
     #* @brief Get current octave index.
     #* @param f SIFT filter.
     #* @return index of the current octave.
     #*/
    inline int    vl_sift_get_octave_index(VlSiftFilt *f)

    # ------------------------------------------------------------------
    # ** @brief Get number of octaves.
    # ** @param f SIFT filter.
    # ** @return number of octaves.
    # **/
    inline int    vl_sift_get_noctaves(VlSiftFilt *f)

    #-------------------------------------------------------------------
    # ** @brief Get first octave.
    # ** @param f SIFT filter.
    # ** @return index of the first octave.
    # **/
    inline int    vl_sift_get_octave_first(VlSiftFilt *f)

    # ------------------------------------------------------------------
    # ** @brief Get current octave width
    # ** @param f SIFT filter.
    # ** @return current octave width.
    # **/
    inline int    vl_sift_get_octave_width(VlSiftFilt *f)

    # ------------------------------------------------------------------
    # ** @brief Get current octave height
    # ** @param f SIFT filter.
    # ** @return current octave height.
    # **/
    inline int    vl_sift_get_octave_height(VlSiftFilt *f)

    # ------------------------------------------------------------------
    # ** @brief Get number of levels per octave
    # ** @param f SIFT filter.
    # ** @return number of leves per octave.
    # **/
    inline int    vl_sift_get_nlevels(VlSiftFilt *f)

    # ------------------------------------------------------------------
    # ** @brief Get number of keypoints.
    # ** @param f SIFT filter.
    # ** @return number of keypoints.
    # **/
    inline int    vl_sift_get_nkeypoints(VlSiftFilt *f)
    # ------------------------------------------------------------------
    # ** @brief Get peaks threshold
    # ** @param f SIFT filter.
    # ** @return threshold
    # **/
    inline double vl_sift_get_peak_thresh(VlSiftFilt *f)
    # ------------------------------------------------------------------
    # ** @brief Get edges threshold
    # ** @param f SIFT filter.
    # ** @return threshold.
    # **/
    inline double vl_sift_get_edge_thresh(VlSiftFilt *f)
    # ------------------------------------------------------------------
    # ** @brief Get norm threshold
    # ** @param f SIFT filter.
    # ** @return threshold.
    # **/
    inline double vl_sift_get_norm_thresh(VlSiftFilt *f)
    # ------------------------------------------------------------------
    # ** @brief Get the magnification factor
    # ** @param f SIFT filter.
    # ** @return magnification factor.
    # **/
    inline double vl_sift_get_magnif(VlSiftFilt *f)
    # ------------------------------------------------------------------
    # ** @brief Get the Gaussian window size.
    # ** @param f SIFT filter.
    # ** @return standard deviation of the Gaussian window(in spatial bin units).
    # **/
    inline double vl_sift_get_window_size(VlSiftFilt *f)

    # ------------------------------------------------------------------
    # ** @brief Get current octave data
    # ** @param f SIFT filter.
    # ** @param s level index.
    # **
    # ** The level index @a s ranges in the interval <tt>s_min = -1</tt>
    # ** and <tt> s_max = S + 2</tt>, where @c S is the number of levels
    # ** per octave.
    # **
    # ** @return pointer to the octave data for level @a s.
    # **/
    inline vl_sift_pix *vl_sift_get_octave(VlSiftFilt *f, int s)

    # ------------------------------------------------------------------
    # ** @brief Get keypoints.
    # ** @param f SIFT filter.
    # ** @return pointer to the keypoints list.
    # **/
    inline  const VlSiftKeypoint *vl_sift_get_keypoints(VlSiftFilt *f)
    # @} */

    # @name Set parameters
    # ** @{
    # **/
    # ------------------------------------------------------------------
    # ** @brief Set peaks threshold
    # ** @param f SIFT filter.
    # ** @param t threshold.
    # **/

    inline void vl_sift_set_peak_thresh(VlSiftFilt *f, double t)

    # ------------------------------------------------------------------
    # ** @brief Set edges threshold
    # ** @param f SIFT filter.
    # ** @param t threshold.
    # **/
    inline void vl_sift_set_edge_thresh(VlSiftFilt *f, double t)

    # ------------------------------------------------------------------
    # ** @brief Set norm threshold
    # ** @param f SIFT filter.
    # ** @param t threshold.
    # **/
    inline void vl_sift_set_norm_thresh(VlSiftFilt *f, double t)

    # ------------------------------------------------------------------
    # ** @brief Set the magnification factor
    # ** @param f SIFT filter.
    # ** @param m magnification factor.
    # **/
    inline void vl_sift_set_magnif(VlSiftFilt *f, double m)

    # ------------------------------------------------------------------
    # ** @brief Set the Gaussian window size
    # ** @param f SIFT filter.
    # ** @param x Gaussian window size(in units of spatial bin).
    # **
    # ** This is the parameter @f$ \hat \sigma_{\text{win}} @f$ of
    # ** the standard SIFT descriptor @ref sift-tech-descriptor-std.
    # **/
    inline void vl_sift_set_window_size(VlSiftFilt *f, double m)
    # @} */

