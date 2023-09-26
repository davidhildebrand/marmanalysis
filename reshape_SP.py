#!/usr/bin/env python3

import glob
import h5py
import json
#import matplotlib.pyplot as plt
#plt.rcParams['figure.dpi'] = 900
import numpy as np
import os
import skimage
import tifffile

raw_fn_must_contain = '_SP_'
raw_fn_sufx = '_00001'
raw_fn_ext = '.tif'
raw_data_dirs = [r'/Data/Cadbury/20230722d',
                 r'/Data/Cadbury/20230809d',
                 r'/Data/Dali/20230511d',
                 r'/Data/Dali/20230515d',
                 r'/Data/Dali/20230517d',
                 r'/Data/Dali/20230522d',
                 r'/Data/Dali/20230525d',
                 r'/Data/Dali/20230606d',
                 r'/Data/Dali/20230608d',
                 r'/Data/Dali/20230611d',
                 r'/Data/Dali/20230615d',
                 r'/Data/Dali/20230618d',
                 r'/Data/Dali/20230620d',
                 r'/Data/Dali/20230622d',
                 r'/Data/Dali/20230627d',
                 r'/Data/Dali/20230629d',
                 r'/Data/Dali/20230704d',
                 r'/Data/Dali/20230727d',
                 r'/Data/Dali/20230804d',
                 r'/Data/Dali/20230810d'] # list containing ≥1 dirs
exclude_h5 = '_depth000um_'

path_input_files = []
for i_dir in raw_data_dirs:
    tmp_paths = sorted(glob.glob(i_dir + os.path.sep + '**' + os.path.sep + '*' + raw_fn_sufx + raw_fn_ext, recursive=True))
    for this_tmp_path in tmp_paths:
        if not raw_fn_must_contain in this_tmp_path:
            continue
        path_input_files.append(this_tmp_path)

#%% Iterate over files
for path_input_file in path_input_files:
    print('Processing {}'.format(path_input_file))

    #%% Get MROI info from tif metadata
    with tifffile.TiffFile(path_input_file) as tif:
        metadata = {}
        for tag in tif.pages[0].tags.values():
            tag_name, tag_value = tag.name, tag.value
            metadata[tag_name] = tag_value
            
    mrois_si_raw = json.loads(metadata["Artist"])['RoiGroups']['imagingRoiGroup']['rois']
    if type(mrois_si_raw) != dict:
        mrois_si = []
        for roi in mrois_si_raw:
            if type(roi['scanfields']) != list:
                scanfield = roi['scanfields']
            else: 
                scanfield = roi['scanfields'][np.where(np.array(roi['zs']) == 0)[0][0]]
            roi_dict = {}
            roi_dict['center'] = np.array(scanfield['centerXY'])
            roi_dict['sizeXY'] = np.array(scanfield['sizeXY'])
            roi_dict['pixXY'] = np.array(scanfield['pixelResolutionXY'])
            mrois_si.append(roi_dict)
    else:
        scanfield = mrois_si_raw['scanfields']
        roi_dict = {}
        roi_dict['center'] = np.array(scanfield['centerXY'])
        roi_dict['sizeXY'] = np.array(scanfield['sizeXY'])
        roi_dict['pixXY'] = np.array(scanfield['pixelResolutionXY'])
        mrois_si = [roi_dict]
    
    # Sort MROIs so they go from left-to-right (but keep the un-sorted 
    # because that matches how they were acquired and saved in the long-tif-strip)
    mrois_centers_si = np.array([mroi_si['center'] for mroi_si in mrois_si])
    x_sorted = np.argsort(mrois_centers_si[:,0])
    mrois_si_sorted_x = [mrois_si[i] for i in x_sorted]
    mrois_centers_si_sorted_x = [mrois_centers_si[i] for i in x_sorted]
        
    #%% Load image data
    tiff_file = tifffile.imread(path_input_file)
    # dim0 = time, dim1 = fast-resonant-scanning, dim2 = slow-galvo-scanning
    tiff_file = np.swapaxes(tiff_file, 1, 2)
    
    #%% Separate long single-stripe frame into MROIs
    # Get the Y coordinates for mrois (and not flybacks)

    n_mrois = len(mrois_si)
    tif_pixels_Y = tiff_file.shape[2]
    mrois_pixels_Y = np.array([mroi_si['pixXY'][1] for mroi_si in mrois_si])
    each_flyback_pixels_Y = (tif_pixels_Y - mrois_pixels_Y.sum()) // (n_mrois - 1)

    # Divide long stripe into mrois
    planes_mrois = np.empty((n_mrois), dtype=np.ndarray)

    y_start = 0
    for i_mroi in range(n_mrois): # We go over the order in which they were acquired
        planes_mrois[i_mroi] = tiff_file[:, :, y_start:y_start+mrois_pixels_Y[x_sorted[i_mroi]]]
        y_start += mrois_pixels_Y[i_mroi] + each_flyback_pixels_Y

    #%% Get location of MROIs in final canvas based on MROI metadata

    sizes_mrois_pix = np.array([mroi_pix.shape[1:] for mroi_pix in planes_mrois])
    sizes_mrois_si = np.array([mroi_si['sizeXY'] for mroi_si in mrois_si_sorted_x])
    pixel_sizes = sizes_mrois_si / sizes_mrois_pix
    psize_x, psize_y = np.mean(pixel_sizes[:,0]), np.mean(pixel_sizes[:,1])
    assert np.product(np.isclose(pixel_sizes[:,1] - psize_y, 0)), "Y-pixels resolution not uniform across MROIs"
    assert np.product(np.isclose(pixel_sizes[:,0] - psize_x, 0)), "X-pixels resolution not uniform across MROIs"
    
    # Calculate the pixel ranges (with their SI locations) that would fit all MROIs
    top_left_corners_si = mrois_centers_si_sorted_x - sizes_mrois_si / 2
    bottom_right_corners_si = mrois_centers_si_sorted_x + sizes_mrois_si / 2
    xmin_si, ymin_si = top_left_corners_si[:,0].min(), top_left_corners_si[:,1].min()
    xmax_si, ymax_si = bottom_right_corners_si[:,0].max(), bottom_right_corners_si[:,1].max()
    reconstructed_xy_ranges_si = [np.arange(xmin_si, xmax_si, psize_x), np.arange(ymin_si, ymax_si, psize_y)]
    
    # Calculate the starting pixel for each MROI when placed in the reconstructed movie
    top_left_corners_pix = np.empty((n_mrois, 2), dtype=int)
    for i_xy in range(2):
        for i_mroi in range(n_mrois):
            closest_xy_pix = np.argmin(np.abs(reconstructed_xy_ranges_si[i_xy] - top_left_corners_si[i_mroi, i_xy]))
            top_left_corners_pix[i_mroi,i_xy] = int(closest_xy_pix)
            closest_xy_si = reconstructed_xy_ranges_si[i_xy][closest_xy_pix]
            if not np.isclose(closest_xy_si, top_left_corners_si[i_mroi, i_xy]):
                print("ROI %d x does not fit perfectly into image, corner is %.4f but closest available is %.4f" %\
                      (i_mroi, closest_xy_si, top_left_corners_si[i_mroi, i_xy]))
    #Sometimes an extra pixel is added because of pixel_size rounding
    for i_xy in range(2):
        if len(reconstructed_xy_ranges_si[i_xy]) == np.sum(sizes_mrois_pix[:,0]) + 1: 
            reconstructed_xy_ranges_si[i_xy] = reconstructed_xy_ranges_si[i_xy][:-1]
      
    
    #%% Place MROIs in canvas
    
    n_f = planes_mrois[0].shape[0] 

    plane_width = len(reconstructed_xy_ranges_si[0]) 
    plane_length = len(reconstructed_xy_ranges_si[1])
    plane_canvas = np.zeros((n_f, plane_width, plane_length), dtype = np.int16)
    for i_mroi in range(n_mrois):
        x_start_canvas = top_left_corners_pix[i_mroi][0]
        x_end_canvas = x_start_canvas + sizes_mrois_pix[i_mroi][0]
        
        y_start_canvas = top_left_corners_pix[i_mroi][1]
        y_end_canvas = y_start_canvas + sizes_mrois_pix[i_mroi][1]
        
        plane_canvas[:, x_start_canvas:x_end_canvas, y_start_canvas:y_end_canvas] = planes_mrois[i_mroi]
        
    #%% Saving outputs
    #plt.imshow(np.mean(plane_canvas, axis = 0))
    #plt.show()
    
    # Swap axes, otherwise it will look rotated. This makes it: dim0 = time, dim1 = slow-galvo-scanning, 
    # dim2 = fast-resonant-scanning
    plane_canvas = np.swapaxes(plane_canvas, 1, 2)
    plane_canvas_mean = np.mean(plane_canvas, axis=0)
    plane_canvas_mean_doub = skimage.exposure.rescale_intensity(plane_canvas_mean, out_range=(0, 1))
    sw_low, sw_high = np.percentile(plane_canvas_mean_doub, (1.0, 99.0))
    plane_canvas_mean_rescale = skimage.exposure.rescale_intensity(plane_canvas_mean_doub, in_range=(sw_low, sw_high))
    plane_disp = skimage.util.img_as_ubyte(plane_canvas_mean_rescale)
    
    save_path = os.path.dirname(path_input_file)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if exclude_h5 not in os.path.basename(path_input_file):
        save_fn_h5 = os.path.splitext(os.path.basename(path_input_file))[0] + '_preprocessed.h5'
        save_path_h5 = save_path + os.path.sep + save_fn_h5
        h5file = h5py.File(save_path_h5, 'w') 
        h5file.create_dataset('data', data=plane_canvas)
        h5file.attrs.create('metadata', str(metadata)) # can load as json into dict
        h5file.close()
        del h5file
    else:
        print('Not saving as hdf5, contains \'' + exclude_h5 + '\': {}'.format(os.path.basename(path_input_file)))
    save_fn_tif = os.path.splitext(os.path.basename(path_input_file))[0] + '_preprocessed.tif'
    save_path_tif = save_path + os.path.sep + save_fn_tif
    tifffile.imwrite(save_path_tif, plane_canvas, metadata=metadata)
    save_fn_meantif = os.path.splitext(os.path.basename(path_input_file))[0] + '_preprocessed_meandisp.tif'
    save_path_meantif = save_path + os.path.sep + save_fn_meantif
    tifffile.imwrite(save_path_meantif, plane_disp)
    #save_fn_meanpng = os.path.splitext(os.path.basename(path_input_file))[0] + '_preprocessed_mean.png'
    #save_path_meanpng = save_path + os.path.sep + save_fn_meanpng
 
