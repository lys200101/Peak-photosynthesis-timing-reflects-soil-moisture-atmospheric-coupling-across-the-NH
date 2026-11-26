
import os
import glob
import datetime
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from fontTools.ttLib.tables.otTables import DeltaSetIndexMap
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from joblib import parallel_backend
from pandas.core.methods.selectn import SelectNSeries
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import matplotlib as mpl
from matplotlib import colormaps
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import linregress
from scipy import stats
from scipy.stats import mode
import pingouin as pg
from scipy.stats import theilslopes
import pymannkendall as mk
from brokenaxes import brokenaxes
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches


######### Function ##########

### Extract dates from the last 8 digits of each TIF filename
def extract_date_from_filename(filename):
    # Extract the filename without path or extension
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # Extract the last 8 characters
    date_str = filename_without_ext[-8:]

    # Validate date format
    if not date_str.isdigit() or len(date_str) != 8:
        raise ValueError(f"The last 8 characters of {filename} are not a valid date (YYYYMMDD required).")

    # Convert to a datetime object
    return datetime.datetime.strptime(date_str, "%Y%m%d")


def clip_by_latitude(gt, rows, cols, lat_min, lat_max):
    """
    Crop data by latitude range.
    Returns: (row_start, row_end, new_gt)
    """
    # Calculate the center latitude of each row
    row_centers = np.arange(rows) * gt[5] + gt[3] + gt[5] / 2

    # Find rows with latitudes between 30 and 90 degrees
    valid_rows = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(valid_rows):
        raise ValueError(f"No valid data found in latitude range {lat_min}-{lat_max}")

    # Find the first and last valid rows
    valid_row_indices = np.where(valid_rows)[0]
    row_start = valid_row_indices[0]
    row_end = valid_row_indices[-1] + 1  # The slice includes the start and excludes the end

    # Calculate the new upper-left coordinates
    new_top_left_x = gt[0] + row_start * gt[2]  # Usually gt[2]=0
    new_top_left_y = gt[3] + row_start * gt[5]  # gt[5] is the pixel height

    # Create the new geotransform
    new_gt = (
        new_top_left_x,
        gt[1],  # Pixel width remains unchanged
        gt[2],  # Row rotation remains unchanged
        new_top_left_y,
        gt[4],  # Column rotation remains unchanged
        gt[5]  # Pixel height remains unchanged
    )

    # print(f"Latitude range: {lat_min}-{lat_max}°N")
    # print(f"Corresponding row range: {row_start} - {row_end - 1}")
    # print(f"Original rows: {rows}, Clipped rows: {row_end - row_start}")
    # print(f"Original upper-left latitude: {gt[3]:.2f}°N, New upper-left latitude: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt


### Read SM and VPD bands
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    stack.append(data)


def cal_pixel_timelength_mean(i, j, data):

    # if len(time_series_clean) > (years_length - 3):
    if len(np.isfinite(data)) > 1:
        result = np.nanmean(data)
        # print(f'pheno mean={result}')
        return (i, j, result)

    else:
        return (i, j, np.nan)


def calculate_senSlope(data, i, j):

    mask = np.isfinite(data)

    data_clean = data[mask]

    years_valid = yearsList[mask]

    if (len(data_clean) >= (years_length/2)) and (len(data_clean) <= years_length):   ### >10 / >17/ >19

        # Sen slope
        result = theilslopes(data_clean, years_valid)

        slope = result.slope
        intercept = result.intercept

        # Mann-Kendall
        mk_result = mk.original_test(data_clean)

        pvalue = mk_result.p

        # result = mk.original_test(data_clean, alpha=0.05)
        # slope = round(result.slope, 4)
        # pvalue = round(result.p, 2)

    else:
        slope = np.nan
        intercept = np.nan
        pvalue = np.nan

        # print(f'slope:{slope}, p:{pvalue}')
    return (i, j, slope, intercept, pvalue)



### Extract data within the preseason growing season

def save_tif_gdal(output_path, data, crs, transform):
    """Save a TIFF file with its data dimensions and geotransform"""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols, rows, 1, gdal.GDT_Float32
    )
    if not output_ds:
        raise RuntimeError(f"Cannot create output file: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # Set NaN as the NoData value

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # Use the adjusted geotransform
    output_ds = None
    return True




# ####################################### 1 Input and output ######################################
# ##################### 1.1 Input ##########################
# #### Input SM and VPD TIF files
# star_year = 2001
# end_year = 2024
#
# sos = 'MOS'
# pos = 'POS'
#
# filter_condition = '5'  ###'5':Exclude the day and following day when Pre > 0.001 m
#
# n_days =7 #####When filter_condition = '5', adjust precipitation effects on SM
#
# coupling_method = 'Pearson'  ### 'Partial'/'Pearson'
# lag_day = 0
#
# partial_for = 'SM_VPDlag'  ###SM and lagged VPD: SM_VPDlag; SM and VPD: SMlag_VPDlag
#
# # test_number = f'20_{lag_day}'  ###Modify carefully here!!!!!!
# test_number = '17_8_3'  ###Modify carefully here!!!!!!
#
# ########################Check and revise the section above carefully########################
# years_length = end_year - star_year + 1
# print('years_length:', years_length)
# years = range(star_year, end_year + 1)
#
# input_path = 'D:/code/ERA5_Land_NH_55km_daily_deseason_detrend'
#
# folder1 = fr'{input_path}\ERA5_Land_NH_55km_daily_0-100cmSM(2001-2024)'  ### Modify carefully here   SM:0-100cmSM  Ta
# folder2 = fr'{input_path}\ERA5_Land_NH_55km_daily_VPD(2001-2024)'  ### Modify carefully here   VPD           Pre
#
# pos_folder = fr'D:/code/{pos}_55km'
# sos_folder = fr'D:/code/{sos}_55km'
# # sos_folder = 'no'
#
# veg_type_tif = r'D:/code/Veg_type/NH_veg_type_55km(Python).tif'
#
# climate_type_tif = r'D:/code/KB_climate/NH30_84_KB(simple)_climate_region_55km.tif'
#
# ai_tif = r'D:/code/AI/NH30_84_AI(graident)_55km.tif'
#
# ###### Data used for filtering
# pre_origin_folder = r'D:/code/ERA5_Land_NH_55km_daily_Pre'
#
# min_lat = 30
# max_lat = 84
#
# ###################### ===================== Output settings ======================== ########################
# ouput_path = 'D:/code/Climate'
#
# output_cor_tif_path = fr'{ouput_path}\Correlation(SM_VPD_pearson){test_number}'
#
# output_cor_mean_slope_png_path = fr'D:/code/3Cor_mean_slope'
# output_cor_mean_slope_tif_path = fr'D:/code/3Cor_mean_slope'
#
#
# ####################################
# tif_files1 = sorted(glob.glob(os.path.join(folder1, '*.tif')))
# tif_files2 = sorted(glob.glob(os.path.join(folder2, '*.tif')))
#
# pos_tif_files = sorted(glob.glob(os.path.join(pos_folder, '*.tif')))
#
# sos_tif_files = sorted(glob.glob(os.path.join(sos_folder, '*.tif')))
#
# pre_origin_files = sorted(glob.glob(os.path.join(pre_origin_folder, '*.tif')))
#
# print('files done!')
#
# if not tif_files1:
#     raise FileNotFoundError("No tif_files1 TIF files found!")
#     raise FileNotFoundError("No tif_files1 TIF files found!")
# if not tif_files2:
#     raise FileNotFoundError("No tif_files2 TIF files found!")
# if not pos_tif_files:
#     raise FileNotFoundError("No POS TIF files found!")
#
#
# ####################### 2 Extract TIF metadata ############################
# first_tif = tif_files1[0]
# sample_tif = gdal.Open(first_tif)
#
# if sample_tif is None:
#     raise RuntimeError(f"Cannot open TIF file: {sample_tif} (unsupported driver or corrupted file)")
#
# # Get geotransform, projection and pixel size
# #Coordinates and projection: the data's spatial reference system
# crs = sample_tif.GetProjectionRef()          # Get the input CRS automatically
# # print('crs:', crs)
# gt = sample_tif.GetGeoTransform()  #Geotransform from pixel coordinates to geographic coordinates.
# proj = sample_tif.GetProjection()  #Projected coordinates: x and y (m)
#
# #pixel
# pixel_width = gt[1]
# pixel_height = gt[5]
#
# top_left_x = gt[0]
# top_left_y = gt[3]
#
# #Rows and columns
# sample_tif = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
#
# rows = sample_tif.shape[0]
# cols = sample_tif.shape[1]
# print('rows:', rows, 'cols:', cols)
#
# row_indices = np.repeat(np.arange(rows), cols)  # Repeat each row index cols times
# col_indices = np.tile(np.arange(cols), rows)  # Tile column indices rows times
#
# # Calculate coordinate bounds with negative pixel height
# lon_min = top_left_x
# lon_max = top_left_x + cols * pixel_width  # Right longitude bound
# lat_min = top_left_y + rows * pixel_height  # Lower latitude bound (southernmost)
# lat_max = top_left_y  # Upper latitude bound (northernmost)
# print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
# print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")
#
#
# ################ For clipping ########################
# row_start, row_end, new_gt = clip_by_latitude(gt, rows, cols, lat_min=min_lat, lat_max=max_lat)
#
# # Calculate clipped dimensions
# new_rows = row_end - row_start
# new_cols = cols  # Column count unchanged
#
# row_indices = np.repeat(np.arange(new_rows), cols)  # Repeat each row index cols times
# col_indices = np.tile(np.arange(cols), new_rows)  # Tile column indices rows times
# rows = new_rows
# gt = new_gt
#
# print(f"After clipping: rows={new_rows}, cols={new_cols}")
# # Calculate coordinate bounds with negative pixel height
# lon_min = new_gt[0]
# lon_max = new_gt[0] + new_gt[1] * cols    # Right longitude bound
# lat_min = new_gt[3] + new_gt[5] * rows # Lower latitude bound (southernmost)
# lat_max = new_gt[3]  # Upper latitude bound (northernmost)
# print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
# print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")
#
# ############################################ 3 Time-Stacking ###################################################
# ## Time
# tif_dates = []
# for tif_file in tif_files1:
#     date = extract_date_from_filename(tif_file)
#     tif_dates.append(date)
#
# dates = pd.to_datetime(tif_dates)
# print('First five dates: ', [d.strftime('%Y%m%d') for d in dates[:5]])
#
#
# print('All stack start!')
#
# ## Stack data
# sm_stack = []
# vpd_stack = []
#
# pos_stack = []
# sos_stack = []
#
# pre_origin_stack = []
#
# for tif_file in tif_files1:
#     get_band(tif_file, sm_stack)
#
# for tif_file in tif_files2:
#     get_band(tif_file, vpd_stack)
#
# for tif_file in pos_tif_files:
#     get_band(tif_file, pos_stack)
#
# for tif_file in sos_tif_files:
#     get_band(tif_file, sos_stack)
#
# for tif_file in pre_origin_files:
#     get_band_clip(tif_file, pre_origin_stack, row_start, row_end)
#
# sm_stack = np.stack(sm_stack, axis=0)#[:, 505:510, 505:510]
# # print('sm_stack shape:\n', sm_stack.shape)
# # print('sm_stack:\n', sm_stack[:,90:100,90:100])
# vpd_stack = np.stack(vpd_stack, axis=0)#[:, 505:510, 505:510]
#
# pos_stack = np.stack(pos_stack, axis=0)#[:, 505:510, 505:510]
#
# sos_stack = np.stack(sos_stack, axis=0)#[:, 505:510, 505:510]
# print('pos_stack shape:\n', pos_stack.shape)
#
# pre_origin_stack = np.stack(pre_origin_stack, axis=0)
#
#
# print('All stack done!')
#
# ############# 4.2 Vegetation type ################
# veg_type_data = gdal.Open(veg_type_tif)
# veg_type_data = veg_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)
#
# # ############# 4.3 Climate zone ################
# climate_type_data = gdal.Open(climate_type_tif)
# climate_type = climate_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)
#
# # ############# 4.4 Climate zone ################
# ai_type_data = gdal.Open(ai_tif)
# ai_type_data = ai_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)
#
#
# ######################################### 4 Annual coupling effect ################################################
# cor_each_year = []
# p_each_year = []
#
# cor_matrix = np.full((years_length, rows, cols), np.nan)  # 3D array    , dtype=object
# p_matrix = np.full((years_length, rows, cols), np.nan)  # 3D array   , dtype=object
#
# for year in years:
#     print(f"Processing year: {year}")
#
#     k = year - 2001
#
#     year_mask = (dates >= f"{year}-01-01") & (dates <= f"{year}-12-31")
#
#     ## Use sequential indexing
#     year_idx = np.where(year_mask)[0]
#     print('year_idx:\n', year_idx)
#     # Slice dates and data for the year without reset_index
#     year_dates = dates[year_idx]
#     sm_stack_year = sm_stack[year_idx, :, :]
#     vpd_stack_year = vpd_stack[year_idx, :, :]
#
#     pre_origin_stack_year = pre_origin_stack[year_idx, :, :]
#
#     pos_year = pos_stack[k, :, :]  # pos_stack is indexed by year
#
#     sos_year = sos_stack[k, :, :]  # pos_stack is indexed by year
#
#     ############ Parallel processing
#     print(f'year={year}pixel Pearson cor and mean calculate start')
#     # Process all pixels in parallel using all CPU cores
#
#     with parallel_backend("threading", n_jobs=18):
#         results = Parallel()(
#             delayed(process_pixel_sm_vpd_coupling)(
#                 i, j, year, pos_year, sos_year, year_dates,
#                 sm_stack_year, vpd_stack_year,
#                 pre_origin_stack_year
#             )
#             for i, j in zip(row_indices, col_indices)
#         )
#
#     print(f'year={year}pixel Pearson cor, sm mean and vpd mean calculate end')
#
#     for i, j, r, p in results:
#         cor_matrix[k, i, j] = r
#         p_matrix[k, i, j] = p
#

# ########################## 5 Cor mean and slope ###############################
## Use this section for 11 km; use the preceding section for 55 km
##########================ 11km Filter Cor and calculate its mean ==================###########

cal_done = 'yes'  # 'yes' means already calculated; 'no' means not yet calculated

data_type = 'MODIS_GPP_8d' # GIMMS_NDVI_16d / MODIS_GPP_16d / MODIS_GPP_8d / MODIS_EVI2_8d / MCD12Q2

mean_or_yearPOS = ''  #  '' uses annual POS; '_meanPOS' uses long-term mean POS

scale = 11

coupling_method = 'Pearson'  ### 'Pearson'/'Partial'
lag_day = 0

partial_for = 'SMlag_VPDlag'  ###SM and lagged VPD: SM_VPDlag; SM and VPD: SMlag_VPDlag

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno uses detrended SOS and POS for partial correlation
                                   ### Based_on_OriginPheno uses original SOS and POS for partial correlation

veg_threshold = 25

##### Path for the optimal time window
if coupling_method == 'Pearson':
    input_path2 = Rf'D:/code/Cor_preseason_mean'

elif coupling_method == 'Partial' and partial_for == 'SM_VPDlag':
    input_path2 = Rf'D:/code/Cor_preseason_mean'

elif coupling_method == 'Partial' and partial_for == 'SMlag_VPDlag':
    input_path2 = Rf'D:/code/Cor_preseason_mean'

###### Path for the SOS window
if coupling_method == 'Pearson':
    input_path2 = fr'D:/code/60days'

elif coupling_method == 'Partial' and partial_for == 'SM_VPDlag':
    input_path2 = fr'D:/code/60days'

elif coupling_method == 'Partial' and partial_for == 'SMlag_VPDlag':
    input_path2 = fr'D:/code/60days'

print(rf'input_path2, {input_path2}')
cor_files = sorted(glob.glob(os.path.join(input_path2, '*.tif')))

start_year = 2001
end_year = 2024
years = range(start_year, end_year + 1)
years_length = end_year - start_year +1

min_lat = 30
max_lat = 84

output_cor_mean_slope_tif_path = rf'D:/code/3Cor_mean_slope'
# output_cor_mean_slope_tif_path = rf'D:/code/3Cor_mean_slope'

####################### 2 Extract TIF metadata ############################
first_tif = cor_files[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"Cannot open TIF file: {sample_tif} (unsupported driver or corrupted file)")

# Get geotransform, projection and pixel size
#Coordinates and projection: the data's spatial reference system
crs = sample_tif.GetProjectionRef()          # Get the input CRS automatically
# print('crs:', crs)
gt = sample_tif.GetGeoTransform()  #Geotransform from pixel coordinates to geographic coordinates.
proj = sample_tif.GetProjection()  #Projected coordinates: x and y (m)

#pixel
pixel_width = gt[1]
pixel_height = gt[5]

top_left_x = gt[0]
top_left_y = gt[3]

#Rows and columns
sample_tif = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

rows = sample_tif.shape[0]
cols = sample_tif.shape[1]
print('rows:', rows, 'cols:', cols)

row_indices = np.repeat(np.arange(rows), cols)  # Repeat each row index cols times
col_indices = np.tile(np.arange(cols), rows)  # Tile column indices rows times

# Calculate coordinate bounds with negative pixel height
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width  # Right longitude bound
lat_min = top_left_y + rows * pixel_height  # Lower latitude bound (southernmost)
lat_max = top_left_y  # Upper latitude bound (northernmost)
print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")

if cal_done == 'no':
    cor_matrix = []

    for tif in cor_files:
        get_band(tif, cor_matrix)

    cor_matrix = np.stack(cor_matrix, axis=0)

    print('Stack done!')

    # ############################# The preceding section applies only to 11 km ###########################

    cor_mean = np.full((rows, cols), np.nan)
    cor_slope = np.full((rows, cols), np.nan)
    cor_intercept = np.full((rows, cols), np.nan)
    cor_slope_pvalue = np.full((rows, cols), np.nan)

    #### Selection criterion: at least 12 valid years
    cor_valid_mask = np.count_nonzero(np.isfinite(cor_matrix), axis=0) >= (years_length / 2)

    cor_matrix_valid = np.where(np.isfinite(cor_matrix) & cor_valid_mask, cor_matrix, np.nan)

    def Outlier_array_IQR(x, i, j, qmin, qmax):
        """
        Compute average of the 25-th to 75-th percentile of the data across specified zonal.
        Spatial statistics after the removal of outliers by quantiles
        https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414
        """
        if not type(x) is np.ndarray:
            x = np.asarray(x, dtype=np.float32)
        # x1 = x1[~np.isnan(x)]
        x_flatten = x[np.isfinite(x)]
        if len(x_flatten) < time_lengths/2:
            # Return all NaN if no valid data are available
            return np.full_like(x, np.nan), i, j
        else:
            # remove Outliers
            upper_quartile, lower_quartile = np.percentile(x_flatten, [qmax, qmin])
            IQR = (upper_quartile - lower_quartile)

            # from scipy.stats import iqr
            # x1= x.copy()
            # x1= np.where(x!=fillvalue, x1, np.nan)
            # IQR = iqr(x1, nan_policy='omit')

            lower_range = lower_quartile - (1.5 * IQR)
            upper_range = upper_quartile + (1.5 * IQR)

            # maxv = np.max(x_flatten)
            # minv = np.min(x_flatten)
            valid_mask = np.logical_and(x <= upper_range, x >= lower_range)
            x_masked = np.where(valid_mask, x, np.nan)

            if np.count_nonzero(np.isfinite(x_masked)) >= (time_lengths/2):
                return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
            else:
                print(f'IQR threshold：{lower_range:.4f} ~ {upper_range:.4f}\n'
                      f'Original pixel data: {x}\n'
                      f'Remove outliers followed by more than 3 invalid values: {x_masked}')
                return np.full_like(x, np.nan), i, j


    time_lengths = cor_matrix_valid.shape[0]

    cor_matrix_valid_2 = np.full((time_lengths, rows, cols), np.nan)

    ### IQRRemove outliers
    results = Parallel(n_jobs=18, verbose=10)(
        delayed(Outlier_array_IQR)(
            cor_matrix_valid[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        cor_matrix_valid_2[:, i, j] = data_mask

    cor_matrix_valid = cor_matrix_valid_2

    # p_matrix_valid = np.where(np.isfinite(p_matrix) & cor_valid_mask, p_matrix, np.nan)
    #
    # ###Export valid pixels only
    # for year in years:
    #     print(f"Exporting year: {year}")
    #     k = year - 2001  # Year index
    #
    #     output_path1 = os.path.join(output_cor_tif_path, f"Cor_pearson_{year}.tif")
    #     save_tif_gdal(
    #         output_path1,
    #         cor_matrix_valid[k, :, :],
    #         crs, gt
    #     )
    #
    #     # output_path1_1 = os.path.join(output_cor_tif_path, rf"Pvalue\Cor_pearson_pvalue_{year}.tif")
    #     # save_tif_gdal(
    #     #     output_path1_1,
    #     #     p_matrix_valid[k, :, :],
    #     #     crs, gt
    #     # )
    #     # print(f"{year} annual coupling, SM and VPD results saved")
    #
    #

    cor_mean = np.nanmean(cor_matrix_valid, axis=0)
    # print('cor_mean[50:55, 500:505]:', cor_mean[50:55, 500:505])
    # print(f'cor_mean max:{np.nanmax(cor_mean):.2f}, cor_mean min:{np.nanmin(cor_mean):.2f}')

    # ### ====== Save Cor mean for SOS–POS only ========== ###
    # tif_output_path = os.path.join(output_cor_mean_slope_tif_path, rf'mean\Cor({coupling_method}_{partial_for}_lag{lag_day})_mean_{scale}km_All{mean_or_yearPOS}.tif')
    #
    # save_tif_gdal(
    #     tif_output_path,
    #     cor_mean,
    #     crs,
    #     gt  # Use the new geotransform
    # )
    # #
    # # sys.exit()
    #
    ## Calculate Sen'slope
    yearsList = np.arange(start_year, start_year + years_length)
    # with parallel_backend('threading', n_jobs=18):
    results = Parallel(n_jobs=18)(
        delayed(calculate_senSlope)(  ##Calculate the mean
            cor_matrix_valid[:, i, j], i, j,
        )
        for i, j in zip(row_indices, col_indices)
    )
    for i, j, slope, intercept, pvalue in results:
        cor_slope[i, j] = slope
        cor_intercept[i, j] = intercept
        cor_slope_pvalue[i, j] = pvalue


    # ####### Detrend coupling for the POS-60 window
    # for t, year in enumerate(years):
    #     cor_trend = cor_slope * year + cor_intercept
    #
    #     cor_detrend = cor_matrix[t, :, :] - cor_trend
    #
    #     output_path = rf'{input_cor_tif_path}\Cor_Pearson_SM_t_VPD_t{mean_or_yearPOS}_detrend\Cor_detrend_{scale}000m_{year}.tif'
    #     save_tif_gdal(
    #         output_path,
    #         cor_detrend,
    #         crs,
    #         gt  # Use the new geotransform
    #     )
    # #
    # # sys.exit()

    # tif_outpt_path1 = os.path.join(output_cor_mean_slope_tif_path, rf'slope\SM_VPD_Cor_{coupling_method}_{partial_for}(lag={lag_day})_{test_number}\Cor_slope_{scale}km_All.tif')
    # tif_outpuut_path2 = os.path.join(output_cor_mean_slope_tif_path, rf'pvalue\SM_VPD_Cor_{coupling_method}_{partial_for}(lag={lag_day})_{test_number}\Cor_pvalue_{scale}km_All.tif')
    tif_output_path1 = os.path.join(output_cor_mean_slope_tif_path, rf'slope\Cor_Perseason_slope_{scale}km_All{mean_or_yearPOS}.tif')
    tif_output_path2 = os.path.join(output_cor_mean_slope_tif_path, rf'pvalue\Cor_Perseason_pvalue_{scale}km_All{mean_or_yearPOS}.tif')

    save_tif_gdal(
        tif_output_path1,
        cor_slope,
        crs,
        gt  # Use the new geotransform
    )
    save_tif_gdal(
        tif_output_path2,
        cor_slope_pvalue,
        crs,
        gt  # Use the new geotransform
    )


    # sys.exit()


elif cal_done == 'yes':
    cor_mean_file = os.path.join(output_cor_mean_slope_tif_path, rf'mean\Cor({coupling_method}_{partial_for}_lag{lag_day})_mean_{scale}km_All{mean_or_yearPOS}.tif')
    cor_slope_file = os.path.join(output_cor_mean_slope_tif_path, rf'slope\Cor_Perseason_slope_{scale}km_All{mean_or_yearPOS}.tif')
    cor_slope_pvalue_file = os.path.join(output_cor_mean_slope_tif_path, rf'pvalue\Cor_Perseason_pvalue_{scale}km_All{mean_or_yearPOS}.tif')

    cor_mean = gdal.Open(cor_mean_file)
    cor_mean = cor_mean.GetRasterBand(1).ReadAsArray().astype(np.float32)

    cor_slope = gdal.Open(cor_slope_file)
    cor_slope = cor_slope.GetRasterBand(1).ReadAsArray().astype(np.float32)

    cor_slope_pvalue = gdal.Open(cor_slope_pvalue_file)
    cor_slope_pvalue = cor_slope_pvalue.GetRasterBand(1).ReadAsArray().astype(np.float32)

##################################### 6 Plot #########################################
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec

scale = 11

path = rf'D:/code/Global_test2_{scale}000m'
#
# cor_mean_path = rf'{path}\Tif_of_fig\3Cor_mean_slope\mean\SM_VPD_Cor_{coupling_method}_{partial_for}(lag={lag_day})_{test_number}\Cor_mean_{scale}km_All.tif'
# cor_slope_path = rf'{path}\Tif_of_fig\3Cor_mean_slope\slope\SM_VPD_Cor_{coupling_method}_{partial_for}(lag={lag_day})_{test_number}\Cor_slope_{scale}km_All.tif'
# cor_slope_pvalue_path = rf'{path}\Tif_of_fig\3Cor_mean_slope\pvalue\SM_VPD_Cor_{coupling_method}_{partial_for}(lag={lag_day})_{test_number}\Cor_pvalue_{scale}km_All.tif'

veg_type_tif = rf'{path}\Data\Classify\Veg_type\NH_permanent_veg_type_{scale}km_gte{veg_threshold}.tif'

ai_tif = rf'{path}\Data\Classify\AI\NH30_84_AI(graident)_{scale}km.tif'

mcd12q2_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
mcd12q2_mask_tif = gdal.Open(mcd12q2_mask_tif)
mcd12q2_mask = mcd12q2_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
mcd12q2_mask = np.isfinite(mcd12q2_mask)

gpp_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
gpp_mask_tif = gdal.Open(gpp_mask_tif)
gpp_mask = gpp_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
gpp_mask = np.isfinite(gpp_mask)

save_png_path = rf'{path}\Fig\Fig 3 SM-VPD coupling mean and trend'



############# 3.2 Vegetation type ################
veg_type_data = gdal.Open(veg_type_tif)
band = veg_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

veg_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    veg_type_data[veg_type_data == nodata] = np.nan

# # ############# 3.3 Climate zone ################
# climate_type_data = gdal.Open(climate_type_tif)
# climate_type = climate_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)

############### 3.4 AI index #################
ai_type_data = gdal.Open(ai_tif)
band = ai_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

ai_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    ai_type_data[ai_type_data == nodata] = np.nan


# sample_tif = gdal.Open(cor_files[0])
#
#
# if sample_tif is None:
#     raise RuntimeError(f"Cannot open TIF file: {sample_tif} (unsupported driver or corrupted file)")
#
# # Get geotransform, projection and pixel size
# #Coordinates and projection: the data's spatial reference system
# crs = sample_tif.GetProjectionRef()          # Get the input CRS automatically
# # print('crs:', crs)
# gt = sample_tif.GetGeoTransform()  #Geotransform from pixel coordinates to geographic coordinates.
# proj = sample_tif.GetProjection()  #Projected coordinates: x and y (m)
#
# #pixel
# pixel_width = gt[1]
# pixel_height = gt[5]
#
# top_left_x = gt[0]
# top_left_y = gt[3]
#
# #Rows and columns
# sample_tif = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
#
# rows = sample_tif.shape[0]
# cols = sample_tif.shape[1]
# print('rows:', rows, 'cols:', cols)
#
# row_indices = np.repeat(np.arange(rows), cols)  # Repeat each row index cols times
# col_indices = np.tile(np.arange(cols), rows)  # Tile column indices rows times
#
# # Calculate coordinate bounds with negative pixel height
# lon_min = top_left_x
# lon_max = top_left_x + cols * pixel_width  # Right longitude bound
# lat_min = top_left_y + rows * pixel_height  # Lower latitude bound (southernmost)
# lat_max = top_left_y  # Upper latitude bound (northernmost)
# print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
# print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")




## a left
def plot_cor_mean_or_slope_and_pvalue_forAllvegType(plot_data, plot_data_pvalue,  colorbarmin, colorbarmax, data_type_mean_or_slope, name, ax):
    # # Create 5 subplots: 4 maps and a colorbar
    # fig = plt.figure(figsize=(6, 4))
    # gs = gridspec.GridSpec(1, 3,
    #                        width_ratios=[4, 1, 1],  #Width ratios for the three columns
    #                        hspace=0.5, wspace=0.2)

    if data_type_mean_or_slope == 'cor slope':
        plot_data = np.clip(plot_data, colorbarmin, colorbarmax)

        epsilon = 1e-6  # Adjust for the precision and units of data_slope

        # pheno_slope = pheno_slope[np.abs(pheno_slope) > epsilon]
        plot_data = np.where(np.abs(plot_data) > epsilon, plot_data, np.nan)

    fig = ax.get_figure()
    if name == 'All':
        ax2_width = 0.8
    else:
        ax2_width = 1
    gs_inner = ax.get_subplotspec().subgridspec(2, 2,
                                                width_ratios=[5, ax2_width],
                                                height_ratios=[5, 0.3],
                                                hspace=0.23, wspace=0.01)

    # Hide the parent ax, which is a placeholder
    ax.axis('off')

    plots = []  # Store the plot object for each subplot

    # ax1 = plt.subplot(gs[0, 0])
    # ax2 = plt.subplot(gs[0, 1])
    # ax3 = plt.subplot(gs[0, 2])
    # ax3 = plt.subplot(gs[0, 2])
    # Create the three inner axes
    ax1 = fig.add_subplot(gs_inner[0, 0])  # map
    ax2 = fig.add_subplot(gs_inner[0, 1])  # Latitude profile
    ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar spans both columns

    # word = 'a'

    ########### Subplot 1: spatial distribution #################
    ax1.set_box_aspect(1)  #Make the map axis square and fill the grid height
    ax1.axis('off')
    ### Create the map
    m = Basemap(ax=ax1,
                projection='npstere',   # North polar projection
                boundinglat=30,         # Minimum displayed latitude (currently 30°N)
                lon_0=0,                # Central longitude: 180 centers the Pacific, 90 centers Asia
                resolution='l')

    # Generate grid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max , lat_min , rows)
    lons, lats = np.meshgrid(lons, lats)

    # Set latitude and longitude ticks
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  #Make meridians converge at the North Pole
                    labels=[0, 0, 0, 0],  #labels=[left, right, top, bottom] Control longitude label visibility
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # Fill continents
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Hide boundary lines


    ### Plotdata
    # Color mapping
    if data_type_mean_or_slope == 'cor slope':
        color_list = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                      '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 11)
    elif data_type_mean_or_slope == 'cor mean':
        color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                      '#fcbba1', '#fee5d9', '#9ecae1']
        cmap = mpl.colors.ListedColormap(color_list)
        bins = np.linspace(colorbarmin, colorbarmax, 8)

    norm = mpl.colors.BoundaryNorm(bins, cmap.N, clip=True)  # clip=True includes colorbarmax


    plot = m.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Avoid tearing near the pole

    plots.append(plot)  # Save the plot object

    if data_type_mean_or_slope == 'cor slope':
        # Add significance markers
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # Convert geographic coordinates to projected coordinates
            sig_x, sig_y = m(sig_lons, sig_lats)

            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=0.05,
                       linewidth=0.1, zorder=2)
    ax1.set_frame_on(False)

    ### Plotboundaries
    terrence = r'D:/code/NH_terrence/NH_Terrence_30_90'
    # m.readshapefile(terrence, 'NH_Terrence', drawbounds=True, linewidth=0.2, color='black', zorder=3)
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        # Convert the list to a NumPy array for calculation
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]

        # Calculate projected distances between adjacent points
        # A large projected distance between adjacent points indicates a line wrapping across the map center
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        # Set a distance threshold in projected coordinates
        # Flag jumps larger than one tenth of the map diameter
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1

        # Find the indices of jumps
        break_indices = np.where(dist > threshold)[0]

        if len(break_indices) == 0:
            # Draw the full line if there are no jumps
            ax1.plot(x, y, color='black', linewidth=0.2, zorder=3)
        else:
            # Split the line at jumps and draw each segment
            # This removes lines crossing the center while retaining valid boundaries
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.2, zorder=3)
                start_idx = break_idx + 1
            # Draw the final segment
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.2, zorder=3)

    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### Clip to the outer boundary
    from matplotlib.patches import Circle

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()

    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    radius = (x1 - x0) / 2

    clip_circle = Circle(center, radius, transform=ax1.transData)

    for artist in ax1.collections + ax1.lines + ax1.patches:
        artist.set_clip_path(clip_circle)

    boundary_circle = Circle(
        center,
        radius,
        transform=ax1.transData,
        facecolor='none',
        edgecolor='black',  # color
        linewidth=0.8,
        clip_on = False,
        zorder=4  # Place on top
    )

    ax1.add_patch(boundary_circle)

    # if name == 'All':
    #     if data_type == 'cor mean':
    #         ax1.set_title(f'(a)', pad=10, fontweight='bold')
    #     elif data_type == 'cor slope':
    #         ax1.set_title(f'(c)', pad=10, fontweight='bold')
    # else:
    #     if name == 'Forest' or name == 'Arid':
    #         word = 'a'
    #     elif name == 'Shrub' or name == 'Semi-arid':
    #         word = 'b'
    #     elif name == 'Savanna' or name == 'Dry sub-humid':
    #         word = 'c'
    #     elif name == 'Grass' or name == 'Humid':
    #         word = 'd'
    #
    #     ax1.set_title(f'({word}) {name}', pad=10, fontweight='bold')


    ## Statistics ##
    data_gte0 = plot_data[(plot_data >= 0) & np.isfinite(plot_data)]
    data_lt0 = plot_data[(plot_data < 0) & np.isfinite(plot_data)]
    sum_count = np.sum(np.isfinite(plot_data))

    data_gte0_count = np.sum(np.isfinite(data_gte0))
    data_lt0_count = np.sum(np.isfinite(data_lt0))

    data_gte0_ratio = data_gte0_count/sum_count * 100
    data_lt0_ratio = data_lt0_count/sum_count * 100


    if data_type_mean_or_slope == 'cor mean':
        h = 0.25
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data):.2f}\n'
                 f'Pos = {np.nanmean(data_gte0):.2f} ({data_gte0_ratio:.1f}%)\n'
                 f'Neg = {np.nanmean(data_lt0):.2f} ({data_lt0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # Use relative coordinates for positioning
                 multialignment='center',  # Center vertically
                 fontsize = 6)
    elif data_type_mean_or_slope == 'cor slope':
        h = 0.25
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data):.3f}\n'
                 f'Pos = {np.nanmean(data_gte0):.3f} ({data_gte0_ratio:.1f}%)\n'
                 f'Neg = {np.nanmean(data_lt0):.3f} ({data_lt0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # Use relative coordinates for positioning
                 multialignment='center',  # Center vertically
                 fontsize = 6)

    ########### Subplot 2: variation by latitude

    # Use actual latitude values on the y-axis
    lat_centers = lats[:, 0]

    plot_data_lat = np.nanmean(plot_data, axis=1)

    if data_type_mean_or_slope == 'cor slope':
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    elif data_type_mean_or_slope == 'cor mean':
        ax2.axvline(x=-0.3, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    if data_type_mean_or_slope == 'cor slope':
        ax2.set_xlim(-0.015, 0.015)
        ax2.set_xticks(np.arange(-0.01, 0.011, 0.01))
        ax2.set_xticklabels(['-1', '0', '1'])  # Set labels manually

        tick_size = plt.rcParams['xtick.labelsize']
        ax2.text(
            0.98,  # x = Tick position in data coordinates
            -0.02,  # y = Slightly lower in axes coordinates
            r'$×10^{-2}$',  # Text to display
            transform=ax2.transAxes,
            ha='left',  # Align rightward to avoid compression
            va='top',
            fontsize=8,
            clip_on=False
        )
    elif data_type_mean_or_slope == 'cor mean' :
        ax2.set_xlim(-0.5, 0)
        ax2.set_xticks(np.arange(-0.3, 0.01, 0.3))
        ax2.set_xticklabels(['-0.3','0'])  # Set labels manually

    ax2.set_ylim(30, 90)
    ticks = np.arange(30, 91, 10)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(f'{x}°' for x in ticks)


    ax2.tick_params(axis='both', which='major', length = 2, pad=3)

    ########### Subplot 3：Colorbar
    ### Create the colorbar in the final position
    cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

    cbar.set_ticks(bins)

    if data_type_mean_or_slope == 'cor slope':
        cbar.set_label('SM-VPD coupling trend (a$^{-1}$)', labelpad = 13)
        cbar.set_ticklabels([ '0' if x == 0 else
            f'<{int((bins[0])*100)}' if x == colorbarmin else
            f'>{int((bins[-1])*100)}' if x == colorbarmax else
            f'{int(x*100)}' if x*100 == int(x*100) else
            f'{x*100:.1f}'
            for x in bins])
        tick_size = plt.rcParams['xtick.labelsize']
        ax3.text(
            0.9,  # x = Tick position in data coordinates
            -2.05,  # y = Slightly lower in axes coordinates
            r'$×10^{-2}$',  # Text to display
            transform=ax3.transAxes,
            ha='left',  # Align rightward to avoid compression
            va='top',
            fontsize=8,
            clip_on=False
        )
    if data_type_mean_or_slope == 'cor mean':
        cbar.set_label('SM-VPD coupling', labelpad = 13)
        cbar.set_ticklabels(['0' if x == 0 else
            f'{x:.1f}' for x in bins])

    ax3.tick_params(axis='both', length=2, pad=3)

    plt.tight_layout()

    # Current lower-left corner of ax1
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    pos3 = ax3.get_position()

    if name == 'All':
        # Reposition ax1
        ax1.set_position([
            pos1.x0 - 0.04,  # Keep the left edge unchanged
            pos2.y0,  # Align the bottom with ax2
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        if data_type_mean_or_slope == 'cor mean':
            if name == 'All':
                xpos = 0.105
            else:
                xpos = 0.145
            ax1.set_position([
                pos1.x0 - xpos,  # Keep the left edge unchanged
                pos2.y0,  # Align the bottom with ax2
                pos2.height,
                pos2.height
            ])  # [left, bottom, width, height]
        if data_type_mean_or_slope == 'cor slope':
            ax1.set_position([
                pos1.x0 - 0.145,  # Keep the left edge unchanged
                pos2.y0,  # Align the bottom with ax2
                pos2.height,
                pos2.height
            ])  # [left, bottom, width, height]

    # ax2.set_position([
    #     pos2.x0 + 0.055,
    #     pos2.y0,
    #     pos2.width,
    #     pos2.height
    # ])

    pos1_new = ax1.get_position()
    ax3.set_position([
        pos1_new.x0,
        pos3.y0,
        pos2.x1 - pos1_new.x0,
        pos3.height])

    # plt.tight_layout()

    # plt.show()

## a right
def plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(plot_data, data_pvalue, data_type_mean_or_slope, colorbarmin, colorbarmax, name, ax):

    if data_type_mean_or_slope == 'cor slope':

        plot_data = np.clip(plot_data, colorbarmin, colorbarmax)

        epsilon = 1e-6  # Adjust for the precision and units of data_slope

        # pheno_slope = pheno_slope[np.abs(pheno_slope) > epsilon]
        plot_data = np.where(np.abs(plot_data) > epsilon, plot_data, np.nan)

        ### Data grouping
        #Group the three categories
        plot_data_lt0_mask = plot_data<0
        plot_data_gte0_mask = plot_data>=0

        pvalue_sig_mask = data_pvalue<=0.05
        #
        # plot_data_lt0_sig = np.where(plot_data_lt0_mask & pvalue_sig_mask, plot_data, np.nan)
        # plot_data_lt0_all = np.where(plot_data_lt0_mask , plot_data, np.nan)
        # plot_data_gte0_sig = np.where(plot_data_gte0_mask & pvalue_sig_mask, plot_data, np.nan)
        # plot_data_gte0_all = np.where(plot_data_gte0_mask , plot_data, np.nan)

        plot_data_sig = np.where(pvalue_sig_mask, plot_data, np.nan)

        bins = np.linspace(colorbarmin, colorbarmax, 11)
        # count_lt0_sig, _ = np.histogram(plot_data_lt0_sig, bins=bins)
        # count_lt0_all, _ = np.histogram(plot_data_lt0_all, bins=bins)
        # count_gte0_sig, _ = np.histogram(plot_data_gte0_sig, bins=bins)
        # count_gte0_all, _ = np.histogram(plot_data_gte0_all, bins=bins)

        count_sig, _ = np.histogram(plot_data_sig, bins=bins)
        count_all, _ = np.histogram(plot_data, bins=bins)



    elif data_type_mean_or_slope == 'cor mean':
        bins = np.arange(colorbarmin, colorbarmax+0.1, 0.1)
        count_mean, _ = np.histogram(plot_data, bins=bins)

    ### plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1,
                                                height_ratios=[5, 0.3],
                                                hspace=0.15)

    # Hide the parent ax, which is a placeholder
    ax.axis('off')

    #Set the broken y-axis
    # if data_type == 'cor slope':
    #     ax.axis('off')
    #     bax = brokenaxes(
    #         ylims=((0, 5000), (10000, 11000)),
    #         hspace=0.1,
    #         height_ratios=[1, 5],  # Upper panel has 1 share and lower panel 2; larger values occupy more space
    #         subplot_spec=sub_gs
    #     )
    #
    #     # Set bar positions
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     print(f'bin_centers:{bin_centers}')
    #
    #     total_width = 0.007  # Total bar width per tick
    #     n = 2  # Number of categories
    #     width = total_width / n  # Width of each bar
    #
    #     bax.bar(bin_centers - width / 2, count_lt0_sig, width=width, color='#b2182b', label='Cor sig-strong')
    #     bax.bar(bin_centers + width / 2, count_lt0_nosig, width=width, color='#fddbc7',
    #             label='Cor nonsig-strong')
    #     bax.bar(bin_centers - width / 2, count_gte0_nosig, width=width, color='#d1e5f0',
    #             label='Cor nonsig-weaken')
    #     bax.bar(bin_centers + width / 2, count_gte0_sig, width=width, color='#2166ac', label='Cor sig-weaken')
    #
    #
    #     ticks = np.arange(colorbarmin, colorbarmax + 0.0001, 0.007)
    #     labels = [f'{(c * 10):.2f}' for c in ticks]
    #
    #     bax.set_xlim(colorbarmin, colorbarmax)
    #
    #     for ax_part in bax.axs:
    #         ax_part.set_xticks(ticks)
    #         ax_part.set_xticklabels(labels)
    #
    #     bax.set_xlabel('VPD-SM coupling trend (per decade)', labelpad=20) #Control spacing between labels and ticks
    #
    #     bax.set_ylabel('Frequency', labelpad=31)  # Control spacing between labels and ticks

    #     # Colorbar
    #     bax.legend(
    #         loc='upper right',
    #         bbox_to_anchor=(1.2, 1),
    #         ncol=1,
    #         frameon=False,  # Control legend frame visibility
    #         handlelength=1,
    #         handleheight=1
    #     )


    # Do not use a broken y-axis
    if data_type_mean_or_slope == 'cor slope':

        ax1 = fig.add_subplot(gs_inner[0])

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        # Set bar positions
        bin_centers = (bins[:-1] + bins[1:]) / 2
        print(f'bin_centers:{bin_centers}')

        color_list = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7',
                      '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']

        cmap = mpl.colors.ListedColormap(color_list)
        norm = mpl.colors.BoundaryNorm(bins, cmap.N, clip=True)

        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_colors = cmap(norm(bin_centers))

        total_width = 0.006  # Total bar width per tick
        n = 2  # Number of categories
        width = total_width / n  # Width of each bar

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # ax1.bar(bin_centers, count_lt0_sig, width=0.004, linewidth = 0.4, hatch='///', facecolor='none', edgecolor='#f46d43', label='Sig-strong', zorder= 2)
        # ax1.bar(bin_centers, count_lt0_all, width=0.004, linewidth = 0.5, color='#fee090', label='Nonsig-strong', zorder= 1)
        # ax1.bar(bin_centers, count_gte0_all, width=0.004, linewidth = 0.5, color='#e0f3f8', label='Nonsig-weaken', zorder= 1)
        # ax1.bar(bin_centers, count_gte0_sig, width=0.004, linewidth = 0.4, hatch='///', facecolor='none', edgecolor='#74add1', label='Sig-weaken', zorder= 2)

        # Plot bars individually to preserve their colors
        for j in range(len(count_all)):
            ax1.bar(
                bin_centers[j],
                count_all[j],
                width=0.004,
                color=bin_colors[j],
                linewidth=0.5,
                zorder=1,
                edgecolor='none'
            )
        ax1.bar(
            bin_centers,
            count_sig,
            width=0.004,
            hatch='/////',
            facecolor='none',
            edgecolor='black',
            linewidth=0.8,
            zorder=2
        )

        ### X-axis settings
        ticks = bins
        ax1.set_xlim(colorbarmin, colorbarmax)
        ax1.set_xticks(ticks)
        ax1.set_xticks(ticks)

        # ax1.set_xticklabels([
        #     f'≤{bins[0] * 100:g}' if np.isclose(x, colorbarmin) else
        #     f'≥{bins[-1] * 100:g}' if np.isclose(x, colorbarmax) else
        #     '0' if np.isclose(x, 0) else
        #     '3' if np.isclose(x * 100, 3) else
        #     '-3' if np.isclose(x * 100, -3) else
        #     f'{x * 100:.1f}'
        #     for x in ticks
        # ], fontsize=8)
        ax1.set_xticklabels(['0' if x == 0 else
                             f'<{int((bins[0]) * 100)}' if x == colorbarmin else
                             f'>{int((bins[-1]) * 100)}' if x == colorbarmax else
                             f'{int(x * 100)}' if x * 100 == int(x * 100) else
                             f'{x * 100:.1f}'
                             for x in bins])
        ax1.tick_params(axis='both', length=2, pad=3)
        if name == 'All':
            ax_labelrotation = 45
        else:
            ax_labelrotation = 90
        ax1.tick_params(axis='x', labelrotation=ax_labelrotation)

        tick_size = plt.rcParams['xtick.labelsize']
        if name == 'All':
            x_pos = 0.88
            y_pos =-0.15
        else:
            x_pos = 0.78
            y_pos = -0.15
        ax1.text(
            x_pos,  # x = Tick position in data coordinates
            y_pos,  # y = Slightly lower in axes coordinates
            r'$×10^{-2}$',  # Text to display
            transform=ax1.transAxes,
            ha='left',  # Align rightward to avoid compression
            va='top',
            rotation=0,
            fontsize=8,
            clip_on=False
        )

        # ax1.set_xticklabels(labels, rotation=45)  # Display the physical position 0.007 with the label 0.07

        if name == 'All':
            if scale == 55:
                if data_type == 'MCD12Q2':
                    ax1.set_ylim(0, 5000)
                    ticks = np.arange(0, 5000.1, 1000)

                elif data_type == 'MODIS_GPP_8d':
                    ax1.set_ylim(0, 4000)
                    ticks = np.arange(0, 4000.1, 1000)

                ax1.set_yticklabels(f'{int(y * 0.001)}' for y in ticks)
                y_text = r'$×10^{3}$'

            elif scale == 11:
                if data_type == 'MCD12Q2':
                    ax1.set_ylim(0, 80000)
                    ticks = np.arange(0, 80000.1, 20000)
                elif data_type == 'MODIS_GPP_8d':
                    ax1.set_ylim(0, 60000)
                    ticks = np.arange(0, 60000.1, 10000)
                ### Y-axis settings
                ax1.set_yticks(ticks)  # Set ticks only at 0, 10000, 20000 and 30000
                ax1.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda value, pos: f'{(value * 0.0001):g}')
                )
                y_text = r'$×10^{4}$'

            # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))


        elif name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands', 'Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
            ax1.set_ylim(0, 40000)
            ticks = np.arange(0, 40000.1, 10000)
            ax1.set_yticks(ticks)

            ax1.yaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, p: '0' if np.isclose(x, 0) else f'{int(x * 0.0001)}'
                )
            )

            y_text = r'$×10^{4}$'

        # elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
            # ax1.set_ylim(0, 30000)
            # ticks = np.arange(0, 30000.1, 10000)
            # ax1.set_yticks(ticks)
            #
            # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(abs(x))}"))
            #
            # ### Y-axis settings
            # ax1.set_yticklabels('0' if x == 0 else
            #     f'{x * 0.0001:.1f}' for x in ticks)
            # y_text = r'$×10^{4}$'

        ax1.set_ylabel('Frequency', labelpad=3)  # Control spacing between labels and ticks

        ax1.text(
            -0.1,  # x = Tick position in data coordinates
            1.12,  # y = Slightly lower in axes coordinates
            y_text,  # Text to display
            transform=ax1.transAxes,
            ha='left',  # Align rightward to avoid compression
            va='top',
            fontsize=8,
            clip_on=False
        )

        if name == 'All':
            x_position = 0.52
        else:
            x_position = 0.4

        # Colorbar
        sig_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            hatch='/////',
            label='Significant'
        )
        ax1.legend(
            handles=[sig_patch],
            loc='lower center',
            bbox_to_anchor=(x_position, -0.4),
            ncol=1,
            frameon=False,  # Control legend frame visibility
            handlelength=1,
            handleheight=1,
            columnspacing=0.5
        )



    elif data_type_mean_or_slope == 'cor mean':
        if name == 'All':

            total_width = 0.2  # Total bar width per tick
            n = 2  # Number of categories
            width = total_width / n  # Width of each bar

            bin_centers = (bins[:-1] + bins[1:]) / 2

            color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                          '#fcbba1', '#fee5d9', '#9ecae1']

            if scale == 55:
                bax = brokenaxes(
                    ylims=((0, 2000), (4000, 7000)),
                    hspace=0.15,
                    height_ratios=[2, 1],  # Upper panel has 1 share and lower panel 2; larger values occupy more space
                    subplot_spec=gs_inner[0],
                    d=0.005
                )
            # elif scale == 11:
            #     bax = brokenaxes(
            #         ylims=((0, 15000), (20000, 35000)),
            #         hspace=0.15,
            #         height_ratios=[1, 1],
            #         subplot_spec=gs_inner[0],
            #         d=0.005
            #     )



                # Plot bars individually to preserve their colors
                for j in range(len(count_mean)):
                    bax.bar(bin_centers[j], count_mean[j], width=0.08,
                            color=color_list[j], edgecolor='none')

                # bax.set_xlim(colorbarmin, colorbarmax)
                # bax.set_xticks(np.arange(colorbarmin, colorbarmax, 0.1))
                # bax.axs[1].set_xticklabels(bax.axs[1].get_xticklabels())

                bax.tick_params(axis='both', length=2, pad=3)

                bax.set_xlim(colorbarmin, colorbarmax)
                xticks = np.arange(colorbarmin, colorbarmax+0.1, 0.1)

                bax.axs[1].set_xticks(xticks)
                bax.axs[1].set_xticklabels(
                    ['0' if np.isclose(x, 0) else f'{x:.1f}'
                        for x in xticks],
                    rotation=45
                )


                # bax.set_xlabel('VPD-SM coupling', labelpad=20) #Control spacing between labels and ticks
                if scale == 55:
                    bax.axs[0].set_yticks([4000, 5000, 6000, 7000])
                    bax.axs[1].set_yticks([0, 1000, 2000])
                elif scale == 11:
                    bax.axs[0].set_yticks([20000, 25000, 30000, 35000])
                    bax.axs[1].set_yticks([0, 5000, 10000, 15000])

                for ax in bax.axs:
                    # Get current y-axis ticks
                    y_ticks = ax.get_yticks()
                    # Generate labels from y-axis ticks
                    ax.set_yticklabels([
                        '0' if y == 0 else
                        f'{int(y * 0.001)}'
                        for y in y_ticks
                    ])
                if scale == 11:
                    if data_type == 'MCD12Q2':
                        ypos = 1.4
                    elif data_type == 'MODIS_GPP_8d':
                        ypos = 1.2

                ## Control the y-axis power-of-ten label
                bax.axs[0].text(
                    -0.1,  # x = Tick position in data coordinates
                    ypos,  # y = Slightly lower in axes coordinates
                    r'$×10^{3}$',  # Text to display
                    transform=bax.axs[0].transAxes,
                    ha='left',  # Align rightward to avoid compression
                    va='top',
                    rotation=0,
                    fontsize=8,
                    clip_on=False
                )


                bax.set_ylabel('Frequency', labelpad=18)  # Control spacing between labels and ticks



            elif scale == 11:

                ax1 = fig.add_subplot(gs_inner[0])

                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)

                # Plot bars individually to preserve their colors

                for j in range(len(count_mean)):
                    ax1.bar(

                        bin_centers[j],

                        count_mean[j],

                        width=0.08,

                        color=color_list[j],

                        edgecolor='none',

                    )

                ax1.set_xlim(colorbarmin, colorbarmax)

                xticks = np.arange(colorbarmin, colorbarmax + 0.1, 0.1)

                ax1.set_xticks(xticks)

                ax1.set_xticklabels(

                    ['0' if np.isclose(x, 0) else f'{x:.1f}' for x in xticks],

                    rotation=45,

                )

                # As for cor slope, use original y-axis counts and display labels scaled by 1000

                if data_type in ['MCD12Q2', 'MODIS_GPP_8d']:

                    ax1.set_ylim(0, 120000)

                    yticks = np.arange(0, 120000.1, 20000)

                # elif data_type == 'MODIS_GPP_8d':
                #
                #     ax1.set_ylim(0, 40000)
                #
                #     yticks = np.arange(0, 40000.1, 10000)

                else:

                    # Use this range if the data type must extend to 35×10³

                    ax1.set_ylim(0, 35000)

                    yticks = np.arange(0, 35000.1, 5000)


                ax1.set_yticklabels([

                    '0' if np.isclose(y, 0) else f'{int(y * 0.0001)}'

                    for y in yticks

                ])

                ax1.tick_params(axis='both', length=2, pad=3)

                ax1.text(

                    -0.1,

                    1.12,

                    r'$×10^{4}$',

                    transform=ax1.transAxes,

                    ha='left',

                    va='top',

                    fontsize=8,

                    clip_on=False,

                )

                ax1.set_ylabel('Frequency', labelpad=3)




        else:
            ax1 = fig.add_subplot(gs_inner[0])

            ax1.spines['top'].set_visible(False)

            ax1.spines['right'].set_visible(False)

            total_width = 0.2  # Total bar width per tick
            n = 2  # Number of categories
            width = total_width / n  # Width of each bar

            bin_centers = (bins[:-1] + bins[1:]) / 2

            color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
                          '#fcbba1', '#fee5d9', '#9ecae1']

            # Plot bars individually to preserve their colors
            for j in range(len(count_mean)):
                ax1.bar(bin_centers[j], count_mean[j], width=0.08,
                       color=color_list[j], edgecolor='none')

            ticks = np.arange(colorbarmin, colorbarmax + 0.1, 0.1)
            ax1.set_xlim(colorbarmin, colorbarmax)
            ax1.set_xticks(ticks)
            ax1.set_xticklabels(
                ['0' if np.isclose(x, 0) else f'{x:.1f}'
                 for x in ticks],
                rotation=45
            )

            # ax1.set_xlabel('VPD-SM coupling', labelpad=3)  # Control spacing between labels and ticks
            if name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands', 'Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
                ax1.set_ylim(0, 50000)
                ticks = np.arange(0, 50000.1, 10000)
                ax1.set_yticks(ticks)

                ax1.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: f'{int(x * 0.0001)}')
                )
            # elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
            #     ax1.set_ylim(0, 40000)
            #     ticks = np.arange(0, 40000.1, 10000)
            #     ax1.set_yticks(ticks)

            ### Y-axis settings
            # ax1.set_yticklabels(f'{int(x * 0.001)}' for x in ticks)

            tick_size = plt.rcParams['xtick.labelsize']

            if data_type_mean_or_slope == 'cor mean':
                if name != 'All':
                    ypos =1.1
            else:
                ypos = 1
            ax1.text(
                -0.1,  # x = Tick position in data coordinates
                ypos,  # y = Slightly lower in axes coordinates
                r'$×10^{4}$',  # Text to display
                transform=ax1.transAxes,
                ha='left',  # Align rightward to avoid compression
                va='top',
                fontsize=8,
                clip_on=False
            )

            ax1.set_ylabel('Frequency', labelpad=3)  # Control spacing between labels and ticks

            ax1.tick_params(axis='both', length=2, pad=3)


    # if name == 'All':
    #     if data_type == 'cor mean':
    #         bax.set_title(f'(b)', pad=10, fontweight='bold')
    #     elif data_type == 'cor slope':
    #         ax.set_title(f'(d)', pad=10, fontweight='bold')
    # plt.show()

## b
def plot_cor_mean_or_slope_forDiffvegType_and_AI(plot_data, veg_data, ai_data, data_type_mean_or_slope, ax):

    # 1. Prepare data
    veg_list = [plot_data[(veg_data == i) & np.isfinite(plot_data)] for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_list = [
        plot_data[(ai_data == 2) & np.isfinite(plot_data)],  # Arid
        plot_data[((ai_data == 3) | (ai_data == 4)) & np.isfinite(plot_data)],  # Semi-Arid (Combine 3 and 4)
        plot_data[(ai_data == 5) & np.isfinite(plot_data)],  # Dry sub-humid
        plot_data[(ai_data == 6) & np.isfinite(plot_data)]  # Humid
    ]
    ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    full_data = veg_list + ai_list
    x_positions = [1, 2, 3, 4, 5, 6, 7, 8]  # X-axis physical position

    # Calculate the mean safely
    full_data_mean = [np.mean(d) if len(d) > 0 else 0 for d in full_data]

    ### plot
    # fig, ax = plt.subplots(figsize=(4, 4))  # Widen slightly to prevent crowded text
    # plt.subplots_adjust(bottom=0.2)  # Leave bottom space for rotated labels

    fig = ax.get_figure()

    gs_inner = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[5, 0.3],
        hspace=0.15
    )

    ax1 = fig.add_subplot(gs_inner[0])
    ax.axis('off')

    # Add showmedians=True
    vio = ax1.violinplot(full_data, positions=x_positions,
                        showmeans=True, showextrema=False)

    # Set colors
    # v_colors = ['#0ebeff', '#ae63e4', '#ffd200', '#ff3c41',  # Vegetation colors
    #             '#0ebeff', '#ae63e4', '#ffd200', '#ff3c41']  # Use distinct AI colors
    # Get the Paired colormap
    paired_colors = plt.cm.Paired(np.linspace(0, 1, 12))  # First select 8 colors
    indices = [1, 3, 7, 9]
    colors = [paired_colors[i] for i in indices]
    v_colors = list(colors) + list(colors)

    for i, pc in enumerate(vio['bodies']):
        pc.set_facecolor(v_colors[i])
        pc.set_edgecolor('none')
        # pc.set_linewidth(0.5)
        # pc.set_alpha(0.7)

    # Set mean and median line colors separately
    # Mean line (Means) - red
    vio['cmeans'].set_edgecolor('red')
    vio['cmeans'].set_linestyle('-')
    vio['cmeans'].set_linewidth(1.5)


    # Style the axes
    ax1.set_xticks(x_positions)
    # if data_type == 'cor mean':
    #     ax1.set_xticklabels([])
    # elif data_type == 'cor slope':
    ax1.set_xticklabels(veg_labels + ai_labels, rotation=90)

    # if data_type == 'cor slope':
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    ax1.axvline(4.5, color='black', linestyle='-', linewidth=1)  # Vertical dividing line

    if data_type_mean_or_slope == 'cor slope':
        ticks = np.arange(-0.03, 0.031, 0.01)
        ax1.set_ylim(-0.03, 0.03)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels('0' if np.isclose(x, 0) else
                            f'{int(round(x*100))}' for x in ticks)
        ax1.set_ylabel('SM-VPD coupling trend (a$^{-1}$)')

        tick_size = plt.rcParams['xtick.labelsize']
        ax1.text(-0.01, 1.06,
                r'$×10^{-2}$',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=8,
                clip_on=False
                )
    elif data_type_mean_or_slope == 'cor mean':
        ticks = np.arange(-0.7, 0.3, 0.1)
        ax1.set_ylim(-0.7, 0.2)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(['0' if np.isclose(x, 0) else
                             f'{x:.1f}' for x in ticks])
        ax1.set_ylabel('SM-VPD coupling')

    for i, m in enumerate(full_data_mean):

        color = v_colors[i]

        if data_type_mean_or_slope == 'cor mean':
            if data_type == 'MCD12Q2':
                if i == 0:  # Forests
                    yheight = 0.08
                elif i == 1:  # Shrublands
                    yheight = 0.15
                elif i == 2:  # Savannas
                    yheight = 0.10
                elif i == 3:  # Grasslands
                    yheight = 0.05
                elif i == 4:  # Arid
                    yheight = 0.03
                elif i == 5:  # Semi-arid
                    yheight = 0.05
                elif i == 6:  # Dry sub-humid
                    yheight = 0.10
                elif i == 7:  # Humid
                    yheight = 0.15
            elif data_type == 'MODIS_GPP_8d':
                if i == 0:  # Forests
                    yheight = 0.08
                elif i == 1:  # Shrublands
                    yheight = 0.14
                elif i == 2:  # Savannas
                    yheight = 0.10
                elif i == 3:  # Grasslands
                    yheight = 0.05
                elif i == 4:  # Arid
                    yheight = 0.03
                elif i == 5:  # Semi-arid
                    yheight = 0.06
                elif i == 6:  # Dry sub-humid
                    yheight = 0.15
                elif i == 7:  # Humid
                    yheight = 0.12

            ax1.text(
                x_positions[i],
                yheight,
                f'{m:.2f}',
                color=color,
                ha='center',
                va='bottom',
                fontsize=6
            )

        elif data_type_mean_or_slope == 'cor slope':
            if data_type == 'MCD12Q2':
                if i == 0 :
                    yheight = 0.02
                elif i == 1:
                    yheight = 0.027
                elif i == 2 :
                    yheight = 0.02
                elif i == 3 :
                    yheight = 0.023
                elif i == 4 :
                    yheight = 0.023
                elif i == 5 :
                    yheight = 0.023
                elif i == 6 :
                    yheight = 0.021
                elif i == 7:
                    yheight = 0.019

                if i == 3:
                    xheight = -0.2
                elif i == 6:
                    xheight = -0.3
                elif i == 4:
                    xheight = +0.2
                elif i == 5:
                    xheight = +0.4
                else:
                    xheight = 0
            elif data_type == 'MODIS_GPP_8d':
                if i in [0, 7]:
                    yheight = 0.020
                elif i == 2:
                    yheight = 0.022
                elif i == 6:
                    yheight = 0.027
                else:
                    yheight = 0.025


                if i == 3:
                    xheight = -0.2
                elif i == 6:
                    xheight = -0.3
                elif i == 4:
                    xheight = +0.2
                elif i == 5:
                    xheight = +0.4
                else:
                    xheight = 0


            ax1.text(
                x_positions[i] + xheight,
                yheight,
                f'{m:.3f}',
                color=color,
                ha='center',
                va='bottom',
                fontsize=6
            )


    ax1.tick_params(axis='both', length=2, pad=3)


    # if data_type == 'cor mean':
    #     ax1.set_title(f'(c)', pad=10, fontweight='bold')
    # elif data_type == 'cor slope':
    #     ax1.set_title(f'(f)', pad=10, fontweight='bold')

    # plt.show()

### All,Fig4
def plot_fig3(data_mean, data_slope, data_pvalue):

    # Set all font sizes consistently
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (recommended)
        'mathtext.default': 'regular',  # Prevent automatic italic formatting

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        # 'text.usetex': False,  # Do not use external LaTeX
    })

    # Create 5 subplots: 4 maps and a colorbar
    fig = plt.figure(figsize=(8.2, 6.5))
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[6, 3.5, 3.5],  # Width ratios for the three columns
                           height_ratios=[1, 1],  # Reserve the last slot for the colorbar
                           hspace=0.6, wspace=0.33)

    ax1 = plt.subplot(gs[0, 0])  ## fig a left
    ax2 = plt.subplot(gs[0, 1])  ## fig a right
    ax3 = plt.subplot(gs[0, 2])  ## fig b

    ax4 = plt.subplot(gs[1, 0])  ## fig a left
    ax5 = plt.subplot(gs[1, 1])  ## fig a right
    ax6 = plt.subplot(gs[1, 2])  ## fig b

    ### fig1 a left
    plot_cor_mean_or_slope_and_pvalue_forAllvegType(data_mean, data_pvalue,  -0.6, 0.1,  'cor mean', 'All',ax = ax1)

    ### fig1 a right
    plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_mean,  data_pvalue, 'cor mean', -0.6, 0.1, 'All',ax = ax2)

    ### fig1 b
    plot_cor_mean_or_slope_forDiffvegType_and_AI(data_mean, veg_type_data, ai_type_data, 'cor mean', ax = ax3)

    ### fig1 c left
    plot_cor_mean_or_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, -0.025, 0.025,'cor slope', 'All',ax = ax4)

    ### fig1 c right
    plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, 'cor slope', -0.025, 0.025, 'All',ax = ax5)

    ### fig1 d
    plot_cor_mean_or_slope_forDiffvegType_and_AI(data_slope, veg_type_data, ai_type_data, 'cor slope', ax = ax6)

    # plt.savefig(fr'D:/code/3Cor_mean_slope/Cor17_8_2_mean_Slope_pvalue.png', dpi=600, bbox_inches='tight')
    # plt.savefig(fr'{save_png_path}\All\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_{test_number}_mean_Slope_pvalue.png', dpi=300, bbox_inches='tight')

    plt.savefig(fr'{save_png_path}\All\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_{Basedon}_mean_Slope_pvalue{mean_or_yearPOS}_VegS{veg_threshold}_newtest.png', dpi=300, bbox_inches='tight')

    # plt.show()

### S9-12
def plot_S9_12_forMean_or_Slope(data_mean, data_slope, data_pvalue, data_type_mean_or_slope):

    if data_type_mean_or_slope == 'cor mean':
        data = data_mean
        colorbarmax = 0.1
        colorbarmin = -0.6
    elif data_type_mean_or_slope == 'cor slope':
        data = data_slope
        colorbarmax = 0.025
        colorbarmin = -0.025

    # 1. Prepare data
    veg_data_list = [np.where(veg_type_data == i, data, np.nan) for i in [1, 2, 3, 4]]
    veg_pvalue_list = [np.where(veg_type_data == i, data_pvalue, np.nan) for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_data_list = [
        np.where(ai_type_data == 2, data, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data, np.nan),  # Semi-Arid (Combine 3 and 4)
        np.where(ai_type_data == 5, data, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data, np.nan)  # Humid
    ]

    ai_pvalue_list = [
        np.where(ai_type_data == 2, data_pvalue, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data_pvalue, np.nan),  # Semi-Arid
        np.where(ai_type_data == 5, data_pvalue, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data_pvalue, np.nan)  # Humid
    ]
    ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    # Set all font sizes consistently
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',

        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        # Optional (recommended)
        'mathtext.default': 'regular',  # Prevent automatic italic formatting

        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        # 'text.usetex': False,  # Do not use external LaTeX
    })

    ############################################## Vegetation type ##################################################
    fig = plt.figure(figsize=(8.4, 6.5))
    gs = gridspec.GridSpec(2, 5,
                           width_ratios=[6, 4, 0.35, 6, 4],  # Width ratios for the three columns
                           height_ratios=[1, 1],  # Reserve the last slot for the colorbar
                           hspace=0.45, wspace=0.6)

    ax1 = plt.subplot(gs[0, 0])  ## Forest left
    ax2 = plt.subplot(gs[0, 1])  ## Forest right
    ax3 = plt.subplot(gs[0, 3])  ## shrub left
    ax4 = plt.subplot(gs[0, 4])  ## shrub right

    ax5 = plt.subplot(gs[1, 0])  ## savanna left
    ax6 = plt.subplot(gs[1, 1])  ## savanna right
    ax7 = plt.subplot(gs[1, 3])  ## grass left
    ax8 = plt.subplot(gs[1, 4])  ## grass right

    ax_pairs = [(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)]

    for veg_data, veg_pvalue, veg_name, (ax_l, ax_r) in zip(veg_data_list, veg_pvalue_list, veg_labels, ax_pairs):
        # Plotleft map group (ax_l)
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(veg_data, veg_pvalue, colorbarmin, colorbarmax, data_type_mean_or_slope, veg_name, ax=ax_l)

        # Plotright bar chart group (ax_r)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(veg_data, veg_pvalue, data_type_mean_or_slope, colorbarmin, colorbarmax, veg_name, ax=ax_r)

    # plt.savefig(fr'D:/code/3Cor_mean_slope/Cor17_8_2_{data_type}_Vegtype.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'{save_png_path}\Veg\Cor2_{data_type_mean_or_slope}_Vegtype{mean_or_yearPOS}_VegS{veg_threshold}.png', dpi=300, bbox_inches='tight')

    # plt.show()

    ############################################## AI type ##################################################
    fig = plt.figure(figsize=(8.4, 6.5))
    gs = gridspec.GridSpec(2, 5,
                           width_ratios=[6, 4, 0.8, 6, 4],  # Width ratios for the three columns
                           height_ratios=[1, 1],  # Reserve the last slot for the colorbar
                           hspace=0.45, wspace=0.5)

    ax1 = plt.subplot(gs[0, 0])  ## Forest left
    ax2 = plt.subplot(gs[0, 1])  ## Forest right
    ax3 = plt.subplot(gs[0, 3])  ## shrub left
    ax4 = plt.subplot(gs[0, 4])  ## shrub right

    ax5 = plt.subplot(gs[1, 0])  ## savanna left
    ax6 = plt.subplot(gs[1, 1])  ## savanna right
    ax7 = plt.subplot(gs[1, 3])  ## grass left
    ax8 = plt.subplot(gs[1, 4])  ## grass right



    ax_pairs = [(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)]

    for ai_data, ai_pvalue, ai_name, (ax_l, ax_r) in zip(ai_data_list, ai_pvalue_list, ai_labels, ax_pairs):
        # Plotleft map group (ax_l)
        # Assumes veg_name was added to the function signature
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(ai_data, ai_pvalue, colorbarmin, colorbarmax, data_type_mean_or_slope, ai_name, ax=ax_l)

        # Plotright bar chart group (ax_r)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(ai_data, ai_pvalue, data_type_mean_or_slope, colorbarmin, colorbarmax, ai_name, ax=ax_r)

    # plt.savefig(fr'D:/code/3Cor_mean_slope/Cor17_8_2_{data_type_mean_or_slope}_AItype.png', dpi=600, bbox_inches='tight')
    plt.savefig(fr'{save_png_path}\AI\Cor2_{data_type_mean_or_slope}_AItype{mean_or_yearPOS}_VegS{veg_threshold}.png', dpi=300, bbox_inches='tight')

    # # plt.show()


####
cor_mean = np.where((mcd12q2_mask & gpp_mask & np.isfinite(veg_type_data)), cor_mean, np.nan)
cor_slope = np.where((mcd12q2_mask & gpp_mask & np.isfinite(veg_type_data)), cor_slope, np.nan)


# plot_fig3(cor_mean, cor_slope, cor_slope_pvalue)
# print('Fig3 plot done!')
plot_S9_12_forMean_or_Slope(cor_mean, cor_slope, cor_slope_pvalue, 'cor mean')
print('S9-10 plot done!')
plot_S9_12_forMean_or_Slope(cor_mean, cor_slope, cor_slope_pvalue, 'cor slope')
print('S11-12 plot done!')


