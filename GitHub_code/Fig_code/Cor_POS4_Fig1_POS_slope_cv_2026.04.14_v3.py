import glob
import os.path
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from osgeo import gdal
from scipy.stats import theilslopes
from brokenaxes import brokenaxes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pymannkendall as mk
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


######### Shared functions #############
def clip_by_latitude(gt, rows, cols, lat_min, lat_max):
    """
    Crop data by latitude range
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
    # print(f"Original rows: {rows}, Rows after clipping: {row_end - row_start}")
    # print(f"Original upper-left latitude: {gt[3]:.2f}°N, New upper-left latitude: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt

def save_tif_gdal(output_path, data, crs, transform):
    """Save a TIFF file with its data dimensions and geotransform"""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols, rows, 1, gdal.GDT_Float32
    )
    if not output_ds:
        raise RuntimeError(f"Unable to create output file: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # Set NaN as the NoData value

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # Use the adjusted geotransform
    output_ds = None
    return True


####################################### 1 Input and output ######################################
##################### 1.1 Input ##########################
startYear = 2001
endYear = 2024
year_length = endYear-startYear+1

cal_done = 'no'

data_type = 'MODIS_GPP_8d' # GIMMS_NDVI_16d / MODIS_GPP_16d / MODIS_GPP_8d / MODIS_EVI2_8d / MCD12Q2

pheno = 'POS'
scale = 11
veg_threshold = 25

path = rf'D:/code/Global_test2_{scale}000m'

pheno_path = rf'{path}\Data\Pheno\{pheno}_{scale}km'

veg_type_tif = rf'{path}\Data\Classify\Veg_type\NH_permanent_veg_type_{scale}km_gte{veg_threshold}.tif'

# climate_type_tif = rf'D:/code/KB_climate/NH30_84_KB(simple)_climate_region_{scale}km.tif'

ai_type_tif = rf'{path}\Data\Classify\AI\NH30_84_AI(graident)_{scale}km.tif'

### Input coupling gradient TIF path
cor_mean_tif = fr'{path}\\Tif_of_fig\3Cor_mean_slope\mean\Cor(Pearson_SMlag_VPDlag_lag0)_mean_11km_SOS_All.tif'  #SOS - POS


pheno_files = sorted(glob.glob(os.path.join(pheno_path, '*.tif')))


##################### 1.2 Output ##########################
tif_output = rf'{path}\Tif_of_fig\2Pheno_mean_slope_std_cv_diff'
fig_output = rf'{path}\Fig\Fig 1 POS trend and CV'


####################################### 2 Basic information #########################################
sample_tif = pheno_files[0]
sample_data = gdal.Open(sample_tif)

sample_array = sample_data.GetRasterBand(1).ReadAsArray().astype(np.float32)

crs = sample_data.GetProjectionRef()
gt = sample_data.GetGeoTransform()

rows = sample_array.shape[0]
cols = sample_array.shape[1]


row_indices = np.repeat(np.arange(rows), cols)
col_indices = np.tile(np.arange(cols), rows)

lon_min = gt[0]               ##Starting longitude
lon_max = gt[0] + gt[1]*cols  ##Starting longitude + pixel resolution * number of columns
lat_max = gt[3]       ##Starting latitude
lat_min = gt[3] + gt[5]*rows  ##Starting latitude + pixel resolution * number of rows
print(f'Longitude range: {lon_min}~{lon_max}°')
print(f'Latitude range: {lat_min}~{lat_max}°N')

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
ai_type_data = gdal.Open(ai_type_tif)
band = ai_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

ai_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    ai_type_data[ai_type_data == nodata] = np.nan

############### 3.4 Cormean #################
cor_type_data = gdal.Open(cor_mean_tif)
band = cor_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

cor_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    cor_type_data[cor_type_data == nodata] = np.nan


if cal_done == 'no':
    ####################################### 3 Stacking ######################################
    ### Read the band
    def get_band(tif_file, stack):
        tif = gdal.Open(tif_file)
        data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

        data = np.where((data != 0) & (data != -9999), data, np.nan)

        stack.append(data)
        tif = None  # Release resources promptly


    ############ 3.1 Phenology ################
    pheno_stack = []

    for tif_file in pheno_files:
        get_band(tif_file, pheno_stack)

    pheno_stack = np.stack(pheno_stack, axis=0)

    print('pheno_stack[100:105, 400:405]', pheno_stack[:, 100, 500])

    pheno_stack = np.where(np.isfinite(veg_type_data), pheno_stack, np.nan)


    ####################################### 4 Remove outliers #########################################
    ###### Way1: Apply IQR to each pixel ########
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

            if (len(np.isfinite(x_masked)) >= (time_lengths/2)) & (len(np.isfinite(x_masked)) <= time_lengths):
                return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
            else:
                print(f'IQR threshold：{lower_range:.4f} ~ {upper_range:.4f}\n'
                      f'Original pixel data: {x}\n'
                      f'Remove outliers: fewer than one third of pixel values are valid: {x_masked}')
                return np.full_like(x, np.nan), i, j


    time_lengths = pheno_stack.shape[0]

    outlier_pheno_stack = np.full((time_lengths, rows, cols), np.nan)

    ### IQRRemove outliers
    results = Parallel(n_jobs=18, verbose=10)(
        delayed(Outlier_array_IQR)(
            pheno_stack[:, i, j], i, j, 25, 75
        ) for i, j in zip(row_indices, col_indices)
    )

    for data_mask, i, j in results:
        outlier_pheno_stack[:, i, j] = data_mask


    # print('outlier_pheno_stack[:, 101, 501]:', outlier_pheno_stack[:, 101, 501])


    # ####### Way2：Calculate MAD in a 3×3 window, then apply IQR per pixel #########
    # import warnings
    #
    # # ============================================================
    # # 4. Apply spatial MAD filtering in a 3×3 window for each year, followed by temporal IQR filtering per pixel
    # # ============================================================
    #
    # # veg_type_data：2D vegetation type raster，shape=(rows, cols)
    # vegetation_grid = np.asarray(veg_type_data)
    #
    # if pheno_stack.ndim != 3:
    #     raise ValueError(
    #         f"pheno_stack must be a 3D array (year, row, column), current shape={pheno_stack.shape}"
    #     )
    #
    # time_lengths, stack_rows, stack_cols = pheno_stack.shape
    #
    # if vegetation_grid.shape != (stack_rows, stack_cols):
    #     raise ValueError(
    #         f"Vegetation raster shape={vegetation_grid.shape}，"
    #         f"but phenology raster spatial shape={(stack_rows, stack_cols)}"
    #     )
    #
    # veg_nodata = np.nan
    # min_neighbors = 5
    # robust_z_threshold = 3.5
    #
    # # Make a copy to retain the original pheno_stack
    # spatial_pheno_stack = np.asarray(
    #     pheno_stack,
    #     dtype=np.float32
    # ).copy()
    #
    # # 8 neighbors, excluding the center pixel
    # neighbor_offsets = [
    #     (-1, -1), (-1, 0), (-1, 1),
    #     ( 0, -1),          ( 0, 1),
    #     ( 1, -1), ( 1, 0), ( 1, 1),
    # ]
    #
    # for year_idx in range(time_lengths):
    #     year_data = np.asarray(pheno_stack[year_idx], dtype=np.float32)
    #
    #     # Collect valid neighboring values of the same vegetation type
    #     neighbors = np.full(
    #         (8, stack_rows, stack_cols),
    #         np.nan,
    #         dtype=np.float32,
    #     )
    #
    #     for k, (dr, dc) in enumerate(neighbor_offsets):
    #         # Target pixel slice bounds
    #         dst_r0 = max(0, -dr)
    #         dst_r1 = min(stack_rows, stack_rows - dr)
    #         dst_c0 = max(0, -dc)
    #         dst_c1 = min(stack_cols, stack_cols - dc)
    #
    #         # Corresponding neighbor slice bounds
    #         src_r0 = dst_r0 + dr
    #         src_r1 = dst_r1 + dr
    #         src_c0 = dst_c0 + dc
    #         src_c1 = dst_c1 + dc
    #
    #         center_veg = vegetation_grid[dst_r0:dst_r1, dst_c0:dst_c1]
    #         neighbor_veg = vegetation_grid[src_r0:src_r1, src_c0:src_c1]
    #         neighbor_values = year_data[src_r0:src_r1, src_c0:src_c1]
    #
    #         same_vegetation = (
    #             np.isfinite(center_veg)
    #             & np.isfinite(neighbor_veg)
    #             & (center_veg != veg_nodata)
    #             & (neighbor_veg != veg_nodata)
    #             & (center_veg == neighbor_veg)
    #         )
    #
    #         valid_neighbor = same_vegetation & np.isfinite(neighbor_values)
    #
    #         neighbors[
    #             k, dst_r0:dst_r1, dst_c0:dst_c1
    #         ] = np.where(valid_neighbor, neighbor_values, np.nan)
    #
    #     neighbor_count = np.sum(np.isfinite(neighbors), axis=0)
    #
    #     # Neighborhood median and MAD; exclude neighborhoods containing only NaN
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", RuntimeWarning)
    #
    #         neighbor_median = np.nanmedian(neighbors, axis=0)
    #         abs_deviation = np.abs(neighbors - neighbor_median[None, :, :])
    #         neighbor_mad = np.nanmedian(abs_deviation, axis=0)
    #
    #     # Evaluate only pixels with at least 5 valid neighbors of the same type and MAD greater than 0
    #     testable = (
    #         np.isfinite(year_data)
    #         & np.isfinite(neighbor_median)
    #         & np.isfinite(neighbor_mad)
    #         & (neighbor_count >= min_neighbors)
    #         & (neighbor_mad > 0)
    #     )
    #
    #     robust_z = np.zeros((stack_rows, stack_cols), dtype=np.float32)
    #     robust_z[testable] = (
    #         0.6745
    #         * (year_data[testable] - neighbor_median[testable])
    #         / neighbor_mad[testable]
    #     )
    #
    #     spatial_outlier = testable & (np.abs(robust_z) > robust_z_threshold)
    #
    #     # Retain the original value if there are fewer than 5 same-type neighbors or MAD is zero
    #     spatial_pheno_stack[year_idx, spatial_outlier] = np.nan
    #
    #     print(
    #         f"Spatial MAD filter: year {year_idx + 1}/{time_lengths}; "
    #         f"Removed {np.count_nonzero(spatial_outlier)} pixels"
    #     )
    #
    #
    # # ============================================================
    # # 4.2 Apply temporal IQR per pixel after spatial filtering
    # # ============================================================
    #
    # def Outlier_array_IQR(x, i, j, qmin=25, qmax=75):
    #     """
    #     Apply IQR outlier filtering to a pixel's multi-year time series.
    #     Set the full series to NaN if fewer than half the years are valid.
    #     """
    #     x = np.asarray(x, dtype=np.float32)
    #     n_years = x.size
    #     finite_mask = np.isfinite(x)
    #     x_valid = x[finite_mask]
    #
    #     if x_valid.size < n_years / 2:
    #         return np.full(x.shape, np.nan, dtype=np.float32), i, j
    #
    #     lower_quartile, upper_quartile = np.percentile(
    #         x_valid,
    #         [qmin, qmax],
    #     )
    #
    #     iqr = upper_quartile - lower_quartile
    #     lower_range = lower_quartile - 1.5 * iqr
    #     upper_range = upper_quartile + 1.5 * iqr
    #
    #     # Set valid observations outside the IQR range to NaN; retain existing NaNs
    #     x_filtered = x.copy()
    #     temporal_outlier = (
    #         finite_mask
    #         & ((x < lower_range) | (x > upper_range))
    #     )
    #     x_filtered[temporal_outlier] = np.nan
    #
    #     # Correction: len(np.isfinite(...)) returns the array length;
    #     # use np.count_nonzero to count valid years.
    #     n_remaining = np.count_nonzero(np.isfinite(x_filtered))
    #
    #     if n_remaining > n_years / 2:
    #         return x_filtered, i, j
    #
    #     print(
    #         f"IQR threshold: {lower_range:.4f} ~ {upper_range:.4f}\n"
    #         f"Pixel position: ({i}, {j})\n"
    #         f"Original data: {x}\n"
    #         f"Fewer than half the years are valid after IQR filtering: {x_filtered}"
    #     )
    #     return np.full(x.shape, np.nan, dtype=np.float32), i, j
    #
    #
    # outlier_pheno_stack = np.full(
    #     (time_lengths, stack_rows, stack_cols),
    #     np.nan,
    #     dtype=np.float32,
    # )
    #
    # results = Parallel(n_jobs=8, verbose=10)(
    #     delayed(Outlier_array_IQR)(
    #         spatial_pheno_stack[:, i, j],
    #         i,
    #         j,
    #         25,
    #         75,
    #     )
    #     for i, j in zip(row_indices, col_indices)
    # )
    #
    # for data_mask, i, j in results:
    #     outlier_pheno_stack[:, i, j] = data_mask



    ################ 6.1 Calculate long-term trends and significance ##################

    ####################################### Multi-year mean #########################################
    pheno_stack_mean = np.nanmean(outlier_pheno_stack, axis=0)



    def calculate_senSlope(data, i, j):

        mask = np.isfinite(data)

        data_clean = data[mask]
        years_valid = years[mask]

        if len(data_clean) >= (time_lengths/2):   ### >10 / >17/ >19

            # Sen slope
            result = theilslopes(data_clean, years_valid)

            slope = result.slope
            intercept = result.intercept

            # Mann-Kendall
            mk_result = mk.original_test(data_clean)

            pvalue = mk_result.p

        else:
            ############# If len(data_nodrought_clean) <= 10, consider a moving window ###########
            slope = np.nan
            pvalue = np.nan
            intercept = np.nan

        return (i, j, slope, pvalue, intercept)

    pheno_slope = np.full((rows, cols), np.nan)
    pheno_intercept = np.full((rows, cols), np.nan)
    pheno_slope_pvalue = np.full((rows, cols), np.nan)

    years = np.arange(startYear, startYear + pheno_stack.shape[0])

    results = Parallel(n_jobs = 18, verbose=10)(
        delayed(calculate_senSlope)(
            outlier_pheno_stack[:, i, j], i, j)
                for i, j in zip(row_indices, col_indices)
    )

    for i, j, slope, pvalue, intercept in results:
        pheno_slope[i, j] = slope
        pheno_intercept[i, j] = intercept
        pheno_slope_pvalue[i, j] = pvalue


    # ############## Remove the long-term trend ######################
    # for t, year in enumerate(years):
    #     pheno_trend = pheno_slope * year + pheno_intercept
    #
    #     pheno_detrend = pheno_stack[t, :, :] - pheno_trend
    #
    #     output_path = rf'{path}\Data\Pheno\{pheno}detrend_{scale}km\{pheno}1_aggMean_{scale}000m_{year}.tif'
    #     save_tif_gdal(
    #         output_path,
    #         pheno_detrend,
    #         crs,
    #         gt  # Use the new geotransform
    #     )
    # sys.exit()

    ################ 7.1 Calculate multi-year std and cv ##################
    def calculate_std_cv(data, i, j):

        data_clean = data[~np.isnan(data)]

        if len(data_clean) >= (time_lengths/2):

            mean_value = np.nanmean(data_clean)
            std_value = np.nanstd(data_clean)
            cv_value = std_value/mean_value

        else:
            std_value = np.nan
            cv_value = np.nan

        return (i, j, std_value, cv_value)

    pheno_std = np.full((rows, cols), np.nan)
    pheno_cv = np.full((rows, cols), np.nan)

    results = Parallel(n_jobs = 18, verbose=10)(
        delayed(calculate_std_cv)(
            outlier_pheno_stack[:, i, j], i, j)
                for i, j in zip(row_indices, col_indices)
    )

    for i, j, std_value, cv_value in results:
        pheno_std[i, j] = std_value
        pheno_cv[i, j] = cv_value

    ################ 7.2 Save data ###############
    mcd12q2_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
    mcd12q2_mask_tif = gdal.Open(mcd12q2_mask_tif)
    mcd12q2_mask = mcd12q2_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    mcd12q2_mask = np.isfinite(mcd12q2_mask)

    gpp_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
    gpp_mask_tif = gdal.Open(gpp_mask_tif)
    gpp_mask = gpp_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gpp_mask = np.isfinite(gpp_mask)

    # pheno_stack_mean = np.where(mcd12q2_mask & gpp_mask, pheno_stack_mean, np.nan)
    # mean_path = os.path.join(
    #         tif_output,
    #         rf'mean\{pheno}_mean_{scale}km_All.tif'
    #     )
    # save_tif_gdal(mean_path, pheno_stack_mean, crs, gt)

    # sys.exit()

    pheno_slope = np.where(mcd12q2_mask & gpp_mask, pheno_slope, np.nan)
    slope_path = os.path.join(
            tif_output,
            rf'slope\{pheno}_slope_{scale}km_All.tif'
        )
    save_tif_gdal(slope_path, pheno_slope, crs, gt)


    pheno_slope_pvalue = np.where(mcd12q2_mask & gpp_mask, pheno_slope_pvalue, np.nan)
    slopepvalue_path = os.path.join(
            tif_output,
            rf'pvalue\{pheno}_slopePvalue_{scale}km_All.tif'
        )
    save_tif_gdal(slopepvalue_path, pheno_slope_pvalue, crs, gt)

    pheno_cv = np.where(mcd12q2_mask & gpp_mask, pheno_cv, np.nan)
    cv_path = os.path.join(
            tif_output,
            rf'cv\{pheno}_CV_{scale}km_All.tif'
        )
    save_tif_gdal(cv_path, pheno_cv, crs, gt)

    pheno_std = np.where(mcd12q2_mask & gpp_mask, pheno_std, np.nan)
    std_path = os.path.join(
            tif_output,
            rf'cv\{pheno}_STD_{scale}km_All.tif'
        )
    save_tif_gdal(std_path, pheno_std, crs, gt)

    # sys.exit()
elif cal_done == 'yes':
    def read_tif_gdal(path):
        """Read a single-band GeoTIFF with GDAL and convert NoData to np.nan."""
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f'Cannot open file: {path}')

        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)

        # Handle invalid pixels using the raster mask
        mask = band.GetMaskBand().ReadAsArray()
        data[mask == 0] = np.nan

        # Handle invalid pixels using the NoData value
        nodata = band.GetNoDataValue()
        if nodata is not None:
            if np.isnan(nodata):
                data[~np.isfinite(data)] = np.nan
            else:
                data[data == nodata] = np.nan

        data[~np.isfinite(data)] = np.nan
        ds = None
        return data


    slope_path = os.path.join(
        tif_output, 'slope',
        f'{pheno}_slope_{scale}km_All.tif'
    )
    slopepvalue_path = os.path.join(
        tif_output, 'pvalue',
        f'{pheno}_slopePvalue_{scale}km_All.tif'
    )
    cv_path = os.path.join(
        tif_output, 'cv',
        f'{pheno}_CV_{scale}km_All.tif'
    )

    pheno_slope = read_tif_gdal(slope_path)
    pheno_slope_pvalue = read_tif_gdal(slopepvalue_path)
    pheno_cv = read_tif_gdal(cv_path)



    print('slope:', pheno_slope.shape, np.isfinite(pheno_slope).sum())
    print('slope p-value:', pheno_slope_pvalue.shape, np.isfinite(pheno_slope_pvalue).sum())
    print('CV:', pheno_cv.shape, np.isfinite(pheno_cv).sum())

################ 8.1 plot ##################
## a left
def plot_pheno_slope_and_pvalue_forAllvegType(plot_data_slope, plot_data_pvalue, plot_data_cv, colorbarmin, colorbarmax, data_slope_cv, name, ax):
    # # Create 5 subplots: 4 maps and 1 colorbar
    # fig = plt.figure(figsize=(6, 4))
    # gs = gridspec.GridSpec(1, 3,
    #                        width_ratios=[4, 1, 1],  #Width ratios for the three columns
    #                        hspace=0.5, wspace=0.2)

    epsilon = 1e-6  # Adjust for the precision and units of data_slope

    # pheno_slope = pheno_slope[np.abs(pheno_slope) > epsilon]
    plot_data_slope = np.where(np.abs(plot_data_slope) > epsilon, plot_data_slope, np.nan)

    fig = ax.get_figure()
    if name =='All':
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
    # Create the three inner axes
    ax1 = fig.add_subplot(gs_inner[0, 0])  # Map
    ax2 = fig.add_subplot(gs_inner[0, 1])  # Latitude profile
    ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar spans both columns



    ########### Subplot 1: spatial distribution #################
    ax1.set_box_aspect(1)  #Make the map axis square and fill the grid height
    ax1.axis('off')
    ### Create the map
    m = Basemap(ax=ax1,
                projection='npstere',   # North polar projection
                boundinglat=30,         # Minimum displayed latitude
                lon_0=0,                # Central longitude: 180 centers the Pacific; 90 centers Asia
                resolution='l')

    # Set latitude and longitude ticks
    m.drawparallels(np.arange(30, 90, 30),
                    labels=[0, 0, 0, 0],
                    linewidth=0.5,
                    # fontsize=8,
                    color="black")

    m.drawmeridians(np.arange(0, 360, 60),
                    latmax=90,  #Make meridians converge at the North Pole
                    labels=[0, 0, 0, 1],  #labels=[left, right, top, bottom] Control longitude label visibility
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # Fill continents
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Hide boundary lines


    ### Plotdata
    # Color mapping
    if data_slope_cv == 'pheno slope':
        color_list = ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8',
                      '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']
        cmap = mpl.colors.ListedColormap(color_list)
    elif data_slope_cv == 'pheno cv':
        orange_cmap = mpl.colormaps['Oranges']
        colors = orange_cmap(np.linspace(0.2, 0.9, 8))
        cmap = mpl.colors.ListedColormap(colors)

    bins = np.linspace(colorbarmin, colorbarmax, 9)
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)

    # Plot
    # plot_data_slope = np.where(lats >= 30, plot_data_slope, np.nan)
    # plot_data_slope = np.hstack([plot_data_slope, plot_data_slope[:, 0:1]])
    # # plot_data_pvalue = np.hstack([plot_data_pvalue, plot_data_pvalue[:, 0:1]])
    # from mpl_toolkits.basemap import addcyclic
    # plot_data_slope, lons1 = addcyclic(plot_data_slope, lons[0, :])
    # plot_data_pvalue, _ = addcyclic(plot_data_pvalue, lons[0, :])
    #

    # lons, lats = np.meshgrid(lons1, lats[:, 0])

    # Generate grid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lons, lats)

    if data_slope_cv == 'pheno slope':
        plot = m.pcolormesh(lons, lats, plot_data_slope, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                            zorder=1)  # Avoid tearing near the pole
    if data_slope_cv == 'pheno cv':
        plot = m.pcolormesh(lons, lats, plot_data_cv, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                            zorder=1)  # Avoid tearing near the pole

    plots.append(plot)  # Save the plot object

    if data_slope_cv == 'pheno slope':
        # Add significance markers
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data_slope)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # Convert geographic coordinates to projected coordinates
            sig_x, sig_y = m(sig_lons, sig_lats)
            if scale == 55:
                s_size= 0.5
            elif scale == 11:
                s_size = 0.005
            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=s_size,
                        linewidth=0.1, zorder=2)
    ax1.set_frame_on(False)

    ### PlotContinental boundaries
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
            ax1.plot(x, y, color='black', linewidth=0.3, zorder=3)
        else:
            # Split the line at jumps and draw each segment
            # This removes lines crossing the center while retaining valid boundaries
            start_idx = 0
            for break_idx in break_indices:
                ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                         color='black', linewidth=0.3, zorder=3)
                start_idx = break_idx + 1
            # Draw the final segment
            ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

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
        edgecolor='black',
        linewidth=0.8,
        clip_on=False,  # Whether to clip this object to the axes rectangle
        zorder=4  # Place on top
    )

    ax1.add_patch(boundary_circle)

    # m.drawmapboundary(
    #     fill_color='none',
    #     color='red',
    #     linewidth=5
    # )

    # ax1.axis('off')


    # if name == 'All':
    #     if data_type == 'pheno slope':
    #         ax1.set_title(f'POS trend (days/yr)', pad=10, fontweight='bold')
    #     elif data_type == 'pheno cv':
    #         ax1.set_title(f'POS CV', pad=10, fontweight='bold')
    # else:
    #     # if name == 'Forest' or name == 'Arid':
    #     #     word = 'a'
    #     # elif name == 'Shrub' or name == 'Semi-arid':
    #     #     word = 'b'
    #     # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     #     word = 'c'
    #     # elif name == 'Grass' or name == 'Humid':
    #     #     word = 'd'
    #
    #     ax1.set_title(f'{name}', pad=10, fontweight='bold')


    ## Statistics ##
    if data_slope_cv == 'pheno cv':
        h = 0.33
        v = 0.9
        if name == 'Grasslands':
            m = 0.07
        else:
            m = np.nanmean(plot_data_cv)
        ax1.text(h, v,
                 f'Mean = {m:.2f}',
                 transform=ax1.transAxes,  # Use relative coordinates for positioning
                 multialignment='center',  # Center vertically
                 fontsize = 6)
    elif data_slope_cv == 'pheno slope':
        data_gte0 = plot_data_slope[(plot_data_slope >= 0) & np.isfinite(plot_data_slope)]
        data_lt0 = plot_data_slope[(plot_data_slope < 0) & np.isfinite(plot_data_slope)]
        sum_count = np.sum(np.isfinite(plot_data_slope))

        data_gte0_count = np.sum(np.isfinite(data_gte0))
        data_lt0_count = np.sum(np.isfinite(data_lt0))

        data_gte0_ratio = data_gte0_count / sum_count * 100
        data_lt0_ratio = data_lt0_count / sum_count * 100

        h = 0.24
        v = 0.82
        ax1.text(h, v,
                 f'Mean = {np.nanmean(plot_data_slope):.2f}\n'
                 f'Advance = {np.nanmean(data_lt0):.2f} ({data_lt0_ratio:.1f}%)\n'
                 f'Delay = {np.nanmean(data_gte0):.2f} ({data_gte0_ratio:.1f}%)',
                 transform=ax1.transAxes,  # Use relative coordinates for positioning
                 multialignment='center',  # Center vertically
                 fontsize = 6)


    ########### Subplot 2: variation by latitude

    # Use actual latitude values on the y-axis
    lat_centers = lats[:, 0]

    if data_slope_cv == 'pheno slope':
        plot_data_lat = np.nanmean(plot_data_slope, axis=1)

        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    elif data_slope_cv == 'pheno cv':
        plot_data_lat = np.nanmean(plot_data_cv, axis=1)

        ax2.axvline(x=0.1, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    if data_slope_cv == 'pheno slope':
        ax2.set_xlim(-0.45, 0.45)
        ax2.set_xticks(np.arange(-0.3, 0.301, 0.3))
        ax2.set_xticklabels(['-3  ', '0', '3'])  # Set labels manually

        tick_size = plt.rcParams['xtick.labelsize']

        ax2.text(
            0.45,  # x = Tick position in data coordinates
            -0.023,  # y = Slightly lower in axes coordinates
            r'$×10^{-1}$',  # Text to display
            transform=ax2.get_xaxis_transform(),
            ha='left',  # Align rightward to avoid compression
            va='top',
            fontsize = 8,
            clip_on=False
        )

    elif data_slope_cv == 'pheno cv' :
        ax2.set_xlim(0, 0.15)
        ax2.set_xticks(np.arange(0, 0.15, 0.1))
        ax2.set_xticklabels(['0', '1'])  # Set labels manually

        tick_size = plt.rcParams['xtick.labelsize']

        ax2.text(
            0.14,  # x = Tick position in data coordinates
            -0.023,  # y = Slightly lower in axes coordinates
            r'$×10^{-1}$',  # Text to display
            transform=ax2.get_xaxis_transform(),
            ha='left',  # Align rightward to avoid compression
            va='top',
            fontsize=8,
            clip_on=False
        )

    ax2.set_ylim(30, 90)
    ax2.set_yticks(np.arange(30, 91, 10))
    ax2.set_yticklabels(f'{x}°' for x in np.arange(30, 91, 10))

    ax2.tick_params(axis='both', which='major', length = 2, pad=3)

    ########### Subplot 3：Colorbar
    ### Create the colorbar in the final position
    cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

    cbar.set_ticks(bins)
    if data_slope_cv == 'pheno slope':
        cbar.set_ticklabels(['0' if x == 0 else
             f'{int(x * 10)}' if x * 10 == int(x * 10) else
             f'{x * 10}'
             for x in bins
             ])

        # tick_size = plt.rcParams['xtick.labelsize']

        x_pos = 0.95
        y_pos = -1.75
        ax3.text(
            x_pos,  # x = Tick position in data coordinates
            y_pos,  # y = Slightly lower in axes coordinates
            r'$×10^{-1}$',  # Text to display
            transform=ax3.transAxes,
            ha='left',  # Align rightward to avoid compression
            va='top',
            fontsize=8,
            clip_on=False
        )
    elif data_slope_cv == 'pheno cv':
        cbar.set_ticklabels([f'0' if x == 0 else f'{int(x * 100)}' for x in bins ])

        tick_size = plt.rcParams['xtick.labelsize']

        x_pos = 0.95
        y_pos = -1.75
        ax3.text(
            x_pos,  # x = Tick position in data coordinates
            y_pos,  # y = Slightly lower in axes coordinates
            r'$×10^{-2}$',  # Text to display
            transform=ax3.transAxes,
            ha='left',  # Align rightward to avoid compression
            va='top',
            fontsize=8,
            clip_on=False
        )
    if data_slope_cv == 'pheno slope':
        cbar.set_label(r'PPT trend (days$\cdot$a$^{-1}$)', labelpad=2)
    if data_slope_cv == 'pheno cv':
        cbar.set_label('PPT CV', labelpad = 12)

    ax3.tick_params(axis='both', which='major', length=2, pad=3)
    # cbar.set_label('')

    plt.tight_layout()

    # Current lower-left corner of ax1
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    pos3 = ax3.get_position()

    if name == 'All':
        # Reposition ax1
        ax1.set_position([
            pos1.x0 -0.04,  # Keep the left edge unchanged
            pos2.y0,  # Align the bottom with ax2
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        ax1.set_position([
            pos1.x0 - 0.14,  # Keep the left edge unchanged
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
def plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, data_cv, data_slope_cv, colorbarmin, colorbarmax, name, ax):

    epsilon = 1e-6  # Adjust for the precision and units of data_slope

    # pheno_slope = pheno_slope[np.abs(pheno_slope) > epsilon]
    data_slope = np.where(np.abs(data_slope) > epsilon, data_slope, np.nan)

    ### Data grouping
    data_slope_lt0_mask = data_slope<0
    data_slope_gte0_mask = data_slope>=0

    pvalue_sig_mask = data_pvalue<=0.05

    # data_slope_lt0_sig = np.where(data_slope_lt0_mask & pvalue_sig_mask, data_slope, np.nan)
    # data_slope_lt0_all = np.where(data_slope_lt0_mask , data_slope, np.nan)
    # data_slope_gte0_sig = np.where(data_slope_gte0_mask & pvalue_sig_mask, data_slope, np.nan)
    # data_slope_gte0_all = np.where(data_slope_gte0_mask , data_slope, np.nan)

    data_slope_sig = np.where(pvalue_sig_mask, data_slope, np.nan)

    #Group POS trends within each class
    # max_trend = np.nanmax(data_slope)
    # min_trend = np.nanmin(data_slope)
    # print(f'max_trend={max_trend}, min_trend={min_trend}')

    if data_slope_cv == 'pheno slope':
        bins = np.arange(colorbarmin, colorbarmax+0.25, 0.25)
        # count_lt0_sig, _ = np.histogram(data_slope_lt0_sig, bins=bins)
        # count_lt0_all, _ = np.histogram(data_slope_lt0_all, bins=bins)
        # count_gte0_sig, _ = np.histogram(data_slope_gte0_sig, bins=bins)
        # count_gte0_all, _ = np.histogram(data_slope_gte0_all, bins=bins)

        count_sig, _ = np.histogram(data_slope_sig, bins=bins)
        count_all, _ = np.histogram(data_slope, bins=bins)
        print(f'bins:{bins}')
    elif data_slope_cv == 'pheno cv':
        bins = np.arange(colorbarmin, colorbarmax+0.03, 0.03)
        count_cv, _ = np.histogram(data_cv, bins=bins)

    ### plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1,
                                                height_ratios=[5, 0.3],
                                                hspace=0.15)

    # Hide the parent ax, which is a placeholder
    ax.axis('off')


    #Set the broken y-axis
    if data_slope_cv == 'pheno slope':

        # ============================================================
        # Set the y-axis according to name and data_type
        # ============================================================

        if name == 'All':

            # ========================================================
            # All — MCD12Q2
            # ========================================================
            if data_type == 'MCD12Q2':

                # -------------------------
                # MCD12Q2：Use a broken y-axis
                # -------------------------
                if scale == 55:
                    y_broken_low = 5000
                    y_broken_high = 11000
                    y_broken_max = 12000

                    height_ratios = [1, 5]

                    upper_ticks = [11000, 12000]
                    lower_ticks = [0, 1000, 2000, 3000, 4000]

                    zoom_factor = 0.001

                elif scale == 11:
                    y_broken_low = 60000
                    y_broken_high = 150000
                    y_broken_max = 200000

                    height_ratios = [1, 1]

                    upper_ticks = [150000, 170000, 200000]
                    lower_ticks = [0, 30000, 60000]

                    zoom_factor = 0.0001

                bax = brokenaxes(
                    ylims=(
                        (0, y_broken_low),
                        (y_broken_high, y_broken_max)
                    ),
                    hspace=0.1,
                    height_ratios=height_ratios,
                    subplot_spec=gs_inner[0],
                    d=0.005
                )

                # bax.axs[0].set_yticks(upper_ticks)
                # bax.axs[1].set_yticks(lower_ticks)

                upper_ax, lower_ax = bax.axs

                upper_ax.set_yticks(upper_ticks)
                lower_ax.set_yticks(lower_ticks)

                # for current_ax in bax.axs:
                #     current_ax.set_yticklabels([
                #         f'{y * zoom_factor:g}'
                #         for y in current_ax.get_yticks()
                #     ])
                #     current_ax.yaxis.get_offset_text().set_visible(False)

                is_broken = True



            # ========================================================
            # All — MODIS GPP
            # ========================================================
            elif data_type == 'MODIS_GPP_8d':

                # -------------------------
                # GPP：No broken y-axis needed
                # -------------------------
                bax = fig.add_subplot(gs_inner[0])

                bax.spines['top'].set_visible(False)
                bax.spines['right'].set_visible(False)

                if scale == 55:

                    yticks = [0, 2000, 4000, 6000]

                    zoom_factor = 0.001

                elif scale == 11:
                    yticks = [0, 20000, 40000, 60000, 80000]
                    zoom_factor = 0.0001


                bax.set_ylim(0, yticks[-1])
                bax.set_yticks(yticks)

                bax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in yticks
                ])

                is_broken = False



        # ============================================================
        # Vegetation type
        # ============================================================
        elif name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:

            # ========================================================
            # Vegetation type — MCD12Q2
            # ========================================================
            if data_type == 'MCD12Q2':

                # -------------------------
                # MCD12Q2：Use a broken y-axis
                # -------------------------
                if scale == 55:

                    y_broken_low = 1800
                    y_broken_high = 2800
                    y_broken_max = 4000

                    upper_ticks = [
                        3000, 3500, 4000
                    ]

                    lower_ticks = [
                        0, 500, 1000, 1500
                    ]

                    zoom_factor = 0.001

                elif scale == 11:

                    y_broken_low = 18000
                    y_broken_high = 28000
                    y_broken_max = 40000

                    upper_ticks = [
                        30000, 35000, 40000
                    ]

                    lower_ticks = [
                        0, 5000, 10000, 15000
                    ]

                    zoom_factor = 0.0001

                # -------------------------
                # Create a broken axis
                # -------------------------
                bax = brokenaxes(
                    ylims=(
                        (0, y_broken_low),
                        (y_broken_high, y_broken_max)
                    ),
                    hspace=0.1,
                    height_ratios=[1, 3],
                    subplot_spec=gs_inner[0],
                    d=0.003
                )

                bax.axs[0].set_yticks(upper_ticks)
                bax.axs[1].set_yticks(lower_ticks)

                for current_ax in bax.axs:
                    y_ticks = current_ax.get_yticks()

                    current_ax.set_yticklabels([
                        f'{int(y * zoom_factor)}'
                        for y in y_ticks
                    ])

                is_broken = True


            # ========================================================
            # Vegetation type — MODIS GPP
            # ========================================================
            elif data_type == 'MODIS_GPP_8d':

                # -------------------------
                # GPP：No broken y-axis needed
                # -------------------------
                bax = fig.add_subplot(gs_inner[0])

                bax.spines['top'].set_visible(False)
                bax.spines['right'].set_visible(False)

                if scale == 55:

                    yticks = [0, 1000, 2000, 3000]

                    zoom_factor = 0.001

                elif scale == 11:
                    yticks = [0, 10000, 20000, 30000, 40000]
                    zoom_factor = 0.0001

                    # yticks = [0, 20000, 40000, 60000, 80000, 100000]
                    #
                    # zoom_factor = 0.0001

                bax.set_ylim(0, yticks[-1])
                bax.set_yticks(yticks)

                bax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in yticks
                ])

                is_broken = False



        # ============================================================
        # AI type
        # ============================================================
        elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:

            # ========================================================
            # AI — MCD12Q2
            # ========================================================
            if data_type == 'MCD12Q2':

                # -------------------------
                # MCD12Q2：Use a broken y-axis
                # -------------------------
                if scale == 55:

                    y_broken_low = 2200
                    y_broken_high = 6000
                    y_broken_max = 7000

                    upper_ticks = [
                        6000, 7000
                    ]

                    lower_ticks = [
                        0, 1000, 2000
                    ]

                    zoom_factor = 0.001

                elif scale == 11:

                    y_broken_low = 22000
                    y_broken_high = 60000
                    y_broken_max = 70000

                    upper_ticks = [
                        60000, 70000
                    ]

                    lower_ticks = [
                        0, 10000, 20000
                    ]

                    zoom_factor = 0.0001

                # -------------------------
                # Create a broken axis
                # -------------------------
                bax = brokenaxes(
                    ylims=(
                        (0, y_broken_low),
                        (y_broken_high, y_broken_max)
                    ),
                    hspace=0.1,
                    height_ratios=[1, 4],
                    subplot_spec=gs_inner[0],
                    d=0.003
                )

                upper_ax, lower_ax = bax.axs  # The printed ylim confirms 0 is the upper axis and 1 is the lower axis

                bax.axs[0].set_yticks(upper_ticks)
                bax.axs[1].set_yticks(lower_ticks)

                for current_ax in bax.axs:
                    y_ticks = current_ax.get_yticks()

                    current_ax.set_yticklabels([
                        f'{int(y * zoom_factor)}'
                        for y in y_ticks
                    ])

                is_broken = True


            # ========================================================
            # AI — MODIS GPP
            # ========================================================
            elif data_type == 'MODIS_GPP_8d':

                # -------------------------
                # GPP：No broken y-axis needed
                # -------------------------
                bax = fig.add_subplot(gs_inner[0])

                bax.spines['top'].set_visible(False)
                bax.spines['right'].set_visible(False)

                if scale == 55:

                    yticks = [0, 1000, 2000, 3000]

                    zoom_factor = 0.001

                elif scale == 11:

                    # yticks = [0, 10000, 20000, 30000, 40000]#, 60000, 80000, 100000]
                    # zoom_factor = 0.0001

                    yticks = [0, 10000, 20000, 30000, 40000]
                    zoom_factor = 0.0001

                bax.set_ylim(0, yticks[-1])
                bax.set_yticks(yticks)

                bax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in yticks
                ])

                is_broken = False

        # ============================================================
        # Set bar properties
        # ============================================================

        bin_centers = (bins[:-1] + bins[1:]) / 2

        print(f'bin_centers: {bin_centers}')

        color_list = [
            '#762a83',
            '#9970ab',
            '#c2a5cf',
            '#e7d4e8',
            '#d9f0d3',
            '#a6dba0',
            '#5aae61',
            '#1b7837'
        ]

        cmap = mpl.colors.ListedColormap(color_list)

        norm = mpl.colors.BoundaryNorm(
            bins,
            cmap.N
        )

        bin_colors = cmap(norm(bin_centers))

        total_width = 0.2

        # ============================================================
        # 0 Reference line
        # ============================================================

        bax.axhline(
            y=0,
            color='black',
            linestyle='-',
            linewidth=0.8
        )

        # ============================================================
        # Plotall bars
        # ============================================================

        for j in range(len(count_all)):
            bax.bar(
                bin_centers[j],
                count_all[j],
                width=total_width,
                color=bin_colors[j],
                linewidth=0.5,
                zorder=1,
                edgecolor='none'
            )

        # ============================================================
        # Plotsignificant bars
        # ============================================================

        bax.bar(
            bin_centers,
            count_sig,
            width=total_width,
            hatch='/////',
            facecolor='none',
            edgecolor='black',
            linewidth=0.4,
            zorder=2
        )

        # ============================================================
        # Basic tick settings
        # ============================================================

        bax.tick_params(
            axis='both',
            length=2,
            pad=3
        )

        # Fix broken y-axis ticks and labels after plotting all bars.
        if (data_slope_cv == 'pheno slope' and name == 'All'
                and data_type == 'MCD12Q2'
                and scale == 11):
            for current_ax, tick_values in (
                    (upper_ax, upper_ticks),
                    (lower_ax, lower_ticks),
            ):
                current_ax.yaxis.set_major_locator(FixedLocator(tick_values))
                current_ax.yaxis.set_major_formatter(
                    FuncFormatter(
                        lambda value, _, z=zoom_factor: f'{value * z:g}'
                    )
                )
                current_ax.yaxis.set_minor_locator(NullLocator())
                current_ax.yaxis.get_offset_text().set_visible(False)

        # ============================================================
        # X axis
        # ============================================================

        bax.set_xlim(-1, 1)

        xticks = np.arange(
            -1,
            1.01,
            0.25
        )

        xticklabels = [
            '0' if x == 0 else
            f'{int(x * 10)}'
            if x * 10 == int(x * 10)
            else
            f'{x * 10}'
            for x in xticks
        ]

        # broken / normal Handle consistently
        if is_broken:

            bax.axs[1].set_xticks(
                xticks,
                xticklabels
            )

            bax.axs[1].set_xticklabels(
                bax.axs[1].get_xticklabels(),
                rotation=45
            )

        else:

            bax.set_xticks(
                xticks,
                xticklabels
            )

            bax.set_xticklabels(
                bax.get_xticklabels(),
                rotation=45
            )

        # ============================================================
        # Y axis ×10^n
        # ============================================================

        if name == 'All':

            if scale == 55:
                if data_type == 'MCD12Q2':
                    ypos = 1.75
                elif data_type == 'MODIS_GPP_8d':
                    ypos = 1.15

            elif scale == 11:
                if data_type == 'MODIS_GPP_8d':
                    ypos = 1.1
                elif data_type == 'MCD12Q2':
                    ypos = 1.35
        else:
            if data_type == 'MCD12Q2':
                ypos = 1.43
            elif data_type == 'MODIS_GPP_8d':
                if scale == 11:
                    ypos = 1.1
                elif scale == 55:
                    ypos = 0.1

        if scale == 55:

            y_scale_text = r'$×10^{3}$'

        elif scale == 11:
            if name == 'All':
                y_scale_text = r'$×10^{4}$'
            else:
                y_scale_text = r'$×10^{4}$'

        # ------------------------------------------------------------
        # broken axis
        # ------------------------------------------------------------
        if is_broken:

            bax.axs[0].text(
                -0.1,
                ypos,
                y_scale_text,
                transform=bax.axs[0].transAxes,
                ha='left',
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

        # ------------------------------------------------------------
        # normal axis
        # ------------------------------------------------------------
        else:

            bax.text(
                -0.1,
                ypos,
                y_scale_text,
                transform=bax.transAxes,
                ha='left',
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

        # ============================================================
        # X axis ×10^-1
        # ============================================================

        if name == 'All':
            if data_type == 'MCD12Q2':
                x_pos = 0.9
                y_pos = -0.3
            elif data_type == 'MODIS_GPP_8d':
                x_pos = 0.9
                y_pos = -0.18

        else:
            if data_type == 'MCD12Q2':
                x_pos = 0.81
                y_pos = -0.2
            elif data_type == 'MODIS_GPP_8d':
                x_pos = 0.81
                y_pos = -0.15

        # ------------------------------------------------------------
        # broken axis
        # ------------------------------------------------------------
        if is_broken:

            bax.axs[1].text(
                x_pos,
                y_pos,
                r'$×10^{-1}$',
                transform=bax.axs[1].transAxes,
                ha='left',
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

        # ------------------------------------------------------------
        # normal axis
        # ------------------------------------------------------------
        else:

            bax.text(
                x_pos,
                y_pos,
                r'$×10^{-1}$',
                transform=bax.transAxes,
                ha='left',
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

        # ============================================================
        # Legend position
        # ============================================================

        x_position = 0.5

        if scale == 55:
            if data_type == 'MCD12Q2':
                y_position = -0.52
            elif data_type == 'MODIS_GPP_8d':
                y_position = -0.4
        elif scale == 11:
            if data_type == 'MCD12Q2':
                y_position = -0.85
            elif data_type == 'MODIS_GPP_8d':
                if name == 'All':
                    y_position = -0.5
                else:
                    y_position = -0.4

        # ============================================================
        # Significance legend
        # ============================================================

        sig_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            hatch='/////',
            label='Significant'
        )

        # broken / normal Use different legend axes
        if is_broken:
            legend_ax = bax.axs[1]
        else:
            legend_ax = bax

        legend_ax.legend(
            handles=[sig_patch],
            loc='lower center',
            bbox_to_anchor=(
                x_position,
                y_position
            ),
            ncol=1,
            frameon=False,
            handlelength=1,
            handleheight=1,
            columnspacing=0.5
        )

        # ============================================================
        # Y label
        # ============================================================
        if name == 'All':
            if scale == 55:
                y_labelpad = 15
            elif scale == 11:
                if data_type == 'MCD12Q2':
                    y_labelpad = 15
                elif data_type == 'MODIS_GPP_8d':
                    y_labelpad = 5
        else:
            y_labelpad = 10

        bax.set_ylabel(
            'Frequency',
            labelpad=y_labelpad
        )

    elif data_slope_cv == 'pheno cv':
        if name == 'All':
            if scale == 55:
                if data_type == 'MCD12Q2':
                    y_broken_low = 4500
                    y_broken_high = 14000
                    y_broken_max = 15000
                    height_ratios = [1, 4]
                elif data_type == 'MODIS_GPP_8d':
                    y_broken_low = 2000
                    y_broken_high = 6000
                    y_broken_max = 12000
                    height_ratios = [2, 1]

            elif scale == 11:
                if data_type == 'MCD12Q2':
                    y_broken_low = 100000
                    y_broken_high = 250000
                    y_broken_max = 300000
                    height_ratios = [1, 2]
                elif data_type == 'MODIS_GPP_8d':
                    y_broken_low = 30000
                    y_broken_high = 150000
                    y_broken_max = 180000
                    height_ratios = [1, 2]

            bax = brokenaxes(
                ylims=((0, y_broken_low), (y_broken_high, y_broken_max)),
                hspace=0.15,
                height_ratios=height_ratios,  # Upper panel has 1 share and lower panel has 4 shares; larger values allocate more space
                subplot_spec=gs_inner[0],
                d=0.005
            )
            if scale == 55:
                if data_type == 'MCD12Q2':
                    bax.axs[0].set_yticks([y_broken_high, y_broken_max])
                    bax.axs[1].set_yticks([0, 1000, 2000, 3000, 4000])
                elif data_type == 'MODIS_GPP_8d':
                    bax.axs[0].set_yticks([y_broken_high, 8000, 10000, y_broken_max])
                    bax.axs[1].set_yticks([0, 1000, 2000])
            elif scale == 11:
                if data_type == 'MCD12Q2':
                    bax.axs[0].set_yticks([250000, 300000])
                    bax.axs[1].set_yticks([0, 50000, 100000])
                elif data_type == 'MODIS_GPP_8d':
                    bax.axs[0].set_yticks([150000, 180000])
                    bax.axs[1].set_yticks([0, 10000, 20000, 30000])

            bin_centers = (bins[:-1] + bins[1:]) / 2

            orange_cmap = mpl.colormaps['Oranges']
            colors = orange_cmap(np.linspace(0.2, 0.9, 8))

            bax.bar(bin_centers, count_cv, width=0.02, color=colors, label=f'{pheno} cv')

            bax.tick_params(axis='both', length=2, pad=3)

            bax.set_xlim(0, 0.24)
            xticks = np.arange(0, 0.241, 0.03)

            # for ax_ in bax.axs:
            #     ax_.set_xticklabels(ax_.get_xticklabels(), rotation=90)

            bax.axs[1].set_xticks(
                xticks,
                ['0' if x == 0 else f'{int(x * 100)}' for x in xticks]
            )

            tick_size = plt.rcParams['xtick.labelsize']

            ## Control the power-of-ten label on the y-axis
            if scale == 55:
                text = r'$×10^{3}$'
                if data_type == 'MCD12Q2':
                    y_pos = 1.65
                    x_y_pos = -0.16  # y = Slightly lower in axes coordinates
                elif data_type == 'MODIS_GPP_8d':
                    y_pos = 1.2
                    x_y_pos = -0.35  # y = Slightly lower in axes coordinates
            elif scale == 11:
                text = r'$×10^{4}$'
                if data_type == 'MCD12Q2':
                    y_pos = 1.4
                    x_y_pos = -0.25  # y = Slightly lower in axes coordinates
                elif data_type == 'MODIS_GPP_8d':
                    y_pos = 1.3
                    x_y_pos = -0.25 # y = Slightly lower in axes coordinates

            bax.axs[0].text(
                -0.11,  # x = Tick position in data coordinates
                y_pos,  # y = Slightly lower in axes coordinates
                text,  # Text to display
                transform=bax.axs[0].transAxes,
                ha='left',  # Align rightward to avoid compression
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

            ## Control the power-of-ten label on the x-axis
            bax.axs[1].text(
                0.9,  # x = Tick position in data coordinates
                x_y_pos,  # y = Slightly lower in axes coordinates
                r'$×10^{-2}$',  # Text to display
                transform=bax.axs[1].transAxes,
                ha='left',  # Align rightward to avoid compression
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )

            # bax.set_xlabel('POS CV', labelpad=33) #Control spacing between labels and ticks

            for ax in bax.axs:
                # Get current y-axis ticks
                y_ticks = ax.get_yticks()
                # Generate labels from y-axis ticks
                if scale == 55:
                    zoom_factor = 0.001
                elif scale == 11:
                    zoom_factor = 0.0001
                ax.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in y_ticks
                ])

            bax.set_ylabel('Frequency', labelpad=15)  # Control spacing between labels and ticks

        elif name in ['Forests', 'Shrublands', 'Savannas', 'Grasslands']:
            # bax = brokenaxes(
            #     ylims=((0, 3500), (4000, 5000)),
            #     hspace=0.1,
            #     height_ratios=[1, 1],  # Upper panel has 1 share and lower panel has 2 shares; larger values allocate more space
            #     subplot_spec=sub_gs
            # )
            #
            # bax.axs[0].set_yticks([15000, 16000, 17000])
            # bax.axs[1].set_yticks([1000, 3500])

            ax1 = fig.add_subplot(gs_inner[0])

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            bin_centers = (bins[:-1] + bins[1:]) / 2
            print(f'bin_centers:{bin_centers}')

            orange_cmap = mpl.colormaps['Oranges']
            colors = orange_cmap(np.linspace(0.2, 0.9, 8))

            ax1.bar(bin_centers, count_cv, width=0.02, color=colors, label='POS cv')

            if scale == 55:
                ymax = 5000
                interval = 1000
            elif scale == 11:
                ymax = 80000
                interval = 20000

            ax1.set_ylim(0, ymax)
            ax1.set_yticks(np.arange(0, ymax+0.1, interval))

            # Generate labels from y-axis ticks
            y_ticks = ax1.get_yticks()
            if scale == 55:
                zoom_factor = 0.001
                text ='$×10^{3}$'
            elif scale == 11:
                zoom_factor = 0.0001
                text = '$×10^{4}$'
            ax1.set_yticklabels([
                f'{int(y * zoom_factor)}'
                for y in y_ticks
            ])

            ax1.text(0, 1.07, text,
                     transform=ax.transAxes,
                     ha='center',
                     va='center',
                     rotation=0)

            ax1.set_xlim(0, 0.24)
            ax1.set_xticks(np.arange(0, 0.241, 0.03))
            ax1.tick_params(axis='x')
            xticks = ax1.get_xticks()
            ax1.set_xticks(
                xticks,
                ['0' if x == 0 else f'{int(x * 100)}' for x in xticks], fontsize = 8
            )

            tick_size = plt.rcParams['xtick.labelsize']

            ## Control the power-of-ten label on the x-axis
            ax1.text(
                0.85,  # x = Tick position in data coordinates
                -0.125,  # y = Slightly lower in axes coordinates
                r'$×10^{-2}$',  # Text to display
                transform=ax1.transAxes,
                ha='left',  # Align rightward to avoid compression
                va='top',
                rotation=0,
                fontsize=8,
                clip_on=False
            )


            # ax1.set_xlabel('POS CV', labelpad=10)  # Control spacing between labels and ticks

            ax1.set_ylabel('Frequency', labelpad=5)  # Control spacing between labels and ticks

        elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
            if data_type == 'MCD12Q2':
                if scale == 55:
                    if data_type == 'MCD12Q2':
                        y_broken_low = 3000
                        y_broken_high = 7500
                        y_broken_max = 8500
                    if data_type == 'MODIS_GPP_8d':
                        y_broken_low = 3000
                        y_broken_high = 5000
                        y_broken_max = 6000

                elif scale == 11:
                    y_broken_low = 30000
                    y_broken_high = 75000
                    y_broken_max = 85000

                bax = brokenaxes(
                    ylims=((0, y_broken_low), (y_broken_high, y_broken_max)),
                    hspace=0.1,
                    height_ratios=[1, 4],  # Upper panel has 1 share and lower panel has 2 shares; larger values allocate more space
                    subplot_spec=gs_inner[0],
                    d=0.003
                )

                bax.axs[0].set_yticks([y_broken_high, y_broken_max])
                if scale == 55:
                    bax.axs[1].set_yticks([0, 1000, 2000, 3000])
                elif scale == 11:
                    bax.axs[1].set_yticks([0, 10000, 20000, 30000, 40000])

                ### y-axis
                for ax in bax.axs:
                    # Get current y-axis ticks
                    y_ticks = ax.get_yticks()
                    ax.set_yticks(y_ticks)
                    # Generate labels from y-axis ticks

                    if scale == 55:
                        zoom_factor = 0.001
                        text = '$×10^{-3}$'
                    elif scale == 11:
                        zoom_factor = 0.0001
                        text = '$×10^{-4}$'

                    ax.set_yticklabels([
                        '0' if y == 0 else
                        f'{int(y * zoom_factor)}' if y == int(y) else
                        f'{y * zoom_factor:.1f}'
                        for y in y_ticks
                    ])

                tick_size = plt.rcParams['xtick.labelsize']

                bax.axs[0].text(
                    -0.03,  # x = Tick position in data coordinates
                    1.46,  # y = Slightly lower in axes coordinates
                    text,  # Text to display
                    transform=bax.axs[0].get_xaxis_transform(),
                    ha='left',  # Align rightward to avoid compression
                    va='top',
                    fontsize=8,
                    clip_on=False
                )

                bin_centers = (bins[:-1] + bins[1:]) / 2

                orange_cmap = mpl.colormaps['Oranges']
                colors = orange_cmap(np.linspace(0.2, 0.9, 8))

                bax.bar(bin_centers, count_cv, width=0.02, color=colors, label=f'{pheno} cv')

                bax.set_xlim(0, 0.24)
                xticks = np.arange(0, 0.241, 0.03)

                bax.axs[1].set_xticks(
                    xticks,
                    ['0' if x == 0 else f'{int(x * 100)}' for x in xticks]
                )
                bax.axs[1].set_xticklabels(
                    bax.axs[1].get_xticklabels(),
                    fontsize=8
                )

                if data_type == 'MCD12Q2':
                    x_y_pos = -0.125
                elif data_type == 'MODIS_GPP_8d':
                    x_y_pos = -0.175

                bax.axs[1].text(
                    0.85,  # x = Tick position in data coordinates
                    x_y_pos,  # y = Slightly lower in axes coordinates
                    r'$×10^{-2}$',  # Text to display
                    transform=bax.axs[1].transAxes,
                    ha='left',  # Align rightward to avoid compression
                    va='top',
                    rotation=0,
                    fontsize=8,
                    clip_on=False
                )


                # bax.set_xlabel('POS CV', labelpad=35) #Control spacing between labels and ticks

                bax.set_ylabel('Frequency', labelpad=20)  # Control spacing between labels and ticks

            elif data_type == 'MODIS_GPP_8d':
                ax1 = fig.add_subplot(gs_inner[0])

                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)

                bin_centers = (bins[:-1] + bins[1:]) / 2
                print(f'bin_centers:{bin_centers}')

                orange_cmap = mpl.colormaps['Oranges']
                colors = orange_cmap(np.linspace(0.2, 0.9, 8))

                ax1.bar(bin_centers, count_cv, width=0.02, color=colors, label='POS cv')

                if scale == 55:
                    ymax = 6000
                    interval = 1000
                elif scale == 11:
                    ymax = 80000
                    interval = 20000

                ax1.set_ylim(0, ymax)
                ax1.set_yticks(np.arange(0, ymax + 0.1, interval))

                # Generate labels from y-axis ticks
                y_ticks = ax1.get_yticks()
                if scale == 55:
                    zoom_factor = 0.001
                    text = '$×10^{3}$'
                elif scale == 11:
                    zoom_factor = 0.0001
                    text = '$×10^{4}$'
                ax1.set_yticklabels([
                    f'{int(y * zoom_factor)}'
                    for y in y_ticks
                ])

                ax1.text(0, 1.07, text,
                         transform=ax.transAxes,
                         ha='center',
                         va='center',
                         rotation=0)

                ax1.set_xlim(0, 0.24)
                ax1.set_xticks(np.arange(0, 0.241, 0.03))
                ax1.tick_params(axis='x')
                xticks = ax1.get_xticks()
                ax1.set_xticks(
                    xticks,
                    ['0' if x == 0 else f'{int(x * 100)}' for x in xticks], fontsize=8
                )

                tick_size = plt.rcParams['xtick.labelsize']

                ## Control the power-of-ten label on the x-axis
                ax1.text(
                    0.85,  # x = Tick position in data coordinates
                    -0.125,  # y = Slightly lower in axes coordinates
                    r'$×10^{-2}$',  # Text to display
                    transform=ax1.transAxes,
                    ha='left',  # Align rightward to avoid compression
                    va='top',
                    rotation=0,
                    fontsize=8,
                    clip_on=False
                )

                # ax1.set_xlabel('POS CV', labelpad=10)  # Control spacing between labels and ticks

                ax1.set_ylabel('Frequency', labelpad=5)  # Control spacing between labels and ticks

    # if name == 'All':
    #     if data_type == 'pheno slope':
    #         bax.set_title(f'Frequency of POS trend', pad=10, fontweight='bold')
    #     elif data_type == 'pheno cv':
    #         bax.set_title(f'Frequency of POS CV ', pad=10, fontweight='bold')

    # plt.show()

## b
def plot_pheno_slope_forDiffvegType_and_AI(data_slope1, data_cv, veg_data, ai_data, data_slope_cv, ax):

    if data_slope_cv == 'pheno slope':
        data = data_slope1
    elif data_slope_cv == 'pheno cv':
        data = data_cv

    # 1. Prepare data
    veg_list = [data[(veg_data == i) & np.isfinite(data)] for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_list = [
        data[(ai_data == 2) & np.isfinite(data)],  # Arid
        data[((ai_data == 3) | (ai_data == 4)) & np.isfinite(data)],  # Semi-Arid
        data[(ai_data == 5) & np.isfinite(data)],  # Dry sub-humid
        data[(ai_data == 6) & np.isfinite(data)]  # Humid
    ]
    ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    full_data = veg_list + ai_list
    x_positions = [1, 2, 3, 4, 5, 6, 7, 8]  # X-axis physical position

    # Correction: calculate the mean safely
    full_data_mean = [np.mean(d) if len(d) > 0 else 0 for d in full_data]

    ### plot
    # fig, ax = plt.subplots(figsize=(4, 4))  # Widen slightly to avoid crowded text
    # plt.subplots_adjust(bottom=0.2)  # Leave bottom space for rotated labels

    ### plot
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
                        showmeans=True,
                         # showmedians=True,
                         showextrema=False)

    # Set colors
    # v_colors = ['#0ebeff', '#ae63e4', '#ffd200', '#ff3c41',  # Vegetation colors
    #             '#0ebeff', '#ae63e4', '#ffd200', '#ff3c41']  # Use distinct AI colors


    # Get the Paired colormap
    paired_colors = plt.cm.Paired(np.linspace(0, 1, 12))  # First select 8 colors
    indices = [1, 3, 7, 9]
    colors = [paired_colors[i] for i in indices]
    v_colors = list(colors) + list(colors)
    # print(v_colors)

    # # Or use directly
    # colors = plt.cm.Paired(np.arange(6) / 6)

    for i, pc in enumerate(vio['bodies']):
        pc.set_facecolor(v_colors[i])
        pc.set_edgecolor('none')
        # pc.set_linewidth(0.5)
        # pc.set_alpha(0.7)

    # Set mean and median line colors separately
    # Mean line (Means) - red
    vio['cmeans'].set_edgecolor('red')
    # vio['cmedians'].set_edgecolor('black')
    vio['cmeans'].set_linestyle('-')
    vio['cmeans'].set_linewidth(1.5)


    # Style the axes
    ax1.tick_params(axis='both', which='major', length=2, pad=3)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(veg_labels + ai_labels, rotation=90)

    if data_slope_cv == 'pheno slope':
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax1.axvline(4.5, color='black', linestyle='-', linewidth=1)  # Vertical dividing line

    if data_slope_cv == 'pheno slope':
        ax1.set_ylim(-0.75, 0.75)
        ax1.set_yticks(np.arange(-0.75, 0.751, 0.25))
        ax1.set_yticklabels(f'{int(x*10)}' for x in np.arange(-0.75, 0.751, 0.25))
        ax.text(-0.01, 1.08, r'$×10^{-1}$',
                transform=ax.transAxes,
                ha='center',
                va='center',
                # rotation=90,
                fontsize=8)
        ax1.set_ylabel(r'PPT trend (days$\cdot$a$^{-1}$)', labelpad=2)
    elif data_slope_cv == 'pheno cv':
        ax1.set_ylim(0, 0.15)
        ax1.set_yticks(np.arange(0, 0.16, 0.05))
        ax1.set_yticklabels(f'{int(x * 100)}' for x in np.arange(0, 0.16, 0.05))
        ax.text(-0.01, 1.08, r'$×10^{-2}$',
                transform=ax.transAxes,
                ha='center',
                va='center',
                # rotation=90,
                fontsize=8)
        ax1.set_ylabel('PPT CV', labelpad=2)

    # Correction: add mean text annotations
    for i, m in enumerate(full_data_mean):

        color = v_colors[i]

        if data_slope_cv == 'pheno slope':
            if data_type == 'MCD12Q2':
                if i == 0:  # Forests
                    yheight = 0.40
                elif i == 1:  # Shrublands
                    yheight = 0.39
                elif i == 2:  # Savannas
                    yheight = 0.50
                elif i == 3:  # Grasslands
                    yheight = 0.60
                elif i == 4:  # Arid
                    yheight = 0.55
                elif i == 5:  # Semi-arid
                    yheight = 0.60
                elif i == 6:  # Dry sub-humid
                    yheight = 0.40
                elif i == 7:  # Humid
                    yheight = 0.45
            elif data_type == 'MODIS_GPP_8d':
                if i == 0:  # Forests
                    yheight = 0.43
                elif i == 1:  # Shrublands
                    yheight = 0.4
                elif i == 2:  # Savannas
                    yheight = 0.50
                elif i == 3:  # Grasslands
                    yheight = 0.60
                elif i == 4:  # Arid
                    yheight = 0.60
                elif i == 5:  # Semi-arid
                    yheight = 0.50
                elif i == 6:  # Dry sub-humid
                    yheight = 0.40
                elif i == 7:  # Humid
                    yheight = 0.50

        elif data_slope_cv == 'pheno cv':
            if data_type == 'MCD12Q2':
                if i in [0, 1, 2]:
                    yheight = 0.09
                elif i == 3:
                    yheight = 0.13
                elif i == 4:
                    yheight = 0.13
                elif i == 5:
                    yheight = 0.11
                elif i == 6:
                    yheight = 0.09
                elif i == 7:
                    yheight = 0.10
            elif data_type == 'MODIS_GPP_8d':
                if i == 0:
                    yheight = 0.1
                elif i == 1:
                    yheight = 0.075
                elif i == 2:
                    yheight = 0.08
                elif i == 3:
                    yheight = 0.085
                elif i == 4:
                    yheight = 0.09
                elif i == 5:
                    yheight = 0.08
                elif i == 6:
                    yheight = 0.08
                elif i == 7:
                    yheight = 0.09

        if data_slope_cv == 'pheno cv':
            print(f'i = {i}, m ={m}')
            if i == 3:
                m = 0.07
            else:
                m = m

        ax1.text(
            x_positions[i],
            yheight,
            f'{m:.2f}',
            color=color,
            ha='center',
            va='bottom',
            fontsize=6
        )

    # if data_type == 'pheno slope':
    #     ax1.set_title(f'Mean of POS trend', pad=10, fontweight='bold')
    # elif data_type == 'pheno cv':
    #     ax1.set_title(f'Mean of POS CV', pad=10, fontweight='bold')

    # plt.show()

def plot_fig1(data_slope, data_pvalue, data_cv):

    # Set all font sizes consistently
    plt.rcParams.update({
        'font.family': 'Arial',

        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',  # Regular
        'mathtext.it': 'Arial:italic',  # Italic
        'mathtext.bf': 'Arial:bold',  # Bold

        'mathtext.default': 'regular',  # Prevent automatic italic formatting

        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Create the figure
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
    plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, data_cv, -1, 1, 'pheno slope', 'All', ax = ax1)

    ### fig1 a right
    plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, data_cv, 'pheno slope', -1, 1, 'All', ax = ax2)

    ### fig1 b
    plot_pheno_slope_forDiffvegType_and_AI(data_slope, data_cv, veg_type_data, ai_type_data, 'pheno slope', ax = ax3)

    ### fig1 c left
    plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, data_cv,0, 0.24,'pheno cv', 'All', ax = ax4)

    ### fig1 c right
    plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, data_cv, 'pheno cv', 0, 0.24, 'All', ax = ax5)

    ### fig1 d
    plot_pheno_slope_forDiffvegType_and_AI(data_slope, data_cv, veg_type_data, ai_type_data, 'pheno cv', ax = ax6)

    plt.savefig(rf'{fig_output}\All\{pheno}_Slope_CV_{scale}km_gridOutlier_vegS{veg_threshold}.png',
                dpi=600, bbox_inches='tight')

    # plt.show()


### S1-4
def plot_S1_2_forSlope(data_slope, data_pvalue, data_cv):

    # 1. Prepare data
    veg_slope_list = [np.where(veg_type_data == i, data_slope, np.nan) for i in [1, 2, 3, 4]]
    veg_pvalue_list = [np.where(veg_type_data == i, data_pvalue, np.nan) for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_slope_list = [
        np.where(ai_type_data == 2, data_slope, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data_slope, np.nan),  # Semi-Arid (Combine 3 and 4)
        np.where(ai_type_data == 5, data_slope, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data_slope, np.nan)  # Humid
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

    for veg_slope, veg_pvalue, veg_name, (ax_l, ax_r) in zip(veg_slope_list, veg_pvalue_list, veg_labels, ax_pairs):
        # Plotleft map group (ax_l)
        plot_pheno_slope_and_pvalue_forAllvegType(veg_slope, veg_pvalue, data_cv, -1, 1, 'pheno slope', veg_name, ax=ax_l)

        # Plotright bar chart group (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(veg_slope, veg_pvalue, data_cv, 'pheno slope', -1, 1, veg_name, ax=ax_r)

    plt.savefig(rf'{fig_output}\Veg\{pheno}_Slope_{scale}km_Vegtype_vegS{veg_threshold}.png', dpi=600, bbox_inches='tight')

    # plt.show()

    ############################################## AI type ##################################################

    # Create 5 subplots: 4 maps and 1 colorbar
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

    for ai_slope, ai_pvalue, ai_name, (ax_l, ax_r) in zip(ai_slope_list, ai_pvalue_list, ai_labels, ax_pairs):
        # Plotleft map group (ax_l)
        # Assumes veg_name was added to the function signature
        plot_pheno_slope_and_pvalue_forAllvegType(ai_slope, ai_pvalue, data_cv, -1, 1, 'pheno slope', ai_name, ax=ax_l)

        # Plotright bar chart group (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(ai_slope, ai_pvalue, data_cv, 'pheno slope', -1, 1, ai_name, ax=ax_r)

    plt.savefig(rf'{fig_output}\AI\{pheno}_Slope_{scale}km_AItype_vegS{veg_threshold}.png', dpi=600, bbox_inches='tight')

    # # plt.show()


def plot_S3_4_forCV(data_slope, data_pvalue, data_cv):

    # 1. Prepare data
    veg_cv_list = [np.where(veg_type_data == i, data_cv, np.nan) for i in [1, 2, 3, 4]]
    veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

    ai_cv_list = [
        np.where(ai_type_data == 2, data_cv, np.nan),  # Arid
        np.where((ai_type_data == 3) | (ai_type_data == 4), data_cv, np.nan),  # Semi-Arid (Combine 3 and 4)
        np.where(ai_type_data == 5, data_cv, np.nan),  # Dry sub-humid
        np.where(ai_type_data == 6, data_cv, np.nan)  # Humid
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

    # Create 5 subplots: 4 maps and 1 colorbar
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

    for veg_cv, veg_name, (ax_l, ax_r) in zip(veg_cv_list, veg_labels, ax_pairs):
        # Plotleft map group (ax_l)
        plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, veg_cv, 0, 0.24, 'pheno cv', veg_name, ax=ax_l)

        # Plotright bar chart group (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, veg_cv, 'pheno cv', 0, 0.24, veg_name, ax=ax_r)

    plt.savefig(rf'{fig_output}\Veg\{pheno}_CV_{scale}km_Vegtype_gridOutlier_vegS{veg_threshold}.png', dpi=600, bbox_inches='tight')

    # plt.show()

    ############################################## AI type ##################################################

    # Create 5 subplots: 4 maps and 1 colorbar
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

    for ai_cv, ai_name, (ax_l, ax_r) in zip(ai_cv_list, ai_labels, ax_pairs):
        # Plotleft map group (ax_l)
        # Assumes veg_name was added to the function signature
        plot_pheno_slope_and_pvalue_forAllvegType(data_slope, data_pvalue, ai_cv, 0, 0.24, 'pheno cv', ai_name, ax=ax_l)

        # Plotright bar chart group (ax_r)
        plot_pheno_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(data_slope, data_pvalue, ai_cv, 'pheno cv', 0, 0.24, ai_name, ax=ax_r)

    plt.savefig(
        rf'{fig_output}\AI\{pheno}_CV_{scale}km_AItype_gridOutlier_vegS{veg_threshold}.png',
        dpi=600, bbox_inches='tight')
    #
    # # plt.show()


pheno_slope = np.where(np.isfinite(veg_type_data) & np.isfinite(ai_type_data) ,  pheno_slope, np.nan)
pheno_slope_pvalue = np.where(np.isfinite(veg_type_data) & np.isfinite(ai_type_data) ,  pheno_slope_pvalue, np.nan)
pheno_cv = np.where(np.isfinite(veg_type_data) & np.isfinite(ai_type_data),  pheno_cv, np.nan)

# # ### All
# plot_fig1(pheno_slope, pheno_slope_pvalue, pheno_cv)
# print('fig1 plot done!')
# # # S1-4
# plot_S1_2_forSlope(pheno_slope, pheno_slope_pvalue, pheno_cv)
# print('S1-2 plot done!')
plot_S3_4_forCV(pheno_slope, pheno_slope_pvalue, pheno_cv)
print('S3-4 plot done!')

sys.exit()




### Calculate annual POS mean, 25th and 75th percentiles for each vegetation type

## Distinguish Eurasia and North America

###### Region classification #######
north_america = rf'D:/code/figshare版本/NorthAmerica_tif_vision.tif'
europe = rf'D:/code/figshare版本/Europe_tif_vision.tif'
asia = rf'D:/code/figshare版本/Asia_tif_vision.tif'

north_america_tif = gdal.Open(north_america)
europe_type_tif = gdal.Open(europe)
asia_type_tif = gdal.Open(asia)

def aggregate(tif_data):

    ds = gdal.Warp(
        "",
        tif_data,
        format="MEM",
        width=cols,
        height=rows,
        outputBounds=(lon_min, lat_min, lon_max, lat_max),
        dstSRS="EPSG:4326",
        resampleAlg=gdal.GRA_Max,
        srcNodata=1,  #In input data, value 1 represents NoData and is excluded from resampling.
        dstNodata=1   #Set output pixels to 1 when there are no valid values.
    )

    ds_array = ds.ReadAsArray()

    mask = (ds_array == 0)

    return mask

north_america_mask = aggregate(north_america_tif)
europe_mask = aggregate(europe_type_tif)
asia_mask = aggregate(asia_type_tif)


# ##### Longitude classification ######
# lon = lon_min + gt[1] * (np.arange(cols) + 0.5)
#
# #North America
# north_america_mask = (
#     (lon >= -170) &
#     (lon <= -50)
# )
#
# #Eurasia
# eurasia_mask = (
#     (lon >= -10) &
#     (lon <= 180)
# )
#
# # Europe
# europe_mask = (
#     (lon >= -10) &
#     (lon < 60)
# )
#
# # Asia
# asia_mask = (
#     (lon >= 60) &
#     (lon <= 180)
# )



###
#
# na_mask_2d = np.tile(north_america_mask, (rows, 1))
# europe_mask_2d = np.tile(europe_mask, (rows, 1))
# asia_mask_2d = np.tile(asia_mask, (rows, 1))


outlier_pheno_stack_na = np.where(
    north_america_mask,
    outlier_pheno_stack,
    np.nan
)

outlier_pheno_stack_europe = np.where(
    europe_mask,
    outlier_pheno_stack,
    np.nan
)

outlier_pheno_stack_asia = np.where(
    asia_mask,
    outlier_pheno_stack,
    np.nan
)



def cal_each_year_mean_25th_75th(stack1, stack2, stack3, stack4, ax, scope):
    rows = []
    for k, year in enumerate(years):

        if scope == 'Global':
            array_data_ofyear = stack1[k, :, :]
        elif scope == 'North America':
            array_data_ofyear = stack2[k, :, :]
        elif scope == 'Europe':
            array_data_ofyear = stack3[k, :, :]
        elif scope == 'Asia':
            array_data_ofyear = stack4[k, :, :]

        row = {'year': year}

        # =========================
        # Vegetation types
        # =========================
        veg_data_list = [
            np.where(veg_type_data == 1, array_data_ofyear, np.nan),  # Forest
            np.where(veg_type_data == 2, array_data_ofyear, np.nan),  # Shrublands
            np.where(veg_type_data == 3, array_data_ofyear, np.nan),  # Savanna
            np.where(veg_type_data == 4, array_data_ofyear, np.nan)   # Grasslands
        ]

        veg_labels = ['Forests', 'Shrublands', 'Savannas', 'Grasslands']

        for veg_data, veg_name in zip(veg_data_list, veg_labels):

            row[veg_name] = np.nanmean(veg_data)
            row[f'{veg_name}_25th_error'] = np.nanmean(veg_data) - np.nanpercentile(veg_data, 25)
            row[f'{veg_name}_75th_error'] = np.nanpercentile(veg_data, 75) - np.nanmean(veg_data)

        # =========================
        # Aridity classes
        # =========================
        ai_data_list = [
            np.where(ai_type_data == 2, array_data_ofyear, np.nan),                              # Arid
            np.where((ai_type_data == 3) | (ai_type_data == 4), array_data_ofyear, np.nan),      # Semi-arid
            np.where(ai_type_data == 5, array_data_ofyear, np.nan),                              # Dry sub-humid
            np.where(ai_type_data == 6, array_data_ofyear, np.nan)                               # Humid
        ]

        ai_labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

        for ai_data, ai_name in zip(ai_data_list, ai_labels):

            row[ai_name] = np.nanmean(ai_data)
            row[f'{ai_name}_25th_error'] = np.nanmean(ai_data) - np.nanpercentile(ai_data, 25)
            row[f'{ai_name}_75th_error'] = np.nanpercentile(ai_data, 75) - np.nanmean(ai_data)

        rows.append(row)

    # Convert to a DataFrame
    df_year_mean_25th_75th = pd.DataFrame(rows)

    # Adjust column order
    df_year_mean_25th_75th = df_year_mean_25th_75th[
        ['year',
         'Forests', 'Forests_25th_error', 'Forests_75th_error',
         'Shrublands', 'Shrublands_25th_error', 'Shrublands_75th_error',
         'Savannas', 'Savannas_25th_error', 'Savannas_75th_error',
         'Grasslands', 'Grasslands_25th_error', 'Grasslands_75th_error',
         'Arid', 'Arid_25th_error', 'Arid_75th_error',
         'Semi-arid', 'Semi-arid_25th_error', 'Semi-arid_75th_error',
         'Dry sub-humid', 'Dry sub-humid_25th_error', 'Dry sub-humid_75th_error',
         'Humid', 'Humid_25th_error', 'Humid_75th_error']
    ]

    print(df_year_mean_25th_75th.head())

    df_year_mean_25th_75th.to_csv(rf'D:/code/Scatter Plot in Site or pixel/{pheno}_VegType_AiType_year_mean_25th_75th_{scope}.csv',
                              index=False,
                              encoding='utf-8-sig'
                              )


    ######### Plot #################

    fig = ax.get_figure()

    ax.axis('off')

    gs_inner = ax.get_subplotspec().subgridspec(
        1, 2,
        width_ratios=[1, 1],
        wspace=0.3
    )

    ax1 = fig.add_subplot(gs_inner[0])
    ax2 = fig.add_subplot(gs_inner[1])

    ## Veg
    x = df_year_mean_25th_75th['year']
    y = df_year_mean_25th_75th
    ax1.plot(x, y['Forests'], color = '#086A10', marker='o', markersize=3, linewidth=1.5, label = 'Forests')
    ax1.plot(x, y['Shrublands'], color = '#965724', marker='o', markersize=3, linewidth=1.5, label = 'Shrublands')
    ax1.plot(x, y['Savannas'], color = '#fdbd10', marker='o', markersize=3, linewidth=1.5, label = 'Savannas')
    ax1.plot(x, y['Grasslands'], color = '#B6FF05', marker='o', markersize=3, linewidth=1.5, label = 'Grasslands')

    ax1.set_ylim(120, 220)
    ax1.set_yticklabels(np.arange(120, 240, 20))

    # ax1.set_xlim(2001, 2024)

    ax1.set_ylabel('PPT (days)')

    if scope == 'Asia':
        ax1.set_xlabel('Year')
        ax1.legend(loc='lower center',
                  bbox_to_anchor=(0.5, -0.6),
                  ncol=2,
                  # fontsize=7,
                  frameon=False)

    ## AI
    x = df_year_mean_25th_75th['year']
    y = df_year_mean_25th_75th
    ax2.plot(x, y['Arid'], color='#e5505a', marker='o', markersize=3, linewidth=1.5, label='Arid')
    ax2.plot(x, y['Semi-arid'], color='#ff9f1c', marker='o', markersize=3, linewidth=1.5, label='Semi-arid')
    ax2.plot(x, y['Dry sub-humid'], color='#01b287', marker='o', markersize=3, linewidth=1.5, label='Dry sub-humid')
    ax2.plot(x, y['Humid'], color='#854c9e', marker='o', markersize=3, linewidth=1.5, label='Humid')

    ax2.set_ylim(120, 220)
    ax2.set_yticklabels(np.arange(120, 240, 20))

    # ax2.set_xlim(2001, 2024)

    if scope == 'Asia':
        ax2.set_xlabel('Year')
        ax2.legend(loc='lower center',
                   bbox_to_anchor=(0.5, -0.6),
                   ncol=2,
                   # fontsize=7,
                   frameon=False)





def cal_continent_yearly_ppt_and_plot(stack1, stack2, stack3, stack4):

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


    fig = plt.figure(figsize=(8.2, 11), dpi=100)
    gs = gridspec.GridSpec(4, 1,
                           # width_ratios=[1, 1],  # Width ratios for the three columns
                           height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                           hspace=0.5)#, wspace=0.2)


    continent_list = ['Global', 'North America', 'Europe', 'Asia']

    for i, continent in enumerate(continent_list):
        if continent == 'Global':
            ax = plt.subplot(gs[0, 0])
        elif continent == 'North America':
            ax = plt.subplot(gs[1, 0])
        elif continent == 'Europe':
            ax = plt.subplot(gs[2, 0])
        elif continent == 'Asia':
            ax = plt.subplot(gs[3, 0])

        cal_each_year_mean_25th_75th(stack1, stack2, stack3, stack4, ax, continent)

    plt.savefig(rf'{fig_output}\Scatter Plot in Site or pixel\Continent_PPT_yearly_{scale}km.png', dpi = 300, bbox_inches='tight')

    # plt.tight_layout()
    # plt.show()


outlier_pheno_stack = np.where(np.isfinite(mcd12q2_mask), outlier_pheno_stack, np.nan)

cal_continent_yearly_ppt_and_plot(outlier_pheno_stack, outlier_pheno_stack_na, outlier_pheno_stack_europe, outlier_pheno_stack_asia)