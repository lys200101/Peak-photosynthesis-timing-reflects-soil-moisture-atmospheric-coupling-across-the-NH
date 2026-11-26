import os
import glob
import datetime
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
import pandas as pd
# from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
# from scipy.stats import pearsonr
from joblib import Parallel, delayed, parallel_backend
# import matplotlib as mpl
from matplotlib import colormaps

# import pymannkendall as mk



######### Functions ##########

### Read time information and extract the date (last 8 digits) from each TIF file
def extract_date_from_filename(filename):
    # Get the filename without its path or extension
    basename = os.path.basename(filename)
    filename_without_ext = os.path.splitext(basename)[0]

    # Take the last 8 digits
    date_str = filename_without_ext[-8:]

    # Validate the date format
    if not date_str.isdigit() or len(date_str) != 8:
        raise ValueError(f"文件 {filename} 的最后8位不是有效日期（需为YYYYMMDD）！")

    # Convert to a datetime object
    return datetime.datetime.strptime(date_str, "%Y%m%d")


def clip_by_latitude(gt, rows, cols, lat_min=30, lat_max=90):
    """
    Clip the data by latitude range.
    Returns: (row_start, row_end, new_gt)
    """
    # Calculate the center latitude of each row
    row_centers = np.arange(rows) * gt[5] + gt[3] + gt[5] / 2

    # Find rows within the 30–90 degree latitude range
    valid_rows = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(valid_rows):
        raise ValueError(f"在纬度范围 {lat_min}-{lat_max} 内没有找到有效数据")

    # Find the first and last valid rows
    valid_row_indices = np.where(valid_rows)[0]
    row_start = valid_row_indices[0]
    row_end = valid_row_indices[-1] + 1  # Slices are left-inclusive and right-exclusive

    # Calculate the new upper-left coordinates
    new_top_left_x = gt[0] + row_start * gt[2]  # Usually gt[2] = 0
    new_top_left_y = gt[3] + row_start * gt[5]  # gt[5] is the pixel height

    # Create the new geotransform parameters
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
    # print(f"Original row count: {rows}, clipped row count: {row_end - row_start}")
    # print(f"Original upper-left latitude: {gt[3]:.2f}°N, new upper-left latitude: {new_top_left_y:.2f}°N")

    return row_start, row_end, new_gt


### Read the SM and VPD bands
def get_band_noclip(tif_file, stack):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(climate_data)
    tif = None  # Release resources promptly


def get_band_clip(tif_file, stack, row_start, row_end):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    climate_data_cropped = climate_data[row_start:row_end, :]
    stack.append(climate_data_cropped)

    tif = None  # Release resources promptly



### Calculate daily climatology (multi-year daily mean)
def calculate_pixel_daily_climatology(pixel_time_series, dates):
    if not hasattr(dates, 'month'):
        dates = pd.Series(dates)

    daily_climatology = {}

    for month in range(1, 13):
        for day in range(1, 32):
            try:
                # Check whether the date is valid
                datetime.datetime(2000, month, day)

                # Select all data for this month and day
                mask = (dates.dt.month == month) & (dates.dt.day == day)
                if np.sum(mask) > 0:
                    # Extract values for this pixel on the corresponding dates
                    daily_values = pixel_time_series[mask]
                    # Calculate the multi-year mean for this day (ignoring NaN)
                    if np.sum(~np.isnan(daily_values)) > 0:
                        climatology = np.nanmean(daily_values, axis=0)
                        # print(f'month:{month}, day:{day}, daily_values:{pixel_time_series_value}, daily_climatology[key]:{daily_climatology_value}')
                        daily_climatology[(month, day)] = climatology

            except ValueError:
                continue

    return daily_climatology


### Calculate daily climatology (multi-year half-month mean)
def calculate_half_month_climatology(pixel_time_series, dates):
    """
    Calculate half-month climatological means - simplified version

    Steps:
    1. Calculate separate means for the first half (days 1–15) and second half (day 16 to month end) of each month across all years.
    2. Create a 366-day climatology array, assigning each DOY the corresponding half-month mean.
    """
    # Ensure dates are in datetime format
    if not hasattr(dates, 'day'):
        dates = pd.to_datetime(dates)

    # Step 1: Calculate the multi-year mean for each half-month
    half_month_sums = {}
    half_month_counts = {}

    for date, value in zip(dates, pixel_time_series):
        if np.isnan(value):
            continue

        month = date.month
        day = date.day

        # Determine whether the date belongs to the first or second half of the month
        if day <= 15:
            half_key = (month, 1)  # (month, first half)
        else:
            half_key = (month, 2)  # (month, second half)

        # Accumulate the sum and count
        if half_key not in half_month_sums:
            half_month_sums[half_key] = 0
            half_month_counts[half_key] = 0

        half_month_sums[half_key] += value
        half_month_counts[half_key] += 1

    # Calculate the mean for each half-month
    half_month_means = {}
    for half_key in half_month_sums:
        if half_month_counts[half_key] > 0:
            half_month_means[half_key] = half_month_sums[half_key] / half_month_counts[half_key]

    # Step 2: Create a 366-day climatology array
    daily_climatology = {}

    # Iterate over every day of the year (1–366)
    for month in range(1, 13):
        for day in range(1, 32):

            datetime.datetime(2000, month, day)

            # Select all data for this month and day
            mask = (dates.dt.month == month) & (dates.dt.day == day)
            if np.sum(mask) > 0:
                if day <= 15:
                    half_key = (month, 1)
                else:
                    half_key = (month, 2)
                # Get values for this pixel on the corresponding dates
                daily_values = half_month_means.get(half_key, np.nan)
                # Assign the multi-year mean for this day (ignoring NaN)
                daily_climatology[(month, day)] = daily_values

    return daily_climatology



### Remove seasonality by subtracting the multi-year mean for each day
def deseasonalize_pixel(
        pixel_time_series, dates, row_i, col_j
):
    deseasonalized_series = np.full_like(
        pixel_time_series, np.nan
    )

    if not np.any(np.isfinite(pixel_time_series)):
        return deseasonalized_series

    for t, date in enumerate(dates):
        key = (date.month, date.day)

        if (
                key in daily_climatology
                and np.isfinite(pixel_time_series[t])
        ):
            daily_mean = daily_climatology[key][row_i, col_j]

            if np.isfinite(daily_mean):
                deseasonalized_series[t] = (
                        pixel_time_series[t] - daily_mean
                )

    return deseasonalized_series

def deseasonal_parelle(
        data, dates, n_jobs, row_offset=0, col_offset=0
):
    time_steps, rows, cols = data.shape

    row_indices = np.repeat(np.arange(rows), cols)
    col_indices = np.tile(np.arange(cols), rows)

    deseason_data = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(deseasonalize_pixel)(
            data[:, i, j],
            dates,
            i + row_offset,
            j + col_offset,
        )
        for i, j in zip(row_indices, col_indices)
    )

    return (
        np.array(deseason_data)
        .reshape(rows, cols, time_steps)
        .transpose(2, 0, 1)
    )





### Detrending
def detrend_with_lowess_matrix(time_series, frac):

    if np.all(~np.isfinite(time_series)):
        detrended_full = np.full_like(time_series, np.nan)

    else:
        # # Check valid data points
        # valid_mask = ~np.isnan(time_series)
        # valid_data = time_series[valid_mask]
        #
        # # Generate the time index
        # y = np.arange(len(valid_data))
        # # Apply LOWESS smoothing
        # smoothed = lowess(valid_data, y, frac=frac, return_sorted=False)

        time_index = np.arange(len(time_series))

        # Select values and their corresponding times together
        valid_mask = np.isfinite(time_series)

        valid_data = time_series[valid_mask]  # [10, 11, 20]
        valid_time = time_index[valid_mask]  # [0, 1, 4]

        smoothed = lowess(
            valid_data,
            valid_time,
            frac,
            return_sorted=False,
        )
        # Remove the trend
        detrended_valid = valid_data - smoothed

        # Reconstruct the full series
        detrended_full = np.full_like(time_series, np.nan)
        detrended_full[valid_mask] = detrended_valid

    return detrended_full


def detrend_stack_parallel(stack, n_jobs, frac=0.4):
    # Get the data dimensions (time steps, rows, columns)
    time_steps, rows, cols = stack.shape

    # Generate row and column indices for all pixels (i: row, j: column)
    row_indices = np.repeat(np.arange(rows), cols)  # Repeat each row index cols times
    col_indices = np.tile(np.arange(cols), rows)  # Tile the column indices for each row

    # print('detrend start2')
    # Process each pixel in parallel (extract time series → detrend → return result)
    # with parallel_backend('threading', n_jobs=n_jobs):
    detrended_pixels = Parallel(n_jobs=n_jobs, verbose = 10)(
        delayed(detrend_with_lowess_matrix)(
            stack[:, i, j],  # Extract the time series for pixel (i, j) (shape: time steps)
            frac=frac
        )
        for i, j in zip(row_indices, col_indices)  # Iterate over all row and column combinations
    )

    # Reshape the 1D results into a 3D image (time steps, rows, columns)
    detrended_stack = np.array(detrended_pixels).reshape(rows, cols, time_steps).transpose(2, 0, 1)

    return detrended_stack


def save_tif_gdal(output_path, data, crs, transform):
    """Save a TIFF file, automatically getting the data dimensions and applying the geotransform."""
    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")

    output_ds = driver.Create(
        output_path,
        cols, rows, 1, gdal.GDT_Float32
    )
    if not output_ds:
        raise RuntimeError(f"无法创建输出文件: {output_path}")

    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data, 0, 0)
    output_band.SetNoDataValue(np.nan)  # Set NaN as the NoData value

    output_ds.SetProjection(crs)
    output_ds.SetGeoTransform(transform)  # Use the adjusted geotransform parameters
    output_ds = None
    return True




###################################### 1 Data reading and output settings ################################

print(list(colormaps))



######  ============== Input settings to modify ============= #########

pixel_resolution = 11

climate_var = 'VPD'  ##0-100cmSM / VPD / Ta / Pre / Srad

path = 'I:\Data\ERA5_Land'

folder = f'{path}\ERA5_Land_NH_{pixel_resolution}km_daily\ERA5_Land_NH_{pixel_resolution}km_daily_{climate_var}'

clip = 'Yes'  ##Set Yes or No    ###Set clip or not with latitude

if clip == 'Yes':
    min_lat = 30
    max_lat = 84


#########################
tif_files = sorted(glob.glob(os.path.join(folder, '*.tif')))
if not tif_files:
    raise FileNotFoundError("未找到任何 TIF 文件！")

# #### Input POS and SOS
# folder2 = r'D:\CAU\phenology_swc_vpd\China_test\Data\test_pheno\POS'
# folder3 = r'D:\CAU\phenology_swc_vpd\China_test\Data\test_pheno\SOS'

######## Output settings
output_detrend_tif_path = f'{path}\ERA5_Land_NH_{pixel_resolution}km_daily_deseason_detrend\ERA5_Land_NH_{pixel_resolution}km_daily_{climate_var}(2001-2024)_new'


# output_tif_path = r'D:\CAU\phenology_swc_vpd\China_test\Cor_result\tif\SM_VPD_Cor'
# output_png_path = r'D:\CAU\phenology_swc_vpd\China_test\Cor_result\fig\SM_VPD_Cor'


########################### 2 Basic information ########################
first_tif = tif_files[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"无法打开 TIF 文件：{sample_tif}（驱动不支持或文件损坏）")

# Get the geotransform parameters
crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()
proj = sample_tif.GetProjection()

# Get the data dimensions
sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
rows = sample_array.shape[0]
cols = sample_array.shape[1]
print('Original: rows=', rows, 'cols=', cols)

if clip == 'Yes':
    # Calculate the latitude clipping range
    print(f'Retaining {min_lat}°–{max_lat}°N')
    row_start, row_end, new_gt = clip_by_latitude(gt, rows, cols, lat_min=min_lat, lat_max=max_lat)

    # Calculate the dimensions after clipping
    new_rows = row_end - row_start
    new_cols = cols  # The number of columns remains unchanged

    print(f"After clipping: rows={new_rows}, cols={new_cols}")

# Release the sample file
sample_tif = None



############################################ 3 Time and stacking ##########################################
## Time
tif_dates = []
for tif_file in tif_files:
    date = extract_date_from_filename(tif_file)
    tif_dates.append(date)
dates_pd = pd.Series(tif_dates)  # Convert to a pandas Series to use the dt accessor

print('First five dates: ', [d.strftime('%Y%m%d') for d in tif_dates[:5]])
print('Last five dates: ', [d.strftime('%Y%m%d') for d in tif_dates[-5:]])


## Stack the data
data_stack = []

for tif_file in tif_files:
    # print('tif_file:\n', tif_file)
    # No clipping
    if clip == 'No':
        get_band_noclip(tif_file, data_stack)

    # # Clipping
    if clip == 'Yes':
        get_band_clip(tif_file, data_stack, row_start, row_end)
    # print('Get band done!')

print('Stack start!')
data_stack = np.stack(data_stack, axis=0)
# Release the file list
tif_files = None
print('data_stack shape:\n', data_stack.shape)
print('Stack end!')

# #### Test:  #####
# # Calculate the new geotransform parameters for the cropped region
# start_row = 400    # Starting row
# start_col = 3000   # Starting column
# end_row = 410      # Ending row
# end_col = 3010     # Ending column
#
# # Add the following code after extracting the subset
# data_stack2 = data_stack[:, start_row:end_row, start_col:end_col]  # test
#
#
# # New upper-left coordinate = original coordinate + column offset * pixel width + row offset * row rotation
# new_top_left_x = gt[0] + start_col * gt[1] + start_row * gt[2]
# new_top_left_y = gt[3] + start_col * gt[4] + start_row * gt[5]
#
# # Create the new geotransform parameters
# new_gt = (
#     new_top_left_x,  # New upper-left X coordinate
#     gt[1],           # Pixel width (unchanged)
#     gt[2],           # Row rotation (usually 0)
#     new_top_left_y,  # New upper-left Y coordinate
#     gt[4],           # Column rotation (usually 0)
#     gt[5]            # Pixel height (unchanged, usually negative)
# )
# new_rows = data_stack2.shape[1]
# new_cols = data_stack2.shape[2]
#
# print(f"After clipping: rows={new_rows}, cols={new_cols}")

# print(f"Original upper-left: ({gt[0]:.2f}, {gt[3]:.2f})")
# print(f"New upper-left: ({new_top_left_x:.2f}, {new_top_left_y:.2f})")


########################################### 5 Deseasonalization ##########################################
print(f'{climate_var} deseasonal start')
### Use this to calculate the multi-year daily mean:
# Calculate the daily climatology for each pixel
daily_climatology = calculate_pixel_daily_climatology(data_stack, dates_pd)

# ### Use this to calculate the multi-year half-month mean:
# daily_climatology = calculate_half_month_climatology(data_stack, dates_pd)

# data_deseasonalized = deseasonal_parelle(data_stack, dates_pd,  n_jobs=18)

############################ 4 Deseasonalization and detrending #############################
import gc

_, rows, cols = data_stack.shape
print(rf'After clipping rows={rows}, cols={cols}')

mid_row1 = rows // 3
mid_row2 = 2 * rows // 3

mid_col1 = cols // 3
mid_col2 = 2 * cols // 3

parts = [
    # First row
    (slice(None), slice(0, mid_row1),      slice(0, mid_col1)),      # Upper left
    (slice(None), slice(0, mid_row1),      slice(mid_col1, mid_col2)), # Upper middle
    (slice(None), slice(0, mid_row1),      slice(mid_col2, None)),      # Upper right

    # Second row
    (slice(None), slice(mid_row1, mid_row2), slice(0, mid_col1)),      # Middle left
    (slice(None), slice(mid_row1, mid_row2), slice(mid_col1, mid_col2)), # Center
    (slice(None), slice(mid_row1, mid_row2), slice(mid_col2, None)),      # Middle right

    # Third row
    (slice(None), slice(mid_row2, None), slice(0, mid_col1)),      # Lower left
    (slice(None), slice(mid_row2, None), slice(mid_col1, mid_col2)), # Lower middle
    (slice(None), slice(mid_row2, None), slice(mid_col2, None)),      # Lower right
]


frac = 0.4

data_detrend = np.empty_like(data_stack, dtype=np.float32)

print(f'{climate_var} deseasonal + detrend start')

for i, idx in enumerate(parts):

    print(f'Processing block {i+1}/6')
    print(f'Deseasonal start')

    # ========================
    # 1. Deseasonalization
    # ========================
    part = deseasonal_parelle(
        data_stack[idx],
        dates_pd,
        n_jobs=18,
        row_offset=idx[1].start or 0,
        col_offset=idx[2].start or 0,
    )

    # ========================
    # 2. Detrending
    # ========================
    print(f'Detrend start')
    part = detrend_stack_parallel(
        part,
        n_jobs=18,
        frac=frac
    )

    # Write directly back to the final array
    data_detrend[idx] = part

    print(f'Block {i+1} finished')

    del part
    gc.collect()

print(f'{climate_var} deseasonal + detrend finished')



############################################ 5 Export ###############################################

# # Important: create the output directory
# os.makedirs(output_detrend_tif_path, exist_ok=True)

# Get the actual dimensions after clipping
output_rows, output_cols = data_detrend.shape[1], data_detrend.shape[2]
print(f'Output image dimensions: {output_rows} x {output_cols}')

### Export the detrended data with dates
for k, date in enumerate(tif_dates):
    date_str = date.strftime("%Y%m%d")  # Format as, for example, 20200101
    output_path = os.path.join(
        output_detrend_tif_path,
        f'{climate_var}_deseason_{date_str}.tif'
    )

    # save_tif_gdal(
    #     output_path,
    #     data_detrend[k, :, :],  # Select the k-th image
    #     new_rows,
    #     new_cols,
    #     crs,
    #     gt
    # )
    # # ### Test:
    save_tif_gdal(
        output_path,
        data_detrend[k, :, :],  # Select the k-th image
        crs,
        new_gt  # Use the new geotransform parameters
    )
print(f'{climate_var} export done!')
