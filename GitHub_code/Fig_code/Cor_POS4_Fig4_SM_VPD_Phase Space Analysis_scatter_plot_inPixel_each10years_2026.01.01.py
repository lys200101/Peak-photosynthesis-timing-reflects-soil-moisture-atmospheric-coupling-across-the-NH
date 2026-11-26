import os.path
import glob
import sys
import numpy as np

from osgeo import gdal

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d



################################ 1 Input and output #############################
##### Input settings #####

star_year = 2001
end_year = 2024
years = np.arange(star_year, end_year + 1)
years_length = end_year - star_year + 1
print('years_length:', years_length)

min_lat = 30
max_lat = 84

data_type = 'MODIS_GPP_8d' # GIMMS_NDVI_16d / MODIS_GPP_16d / MODIS_GPP_8d / MODIS_EVI2_8d / MCD12Q2

mean_or_yearPOS = ''  #  '' uses annual POS; '_meanPOS' uses long-term mean POS

scale = 11

coupling_method = 'Pearson'  ### 'Pearson'/'Partial'
lag_day = 0

partial_for = 'SMlag_VPDlag'  ###SM and lagged VPD: SM_VPDlag; SM and VPD: SMlag_VPDlag

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno uses detrended SOS and POS for partial correlation
                                   ### Based_on_OriginPheno uses original SOS and POS for partial correlation

veg_threshold = 25

climate_test_number = '0'
only_analyses_significant = 'No'   ###Yes analyzes only pixels with POS or Cor p-value < 0.05; No analyzes all pixels

Phase_Space_Analysis_year = 12 # 5 / 12

input_data = rf'D:/code/Global_test2_{scale}000m'

# sm_input = rf'{input_data}\Data\Climate\SM_preseason_mean{climate_test_number}'
# vpd_input = rf'{input_data}\Data\Climate\VPD_preseason_mean{climate_test_number}'
if coupling_method == 'Pearson':
    input_path2 = Rf'{input_data}\Data\Climate\climate_in_best_preseason\Cor_Pearson_SM_t_VPD_t{mean_or_yearPOS}'

elif coupling_method == 'Partial' and partial_for == 'SM_VPDlag':
    input_path2 = Rf'{input_data}\Data\Climate\climate_in_best_preseason\Cor_Pcor_SM_lag{lag_day}_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}'

elif coupling_method == 'Partial' and partial_for == 'SMlag_VPDlag':
    input_path2 = Rf'{input_data}\Data\Climate\climate_in_best_preseason\Cor_Pcor_SM_t_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}'

sm_input = fr'{input_path2}\SM_preseason_mean'  #(POS-30) - POS
vpd_input = fr'{input_path2}\VPD_preseason_mean'  #(POS-30) - POS


cor_input =  rf'{input_path2}\Cor_preseason_mean'

sm_tiffiles = sorted(glob.glob(os.path.join(sm_input, '*.tif')))
vpd_tiffiles = sorted(glob.glob(os.path.join(vpd_input, '*.tif')))

cor_tiffiles = sorted(glob.glob(os.path.join(cor_input, '*.tif')))

#### Input vegetation type TIF path
veg_type_tif = rf'{input_data}\Data\Classify\Veg_type\NH_permanent_veg_type_{scale}km_gte{veg_threshold}.tif'

#### Input AI TIF path
ai_tif = rf'{input_data}\Data\Classify\AI\NH30_84_AI(graident)_{scale}km.tif'

#### Input coupling gradient TIF path
cor_mean_tif = fr'{input_data}\Tif_of_fig\3Cor_mean_slope\mean\Cor(Pearson_SMlag_VPDlag_lag0)_mean_11km_SOS_All.tif'  #SOS - POS

mcd12q2_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
mcd12q2_mask_tif = gdal.Open(mcd12q2_mask_tif)
mcd12q2_mask = mcd12q2_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
mcd12q2_mask = np.isfinite(mcd12q2_mask)

gpp_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
gpp_mask_tif = gdal.Open(gpp_mask_tif)
gpp_mask = gpp_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
gpp_mask = np.isfinite(gpp_mask)


##### Output settings #####
output_fig_path = rf'{input_data}\Fig\Fig 4  SM and VPD'


############################### Basic information ###############################
sample = cor_tiffiles[0]
sample_tif = gdal.Open(sample)

sample_array = sample_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

crs = sample_tif.GetProjectionRef()
gt = sample_tif.GetGeoTransform()

rows = sample_array.shape[0]
cols = sample_array.shape[1]
print(f'rows={rows} , cols={cols}')

row_indices = np.repeat(np.arange(rows), cols)
col_indices = np.tile(np.arange(cols), rows)

lon_min = gt[0]
lon_max = gt[0] + gt[1]*cols
lat_min = gt[3] + gt[5]*rows
lat_max = gt[3]



################################ 2 Stacking ################################
def get_band(tif, stack):
    tif_data = gdal.Open(tif)
    tif_array = tif_data.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(tif_array)

sm_stack = []
vpd_stack = []

cor_stack = []

for tif_file in sm_tiffiles:
    get_band(tif_file, sm_stack)
for tif_file in vpd_tiffiles:
    get_band(tif_file, vpd_stack)

for tif_file in cor_tiffiles:
    get_band(tif_file, cor_stack)


sm_stack = np.stack(sm_stack, axis=0)
vpd_stack = np.stack(vpd_stack, axis=0)

cor_stack = np.stack(cor_stack, axis=0)


############# 3.2 Vegetation type ################
veg_type_data = gdal.Open(veg_type_tif)
band = veg_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

veg_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    veg_type_data[veg_type_data == nodata] = np.nan
print('veg_type_data shape:', veg_type_data.shape)

# # ############# 3.3 Climate zone ################
# climate_type_data = gdal.Open(climate_type_tif)
# climate_type = climate_type_data.GetRasterBand(1).ReadAsArray().astype(np.float32)

############### 3.4 AI index #################
ai_data = gdal.Open(ai_tif)
band = ai_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

ai_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    ai_data[ai_data == nodata] = np.nan
print('ai_data shape:', ai_data.shape)

############### 3.4 Coupling gradient data #################
cor_mean_data = gdal.Open(cor_mean_tif)
band = cor_mean_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

cor_mean_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    cor_mean_data[cor_mean_data == nodata] = np.nan
print('cor_mean_data shape:', cor_mean_data.shape)

# #### POS p-value data
# pos_slope_pvalue_tif = gdal.Open(pos_slope_pvalue_tif)
# pos_slope_pvalue= pos_slope_pvalue_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
# print('pos_slope_pvalue shape:', pos_slope_pvalue.shape)

sm_stack = np.where(np.isfinite(veg_type_data), sm_stack, np.nan)
vpd_stack = np.where(np.isfinite(veg_type_data), vpd_stack, np.nan)

print(f'SM valid pixels: {np.count_nonzero(np.isfinite(sm_stack[0, :, :]))}')
print(f'VPD valid pixels: {np.count_nonzero(np.isfinite(vpd_stack[0, :, :]))}')
print(f'Cor valid pixels: {np.count_nonzero(np.isfinite(cor_stack[0, :, :]))}')
print('Stack done!')

############################### 3 Remove outliers ###################################
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
    if len(x_flatten) < (years_length/2):
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

        if np.count_nonzero(np.isfinite(x_masked)) >= years_length / 2:
            return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
        else:
            print(f'IQR threshold：{lower_range:.4f} ~ {upper_range:.4f}\n'
                  f'Original pixel data: {x}\n'
                  f'More than half the pixel values are invalid after outlier removal: {x_masked}')
            return np.full_like(x, np.nan), i, j


time_lengths = sm_stack.shape[0]

outlier_sm_stack = np.full((time_lengths, rows, cols), np.nan)
outlier_vpd_stack = np.full((time_lengths, rows, cols), np.nan)

outlier_cor_stack = np.full((time_lengths, rows, cols), np.nan)

### Remove SM outliers using IQR
results = Parallel(n_jobs=18, verbose=10)(
    delayed(Outlier_array_IQR)(
        sm_stack[:, i, j], i, j, 25, 75
    ) for i, j in zip(row_indices, col_indices)
)

for data_mask, i, j in results:
    outlier_sm_stack[:, i, j] = data_mask

### Remove VPD outliers using IQR
results = Parallel(n_jobs=18, verbose=10)(
    delayed(Outlier_array_IQR)(
        vpd_stack[:, i, j], i, j, 25, 75
    ) for i, j in zip(row_indices, col_indices)
)

for data_mask, i, j in results:
    outlier_vpd_stack[:, i, j] = data_mask


### Remove Cor outliers using IQR
results = Parallel(n_jobs=18, verbose=10)(
    delayed(Outlier_array_IQR)(
        cor_stack[:, i, j], i, j, 25, 75
    ) for i, j in zip(row_indices, col_indices)
)

for data_mask, i, j in results:
    outlier_cor_stack[:, i, j] = data_mask





############# Check paired pos_stack and cor_stack values
mask = np.isfinite(outlier_sm_stack) & np.isfinite(outlier_vpd_stack)  #& np.isfinite(outlier_cor_stack)


# Count valid time-series observations for each pixel
valid_pixel_count = np.nansum(mask, axis=0)

# Create a mask of valid pixels
valid_pixel_mask = valid_pixel_count >= (years_length/2)  # (years_length - 3)
print(f'Pixels with more than 12 paired valid years after outlier removal: {np.count_nonzero(valid_pixel_mask)}')

paired_mask = mask & (valid_pixel_mask & np.isfinite(ai_data) & np.isfinite(veg_type_data) & np.isfinite(cor_mean_data))[np.newaxis, :, :]

sm_stack_clean = np.where(
    paired_mask, outlier_sm_stack, np.nan
)

vpd_stack_clean = np.where(
    paired_mask, outlier_vpd_stack, np.nan
)

cor_stack_clean = np.where(
    paired_mask, outlier_cor_stack, np.nan
)

print('IQR done!')

########################## 4 Standardization ##################################
def standardize_data(data_stack):
    """Temporal standardization"""

    def standardize_func(data_pixel, i, j):
        original_nan_mask = np.isnan(data_pixel)

        spatio_mean = np.nanmean(data_pixel)
        spatio_std = np.nanstd(data_pixel)

        standardized_pixel_data = (data_pixel - spatio_mean) / spatio_std

        standardized_pixel_data[original_nan_mask] = np.nan

        return standardized_pixel_data, i, j

    standardized_data = np.full((years_length, rows, cols), np.nan)

    results = Parallel(n_jobs=15, verbose=10)(
        delayed(standardize_func)(
            data_stack[:, i, j],
            i, j
        ) for i, j in zip(row_indices, col_indices)
    )

    for standardized_pixel_data, i, j in results:
        standardized_data[:, i, j] = standardized_pixel_data


    return standardized_data


sm_standardized = standardize_data(sm_stack_clean)
vpd_standardized = standardize_data(vpd_stack_clean)
cor_standardized = standardize_data(cor_stack_clean)


from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_phase_space_by_veg_AI(x_data, y_data, year_length, gradient_by):

    ##Set rcParams before creating subplots so plot fonts are updated
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

    if gradient_by == 'All':
        types = ['All']
        codes = [1]

        fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))

    elif gradient_by == 'Veg':
        types = ['Forest', 'Shrub', 'Savanna', 'Grass']
        codes = [1, 2, 3, 4]

        fig, axes = plt.subplots(
            3, 2,
            figsize=(6.5, 6.1),
            gridspec_kw={'height_ratios': [1, 0.005, 1]}
        )
        fig.subplots_adjust(
            hspace=0.01,
            wspace=0.3
        )
        # Hide the two empty middle subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes = axes.flatten()

    elif gradient_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig, axes = plt.subplots(
            3, 2,
            figsize=(6.5, 6.1),
            gridspec_kw={'height_ratios': [1, 0.005, 1]}
        )
        fig.subplots_adjust(
            hspace=0.01,
            wspace=0.3
        )

        # Hide the two empty middle subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes = axes.flatten()


    elif gradient_by == 'Cor mean':

        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(-0.5~-0.4)']#, 'Cor(<-0.5)']
        codes = [0, 1, 2, 3, 4]

        fig, axes = plt.subplots(
            5, 2,
            figsize=(6.5, 9),
            gridspec_kw={'height_ratios': [1, 0.005, 1, 0.005, 1]}
        )
        fig.subplots_adjust(
            hspace=0.01,
            wspace=0.3
        )

        # Hide the two empty middle subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        axes[3, 0].axis('off')
        axes[3, 1].axis('off')

        axes[4, 1].axis('off')

        axes = axes.flatten()


    elif gradient_by == 'POS_slope_sig':
        types = ['Significant']
        codes = [1]

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))

    if year_length == 5:
        period_labels = [
            "2001–2005",
            "2006–2010",
            "2011–2015",
            "2016–2020",
            "2021–2024"
        ]

        balance_hex = [
            "#543729",
            "#2baf2b",
            "#00acee",
            "#ef5734",
            "#ffcc2f"
        ]
    elif year_length == 12:
        period_labels = [
            "2001–2012",
            "2013–2024"
        ]

        balance_hex = [
            "#2166AC",
            "#B2182B"
        ]

    for i, (type, code) in enumerate(zip(types, codes)):



        if gradient_by == 'All' or gradient_by == 'POS_slope_sig':
            ax = axes
        else:
            if i < 2:
                ax = axes[i]
            elif i >= 2 and i <4:
                ax = axes[i+2]
            elif i >= 4:
                ax = axes[i + 4]



        legend_handles = []

        centroid_x = []
        centroid_y = []

        # === Create marginal axes ===
        ### Build marginal distributions
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", size="25%", pad=0.15, sharex=ax)
        ax_histy = divider.append_axes("right", size="25%", pad=0.15, sharey=ax)

        ax_histx.axis("off")
        ax_histy.axis("off")

        ax.set_xlim(-1, 1)
        ax.set_xticks(np.arange(-1, 1.001, 0.25))
        ax.set_xticklabels(f'{int(x)}' if x == int(x) else f'{x}' for x in np.arange(-1, 1.001, 0.25))
        ax.set_ylim(-1, 1)
        ax.set_yticks(np.arange(-1, 1.001, 0.25))
        ax.set_yticklabels(f'{int(x)}' if x == int(x) else f'{x}' for x in np.arange(-1, 1.001, 0.25))

        # print(np.nanmean(x_data[0]), np.nanmean(x_data[1]))
        # print(np.nanmean(y_data[0]), np.nanmean(y_data[1]))

        for t, (label, c) in enumerate(zip(period_labels, balance_hex)):

            if code is None:
                mask = np.isfinite(x_data[t]) & np.isfinite(y_data[t])
            else:
                # === Vegetation mask ===
                if gradient_by == 'Veg':
                    mask = (
                            (veg_type_data == code) &
                            np.isfinite(x_data[t]) &
                            np.isfinite(y_data[t])
                    )
                # === Aridity zone mask ===
                elif gradient_by == 'AI':
                    if code == 3:
                        mask = (
                                ((ai_data == code) | (ai_data == 4)) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )
                    else:
                        mask = (
                                (ai_data == code) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )

                # === Coupling gradient mask ===
                elif gradient_by == 'Cor mean':
                    if code == 4:
                        mask = (
                                (cor_mean_data <= -0.4) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )
                    else:
                        mask = (
                                (cor_mean_data <= -(code*0.1)) & (cor_mean_data > -((code+1)*0.1)) &
                                np.isfinite(x_data[t]) &
                                np.isfinite(y_data[t])
                        )
                    print(f'code={code}, valid pixel count: {np.sum(mask)}')

                # elif gradient_by == 'POS_slope_sig':
                #     mask = (
                #             (pos_slope_pvalue < 0.05) & np.isfinite(pos_slope_pvalue) &
                #             np.isfinite(x_data[t]) &
                #             np.isfinite(y_data[t])
                #     )
                elif gradient_by == 'All':
                    mask = (
                            np.isfinite(x_data[t]) &
                            np.isfinite(y_data[t])
                    )

            x = x_data[t][mask]
            y = y_data[t][mask]

            # === Scatter points for all pixels ===
            ax.plot([-1.5, 1.5], [1.5, -1.5], '--', color='black', linewidth=1)

            ax.scatter(
                x, y,
                s=5,
                color=c,
                alpha=0.05,
                linewidths=0,
                zorder=1
            )

            # === Centroid ===
            cx = np.nanmean(x)
            cy = np.nanmean(y)

            centroid_x.append(cx)
            centroid_y.append(cy)

            h = ax.scatter(
                cx, cy,
                s=60,
                color=c,
                edgecolor='k',
                alpha=1,
                zorder=5,
                label=label
            )

            legend_handles.append(h)

            # ===== 2D KDE density contours around the scatter=====
            # if len(x) > 50:  # Ensure stability
            values = np.vstack([x, y])
            # print(f'code={code}')
            kde2d = gaussian_kde(values, bw_method=0.1)

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            xx, yy = np.mgrid[
                     xmin:xmax:100j,
                     ymin:ymax:100j
                     ]
            grid = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde2d(grid).reshape(xx.shape)

            ax.contour(
                xx, yy, zz,
                levels=[np.percentile(zz, 95)],  # The contour marks the high-density core at the 95th percentile of KDE values.
                colors=[c],
                linewidths=2,
                # alpha=1,
                zorder=3
            )

            # # ===== KDE（marginal density）=====
            # ## x
            # x_kde = gaussian_kde(x, bw_method=0.1)
            # x_grid = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
            # ax_histx.plot(x_grid, x_kde(x_grid), color=c, lw=1)
            # ax_histx.fill_between(x_grid, x_kde(x_grid), color=c, alpha=0.3)
            #
            # # Add mean line
            # y1 = x_kde(cx)[0]
            # ax_histx.plot([cx, cx],
            #                  [0, y1],
            #                  color=c, linestyle='--', linewidth=1.5)
            #
            # ## y
            # y_kde = gaussian_kde(y, bw_method=0.1)
            # y_grid = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 200)
            # ax_histy.plot(y_kde(y_grid), y_grid, color=c, lw=1)
            # ax_histy.fill_betweenx(y_grid, y_kde(y_grid), color=c, alpha=0.3)
            #
            # # Add mean line
            # x1 = y_kde(cy)[0]
            # ax_histy.plot([0, x1],
            #               [cy, cy],
            #               color=c, linestyle='--', linewidth=1.5)


            # ========= Frequency distribution (marginal distribution) ===============
            ## x
            counts, bin_edges, _ = ax_histx.hist(
                x,
                bins=100,
                range=(-1, 1),
                density=True,
                histtype='step',
                linewidth=0
            )

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            smooth_counts = gaussian_filter1d(counts, sigma=2)

            # Correct orientation
            ax_histx.plot(bin_centers, smooth_counts, color=c, lw=1.5)
            ax_histx.fill_between(bin_centers, smooth_counts, color=c, alpha=0.3)

            # Mean line
            idx = np.argmin(np.abs(bin_centers - cx))
            y1 = smooth_counts[idx]

            ax_histx.plot([cx, cx], [0, y1], '--', color=c, linewidth=1.5)

            # Align closely with the main plot
            ax_histx.set_ylim(0, max(smooth_counts) * 1.1)
            ax_histx.margins(y=0)


            ## y
            # histogram（Used for statistics only）
            counts, bin_edges, _ = ax_histy.hist(
                y,
                bins=100,
                range=(-1, 1),
                density=True,
                histtype='step',
                linewidth=0
            )

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Smooth
            smooth_counts = gaussian_filter1d(counts, sigma=2)

            # Plot
            ax_histy.plot(smooth_counts, bin_centers, color=c, lw=1.5)
            ax_histy.fill_betweenx(bin_centers, smooth_counts, color=c, alpha=0.3)

            # Mean line
            idx = np.argmin(np.abs(bin_centers - cy))
            x1 = smooth_counts[idx]

            ax_histy.plot([0, x1], [cy, cy], '--', color=c, linewidth=1.5)

            ax_histy.set_xlim(0, max(smooth_counts) * 1.1)
            ax_histy.margins(x=0)


        # === Mean arrow ===
        for k in range(len(centroid_x) - 1):
            ax.annotate(
                '',
                xy=(centroid_x[k + 1], centroid_y[k + 1]),
                xytext=(centroid_x[k], centroid_y[k]),
                arrowprops=dict(
                    arrowstyle='->',
                    mutation_scale=20,  # 1 = default size, >1 larger, <1 smaller
                    color='k',
                    lw=2,
                    alpha=1
                ),
                zorder=4
            )


            # ### Connect means on the marginal density plots
            # # Get the two mean points
            # x1, x2 = np.nanmean(x_data[k]), np.nanmean(x_data[k+1])
            # y1, y2 = np.nanmean(y_data[k]), np.nanmean(y_data[k + 1])
            # y1_val = x_kde(x1)[0] if len(x) > 0 else 0  # Note: x_kde here belongs to the last period
            # y2_val = x_kde(x2)[0] if len(x) > 0 else 0
            #
            # # Add an arrow to the upper marginal plot
            # ax_histx.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),)
            #                   # arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

        # Reference line
        ax.axhline(0, ls='--', lw=0.8, color='gray', zorder=2)
        ax.axvline(0, ls='--', lw=0.8, color='gray', zorder=2)

        if gradient_by == 'Veg' or gradient_by == 'AI' :
            if type == 'Forest' or type == 'Arid':
                word = 'a'
            elif type == 'Shrub' or type == 'Semi-arid':
                word = 'b'
            elif type == 'Savanna' or type == 'Dry sub-humid':
                word = 'c'
            elif type == 'Grass' or type == 'Humid':
                word = 'd'
            # ax_histx.set_title(f'({word}) {type}', fontsize=10, pad=6)

        ax.set_xlabel('SM (standardized)')
        ax.set_ylabel('VPD (standardized)')

        ax.legend(
            handles=legend_handles,
            loc='upper right',
            frameon=False,  ##Legend frame and background color
            # fontsize=7,
            handletextpad=0.1,  # Reduce spacing between legend lines and text
        )




    plt.tight_layout()

    # output_path = os.path.join(output_fig_path, rf'SM_VPD{climate_test_number}_Phase_Space_Analysis_{gradient_by}.png')
    output_path = os.path.join(output_fig_path, rf'{gradient_by}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_{Basedon}_Phase_Space_Analysis{mean_or_yearPOS}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # plt.show()
    # plt.close()

if Phase_Space_Analysis_year == 5 :
    #########Five-year averages
    sm_2001_2005 = np.nanmean(sm_standardized[:5, :, :], axis=0)
    sm_2006_2010 = np.nanmean(sm_standardized[5:10, :, :], axis=0)
    sm_2011_2015 = np.nanmean(sm_standardized[10:15, :, :], axis=0)
    sm_2016_2020 = np.nanmean(sm_standardized[15:20, :, :], axis=0)
    sm_2021_2024 = np.nanmean(sm_standardized[20:24, :, :], axis=0)

    vpd_2001_2005 = np.nanmean(vpd_standardized[:5, :, :], axis=0)
    vpd_2006_2010 = np.nanmean(vpd_standardized[5:10, :, :], axis=0)
    vpd_2011_2015 = np.nanmean(vpd_standardized[10:15, :, :], axis=0)
    vpd_2016_2020 = np.nanmean(vpd_standardized[15:20, :, :], axis=0)
    vpd_2021_2024 = np.nanmean(vpd_standardized[20:24, :, :], axis=0)

    sm_each_n_year = np.stack([
        sm_2001_2005,
        sm_2006_2010,
        sm_2011_2015,
        sm_2016_2020,
        sm_2021_2024
    ], axis=0)

    vpd_each_n_year = np.stack([
        vpd_2001_2005,
        vpd_2006_2010,
        vpd_2011_2015,
        vpd_2016_2020,
        vpd_2021_2024
    ], axis=0)
    print('Standardized done!')



if Phase_Space_Analysis_year == 12:
    ########### Twelve-year averages
    sm_2001_2012 = np.nanmean(sm_standardized[:12, :, :], axis=0)
    sm_2013_2024 = np.nanmean(sm_standardized[12:, :, :], axis=0)

    vpd_2001_2012 = np.nanmean(vpd_standardized[:12, :, :], axis=0)
    vpd_2013_2024 = np.nanmean(vpd_standardized[12:, :, :], axis=0)

    sm_2001_2012 = np.where((mcd12q2_mask & gpp_mask), sm_2001_2012, np.nan)
    sm_2013_2024 = np.where((mcd12q2_mask & gpp_mask), sm_2013_2024, np.nan)

    vpd_2001_2012 = np.where((mcd12q2_mask & gpp_mask), vpd_2001_2012, np.nan)
    vpd_2013_2024 = np.where((mcd12q2_mask & gpp_mask), vpd_2013_2024, np.nan)

    sm_each_n_year = np.stack([
        sm_2001_2012,
        sm_2013_2024
    ], axis=0)

    vpd_each_n_year = np.stack([
        vpd_2001_2012,
        vpd_2013_2024
    ], axis=0)
    print('Standardized done!')

    common_pixels = np.all(
        np.isfinite(sm_each_n_year)
        & np.isfinite(vpd_each_n_year),
        axis=0,
    )

    sm_each_n_year = np.where(
        common_pixels[None, :, :], sm_each_n_year, np.nan
    )
    vpd_each_n_year = np.where(
        common_pixels[None, :, :], vpd_each_n_year, np.nan
    )


# ### Fig4
plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'All')
print('Fig4 plot done!')
# ## Fig S13-15
# plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'Veg')
# print('S13 plot done!')
# plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'AI')
# print('S14 plot done!')
# plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'Cor mean')
# print('S15 plot done!')
# plot_phase_space_by_veg_AI(sm_each_n_year, vpd_each_n_year, Phase_Space_Analysis_year, 'POS_slope_sig')
