import os.path
import glob
import sys

import numpy as np

from osgeo import gdal, osr
import re
import gc
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats
from sklearn.metrics import r2_score
from pygam import LinearGAM, s
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score

from mpl_toolkits.basemap import Basemap
from joblib import Parallel, delayed
from scipy.stats import theilslopes
from scipy.stats import gaussian_kde
from matplotlib.ticker import FormatStrFormatter



# 【修改1】统一将源文件的NoData/掩膜转换为NaN；0是否有效由文件元data决定。
def read_raster_nan(filename):
    ds = gdal.Open(filename)
    if ds is None:
        raise RuntimeError(f"无法打开：{filename}")
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    data[band.GetMaskBand().ReadAsArray() == 0] = np.nan
    data[~np.isfinite(data)] = np.nan
    ds = None
    return data


# 【修改2】按文件名中的年份配对；禁止不同年份错位或重复年份。
def annual_files(files, label, start, end):
    by_year = {}
    for filename in files:
        found = set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?:\d{4})?(?!\d)",
                               os.path.splitext(os.path.basename(filename))[0]))
        if len(found) != 1:
            raise ValueError(f"{label}无法唯一识别年份：{filename}")
        year = int(found.pop())
        if start <= year <= end:
            if year in by_year:
                raise ValueError(f"{label}年份重复：{year}")
            by_year[year] = filename
    missing = [y for y in range(start, end + 1) if y not in by_year]
    if missing:
        raise ValueError(f"{label}缺少年度文件：{missing}")
    return [by_year[y] for y in range(start, end + 1)]


def check_grid(filename, reference):
    ds = gdal.Open(filename)
    if ds is None:
        raise RuntimeError(f"无法打开：{filename}")
    same_size = (ds.RasterYSize, ds.RasterXSize) == (reference.RasterYSize, reference.RasterXSize)
    same_gt = np.allclose(ds.GetGeoTransform(), reference.GetGeoTransform(), rtol=0, atol=1e-9)
    a, b = osr.SpatialReference(), osr.SpatialReference()
    a.ImportFromWkt(ds.GetProjection())
    b.ImportFromWkt(reference.GetProjection())
    if not (same_size and same_gt and a.IsSame(b)):
        raise ValueError(f"输入网格不一致：{filename}")
    ds = None

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


# 【修改9】Windows并行入口。
if __name__ == "__main__":
    ################################ 1 Input and output #############################
    ##### Input settings #####

    star_year = 2001
    end_year = 2024
    years_length = end_year - star_year + 1
    print('years_length:', years_length)

    cal_done = 'no'  # 'yes' means already calculated; 'no' means not yet calculated

    data_type = 'MCD12Q2' # GIMMS_NDVI_16d / MODIS_GPP_16d / MODIS_GPP_8d / MODIS_EVI2_8d / MCD12Q2

    mean_or_yearPOS = ''  # '' uses annual POS; '_meanPOS' uses long-term mean POS

    coupling_method = 'Pearson'  ### 'Pearson'/'Partial'
    lag_day = 0

    partial_for = 'SMlag_VPDlag'  ###SM and lagged VPD: SM_VPDlag; SM and VPD: SMlag_VPDlag

    Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno uses detrended SOS and POS for partial correlation
                                       ### Based_on_OriginPheno uses original SOS and POS for partial correlation

    veg_threshold = 25

    scale = 11

    min_lat = 30
    max_lat = 84

    test_number = '0'
    only_analyses_significant = 'No'   ###Yes analyzes only pixels with POS or Cor p-value < 0.05; No analyzes all pixels

    Outlier = 'Yes'  # Yes / No
    OutnosigCor = 'No'  # Yes / No

    input_path = rf'D:/code/Global_test2_{scale}000m'

    pos_input = fr'{input_path}\Data\Pheno\POS_{scale}km'

    # folder_cor_1 = fr'{input_path}\Data\Climate\Correlation(SM_VPD_pearson)1'  #(POS-30) - POS  #17_8_1
    # folder_cor_2 = fr'{input_path}\Data\Climate\Correlation(SM_VPD_pearson)2'  #(POS-60) - POS  #17_8_2
    # folder_cor_3 = fr'{input_path}\Data\Climate\Correlation(SM_VPD_pearson)3'  #(POS-90) - POS  #17_8_3

    if coupling_method == 'Pearson':
        folder_cor_1 = fr'{input_path}\Data\Climate\climate_in_best_preseason\Cor_Pearson_SM_t_VPD_t{mean_or_yearPOS}\Cor_preseason_mean'  # (POS-30) - POS : Correlation(SM_VPD_pearson)17_8_1

    elif coupling_method == 'Partial' and partial_for == 'SM_VPDlag':
        folder_cor_1 = fr'{input_path}\Data\Climate\climate_in_best_preseason\Cor_Pcor_SM_lag{lag_day}_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}\Cor_preseason_mean'

    elif coupling_method == 'Partial' and partial_for == 'SMlag_VPDlag':
        folder_cor_1 = fr'{input_path}\Data\Climate\climate_in_best_preseason\Cor_Pcor_SM_t_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}\Cor_preseason_mean'

    # folder_cor_pvalue_1 = fr'{input_path}\Data\Climate\Correlation(SM_VPD_pearson)1\Pvalue'  #(POS-30) - POS
    # folder_cor_pvalue_2 = fr'{input_path}\Data\Climate\Correlation(SM_VPD_pearson)2\Pvalue'  #(POS-60) - POS
    # folder_cor_pvalue_3 = fr'{input_path}\Data\Climate\Correlation(SM_VPD_pearson)3\Pvalue'  #(POS-90) - POS
    folder_cor_pvalue_1 = fr'{folder_cor_1}\Pvalue'  #(POS-30) - POS

    pos_tiffiles = sorted(glob.glob(os.path.join(pos_input, '*.tif')))

    tif_files_cor_1 = sorted(glob.glob(os.path.join(folder_cor_1, '*.tif')))
    # tif_files_cor_2 = sorted(glob.glob(os.path.join(folder_cor_2, '*.tif')))
    # tif_files_cor_3 = sorted(glob.glob(os.path.join(folder_cor_3, '*.tif')))
    # cor_length_tif = os.path.join(input_path, fr'{input_path}\Data\Climate\Best_preseason_length\Cor_preseason_length.tif')

    tif_files_cor_pvalue_1 = sorted(glob.glob(os.path.join(folder_cor_pvalue_1, '*.tif')))
    # tif_files_cor_pvalue_2 = sorted(glob.glob(os.path.join(folder_cor_pvalue_2, '*.tif')))
    # tif_files_cor_pvalue_3 = sorted(glob.glob(os.path.join(folder_cor_pvalue_3, '*.tif')))


    #### Input vegetation type TIF path
    veg_type_tif = rf'{input_path}\Data\Classify\Veg_type\NH_permanent_veg_type_{scale}km_gte{veg_threshold}.tif'

    #### Input的AI index tif path
    ai_tif = fr'{input_path}\Data\Classify\AI\NH30_84_AI(graident)_{scale}km.tif'

    #### Input coupling gradient TIF path
    cor_mean_file = fr'{input_path}\Tif_of_fig\3Cor_mean_slope\mean\Cor(Pearson_SMlag_VPDlag_lag0)_mean_{scale}km_SOS_All.tif'  #SOS - POS


    mcd12q2_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
    mcd12q2_mask_file = mcd12q2_mask_tif
    mcd12q2_mask = np.isfinite(read_raster_nan(mcd12q2_mask_file))

    gpp_mask_file = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
    gpp_mask = np.isfinite(read_raster_nan(gpp_mask_file))

    ##### Output settings #####
    fig_output = rf'{input_path}\Fig\Fig 5 Relationship between POS and Cor'
    output_cor_mean_slope_tif_path = rf'{input_path}\Tif_of_fig\5Relationship between POS and Cor'


    ############################### Basic information ###############################
    # 【修改2】先按年度配对，再核对所有Input网格。
    pos_tiffiles = annual_files(pos_tiffiles, 'PPT', star_year, end_year)
    tif_files_cor_1 = annual_files(tif_files_cor_1, 'Coupling', star_year, end_year)
    if OutnosigCor == 'Yes':
        tif_files_cor_pvalue_1 = annual_files(tif_files_cor_pvalue_1, 'Coupling p', star_year, end_year)
    else:
        tif_files_cor_pvalue_1 = []

    sample = tif_files_cor_1[0]
    sample_tif = gdal.Open(sample)

    if sample_tif is None:
        raise RuntimeError(f"无法打开：{sample}")
    for filename in (pos_tiffiles + tif_files_cor_1 + tif_files_cor_pvalue_1
                     + [veg_type_tif, ai_tif, cor_mean_file, mcd12q2_mask_file, gpp_mask_file]):
        check_grid(filename, sample_tif)

    sample_array = read_raster_nan(sample)

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

    #### Vegetation typedata
    veg_type_data = read_raster_nan(veg_type_tif)
    print('veg_type_data shape:', veg_type_data.shape)

    # climate_type_tif = gdal.Open(climate_type_tif)
    # climate_type_data = climate_type_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    # print('climate_type_data shape:', climate_type_data.shape)

    ai_type_data = read_raster_nan(ai_tif)
    print('ai_type_data shape:', ai_type_data.shape)

    cor_type_data = read_raster_nan(cor_mean_file)
    print('cor_type_data shape:', cor_type_data.shape)

    if cal_done == 'no':

        ################################ 2 Stacking ################################
        def get_band(tif, stack):
            stack.append(read_raster_nan(tif))


        pos_stack = []
        cor_stack_1 = []
        cor_stack_2 = []
        cor_stack_3 = []

        cor_pvalue_stack_1 = []
        cor_pvalue_stack_2 = []
        cor_pvalue_stack_3 = []

        for tif_file in pos_tiffiles:
            get_band(tif_file, pos_stack)

        for tif_file in tif_files_cor_1:
            get_band(tif_file, cor_stack_1)
        # for tif_file in tif_files_cor_2:
        #     get_band(tif_file, cor_stack_2)
        # for tif_file in tif_files_cor_3:
        #     get_band(tif_file, cor_stack_3)

        # cor_length_tif = gdal.Open(cor_length_tif)
        # cor_length = cor_length_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

        for tif_file in tif_files_cor_pvalue_1:
            get_band(tif_file, cor_pvalue_stack_1)
        # for tif_file in tif_files_cor_pvalue_2:
        #     get_band(tif_file, cor_pvalue_stack_2)
        # for tif_file in tif_files_cor_pvalue_3:
        #     get_band(tif_file, cor_pvalue_stack_3)

        pos_stack = np.stack(pos_stack, axis=0)

        cor_stack_1 = np.stack(cor_stack_1, axis=0)#[:, 505:510, 505:510]
        # cor_stack_2 = np.stack(cor_stack_2, axis=0)
        # cor_stack_3 = np.stack(cor_stack_3, axis=0)
        # cor_stack = np.stack([cor_stack_1[:years_length, :, :], cor_stack_2[:years_length, :, :], cor_stack_3[:years_length, :, :]], axis=0)

        cor_pvalue_stack_1 = (np.stack(cor_pvalue_stack_1, axis=0)
                              if OutnosigCor == 'Yes' else None)#[:, 505:510, 505:510]
        # cor_pvalue_stack_2 = np.stack(cor_pvalue_stack_2, axis=0)
        # cor_pvalue_stack_3 = np.stack(cor_pvalue_stack_3, axis=0)
        # cor_pvalue_stack = np.stack([cor_pvalue_stack_1[:years_length, :, :], cor_pvalue_stack_2[:years_length, :, :], cor_pvalue_stack_3[:years_length, :, :]], axis=0)
        # cor_pvalue_stack = cor_stack


        # #### Coupling gradient data
        # cor_mean_data = read_raster_nan(cor_mean_file)
        # print('cor_mean_data shape:', cor_mean_data.shape)

        print(f'原POS valid pixels: {np.count_nonzero(np.isfinite(pos_stack[0, :, :]))}')
        # print(f'原Cor valid pixels: {np.count_nonzero(np.isfinite(cor_stack_1[0, :, :]))}')

        pos_stack = np.where(np.isfinite(veg_type_data), pos_stack, np.nan)
        cor_stack_1 = np.where(np.isfinite(veg_type_data), cor_stack_1, np.nan)


        ############################### 3 Remove outliers ###################################
        if Outlier == 'Yes':
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

                    if np.count_nonzero(np.isfinite(x_masked)) >= (years_length/2):
                        return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
                    else:
                        # print(f'IQR threshold：{lower_range:.4f} ~ {upper_range:.4f}\n'
                        #       f'Original pixel data: {x}\n'
                        #       f'More than half the pixel values are invalid after outlier removal: {x_masked}')
                        return np.full_like(x, np.nan), i, j


            outlier_pos_stack = np.full((years_length, rows, cols), np.nan, dtype=np.float32)

            outlier_cor_stack = np.full((years_length, rows, cols), np.nan, dtype=np.float32)

            ### IQR去 POS 异常值
            results = Parallel(n_jobs=18, verbose=10)(
                delayed(Outlier_array_IQR)(
                    pos_stack[:, i, j], i, j, 25, 75
                ) for i, j in zip(row_indices, col_indices)
            )

            for data_mask, i, j in results:
                outlier_pos_stack[:, i, j] = data_mask


            ### Remove Cor outliers using IQR
            results = Parallel(n_jobs=18, verbose=10)(
                delayed(Outlier_array_IQR)(
                    cor_stack_1[:, i, j], i, j, 25, 75
                ) for i, j in zip(row_indices, col_indices)
            )

            for data_mask, i, j in results:
                outlier_cor_stack[:, i, j] = data_mask

            mask = np.isfinite(outlier_pos_stack) & np.isfinite(outlier_cor_stack)

        elif Outlier == 'No':
            mask = np.isfinite(pos_stack) & np.isfinite(cor_stack_1)

            # cor_mean = np.nanmean(cor_pre_stack, axis=0)

        if OutnosigCor == 'Yes':
            mask = (cor_pvalue_stack_1[:years_length,:,:] <= 0.1) & mask

        valid_pixel_count = np.nansum(mask, axis=0)

        # Create a mask of valid pixels
        valid_pixel_mask = valid_pixel_count >= (years_length/2)  # (years_length - 3)
        print(f'Remove outliers且配对后有效年份至少一半的像元数量:{np.count_nonzero(valid_pixel_mask)}')

        paired_mask = mask & valid_pixel_mask[None, :, :]

        if Outlier == 'Yes':
            pos_stack_clean = np.where(paired_mask, outlier_pos_stack, np.nan)
            cor_stack_clean = np.where(paired_mask, outlier_cor_stack, np.nan)
        else:
            pos_stack_clean = np.where(paired_mask, pos_stack, np.nan)
            cor_stack_clean = np.where(paired_mask, cor_stack_1, np.nan)
        print(f'pos_stack_clean shape:{pos_stack_clean.shape[0]}')



        ################################ 4 POS和Cor sen’s slope/mean 计算 #########################
        def calculate_senSlope(data):

            mask = np.isfinite(data)

            data_clean = data[mask]
            years_valid = years[mask]

            if len(data_clean) >= (years_length/2):   ### >10 / >17/ >19

                # Sen slope
                result = theilslopes(data_clean, years_valid)

                slope = result.slope
                # intercept = result.intercept

                # Mann-Kendall
                mk_result = mk.original_test(data_clean)

                pvalue = mk_result.p

            else:
                ############# 如果 len(data_nodrought_clean) <= 10 可以考虑用滑动窗口来计算 ###########
                slope = np.nan
                pvalue = np.nan
                # intercept = np.nan

            return slope, pvalue


        # ###POS mean
        pos_mean = np.nanmean(pos_stack_clean, axis=0)
        # print('pos_mean:', pos_mean[50:55, 550:555])


        years = np.arange(star_year, star_year + pos_stack.shape[0])
        print(f'years shape:{years.shape[0]}')

        ###POS sen's slope
        pos_results = np.apply_along_axis(calculate_senSlope,
                                            axis=0,  # 沿着Time维度
                                            arr=pos_stack_clean)
        pos_slope = pos_results[0, :, :]
        pos_slope_p = pos_results[1, :, :]

        ### Cor mean
        cor_mean = np.nanmean(cor_stack_clean, axis=0)

        ###Cor sen's slope
        cor_results = np.apply_along_axis(calculate_senSlope,
                                            axis=0,  # 沿着Time维度
                                            arr=cor_stack_clean)
        cor_slope = cor_results[0, :, :]
        cor_slope_p = cor_results[1, :, :]

        cor_mean = np.where((mcd12q2_mask & gpp_mask), cor_mean, np.nan)
        cor_slope = np.where((mcd12q2_mask & gpp_mask), cor_slope, np.nan)
        cor_slope_p = np.where((mcd12q2_mask & gpp_mask), cor_slope_p, np.nan)

        pos_mean = np.where((mcd12q2_mask & gpp_mask), pos_mean, np.nan)
        pos_slope = np.where((mcd12q2_mask & gpp_mask), pos_slope, np.nan)
        pos_slope_p = np.where((mcd12q2_mask & gpp_mask), pos_slope_p, np.nan)


        tif_output_path = os.path.join(output_cor_mean_slope_tif_path, rf'Cor_mean\Cor({coupling_method}_{partial_for}_lag{lag_day})_mean_{scale}km_All{mean_or_yearPOS}.tif')
        os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
        save_tif_gdal(
            tif_output_path,
            cor_mean,
            crs,
            gt  # Use the new geotransform
        )
        tif_output_path = os.path.join(output_cor_mean_slope_tif_path, rf'POS_mean\POS_mean_{scale}km_All{mean_or_yearPOS}.tif')
        os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
        save_tif_gdal(
            tif_output_path,
            pos_mean,
            crs,
            gt  # Use the new geotransform
        )

        tif_output_path = os.path.join(output_cor_mean_slope_tif_path,
                                       rf'Cor_slope\Cor({coupling_method}_{partial_for}_lag{lag_day})_slope_{scale}km_All{mean_or_yearPOS}.tif')
        os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
        save_tif_gdal(
            tif_output_path,
            cor_slope,
            crs,
            gt  # Use the new geotransform
        )
        tif_output_path = os.path.join(output_cor_mean_slope_tif_path,
                                       rf'POS_slope\POS_slope_{scale}km_All{mean_or_yearPOS}.tif')
        os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
        save_tif_gdal(
            tif_output_path,
            pos_slope,
            crs,
            gt  # Use the new geotransform
        )

        tif_output_path = os.path.join(output_cor_mean_slope_tif_path,
                                       rf'Cor_slope\Cor({coupling_method}_{partial_for}_lag{lag_day})_slopePvalue_{scale}km_All{mean_or_yearPOS}.tif')
        os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
        save_tif_gdal(
            tif_output_path,
            cor_slope_p,
            crs,
            gt  # Use the new geotransform
        )
        tif_output_path = os.path.join(output_cor_mean_slope_tif_path,
                                       rf'POS_slope\POS_slopePvalue_{scale}km_All{mean_or_yearPOS}.tif')
        os.makedirs(os.path.dirname(tif_output_path), exist_ok=True)
        save_tif_gdal(
            tif_output_path,
            pos_slope_p,
            crs,
            gt  # Use the new geotransform
        )

    elif cal_done == 'yes':

        def read_tif_as_array(path):
            """用 GDAL 读取 GeoTIFF，返回二维数组；NoData 转成 np.nan。"""
            dataset = gdal.Open(path, gdal.GA_ReadOnly)
            if dataset is None:
                raise FileNotFoundError(f'无法打开文件：{path}')

            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray().astype(np.float32)

            nodata = band.GetNoDataValue()
            if nodata is not None:
                data[data == nodata] = np.nan

            dataset = None  # 释放 GDAL data集
            return data


        cor_mean_path = os.path.join(
            output_cor_mean_slope_tif_path,
            rf'Cor_mean\Cor({coupling_method}_{partial_for}_lag{lag_day})_mean_{scale}km_All{mean_or_yearPOS}.tif'
        )
        pos_mean_path = os.path.join(
            output_cor_mean_slope_tif_path,
            rf'POS_mean\POS_mean_{scale}km_All{mean_or_yearPOS}.tif'
        )
        cor_slope_path = os.path.join(
            output_cor_mean_slope_tif_path,
            rf'Cor_slope\Cor({coupling_method}_{partial_for}_lag{lag_day})_slope_{scale}km_All{mean_or_yearPOS}.tif'
        )
        pos_slope_path = os.path.join(
            output_cor_mean_slope_tif_path,
            rf'POS_slope\POS_slope_{scale}km_All{mean_or_yearPOS}.tif'
        )
        cor_slope_p_path = os.path.join(
            output_cor_mean_slope_tif_path,
            rf'Cor_slope\Cor({coupling_method}_{partial_for}_lag{lag_day})_slopePvalue_{scale}km_All{mean_or_yearPOS}.tif'
        )
        pos_slope_p_path = os.path.join(
            output_cor_mean_slope_tif_path,
            rf'POS_slope\POS_slopePvalue_{scale}km_All{mean_or_yearPOS}.tif'
        )

        cor_mean = read_tif_as_array(cor_mean_path)
        pos_mean = read_tif_as_array(pos_mean_path)
        cor_slope = read_tif_as_array(cor_slope_path)
        pos_slope = read_tif_as_array(pos_slope_path)
        cor_slope_p = read_tif_as_array(cor_slope_p_path)
        pos_slope_p = read_tif_as_array(pos_slope_p_path)

        valid_pos_slope = pos_slope[np.isfinite(pos_slope)]

        # 四舍五入仅用于合并浮点数表示上的微小差异，不会修改原始 pos_slope
        values, counts = np.unique(
            np.round(valid_pos_slope, 6),
            return_counts=True
        )

        top = np.argsort(counts)[-20:][::-1]

        print('有效像元数:', valid_pos_slope.size)
        print('POS slope 不同取值数:', len(values))
        print('前20个最常见的 POS slope:')
        for i in top:
            print(f'{values[i]:.6f}: {counts[i]} 个像元')

        # sys.exit()

        cor_mean = np.where(np.isfinite(veg_type_data)&np.isfinite(ai_type_data)&np.isfinite(cor_type_data), cor_mean, np.nan)
        pos_mean = np.where(np.isfinite(veg_type_data)&np.isfinite(ai_type_data)&np.isfinite(cor_type_data), pos_mean, np.nan)
        cor_slope = np.where(np.isfinite(veg_type_data)&np.isfinite(ai_type_data)&np.isfinite(cor_type_data), cor_slope, np.nan)
        pos_slope = np.where(np.isfinite(veg_type_data)&np.isfinite(ai_type_data)&np.isfinite(cor_type_data), pos_slope, np.nan)
        cor_slope_p = np.where(np.isfinite(veg_type_data)&np.isfinite(ai_type_data)&np.isfinite(cor_type_data), cor_slope_p, np.nan)
        pos_slope_p = np.where(np.isfinite(veg_type_data)&np.isfinite(ai_type_data)&np.isfinite(cor_type_data), pos_slope_p, np.nan)

    print('--此处以下是二维密度图--')

    ################################ 5 Plot(二维密度图+散点图） #####################################
    ## ！！！只保留显著像元：
    if only_analyses_significant == 'Yes':
        significant_mask = (np.isfinite(pos_slope_p)) & (np.isfinite(cor_slope_p)) & ((pos_slope_p < 0.05)|(cor_slope_p < 0.05))
        print(f'POS或Cor变化显著的像元数量：{np.count_nonzero(significant_mask)}')
        pos_slope = np.where(significant_mask, pos_slope, np.nan)
        cor_slope = np.where(significant_mask, cor_slope, np.nan)

    # sys.exit()



    ### 线性拟合
    def linegres_fit(data1, data2):
        mask = np.isfinite(data1) & np.isfinite(data2)
        x_clean, y_clean = data1[mask], data2[mask]
        if len(x_clean) < 3 or np.ptp(x_clean) == 0:
            return np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan
        slope, intercept, r_value, p_value, stderr = stats.linregress(x_clean, y_clean)
        x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
        return x_fit, slope * x_fit + intercept, slope, intercept, r_value**2, p_value


    # ### 多项式拟合
    # def polynomial_fit_for_spatial_data(data1, data2, degree):
    #     """多项式回归 - 使用训练-测试分割"""
    #     mask = np.isfinite(data1) & np.isfinite(data2)
    #     x_clean = data1[mask].reshape(-1, 1)
    #     y_clean = data2[mask]
    #
    #     # 多项式特征转换（只在训练集上拟合）
    #     poly_features = PolynomialFeatures(degree=degree)
    #     x_train_poly = poly_features.fit_transform(x_clean)
    #
    #     # 模型训练
    #     model = LinearRegression()
    #     model.fit(x_train_poly, y_clean)
    #
    #     # 生成拟合曲线（使用完整data范围）
    #     x_fit = np.linspace(x_clean.min(), x_clean.max(), 100).reshape(-1, 1)
    #     x_fit_poly = poly_features.transform(x_fit)
    #     y_fit = model.predict(x_fit_poly)
    #
    #     return x_fit.flatten(), y_fit


    # ### GAM拟合
    # # def gam_fit_for_spatial_data_onlyXY(data1, data2, n_splines):
    # #     """GAM拟合 - 使用训练-测试分割"""
    # #     mask = np.isfinite(data1) & np.isfinite(data2)
    # #     x_clean = data1[mask]
    # #     y_clean = data2[mask]
    # #
    # #     # 分割data
    # #     x_train, x_test, y_train, y_test = train_test_split(
    # #         x_clean, y_clean,
    # #         test_size=0.1,  # 固定比例，如0.2
    # #         random_state=42  # 固定整数，如42
    # #     )
    # #
    # #     # # 按x坐标排序（用于无分割：
    # #     # sort_idx = np.argsort(x_clean)
    # #     # x_train_sorted = x_clean[sort_idx]
    # #     # y_train_sorted = y_clean[sort_idx]
    # #
    # #     # 按x坐标排序（用于有分割：
    # #     sort_idx = np.argsort(x_train)
    # #     x_train_sorted = x_train[sort_idx]
    # #     y_train_sorted = y_train[sort_idx]
    # #
    # #     # 创建并拟合GAM模型
    # #     gam = LinearGAM(s(0, n_splines=n_splines)).fit(x_train_sorted, y_train_sorted)
    # #
    # #     # 在测试集上评估
    # #     y_test_pred = gam.predict(x_test)
    # #     test_r2 = r2_score(y_test, y_test_pred)
    # #
    # #     # 生成拟合曲线
    # #     x_fit = np.linspace(x_clean.min(), x_clean.max(), 300)
    # #     y_fit = gam.predict(x_fit)
    # #
    # #     # # 获得趋势显著性
    # #     # p_values = gam.statistics_['p_values']
    # #
    # #     return x_fit, y_fit, test_r2#, p_values
    #
    def gam_fit_for_spatial_data_onlyXY(data1, data2, n_splines, weights):

        mask = np.isfinite(data1) & np.isfinite(data2)

        x_clean = data1[mask]
        y_clean = data2[mask]

        if weights is not None:
            weights_clean = weights[mask]
        else:
            weights_clean = None

        if len(x_clean) < 5 or np.unique(x_clean).size < 4:
            return np.array([]), np.array([]), np.nan

        # train-test split
        if weights_clean is not None:
            x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
                x_clean,
                y_clean,
                weights_clean,
                test_size=0.2,
                random_state=42
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_clean,
                y_clean,
                test_size=0.2,
                random_state=42
            )

        # sort
        sort_idx = np.argsort(x_train)

        x_train_sorted = x_train[sort_idx]
        y_train_sorted = y_train[sort_idx]

        if weights_clean is not None:
            w_train_sorted = w_train[sort_idx]

        # GAM
        gam = LinearGAM(
            s(0, n_splines=n_splines)
        )

        if weights_clean is not None:
            gam.fit(
                x_train_sorted,
                y_train_sorted,
                weights=w_train_sorted
            )
        else:
            gam.fit(
                x_train_sorted,
                y_train_sorted
            )

        # test
        y_test_pred = gam.predict(x_test)

        if weights_clean is not None:
            test_r2 = r2_score(
                y_test,
                y_test_pred,
                sample_weight=w_test
            )
        else:
            test_r2 = r2_score(
                y_test,
                y_test_pred
            )

        # fit curve
        x_fit = np.linspace(
            x_clean.min(),
            x_clean.max(),
            300
        )

        y_fit = gam.predict(x_fit)

        return x_fit, y_fit, test_r2
    #
    #
    # 【修改】双变量色标及量化。
    ## ======== 5.1 二维colorbar定义 ======== ##
    #### mean:
    # 【修改】4×4分级。Coupling长期均值的4个区间：
    # [-0.6,-0.4)、[-0.4,-0.2)、[-0.2,0)、[0,0.2]。
    # 低于-0.6/高于0.2的有效像元沿用原代码，归入两端color。
    mean_lower_bound_x = -0.6
    mean_upper_bound_x = 0.2
    mean_bin_size_x = 0.2

    # PPT原显示范围不变，分成4级。
    mean_lower_bound_y = 120
    mean_upper_bound_y = 280
    mean_bin_size_y = 40
    mean_bins = 4

    # 趋势图原显示范围不变，分成4级。
    slope_lower_bound_x = -0.024
    slope_upper_bound_x = 0.024
    slope_lower_bound_y = -1
    slope_upper_bound_y = 1
    slope_bin_size_x = 0.012
    slope_bin_size_y = 0.5
    slope_bins = 4

    ## ======== 5.2 准备双变量color矩阵 ======== ##
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    import matplotlib.gridspec as gridspec


    ## 插值
    def create_bivariate_colormap(nquantiles, x_left_color, x_right_y_bottom_color, y_top_color):

        n = nquantiles

        col_matrix = np.zeros((n, n, 3))

        for i in range(n):      # i: 上→下
            for j in range(n):  # j: 左→右

                x = j / (n - 1)          # 左→右：0→1
                y = 1-i / (n - 1)      # 上→下：0→1（修正：1-i/(n-1) 需要加括号）

                # 四角权重（标准双线性插值）
                w_ll = (1 - x) * (1 - y)   # 左上
                w_lr = x * (1 - y)         # 右上
                w_ul = (1 - x) * y         # 左下
                w_ur = x * y               # 右下

                # 四个角的color
                c_ll = np.array([0, 1, 0])  # 左上
                c_lr = y_top_color   # 右上
                c_ul = x_left_color   # 左下
                c_ur = x_right_y_bottom_color  # 右下

                color = (
                    w_ll * c_ll +
                    w_lr * c_lr +
                    w_ul * c_ul +
                    w_ur * c_ur
                )

                col_matrix[i, j] = color

        return col_matrix

    def plot_bivariate_colorbar(col_matrix, xlabel='x', ylabel='y'):
        """
        绘制二维 colorbar
        """

        n = col_matrix.shape[0]

        fig, ax = plt.subplots(figsize=(4, 4))

        # imshow 直接画color矩阵
        ax.imshow(col_matrix, origin='lower')

        # 设置刻度（可选）
        ax.set_xticks([0, n-1])
        ax.set_yticks([0, n-1])

        ax.set_xticklabels(['Low', 'High'])
        ax.set_yticklabels(['Low', 'High'])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # 去掉网格
        ax.grid(False)

        plt.tight_layout()
        plt.show()


    x_left = np.array([1, 1, 0])
    y_top = np.array([0, 0, 1])
    x_right_y_bottom  = np.array([1, 1, 1])

    # 【仅配色修改】4×4分级，使用参考图方向的color。
    # 【仅配色修改】使用4×4分级，四角为蓝、绿、橙、黄。
    # 第0行是低PPT，最后一行是高PPT；color与像元索引共用col_matrix。
    def reference_color_matrix(n=8):
        matrix = np.empty((n, n, 3), dtype=float)
        bottom_left = np.array([0.0, 0.0, 1.0])
        bottom_right = np.array([0.0, 1.0, 0.0])
        top_left = np.array([1.0, 0.5, 0.0])
        top_right = np.array([1.0, 1.0, 0.0])
        for row in range(n):
            y = row / (n - 1)
            for col in range(n):
                x = col / (n - 1)
                matrix[row, col] = (
                    (1-x)*(1-y)*bottom_left + x*(1-y)*bottom_right
                    + (1-x)*y*top_left + x*y*top_right
                )
        return matrix

    col_matrix = reference_color_matrix(mean_bins)
    if mean_bins != slope_bins:
        raise ValueError("均值与趋势分级数量不同，需要分别建立色板")
    # plot_bivariate_colorbar(col_matrix)


    #### ========== 5.3 data量化 =========== ####
    def read_and_process_raster(data_x, data_y, lower_bound_x, upper_bound_x,
                                lower_bound_y, upper_bound_y, bin_size_x, bin_size_y):
        nx = int(round((upper_bound_x - lower_bound_x) / bin_size_x))
        ny = int(round((upper_bound_y - lower_bound_y) / bin_size_y))
        if col_matrix.shape[:2] != (ny, nx):
            raise ValueError("色板尺寸与分级数不一致")
        x_edges = np.linspace(lower_bound_x, upper_bound_x, nx + 1)
        y_edges = np.linspace(lower_bound_y, upper_bound_y, ny + 1)
        valid = np.isfinite(data_x) & np.isfinite(data_y)
        qx = np.full(data_x.shape, np.nan)
        qy = np.full(data_y.shape, np.nan)
        qx[valid] = np.clip(np.searchsorted(x_edges, data_x[valid], side='right') - 1, 0, nx - 1)
        qy[valid] = np.clip(np.searchsorted(y_edges, data_y[valid], side='right') - 1, 0, ny - 1)
        return qx, qy


    mean_x, mean_y = cor_mean, pos_mean

    if only_analyses_significant == 'Yes':
        ###  ！！！只保留POSslope显著的像元
        significant_mask = ((pos_slope_p < 0.05)|(cor_slope_p < 0.05)) & np.isfinite(pos_slope_p) & np.isfinite(cor_slope_p)

        pos_slope_significant = np.where(significant_mask, pos_slope, np.nan)
        cor_slope_significant = np.where(significant_mask, cor_slope, np.nan)

        slope_y = pos_slope_significant
        ### 注意，下面的pos_slope要记得更改！！！
        slope_x = cor_slope_significant

    else:
        mean_y = pos_mean
        mean_x = cor_mean

        slope_y = pos_slope
        slope_x = cor_slope


    cormean_quantily, posmean_quantily = read_and_process_raster(mean_x, mean_y,
                                                                  mean_lower_bound_x, mean_upper_bound_x,
                                                                  mean_lower_bound_y, mean_upper_bound_y,
                                                                  mean_bin_size_x, mean_bin_size_y)

    corslope_quantily, posslope_quantily = read_and_process_raster(slope_x, slope_y,
                                                                  slope_lower_bound_x, slope_upper_bound_x,
                                                                  slope_lower_bound_y, slope_upper_bound_y,
                                                                  slope_bin_size_x, slope_bin_size_y)


    def assign_rgb_to_combined_matrix(data_x, data_y, combined_matrix):
        rgb_matrix = np.zeros((rows, cols, 3))  # 存储提取的RGBcolor

        # 根据量化后的坐标组合提取RGBcolor
        for i in range(rows):
            for j in range(cols):
                # 获取第一张栅格量化值 x 和第二张栅格量化值 y
                x = data_x[i, j]
                y = data_y[i, j]

                # 如果 x 或 y 是 NaN，则赋值为空白透明色
                if np.isnan(x) or np.isnan(y):
                    rgb_matrix[i, j] = [1, 1, 1]  # 设置透明色（RGB全0）
                else:
                    x = int(x)
                    y = int(y)
                    # 从combined_matrix提取RGB值
                    rgb_matrix[i, j] = combined_matrix[y, x, :3]  # 提取RGBcolor，忽略透明度

        return rgb_matrix

    biv_map_cormean_posmean = assign_rgb_to_combined_matrix(cormean_quantily, posmean_quantily, col_matrix)
    biv_map_corslope_posslope = assign_rgb_to_combined_matrix(corslope_quantily, posslope_quantily, col_matrix)


    ########### ====== 5.4 像元上的POS与Cor的线性关系 ###########
    def bivariate_map(ax, data, lower_bound_x, upper_bound_x, size_x,
                      lower_bound_y, upper_bound_y, size_y,
                      colorbar_matrix, xlabel, ylabel):

        plots = []

        plot_data_filter = data


        ########### Subplot 1: spatial distribution #################
        ax.set_box_aspect(1)  # Make the map axis square and fill the grid height
        ax.axis('off')
        ### Create the map
        m = Basemap(ax=ax,
                    projection='npstere',  # North polar projection
                    boundinglat=30,  # Minimum displayed latitude (currently 30°N)
                    lon_0=0,  # Central longitude: 180 centers the Pacific, 90 centers Asia
                    resolution='l')

        # Generate grid coordinates
        lons = np.linspace(-180, 180, cols, endpoint=False)
        lats = np.linspace(lat_max, lat_min, rows)
        lons, lats = np.meshgrid(lons, lats)

        # Set latitude and longitude ticks
        m.drawparallels(np.arange(30, 90, 30),
                        labels=[0, 0, 0, 0],
                        linewidth=0.5,
                        # fontsize=8,
                        color="black")

        m.drawmeridians(np.arange(0, 360, 60),
                        latmax=90,  # Make meridians converge at the North Pole
                        labels=[0, 0, 0, 0],  # labels=[left, right, top, bottom] Control longitude label visibility
                        linewidth=0.5,
                        # fontsize=8,
                        color="gray")

        ### Plotdata
        # Plot
        plot = m.pcolormesh(lons, lats,
                            plot_data_filter,
                            shading='nearest',
                            latlon=True,
                            zorder=1)  #控制所画的图所在层，zorder=1（底层），zorder=3（顶层）

        plots.append(plot)  # Save the plot object

        # Plotboundaries
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
            threshold = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.1

            # Find the indices of jumps
            break_indices = np.where(dist > threshold)[0]

            if len(break_indices) == 0:
                # Draw the full line if there are no jumps
                ax.plot(x, y, color='black', linewidth=0.3, zorder=3)
            else:
                # Split the line at jumps and draw each segment
                # This removes lines crossing the center while retaining valid boundaries
                start_idx = 0
                for break_idx in break_indices:
                    ax.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                             color='black', linewidth=0.3, zorder=3)
                    start_idx = break_idx + 1
                # Draw the final segment
                ax.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

        # x_pole, y_pole = m(0, 90)
        # ax.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

        ### Clip to the outer boundary
        from matplotlib.patches import Circle

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        center = [(x0 + x1) / 2, (y0 + y1) / 2]
        radius = (x1 - x0) / 2

        clip_circle = Circle(center, radius, transform=ax.transData)

        for artist in ax.collections + ax.lines + ax.patches:
            artist.set_clip_path(clip_circle)

        boundary_circle = Circle(
            center,
            radius,
            transform=ax.transData,
            facecolor='none',
            edgecolor='black',  # color
            linewidth=0.8,
            zorder=4  # Place on top
        )

        ax.add_patch(boundary_circle)

        #######  colorbar  ############
        inset_pos = [0.01, 0, 0.28, 0.28]
        axins = ax.inset_axes(inset_pos)
        cbar = axins

        n = colorbar_matrix.shape[0]

        # 设置正确的extent，但通过调整ax_cbar的宽高比来保持正方形
        cbar.imshow(colorbar_matrix,
                    interpolation='nearest',
                    extent=[0, n, 0, n],
                    origin='lower')  # 数组的第 0 行（row=0）显示在图的“底部”

        def value_to_index(value, lower, size):
            return (value - lower) / size

        if ylabel == 'POS trend':
            x0 = value_to_index(0, lower_bound_x, size_x)
            y0 = value_to_index(0, lower_bound_y, size_y)

            cbar.axvline(x=x0, color='black', linestyle='--', linewidth=0.5)
            cbar.axhline(y=y0, color='black', linestyle='--', linewidth=0.5)

        # elif ylabel == 'POS mean':
        #     x_line = value_to_index(-0.3, lower_bound_x, size_x)
        #     y_line = value_to_index(200, lower_bound_y, size_y)
        #
        #     cbar.axvline(x=x_line, color='black', linestyle='--', linewidth=0.5)
        #     cbar.axhline(y=y_line, color='black', linestyle='--', linewidth=0.5)

        ### 修改成隔一个position取1个刻度
        tick_idx = np.arange(0, n + 1)  # 显示全部5个boundaries刻度 # 0表示从第一个数开始取

        cbar.set_xticks(tick_idx)
        cbar.set_yticks(tick_idx)
        cbar.tick_params(axis='both', direction='in')

        x_labels = np.linspace(lower_bound_x, upper_bound_x, n+1)
        y_labels = np.linspace(lower_bound_y, upper_bound_y, n+1)



        if ylabel == 'POS mean':
            cbar.set_xlabel(xlabel)
            cbar.set_ylabel('PPT (days)')

            cbar.set_xticklabels([f'{0}' if abs(x) < 1e-6 else f'{x:.1f}' for x in x_labels[tick_idx]], fontsize = 9, rotation=90)
            cbar.set_yticklabels([f'{int(y)}' for y in y_labels[tick_idx]], fontsize = 9)

        elif ylabel == 'POS trend':

            cbar.set_xlabel(r'SM-VPD coupling trend (days$\cdot$a$^{-1}$)')
            cbar.xaxis.set_label_coords(0.7, -0.45)

            cbar.set_ylabel(r'PPT trend (days$\cdot$a$^{-1}$)')

            cbar.set_xticklabels([f'{0}' if abs(x) < 1e-6 else f'{(x * 100):.1f}' for x in x_labels[tick_idx]], fontsize = 9, rotation=90)
            cbar.set_yticklabels([f'{0}' if abs(y) < 1e-6 else
                                  f'{int(y)}' if y == int(y) else
                                  f'{y:.1f}' for y in y_labels[tick_idx]], fontsize = 9)

            ## x
            cbar.text(1.35, -0.15, r'$×10^{-2}$',
                      transform=cbar.transAxes,
                      ha='center',
                      va='center',
                      fontsize = 9)

        plt.tight_layout()

    def plot_combined_hexbin_with_WeightedPointTrendline(data1, data2, xlabel, ylabel, grade_by, type, plot_weighted_point, ax):

        ### === 0 维度划分 === ###
        if grade_by == 'All':

            fig = ax.figure

            # 定义四种Vegetation type
            if only_analyses_significant == 'Yes':
                vmax_values = [6]  # 对应的vmax值  ### 这行vmax值仅适用于像元POS slope显著时
            else:
                if ylabel == 'POS trend':
                    vmax_values = [40]  # Cor50:[55, 20, 12, 25, 12, 6]
                if ylabel == 'POS mean':
                    vmax_values = [40]

            types = ['All']

        elif grade_by == 'Veg':

            fig = ax.figure

            types = ['Forest', 'Shrub', 'Savanna', 'Grass']

            if ylabel == 'POS trend':
                vmax_values = [15, 10, 20, 6]  #Cor50:[55, 20, 12, 25, 12, 6]

            if ylabel == 'POS mean':
                vmax_values = [15, 10, 20, 6]

            i = types.index(type)

            vmax_values = vmax_values[i]
            vmax_values = [vmax_values]

            types = types[i]
            types = [types]


        elif grade_by == 'AI':

            fig = ax.figure

            types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

            if ylabel == 'POS trend':
                vmax_values = [6, 10, 10, 20]  # Cor50:[55, 20, 12, 25, 12, 6]

            if ylabel == 'POS mean':
                vmax_values = [6, 10, 10, 20]

            i = types.index(type)

            vmax_values = vmax_values[i]
            vmax_values = [vmax_values]

            types = types[i]
            types = [types]

        elif grade_by == 'Cor mean':

            fig = ax.figure

            types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                     'Cor(-0.4~-0.3)', 'Cor(<-0.4)']

            if ylabel == 'POS trend':
                vmax_values = [10, 10, 20, 20, 6]  # Cor50:[55, 20, 12, 25, 12, 6]

            if ylabel == 'POS mean':
                vmax_values = [10, 10, 10, 20, 6]

            i = types.index(type)

            vmax_values = vmax_values[i]
            vmax_values = [vmax_values]

            types = types[i]
            types = [types]


        # 预先展平data
        data1_flat = data1.flatten()
        data2_flat = data2.flatten()

        veg_type_flat = veg_type_data.flatten()  # 展平Vegetation typedata
        ai_type_flat = ai_type_data.flatten()  # 展平Vegetation typedata
        cor_type_flat = cor_type_data.flatten()  # 展平Vegetation typedata
        print(f'veg_type_flat count:{np.count_nonzero(np.isfinite(veg_type_flat))}')

        for i, (type, vmax) in enumerate(zip(types, vmax_values)):
            ax1 = ax

            # 创建Vegetation type掩码
            if grade_by == 'All':
                mask = np.isfinite(data1_flat) & np.isfinite(data2_flat)
            elif grade_by == 'Veg':
                if type == 'Forest':
                    mask = (veg_type_flat == 1) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
                elif type == 'Shrub':
                    mask = (veg_type_flat == 2) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
                elif type == 'Savanna':
                    mask = (veg_type_flat == 3) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
                elif type == 'Grass':
                    mask = (veg_type_flat == 4) #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # & (data1_flat >= -0.5) & (data1_flat < 0)
            elif grade_by == 'AI':
                if type == 'Hyper Arid':
                    mask = (ai_type_flat == 1)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0-0.03
                elif type == 'Arid':
                    mask = (ai_type_flat == 2)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.03-0.2
                elif type == 'Semi-arid':
                    mask = ((ai_type_flat == 3) | (ai_type_flat == 4))  #& np.isfinite(data1_flat) & np.isfinite(data2_flat)# AI 0.2-0.35
                elif type == 'Dry sub-humid':
                    mask = (ai_type_flat == 5)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.5-0.65
                elif type == 'Humid':
                    mask = (ai_type_flat == 6)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI > 0.65
            elif grade_by == 'Cor mean':
                if type == 'Cor(-0.1~0)':
                    mask = (-0.1 <= cor_type_flat) & (cor_type_flat < 0)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0-0.03
                elif type == 'Cor(-0.2~-0.1)':
                    mask = (-0.2 <= cor_type_flat) & (cor_type_flat < -0.1)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.03-0.2
                elif type == 'Cor(-0.3~-0.2)':
                    mask = (-0.3 <= cor_type_flat) & (cor_type_flat < -0.2)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat)# AI 0.2-0.35
                elif type == 'Cor(-0.4~-0.3)':
                    mask = (-0.4 <= cor_type_flat) & (cor_type_flat < -0.3)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI 0.5-0.65
                elif type == 'Cor(<-0.4)':
                    mask = (cor_type_flat < -0.4) #& (-0.5 <= cor_mean_data)  #& np.isfinite(data1_flat) & np.isfinite(data2_flat) # AI > 0.65

            valid_mask = (
                    np.isfinite(data1_flat) &
                    np.isfinite(data2_flat) &
                    mask
            )

            x = data1_flat[valid_mask]
            y = data2_flat[valid_mask]

            ### ============= 1 Plot原始散点 ============== ###

            pos_level = np.round(y, 6)
            _, inverse, counts = np.unique(
                pos_level,
                return_inverse=True,
                return_counts=True
            )

            max_points_per_level = 1000
            show_mask = counts[inverse] <= max_points_per_level

            ax1.scatter(
                x,
                y,
                s=8,
                facecolors='none',  # 空心
                edgecolors='gray',  # red边框
                linewidths=0.3,
                alpha=0.3,
                zorder=1
            )


            ### ============== 2 KDE ================ ###
            if ylabel == 'POS trend':
                bandwidth = 0.3
                if data_type == 'MCD12Q2':
                    xmin, xmax = -0.03, 0.03
                    ymin, ymax = -0.6, 0.6
                elif data_type == 'MODIS_GPP_8d':
                    xmin, xmax = -0.04, 0.04
                    ymin, ymax = -1, 1

            elif ylabel == 'POS mean':
                bandwidth = 0.3
                if data_type == 'MODIS_GPP_8d':
                    xmin, xmax = -0.7, 0.2
                    ymin, ymax = 100, 250
                elif data_type == 'MCD12Q2':
                    xmin, xmax = -0.7, 0.2
                    ymin, ymax = 100, 250


            if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
                ax1.text(0.5, 0.5, 'Insufficient variation for KDE',
                         transform=ax1.transAxes, ha='center')
                return
            xy = np.vstack([x, y])


            kde = gaussian_kde(xy, bw_method=bandwidth)  #创建 KDE 模型
            # print(kde.factor)

            # 创建统一网格
            xgrid = np.linspace(xmin, xmax, 200)
            ygrid = np.linspace(ymin, ymax, 200)

            xx, yy = np.meshgrid(xgrid, ygrid)

            # 网格坐标
            grid_coords = np.vstack([xx.ravel(), yy.ravel()])

            # KDE density
            zz = kde(grid_coords).reshape(xx.shape)

            print(f'{ylabel} - {type}, zz min: {np.nanmin(zz)}')
            print(f'{ylabel} - {type}, zz max: {np.nanmax(zz)}')
            print(f'{ylabel} - {type}, zz max * 70%: {np.nanpercentile(zz, 70)}')
            print(f'{ylabel} - {type}, zz max * 75%: {np.nanpercentile(zz, 75)}')
            print(f'{ylabel} - {type}, zz max * 80%: {np.nanpercentile(zz, 80)}')
            print(f'{ylabel} - {type}, zz max * 90%: {np.nanpercentile(zz, 90)}')

            # 固定 density levels
            if ylabel == 'POS mean':
                if grade_by == 'All' or (grade_by == 'AI' and type == 'Semi-arid') or (grade_by == 'Cor mean' and type == 'Cor(<-0.4)'):
                    density_min = 0.01
                    density_max = 0.11
                elif (grade_by == 'Veg' and type == 'Grass') or (grade_by == 'AI' and type == 'Arid'):
                    density_min = 0.01
                    density_max = 0.06
                elif grade_by == 'AI' and type == 'Humid':
                    density_min = 0.01
                    density_max = 0.13
                elif grade_by == 'Cor mean' and type in ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)']:
                    density_min = 0.01
                    density_max = 0.25
                elif grade_by == 'Cor mean' and type in ['Cor(-0.3~-0.2)', 'Cor(-0.4~-0.3)']:
                    density_min = 0.01
                    density_max = 0.15
                else:
                    density_min = 0.01
                    density_max = 0.2

            elif ylabel == 'POS trend':
                if grade_by == 'All':
                    if data_type == 'MODIS_GPP_8d':
                        density_min = 5
                        density_max = 35
                    elif data_type == 'MCD12Q2':
                        density_min = 10
                        density_max = 85
                elif (grade_by == 'Veg' and type == 'Grass') or (grade_by == 'AI' and type == 'Arid'):
                    density_min = 10
                    density_max = 40
                elif grade_by == 'AI' and type == 'Semi-arid':
                    density_min = 10
                    density_max = 50
                elif (grade_by == 'AI' and type == 'Dry sub-humid') or (grade_by == 'AI' and type == 'Humid'):
                    density_min = 10
                    density_max = 30
                elif grade_by == 'Cor mean' and type == 'Cor(-0.2~-0.1)':
                    density_min = 10
                    density_max = 100
                elif grade_by == 'Cor mean' and type in ['Cor(-0.3~-0.2)', 'Cor(-0.1~0)']:
                    density_min = 10
                    density_max = 90
                elif grade_by == 'Cor mean' and type == 'Cor(-0.4~-0.3)':
                    density_min = 10
                    density_max = 80
                elif grade_by == 'Cor mean' and type == 'Cor(<-0.4)':
                    density_min = 15
                    density_max = 60
                else:
                    density_min = 10
                    density_max = 30

            levels = np.linspace(
                density_min,
                density_max,
                11
            )

            # Plot KDE 密度面
            cf = ax1.contourf(
                xx,
                yy,
                zz,
                levels=levels,
                cmap='YlGnBu',
                extend='max',
                alpha=0.75,
                zorder=2
            )

            ### =========== 3 基于分箱密度加权值============= ###
            if ylabel == 'POS trend':
                extent = [-0.04, 0.04, -2, 2]  ##Cor mean: extent = [-0.8, 0.4, -2, 2]
                gridsize = 160
            if ylabel == 'POS mean':
                if data_type == 'MCD12Q2':
                    extent = [-0.7, 0.2, 100, 250]  ##Cor mean: extent = [-0.8, 0.4, -15, 10]
                    gridsize = 160            #设置六边形网格的数量级,数值越大，分箱越细；数值越小，分箱越粗。
                elif data_type == 'MODIS_GPP_8d':
                    extent = [-0.7, 0.2, 100, 250]  ##Cor mean: extent = [-0.8, 0.4, -15, 10]
                    gridsize = 160

            hb = ax1.hexbin(x, y, gridsize=gridsize, extent=extent, visible=False)  # mincnt=1：显示所有包含至少1个data点的六边形；vmin=1：Color mapping的最小值从1开始
            print(f'type: {type}')

            # == 2 密度加权 == #
            def Density_Average_Specific_Gravity(hb):
                counts = hb.get_array()  # 每个六边形的密度（计数）
                offsets = hb.get_offsets()  # 每个六边形的中心坐标 (x, y)

                # 移除零计数的六边形
                valid_mask = counts > 0
                hex_x = offsets[valid_mask, 0]
                hex_y = offsets[valid_mask, 1]
                hex_weights = counts[valid_mask]

                if xlabel == 'SM-VPD coupling':
                    bin_width = 0.005
                    # if veg_type == 'All':
                    # x_bins = np.linspace(-0.5, -0.1, 101) #用于Cor17_5-8
                    # x_bins = np.linspace(-0.5, -0.1, 81)
                    bincount = int(0.8 / bin_width)
                    x_bins = np.linspace(-0.6, 0.2, bincount + 1)

                    # if grade_by != 'Cor mean':
                    #     if lag_day in [0, 3]:
                    #         if type not in ['Grass', 'Arid', 'All']:
                    #             #     x_bins = np.linspace(-0.5, -0.1, 101) #用于Cor17_5-8
                    #             bincount = int(0.45 / bin_width)
                    #             x_bins = np.linspace(-0.45, 0, bincount+1)
                    #         elif type in ['Grass', 'Arid']:
                    #             bincount = int(0.3 / bin_width)
                    #             x_bins = np.linspace(-0.5, -0.2, bincount+1)
                    #             # x_bins = np.linspace(-0.4, 0, 101)
                    #         elif type == 'All':
                    #             bincount = int(0.5 / bin_width)
                    #             x_bins = np.linspace(-0.4, 0.1, bincount+1)
                    #     elif lag_day == 1:
                    #         bincount = int(0.3 / bin_width)
                    #         x_bins = np.linspace(-0.3, 0, bincount+1)
                    # else:
                    #     if type in ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)']:
                    #         bincount = int(0.4 / bin_width)
                    #         x_bins = np.linspace(-0.4, 0, bincount+1)
                    #     elif type == 'Cor(-0.3~-0.2)':
                    #         bincount = int(0.3 / bin_width)
                    #         x_bins = np.linspace(-0.4, -0.1, bincount+1)
                    #     elif type == 'Cor(-0.4~-0.3)':
                    #         bincount = int(0.3 / bin_width)
                    #         x_bins = np.linspace(-0.5, -0.2, bincount+1)
                    #     elif type == 'Cor(<-0.4)':
                    #         bincount = int(0.3 / bin_width)
                    #         x_bins = np.linspace(-0.6, -0.3, bincount+1)

                elif xlabel == 'SM-VPD coupling trend':

                    bin_width = 0.001

                    bincount = int(0.05 / bin_width)
                    x_bins = np.linspace(-0.025, 0.025, bincount + 1)
                    # bincount = int(0.06 / bin_width)
                    # x_bins = np.linspace(-0.03, 0.03, bincount + 1)

                    # if type in ['All']:
                    #     if OutnosigCor == 'No':
                    #         if lag_day in [0, 3]:
                    #             bincount = int(0.04 / bin_width)
                    #             x_bins = np.linspace(-0.02, 0.02, bincount+1)
                    #         elif lag_day == 1:
                    #             bincount = int(0.04 / bin_width)
                    #             x_bins = np.linspace(-0.02, 0.02, bincount+1)
                    #     else:
                    #         bincount = int(0.035 / bin_width)
                    #         x_bins = np.linspace(-0.02, 0.015, bincount+1)
                    # elif type in ['Forest', 'Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)']:
                    #     bincount = int(0.035 / bin_width)
                    #     x_bins = np.linspace(-0.02, 0.015, bincount+1)
                    # elif type  in ['Arid']:
                    #     bincount = int(0.03 / bin_width)
                    #     x_bins = np.linspace(-0.015, 0.015, bincount+1)
                    # elif type in ['Semi-arid']:
                    #     bincount = int(0.035 / bin_width)
                    #     x_bins = np.linspace(-0.02, 0.015, bincount+1)
                    # elif type in ['Dry sub-humid']:
                    #     bincount = int(0.045 / bin_width)
                    #     x_bins = np.linspace(-0.025, 0.02, bincount+1)
                    # elif type in ['Humid']:
                    #     bincount = int(0.045 / bin_width)
                    #     x_bins = np.linspace(-0.025, 0.02, bincount+1)
                    #
                    # elif type in ['Shrub']:
                    #     bincount = int(0.03 / bin_width)
                    #     x_bins = np.linspace(-0.02, 0.01, bincount+1)
                    # elif type in ['Savanna']:
                    #     bincount = int(0.03 / bin_width)
                    #     x_bins = np.linspace(-0.015, 0.015, bincount+1)
                    # elif type  in ['Grass']:
                    #     bincount = int(0.03 / bin_width)
                    #     x_bins = np.linspace(-0.015, 0.015, bincount+1)
                    #
                    # elif type in ['Cor(-0.4~-0.3)', 'Cor(<-0.4)']:
                    #     bincount = int(0.025 / bin_width)
                    #     x_bins = np.linspace(-0.015, 0.01, bincount+1)

                x_centers = (x_bins[:-1] + x_bins[1:]) / 2

                x_weighted = []
                y_weighted = []
                max_densities = []  # 存储每个区间的最大密度
                y_max = []
                y_min = []

                for i in range(len(x_bins) - 1):
                    bin_mask = (hex_x >= x_bins[i]) & (hex_x < x_bins[i + 1])

                    if i == len(x_bins) - 2:
                        bin_mask = (hex_x >= x_bins[i]) & (hex_x <= x_bins[i + 1])
                    else:
                        bin_mask = (hex_x >= x_bins[i]) & (hex_x < x_bins[i + 1])

                    if np.sum(bin_mask) > 0:
                        hex_x_bin = hex_x[bin_mask]
                        hex_y_bin = hex_y[bin_mask]
                        hex_weights_bin = hex_weights[bin_mask]

                        # 使用密度作为权重计算加权平均值
                        hex_x_weighted = np.average(hex_x_bin, weights=hex_weights_bin)
                        hex_y_weighted = np.average(hex_y_bin, weights=hex_weights_bin)
                        hex_weight_max = np.max(hex_weights_bin)

                        hex_y_max = np.nanmax(hex_y_bin)
                        hex_y_min = np.nanmin(hex_y_bin)

                    else:
                        hex_x_weighted = np.nan
                        hex_y_weighted = np.nan
                        hex_weight_max = np.nan
                        hex_y_max = np.nan
                        hex_y_min = np.nan

                    # ## x直接用中心点
                    # x_weighted.append(x_centers[i])

                    ## x也根据加权得到
                    x_weighted.append(hex_x_weighted)  # 【修改7】加权结果是标量，不能再[i]索引
                    y_weighted.append(hex_y_weighted)
                    max_densities.append(hex_weight_max)
                    y_max.append(hex_y_max)
                    y_min.append(hex_y_min)

                x_weighted = np.array(x_weighted)
                y_weighted = np.array(y_weighted)
                max_densities = np.array(max_densities)
                y_max = np.array(y_max)
                y_min = np.array(y_min)

                return x_weighted, y_weighted, max_densities, y_max, y_min

            x_weighted, y_weighted, max_densities, y_max, y_min = Density_Average_Specific_Gravity(hb)

            # 计算Color mapping（基于最大密度）
            vaild_mask = np.isfinite(x_weighted) & np.isfinite(y_weighted)
            x_vaild = x_weighted[vaild_mask]
            y_vaild = y_weighted[vaild_mask]

            # # 【修改8】显示密度加权汇总散点；后续线性/GAM仍等权拟合这些点。
            # ax1.scatter(x_vaild, y_vaild, s=10, color='#222222',
            #             edgecolors='white', linewidths=0.25, zorder=4,
            #             label='Density-weighted points')

            ### ========== 4 拟合 ============  ###
            degree = 3
            n_splines = 15

            # mask_neg = x_vaild < 0
            # mask_pos = x_vaild > 0
            #
            # x_vaild_neg = x_vaild[mask_neg]
            # y_vaild_neg = y_vaild[mask_neg]
            #
            # x_vaild_pos = x_vaild[mask_pos]
            # y_vaild_pos = y_vaild[mask_pos]

            # # 基于原始点的 线性回归
            # x_fit_linear_origin, y_fit_linear_origin, linear_slope_origin, linear_r2_origin, linear_p_origin = linegres_fit(data1_clean, data2_clean)

            # 基于密度加权点的 线性回归
            # x_fit_linear_neg, y_fit_linear_neg, linear_slope_neg, linear_r2_neg, linear_p_neg = linegres_fit(x_vaild_neg, y_vaild_neg)
            # x_fit_linear_pos, y_fit_linear_pos, linear_slope_pos, linear_r2_pos, linear_p_pos = linegres_fit(x_vaild_pos, y_vaild_pos)
            x_fit_linear, y_fit_linear, linear_slope, linear_intercept, linear_r2, linear_p = linegres_fit(x_vaild, y_vaild)
            print(rf'linear_slope={linear_slope}, linear_p={linear_p}')
            # 基于密度加权点的 GAM拟合
            # x_weighted_gam_neg, y_weighted_gam_neg, weighted_gam_r2_neg = \
            #     gam_fit_for_spatial_data_onlyXY(
            #         x_vaild_neg,
            #         y_vaild_neg,
            #         n_splines=n_splines,
            #         weights=None
            #     )
            #
            # x_weighted_gam_pos, y_weighted_gam_pos, weighted_gam_r2_pos = \
            #     gam_fit_for_spatial_data_onlyXY(
            #         x_vaild_pos,
            #         y_vaild_pos,
            #         n_splines=n_splines,
            #         weights=None
            #     )
            x_weighted_gam, y_weighted_gam, weighted_gam_r2 = gam_fit_for_spatial_data_onlyXY(
                                                                x_vaild,
                                                                y_vaild,
                                                                n_splines=n_splines,
                                                                weights=None
                                                            )

            # # 基于KDE的 GAM拟合
            # point_density = kde(xy)
            # # x_mask_1 = x < 0
            # # x_1 = x[x_mask_1]
            # # y_1 = y[x_mask_1]
            # # point_density_1 = point_density[x_mask_1]
            # # x_kde_gam_1, y_kde_gam_1, kde_gam_r2_1 = gam_fit_for_spatial_data_onlyXY(x_1, y_1, n_splines=n_splines, weights=point_density_1)
            # #
            # # x_mask_2 = x >= 0
            # # x_2 = x[x_mask_2]
            # # y_2 = y[x_mask_2]
            # # point_density_2 = point_density[x_mask_2]
            # x_kde_gam, y_kde_gam, kde_gam_r2 = gam_fit_for_spatial_data_onlyXY(x, y, n_splines=n_splines,
            #                                                                        weights=point_density)

            # # 基于原始点的线性回归
            # ax1.plot(x_fit_linear_origin, y_fit_linear_origin, 'purple', linewidth=1,
            #          label=f'Density-Origin(Slope={linear_slope_origin:.2f},R2={linear_r2_origin:.2f}, P={linear_p_origin:.2f})', zorder=2)

            # 基于密度加权点的线性回归
            # ax1.plot(x_fit_linear_neg, y_fit_linear_neg, 'green', linewidth=1,
            #          label=f'Density-Weighted(Slope={linear_slope_neg:.1f},R2={linear_r2_neg:.2f}, P={linear_p_neg:.2f})', zorder=3)
            # ax1.plot(x_fit_linear_pos, y_fit_linear_pos, 'green', linewidth=1,
            #          label=f'Density-Weighted(Slope={linear_slope_pos:.1f},R2={linear_r2_pos:.2f}, P={linear_p_pos:.2f})', zorder=3)
            if linear_p < 0.01:
                ax1.plot(x_fit_linear, y_fit_linear, '#11a579', linewidth=1.5,
                         label=rf'Linear (y = {linear_slope:.1f}x {linear_intercept:+.1f}, $\it{{P}}$ < 0.01)', zorder=3)  #青绿色
            elif linear_p < 0.05:
                ax1.plot(x_fit_linear, y_fit_linear, '#11a579', linewidth=1.5,
                         label=rf'Linear (y = {linear_slope:.1f}x {linear_intercept:+.1f}, $\it{{P}}$ <0.05)',
                         zorder=3)  # 青绿色
            elif linear_p >= 0.05:
                ax1.plot(x_fit_linear, y_fit_linear, '#11a579', linewidth=1.5,
                         label=rf'Linear (y = {linear_slope:.1f}x {linear_intercept:+.1f}, $\it{{P}}$ = {linear_p:.2f})',
                         zorder=3)  # 青绿色
            # weight-GAM的趋势线
            # ax1.plot(x_weighted_gam_neg, y_weighted_gam_neg, 'blue', linewidth=1, label=f'Weighted-GAM(R2={weighted_gam_r2_neg:.2f})', zorder=3)
            # ax1.plot(x_weighted_gam_pos, y_weighted_gam_pos, 'blue', linewidth=1, label=f'Weighted-GAM(R2={weighted_gam_r2_pos:.2f})', zorder=3)
            ax1.plot(x_weighted_gam, y_weighted_gam, '#e73f74', alpha = 0.7, linewidth=1.5,
                     label=f'GAM ($R^{2}={weighted_gam_r2:.2f}$)', zorder=3)#玫red

            # KDE-GAM的趋势线
            # ax1.plot(x_kde_gam, y_kde_gam, 'black', linewidth=1, label=f'KDE-GAM(R2={kde_gam_r2:.2f})', zorder=3)
            # ax1.plot(x_kde_gam, y_kde_gam, '#f2b701', linewidth=1, label=f'KDE-GAM(R2={kde_gam_r2:.2f})', zorder=3)#土黄色


            ### ========== 5 细节设置 ========== ###
            # 设置坐标轴
            if xlabel == 'SM-VPD coupling':
                ax1.set_xlim(-0.7, 0.2)  ### 这行vmax值仅适用于像元POS slope显著时
                ax1.set_xticks(np.arange(-0.6, 0.201, 0.2))  ### 这行vmax值仅适用于像元POS slope显著时
                ax1.set_xticklabels('0' if np.isclose(x, 0)  else
                                    f'{x:.1f}' for x in np.arange(-0.6, 0.201, 0.2))  ### 这行vmax值仅适用于像元POS slope显著时


            elif xlabel == 'SM-VPD coupling trend':
                ticks = np.arange(-0.04, 0.0401, 0.02)

                ax1.set_xlim(-0.04, 0.04)
                ax1.set_xticks(ticks)
                ax1.set_xticklabels(
                    '0' if np.isclose(x, 0) else f'{round(x * 100):.0f}'
                    for x in ticks)
                ## x
                ax1.text(1.05, -0.1, r'$×10^{-2}$',
                          transform=ax1.transAxes,
                          ha='center',
                          va='center')

            if ylabel == 'POS trend':
                ticks = np.arange(-1, 1.01, 0.2)
                ax1.set_ylim(-1, 1)  ### 这行vmax值仅适用于像元POS slope显著时

                ax1.set_yticks(ticks)  ### 这行vmax值仅适用于像元POS slope显著时
                ax1.set_yticklabels('0' if np.isclose(x * 10, 0) else
                                    f'{x:.1f}' for x in ticks)  ### 这行vmax值仅适用于像元POS slope显著时

            elif ylabel == 'POS mean':
                ticks = np.arange(100, 250.01, 25)
                ax1.set_ylim(100, 250)
                ax1.set_yticks(ticks)

                ax1.set_yticks(ticks)  ### 这行vmax值仅适用于像元POS slope显著时
                ax1.set_yticklabels(f'{int(x)}' for x in ticks)  ### 这行vmax值仅适用于像元POS slope显著时

            # if type in ['All', 'Savanna', 'Grass', 'Dry sub-humid', 'Humid']:

            # if type in ['All', 'Forest', 'Savanna', 'Arid', 'Dry sub-humid']:
            if ylabel == 'POS trend':
                ax1.set_ylabel(r'PPT trend (days$\cdot$a$^{-1}$)')
                ax1.set_xlabel(r'SM-VPD coupling trend (days$\cdot$a$^{-1}$)', labelpad=5)
            if ylabel == 'POS mean':
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel('PPT (days)')

            # ax1.set_title(f'({word}) {type}')

            # 添加colorbar
            cbar1 = fig.colorbar(cf, ax=ax1)

            cbar1.set_label('Density')
            if ylabel== 'POS mean':
                # 保留两位小数
                cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


            # 添加Reference line
            ax1.axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
            ax1.axvline(x=0, linestyle='--', color='gray', linewidth=0.8)

            # 添加图例
            ax1.legend(loc='upper right', frameon=False, fancybox=True,
                       framealpha=0.1,     # 图例线长度
                       handletextpad=0.3,  # 线和文字距离
                       labelspacing=0.2,   # 行间距
                       fontsize = 9)


    ### Plot Fig5
    def plot_fig5(bivariate_data1,
                  lower_bound_x1, upper_bound_x1, lower_bound_y1, upper_bound_y1, size_x1, size_y1,
                  bivariate_data2,
                  lower_bound_x2, upper_bound_x2, lower_bound_y2, upper_bound_y2, size_x2, size_y2,
                  colorbar_matrix,  ## Bivariate map
                  data_x_1, data_y_1,
                  data_x_2, data_y_2,
                  grade_by, plot_weighted_point): ## Density scatter plots
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

        ########   主图   #######
        # Create 5 subplots: 4 maps and a colorbar
        fig = plt.figure(figsize=(8.2, 8.2))
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[1, 1.15],  # 列的宽度比
                               height_ratios=[5, 5],  # Reserve the last slot for the colorbar
                               hspace=0.2, wspace=0.3)


        ### Cormean 与 POSmean 的二维空间分布
        ax1 = plt.subplot(gs[0, 0])
        bivariate_map(ax1, bivariate_data1, lower_bound_x1, upper_bound_x1, size_x1,
                      lower_bound_y1, upper_bound_y1, size_y1,
                      colorbar_matrix, 'SM-VPD coupling', 'POS mean')

        ### Cormean 与 POSmean 的二维密度图
        ax2 = plt.subplot(gs[0, 1])
        plot_combined_hexbin_with_WeightedPointTrendline(data_x_1, data_y_1, 'SM-VPD coupling', 'POS mean', grade_by, 'All',plot_weighted_point, ax = ax2)

        # sys.exit()

        ### Corslope 与 POSslope 的二维空间分布
        ax3 = plt.subplot(gs[1, 0])
        bivariate_map(ax3, bivariate_data2, lower_bound_x2, upper_bound_x2, size_x2,
                      lower_bound_y2, upper_bound_y2, size_y2,
                      colorbar_matrix, 'SM-VPD coupling trend', 'POS trend')

        ### Corslope 与 POSslope 的二维密度图
        ax4 = plt.subplot(gs[1, 1])
        plot_combined_hexbin_with_WeightedPointTrendline(data_x_2, data_y_2, 'SM-VPD coupling trend', 'POS trend', grade_by, 'All',plot_weighted_point, ax = ax4)

        # plt.savefig(fr'D:/code/{grade_by}/SM_VPD_Cor(preseason)_BivariateMap_ScatterPlot.png',

        os.makedirs(os.path.join(fig_output, grade_by, Basedon), exist_ok=True)
        plt.savefig(fr'{fig_output}\{grade_by}\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_{Basedon}_Biv_OutnosigCor({OutnosigCor}){mean_or_yearPOS}.png',
                    dpi=300, bbox_inches='tight')
        # plt.close(fig)
        print('Fig5 plot done!')
        # plt.show()

    plot_fig5(biv_map_cormean_posmean,
              mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x, mean_bin_size_y,
              biv_map_corslope_posslope,
              slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x, slope_bin_size_y,
              col_matrix,
              cor_mean, pos_mean,
              cor_slope, pos_slope,
              'All', 'No')
    print('Fig5 plot done!')
    #
    # sys.exit()

    ### Plot S Fig
    def plot_S16_21(biv_map_data,
                                   lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y, size_x, size_y,
                                   col_matrix,
                                   data_x, data_y,
                                   xlabel, ylabel,
                                   grade_by):

        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from mpl_toolkits.basemap import Basemap
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.gridspec as gridspec
        import numpy as np
        import os
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

        if grade_by == 'Veg':
            fig = plt.figure(figsize=(8.2, 14.5))
            gs = gridspec.GridSpec(4, 2,
                                   width_ratios=[1, 1],  # 四列的宽度比
                                   height_ratios=[5, 5, 5, 5],  # Reserve the last slot for the colorbar
                                   hspace=0.3, wspace=0.15)
            # 定义四种Vegetation type
            types = ['Forest', 'Shrub', 'Savanna', 'Grass']#, 'Wet']
            titles = ['Forest', 'Shrub', 'Savanna', 'Grass']# 'Wet']

        elif grade_by == 'AI':
            fig = plt.figure(figsize=(8.2, 14.5))
            gs = gridspec.GridSpec(4, 2,
                                   width_ratios=[1, 1],  # 四列的宽度比
                                   height_ratios=[5, 5, 5, 5],  # Reserve the last slot for the colorbar
                                   hspace=0.3, wspace=0.15)
            types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
            titles = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

        elif grade_by == 'Cor mean':
            fig = plt.figure(figsize=(8.2, 16.5))
            gs = gridspec.GridSpec(5, 2,
                                   width_ratios=[1, 1],  # 四列的宽度比
                                   height_ratios=[5, 5, 5, 5, 5],  # Reserve the last slot for the colorbar
                                   hspace=0.3, wspace=0.15)
            types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                     'Cor(-0.4~-0.3)', 'Cor(<-0.4)']
            titles = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                     'Cor(-0.4~-0.3)', 'Cor(<-0.4)']



        for i, (type, title) in enumerate(zip(types, titles)):

            if grade_by == 'Veg':
                # 创建Vegetation type掩码
                if type == 'All':
                    mask = np.isfinite(veg_type_data)
                elif type == 'Forest':
                    ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                    mask = (veg_type_data == 1)
                elif type == 'Shrub':
                    ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                    mask = (veg_type_data == 2)
                elif type == 'Savanna':
                    ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                    mask = (veg_type_data == 3)
                elif type == 'Grass':
                    ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                    mask = (veg_type_data == 4)
                elif type == 'Wet':
                    mask = (veg_type_data == 5)
            elif grade_by == 'AI':
                # 创建Vegetation type掩码
                if type == 'All':
                    mask = np.isfinite(ai_type_data)
                elif type == 'Hyper Arid':
                    mask = (ai_type_data == 1)  # AI 0-0.03
                elif type == 'Arid':
                    ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                    mask = (ai_type_data == 2)  # AI 0.03-0.2
                elif type == 'Semi-arid':
                    ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                    mask = (ai_type_data == 3) | (ai_type_data == 4)  # AI 0.2-0.35
                elif type == 'Dry sub-humid':
                    ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                    mask = (ai_type_data == 5)  # AI 0.5-0.65
                elif type == 'Humid':
                    ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                    mask = (ai_type_data == 6)  # AI > 0.65

            elif grade_by == 'Cor mean':

                if type == 'All':
                    mask = np.isfinite(veg_type_data)
                elif type == 'Cor(-0.1~0)':
                    ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                    mask = (-0.1 <= cor_type_data) & (cor_type_data < 0)
                elif type == 'Cor(-0.2~-0.1)':
                    ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                    mask = (-0.2 <= cor_type_data) & (cor_type_data < -0.1)
                elif type == 'Cor(-0.3~-0.2)':
                    ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                    mask = (-0.3 <= cor_type_data) & (cor_type_data < -0.2)
                elif type == 'Cor(-0.4~-0.3)':
                    ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                    mask = (-0.4 <= cor_type_data) & (cor_type_data < -0.3)
                elif type == 'Cor(<-0.4)':
                    ax1 = plt.subplot(gs[4, 0])  # 第一个变量空间分布
                    ax2 = plt.subplot(gs[4, 1])  # 第一个变量空间分布
                    mask = (cor_type_data < -0.4) #& (-0.5 <= cor_mean_data)

            # 应用掩码
            plot_varname_data = np.where(mask[:, :, np.newaxis], biv_map_data, np.nan)

            plot_data_x = np.where(mask, data_x, np.nan)
            plot_data_y = np.where(mask, data_y, np.nan)


            ### 二维空间分布
            bivariate_map(ax1, plot_varname_data, lower_bound_x, upper_bound_x, size_x,
                          lower_bound_y, upper_bound_y, size_y,
                          col_matrix, xlabel, ylabel)

            ### 二维密度图
            plot_combined_hexbin_with_WeightedPointTrendline(plot_data_x, plot_data_y, xlabel, ylabel, grade_by, type,
                                                             'No', ax=ax2)

        # plt.savefig(fr'D:/code/{grade_by}/SM_VPD_Cor(preseason)_BivariateMap_{ylabel}_{grade_by}.png', dpi = 600, bbox_inches='tight')
        os.makedirs(os.path.join(fig_output, grade_by), exist_ok=True)
        plt.savefig(fr'{fig_output}\{grade_by}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_{ylabel}_{Basedon}_Biv_OutnosigCor({OutnosigCor}).png',
                    dpi = 300, bbox_inches='tight')
        plt.close(fig)
        print('S Fig plot done!')

        # plt.show()


    # plot_S16_21(biv_map_cormean_posmean,
    #            mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x, mean_bin_size_y,
    #            col_matrix,
    #            cor_mean, pos_mean,
    #            'SM-VPD coupling',
    #            'POS mean',
    #            'Veg')
    # print('S16 plot done!')
    # plot_S16_21(biv_map_corslope_posslope,
    #            slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x, slope_bin_size_y,
    #            col_matrix,
    #            cor_slope, pos_slope,
    #            'SM-VPD coupling trend',
    #            'POS trend',
    #            'Veg')
    # print('S19 plot done!')
    #
    #
    # plot_S16_21(biv_map_cormean_posmean,
    #                            mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x, mean_bin_size_y,
    #                            col_matrix,
    #                            cor_mean, pos_mean,
    #                            'SM-VPD coupling',
    #                            'POS mean',
    #                            'AI')
    # print('S17 plot done!')
    # plot_S16_21(biv_map_corslope_posslope,
    #                            slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x, slope_bin_size_y,
    #                            col_matrix,
    #                            cor_slope, pos_slope,
    #                            'SM-VPD coupling trend',
    #                            'POS trend',
    #                            'AI')
    # print('S20 plot done!')

    # plot_S16_21(biv_map_cormean_posmean,
    #             mean_lower_bound_x, mean_upper_bound_x, mean_lower_bound_y, mean_upper_bound_y, mean_bin_size_x,
    #             mean_bin_size_y,
    #             col_matrix,
    #             cor_mean, pos_mean,
    #             'SM-VPD coupling',
    #             'POS mean',
    #             'Cor mean')
    # print('S18 plot done!')
    # plot_S16_21(biv_map_corslope_posslope,
    #             slope_lower_bound_x, slope_upper_bound_x, slope_lower_bound_y, slope_upper_bound_y, slope_bin_size_x,
    #             slope_bin_size_y,
    #             col_matrix,
    #             cor_slope, pos_slope,
    #             'SM-VPD coupling trend',
    #             'POS trend',
    #             'Cor mean')
    # print('S21 plot done!')