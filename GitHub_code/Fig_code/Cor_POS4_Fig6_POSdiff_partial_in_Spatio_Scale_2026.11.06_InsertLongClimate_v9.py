import os
import glob
import datetime
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
from duckdb.experimental.spark.sql.functions import isnan
from matplotlib.pyplot import subplot
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import pingouin as pg
from scipy.stats import pearsonr, alpha
from joblib import Parallel, delayed, parallel_backend
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
import pymannkendall as mk
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
from sklearn.metrics import r2_score
import gc
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


######### Function #########
### Read SM and VPD bands
def get_SPEIband(tif, stack, spei_length):
    tif_data = gdal.Open(tif)
    tif_array = tif_data.GetRasterBand(spei_length).ReadAsArray().astype(np.float32)
    stack.append(tif_array)


def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)


def save_tif_gdal(output_path, data, rows, cols, crs, transform):
    driver = gdal.GetDriverByName("GTiff")

    # 创建Outputdata集（覆盖模式）
    output_ds = driver.Create(
        output_path,
        cols,  # 宽度（列数）
        rows,  # 高度（行数）
        1,  # 波段数
        gdal.GDT_Float32  # 默认data类型（可根据需求修改）
    )
    if not output_ds:
        raise RuntimeError(f"Cannot create output file: {output_path}")

    output_band = output_ds.GetRasterBand(1)  # 单波段索引为 1
    output_band.WriteArray(data, 0, 0)  # 写入data（0,0 表示左上角起始）

    output_ds.SetProjection(crs)

    output_ds.SetGeoTransform(transform)

    output_ds = None  # 关闭data集（必须！否则文件可能损坏）
    return True


def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):
    # def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix):

    # mask = (~np.isnan(pos_stack[:, i, j]) &
    #         ~np.isnan(cor_matrix[:, i, j]) &
    #         ~np.isnan(sm_matrix[:, i, j]) &
    #         ~np.isnan(vpd_matrix[:, i, j]) &
    #         ~np.isnan(ta_matrix[:, i, j]) &
    #         ~np.isnan(pre_matrix[:, i, j]) &
    #         ~np.isnan(srad_matrix[:, i, j]))\

    all_vars = np.stack([
        pos_stack[:, i, j],
        cor_matrix[:, i, j],
        sm_matrix[:, i, j],
        vpd_matrix[:, i, j],
        ta_matrix[:, i, j],
        pre_matrix[:, i, j],
        srad_matrix[:, i, j]
    ], axis=0)
    # mask: 哪些Time点所有变量都不是 NaN
    mask = np.all(np.isfinite(all_vars), axis=0)  #如果np前加~就是返回了无效值

    if mask.sum() > 3:  #Statistics有效值数量

        pos = np.array(pos_stack[:, i, j][mask])

        cor_time_series = np.array(cor_matrix[:, i, j][mask])
        sm_time_series = np.array(sm_matrix[:, i, j][mask])
        vpd_time_series = np.array(vpd_matrix[:, i, j][mask])
        ta_time_series = np.array(ta_matrix[:, i, j][mask])
        pre_time_series = np.array(pre_matrix[:, i, j][mask])
        srad_time_series = np.array(srad_matrix[:, i, j][mask])

        pixel_data = pd.DataFrame({
            'POS': pos,
            'cor': cor_time_series,  #cor
            'sm': sm_time_series,  #sm
            'vpd': vpd_time_series,  #vpd
            # })
            'ta': ta_time_series,  #ta
            'pre': pre_time_series,  #pre
            'srad': srad_time_series  #srad
        })

        # print('pixel_data:\n', pixel_data)

        # partial_coefficient[i, j] = partial_cor['r'].iloc[0].round(4)
        # partial_coefficient_p[i, j] = partial_cor['p-val'].iloc[0].round(4)

        res_list = []
        for var in vars:
            # Perform partial correlation for each variable with other variables
            res = pg.partial_corr(pixel_data, y=y, x=var, covar=[item for item in vars if item != var])
            res['var'] = var
            # print('res:\n', res)
            res_list.append(res)
        result_df = pd.concat(res_list, axis=0)
        # print('result_df:\n', result_df)

        # 排序并去重：按相关系数绝对值从大到小排序
        # top_vars = result_df.reindex(result_df['r'].abs().sort_values(ascending=False).index)
        top_vars = result_df.sort_values(by='r', key=lambda x: x.abs(), ascending=False)
        # print('top_vars:', top_vars)

        # 获取第一个变量（可用 drop_duplicates 确保不重复）
        top_vars_unique = top_vars.drop_duplicates(subset='var').head(1)

        # 显示前三大变量
        # print("Top 3 strongest partial correlations:")
        # print(top_vars_unique[['var', 'r', 'p-val']])

        varname = top_vars_unique['var'][0]
        # print('top var:', varname)
        r = top_vars_unique['r'][0]
        p = top_vars_unique['p-val'][0]

        sm_pcor = result_df.loc[result_df['var'] == 'sm', 'r'].iloc[0]
        vpd_pcor = result_df.loc[result_df['var'] == 'vpd', 'r'].iloc[0]
        cor_pcor = result_df.loc[result_df['var'] == 'cor', 'r'].iloc[0]
        ta_pcor = result_df.loc[result_df['var'] == 'ta', 'r'].iloc[0]
        pre_pcor = result_df.loc[result_df['var'] == 'pre', 'r'].iloc[0]
        srad_pcor = result_df.loc[result_df['var'] == 'srad', 'r'].iloc[0]


    else:
        varname = np.nan
        r = np.nan
        p = np.nan

        sm_pcor = np.nan
        vpd_pcor = np.nan
        cor_pcor = np.nan
        ta_pcor = np.nan
        pre_pcor = np.nan
        srad_pcor = np.nan

    return (i, j, varname, r, p, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor)


###################################### 1 data读取及Output settings ################################################

print(list(colormaps))

###########################  ==== Input settings ==== #################################
star_year = 2001
end_year = 2024
years_length = end_year - star_year + 1
print('years_length:', years_length)

scale = 11
min_lat = 30
max_lat = 84

cal_done = 'yes'  ###

data_type = 'MCD12Q2'  # GIMMS_NDVI_16d / MODIS_GPP_16d / MODIS_GPP_8d / MODIS_EVI2_8d / MCD12Q2

mean_or_yearPOS = ''  #  '' uses annual POS; '_meanPOS' uses long-term mean POS

coupling_method = 'Pearson'  ### 'Pearson'/'Partial'
lag_day = 0

partial_for = 'SMlag_VPDlag'  ###SM and lagged VPD: SM_VPDlag; SM and VPD: SMlag_VPDlag

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno uses detrended SOS and POS for partial correlation
                                    ### Based_on_OriginPheno uses original SOS and POS for partial correlation   差Pearson0 Partial 1 3

veg_threshold = 25

y_list = [
    'POS drought normal diff']  #, 'POS slope', 'POS std']  'POS drought normal diff' / 'POS wet normal diff' / 'POS corhigh corlow diff'
# y_list = ['POS std']#, 'POS slope', 'POS std']
# y_list = ['POS slope']

months_before_pos = '2'  # 1 / 2 / 3/ Preseason

if months_before_pos == '1':
    cor_test_number = climate_test_number = spei_length = '1'  # 17_8_1
    # climate_test_number = '1'  #10
elif months_before_pos == '2':
    cor_test_number = climate_test_number = spei_length = '2'
    # climate_test_number = '2' #11
    # spei_length = '2'
elif months_before_pos == '3':
    cor_test_number = climate_test_number = spei_length = '3'
    # climate_test_number = '3' #12
    # spei_length = '3'
elif months_before_pos == 'Preseason':
    cor_test_number = 'Preseason'
    climate_test_number = 'Preseason'
    spei_length = '2'

drought_distinguish_way = 3

pheno = 'pos'  ##### 如果是POS就是'pos'，如果是Mat-Sen就是'length of pos'

Outlier = 'No'
SigCorPvalue = 'No'
analyze_by = 'All'  ## 'All'  /  'advance PPT'  /  'delay PPT'

drought_or_wet_times = 2  ### !!!重点修改
spei_strength = -1

normal_threshlod_low = -1
normal_threshlod_high = 1

# 全植被、各Vegetation type、各AI梯度统一使用同一重复RF及偏相关函数。
RUN_GROUPED_VEG_RF = False
RUN_GROUPED_AI_RF =  False

ML_model_in_spatio = 'RF'  ##RF / XGBoost
test_size = 0.2

# ===== REPEATED_GRID_RF: 新流程入口配置 =====
RUN_REPEATED_GRID_RF = False   # True：训练、保存并绘图； False：只读取已保存结果Plot旧Fig6
REPEAT_GRID_FRACTIONS = (0.5)  # 每个比例分别重复50次
REPEAT_RF_N = 50
REPEAT_RF_VALIDATION_FRACTION = 0.2
REPEAT_RF_TRIM_FRACTION = 0.2
REPEAT_RF_RANDOM_STATE = 42
REPEAT_RF_SPLIT_UNIT = 'pixel'  # 原始11km像元8:2；可选'grid'为空间网格划分
# 固定参数，不在50次验证结果上调参。可替换为之前训练集调参得到的最优参数。
REPEAT_RF_PARAMS = dict(n_estimators=500, max_depth=20,
                        min_samples_split=5, min_samples_leaf=1,
                        max_features=0.7, n_jobs=18, bootstrap=True)




########### 12Q2 mask
mcd12q2_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
mcd12q2_mask_tif = gdal.Open(mcd12q2_mask_tif)
mcd12q2_mask = mcd12q2_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
mcd12q2_mask = np.isfinite(mcd12q2_mask)

gpp_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
gpp_mask_tif = gdal.Open(gpp_mask_tif)
gpp_mask = gpp_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
gpp_mask = np.isfinite(gpp_mask)
#
# r2_mcd12q2_ppt_mask_tif = rf'D:/code/r2_with_12q2/gpp_and_12q2_ppt_r2.tif'
# r2_mcd12q2_ppt_mask_tif = gdal.Open(r2_mcd12q2_ppt_mask_tif)
# r2_mcd12q2_ppt_mask = r2_mcd12q2_ppt_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
# r2_mcd12q2_ppt_mask = np.isfinite(r2_mcd12q2_ppt_mask)
# # r2_mcd12q2_ppt_mask = np.where(np.isfinite(r2_mcd12q2_ppt_mask) & (r2_mcd12q2_ppt_mask > 0.2), r2_mcd12q2_ppt_mask,
# #                                np.nan)
#
# r2_mcd12q2_sos_mask_tif = rf'D:/code/r2_with_12q2/gpp_and_12q2_ppt_r2.tif'
# r2_mcd12q2_sos_mask_tif = gdal.Open(r2_mcd12q2_sos_mask_tif)
# r2_mcd12q2_sos_mask = r2_mcd12q2_sos_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
# r2_mcd12q2_sos_mask = np.isfinite(r2_mcd12q2_sos_mask)
# # r2_mcd12q2_sos_mask = np.where(np.isfinite(r2_mcd12q2_sos_mask) & (r2_mcd12q2_sos_mask > 0.2), r2_mcd12q2_sos_mask,
# #                                np.nan)

folder_sm_12q2 = fr'D:/code/SM_preseason_mean{climate_test_number}{mean_or_yearPOS}'  # (POS-30) - POS
folder_sm_gpp = fr'D:/code/SM_preseason_mean{climate_test_number}_meanPOS'  # (POS-30) - POS

folder_vpd_12q2 = fr'D:/code/VPD_preseason_mean{climate_test_number}{mean_or_yearPOS}'  # (POS-30) - POS
folder_vpd_gpp = fr'D:/code/VPD_preseason_mean{climate_test_number}_meanPOS'  # (POS-30) - POS

folder_ta_12q2 = fr'D:/code/Ta_preseason_mean{climate_test_number}{mean_or_yearPOS}'  # (POS-30) - POS
folder_ta_gpp = fr'D:/code/Ta_preseason_mean{climate_test_number}_meanPOS'  # (POS-30) - POS

#### InputClimate的tif path
input_path = rf'D:/code/Global_test2_{scale}000m'

if cor_test_number == 'Preseason':

    # folder_cor_1 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_1'  #(POS-30) - POS
    # folder_cor_2 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_2'  #(POS-60) - POS
    # folder_cor_3 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_3'  #(POS-90) - POS
    #
    # folder_cor_pvalue_1 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_1\Pvalue'  #(POS-30) - POS
    # folder_cor_pvalue_2 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_2\Pvalue'  #(POS-60) - POS
    # folder_cor_pvalue_3 = fr'{input_path}\Correlation(SM_VPD_pearson)17_8_3\Pvalue'  #(POS-90) - POS
    #
    # folder_sm_1 = fr'{input_path}\SM_preseason_mean10'  #(POS-30) - POS
    # folder_sm_2 = fr'{input_path}\SM_preseason_mean11'  #(POS-60) - POS
    # folder_sm_3 = fr'{input_path}\SM_preseason_mean12'  #(POS-90) - POS
    #
    # folder_vpd_1 = fr'{input_path}\VPD_preseason_mean10'  #(POS-30) - POS
    # folder_vpd_2 = fr'{input_path}\VPD_preseason_mean11'  #(POS-60) - POS
    # folder_vpd_3 = fr'{input_path}\VPD_preseason_mean12'  #(POS-90) - POS
    #
    # folder_ta_1 = fr'{input_path}\Ta_preseason_mean10'
    # folder_ta_2 = fr'{input_path}\Ta_preseason_mean11'
    # folder_ta_3 = fr'{input_path}\Ta_preseason_mean12'
    #
    # folder_pre_1 = fr'{input_path}\Pre_preseason_sum10'
    # folder_pre_2 = fr'{input_path}\Pre_preseason_sum11'
    # folder_pre_3 = fr'{input_path}\Pre_preseason_sum12'
    #
    # folder_srad_1 = fr'{input_path}\Srad_preseason_sum10'
    # folder_srad_2 = fr'{input_path}\Srad_preseason_sum11'
    # folder_srad_3 = fr'{input_path}\Srad_preseason_sum12'

    folder_cor = fr'{input_path}\Data\Climate\climate_in_best_preseason\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})\Cor_preseason_mean'  #(POS-30) - POS : Correlation(SM_VPD_pearson)17_8_1

    folder_cor_pvalue = fr'{input_path}\Data\Climate\climate_in_best_preseason\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})\Cor_preseason_mean\Pvalue'  # (POS-30) - POS

    folder_sm = fr'{input_path}\Data\Climate\climate_in_best_preseason\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})\SM_preseason_mean'  #(POS-30) - POS  # (POS-30) - POS

    folder_vpd = fr'{input_path}\Data\Climate\climate_in_best_preseason\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})\VPD_preseason_mean'  #(POS-30) - POS  # (POS-30) - POS

    folder_ta = fr'{input_path}\Data\Climate\climate_in_best_preseason\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})\Ta_preseason_mean'

    folder_pre = fr'{input_path}\Data\Climate\climate_in_best_preseason\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})\Pre_preseason_sum'

    folder_srad = fr'{input_path}\Data\Climate\climate_in_best_preseason\{Basedon}\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})\Srad_preseason_sum'


else:
    if coupling_method == 'Pearson':
        folder_cor = fr'{input_path}\Data\Climate\Cor_Pearson_SM_t_VPD_t{mean_or_yearPOS}\60days'

    elif coupling_method == 'Partial' and partial_for == 'SM_VPDlag':
        folder_cor = fr'{input_path}\Data\Climate\Cor_Pcor_SM_lag{lag_day}_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}\60days'

    elif coupling_method == 'Partial' and partial_for == 'SMlag_VPDlag':
        folder_cor = fr'{input_path}\Data\Climate\Cor_Pcor_SM_t_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}\60days'

    # folder_cor = fr'{input_path}\Data\Climate\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_{cor_test_number}_meanPOS_detrend'  # (POS-30) - POS

    folder_cor_pvalue = fr'{folder_cor}\Pvalue'  # (POS-30) - POS

    # folder_sm = fr'{input_path}\Data\Climate\SM_preseason_mean{climate_test_number}'  # (POS-30) - POS
    #
    # folder_vpd = fr'{input_path}\Data\Climate\VPD_preseason_mean{climate_test_number}'  # (POS-30) - POS
    #
    # folder_ta = fr'{input_path}\Data\Climate\Ta_preseason_mean{climate_test_number}'

    folder_pre = fr'{input_path}\Data\Climate\Pre_preseason_sum{climate_test_number}{mean_or_yearPOS}'

    folder_srad = fr'{input_path}\Data\Climate\Srad_preseason_sum{climate_test_number}{mean_or_yearPOS}'

#### InputPOS的tif path
pos_origin_folder = fr'{input_path}\Data\Pheno\POS_{scale}km'  # start
sos_origin_folder = fr'{input_path}\Data\Pheno\SOS_{scale}km'  # start

pos_detrend_folder = fr'{input_path}\Data\Pheno\POSdetrend_{scale}km'  #start
sos_detrend_folder = fr'{input_path}\Data\Pheno\SOSdetrend_{scale}km'  #start


#### InputSPEI识别的干旱年份的tif path
spei_strength_input_path_gpp = rf'{input_path}\Data\SPEI\NH_SPEI{spei_length}_{spei_length}monthBeforePOS'
drought_path_gpp = fr'{input_path}\Data\SPEI\Drought_Event\drought_event(POS_SPEI{spei_length}_threshold10%_way3)'

spei_strength_input_path_12q2 = rf'{input_path}\Data\SPEI\NH_SPEI{spei_length}_{spei_length}monthBeforePOS'
drought_path_12q2 = fr'{input_path}\Data\SPEI\Drought_Event\drought_event(POS_SPEI{spei_length}_threshold10%_way3)'

#### InputGPPmax的异常-Regular年的tif path
#### 2：为低于10%阈值的年份，1为10-25%阈值的年份，0为25-75%阈值的年份，-1为75-100%阈值的年份
gppmax_abnormal_normal = rf'D:/code/GPPmax_for_abnormal_normal'


##### 长期气候状态
# 日尺度原始气候data目录
sm_tif_file = fr'{input_path}\Data\Climate\LongTerm_meanSOS_meanPOS_2001_2024\SM_meanSOS_meanPOS_MultiYearMean_2001_2024.tif'

vpd_tif_file = fr'{input_path}\Data\Climate\LongTerm_meanSOS_meanPOS_2001_2024\VPD_meanSOS_meanPOS_MultiYearMean_2001_2024.tif'

ta_tif_file = fr'{input_path}\Data\Climate\LongTerm_meanSOS_meanPOS_2001_2024\Ta_meanSOS_meanPOS_MultiYearMean_2001_2024.tif'

pre_tif_file = fr'{input_path}\Data\Climate\LongTerm_meanSOS_meanPOS_2001_2024\Pre_meanSOS_meanPOS_MultiYearMeanOfAnnualSum_2001_2024.tif'

srad_tif_file = fr'{input_path}\Data\Climate\LongTerm_meanSOS_meanPOS_2001_2024\Srad_meanSOS_meanPOS_MultiYearMeanOfAnnualSum_2001_2024.tif'

#### InputAI的tif path
ai_tif_file = fr'{input_path}\Data\Classify\AI\NH30_84_AI(graident)_{scale}km.tif'

#### Input vegetation type TIF path
veg_type_file = rf'{input_path}\Data\Classify\Veg_type\NH_permanent_veg_type_{scale}km_gte{veg_threshold}.tif'

### Input coupling gradient TIF path
cor_mean_file = fr'{input_path}\Tif_of_fig\3Cor_mean_slope\mean\Cor(Pearson_SMlag_VPDlag_lag0)_mean_11km_SOS_All.tif'  #SOS - POS

###########################  ==== Output settings ==== #################################
output_partial_max_var_png_path = fr'{input_path}\Fig\Fig 6 Driver of POS difference in drought event'

output_partial_max_var_r_tif_path = fr'{input_path}\Tif_of_fig\6Partial'


####################################### 2 data读取 #################################################
if months_before_pos == 'Preseason':
    # tif_files_cor_1 = sorted(glob.glob(os.path.join(folder_cor_1, '*.tif')))
    # tif_files_cor_2 = sorted(glob.glob(os.path.join(folder_cor_2, '*.tif')))
    # tif_files_cor_3 = sorted(glob.glob(os.path.join(folder_cor_3, '*.tif')))
    # cor_length_tif = os.path.join(input_path, 'Best_preseason_length\Cor_preseason_length.tif')
    #
    # tif_files_cor_pvalue_1 = sorted(glob.glob(os.path.join(folder_cor_pvalue_1, '*.tif')))
    # tif_files_cor_pvalue_2 = sorted(glob.glob(os.path.join(folder_cor_pvalue_2, '*.tif')))
    # tif_files_cor_pvalue_3 = sorted(glob.glob(os.path.join(folder_cor_pvalue_3, '*.tif')))
    #
    #
    # tif_files_sm_1 = sorted(glob.glob(os.path.join(folder_sm_1, '*.tif')))
    # tif_files_sm_2 = sorted(glob.glob(os.path.join(folder_sm_2, '*.tif')))
    # tif_files_sm_3 = sorted(glob.glob(os.path.join(folder_sm_3, '*.tif')))
    # sm_length_tif = os.path.join(input_path, 'Best_preseason_length\SM_preseason_length.tif')
    #
    #
    # tif_files_vpd_1 = sorted(glob.glob(os.path.join(folder_vpd_1, '*.tif')))
    # tif_files_vpd_2 = sorted(glob.glob(os.path.join(folder_vpd_2, '*.tif')))
    # tif_files_vpd_3 = sorted(glob.glob(os.path.join(folder_vpd_3, '*.tif')))
    # vpd_length_tif = os.path.join(input_path, 'Best_preseason_length\VPD_preseason_length.tif')
    #
    #
    # tif_files_ta_1 = sorted(glob.glob(os.path.join(folder_ta_1, '*.tif')))
    # tif_files_ta_2 = sorted(glob.glob(os.path.join(folder_ta_2, '*.tif')))
    # tif_files_ta_3 = sorted(glob.glob(os.path.join(folder_ta_3, '*.tif')))
    # ta_length_tif = os.path.join(input_path, 'Best_preseason_length\Ta_preseason_length.tif')
    #
    # tif_files_pre_1 = sorted(glob.glob(os.path.join(folder_pre_1, '*.tif')))
    # tif_files_pre_2 = sorted(glob.glob(os.path.join(folder_pre_2, '*.tif')))
    # tif_files_pre_3 = sorted(glob.glob(os.path.join(folder_pre_3, '*.tif')))
    # pre_length_tif = os.path.join(input_path, 'Best_preseason_length\Pre_preseason_length.tif')
    #
    # tif_files_srad_1 = sorted(glob.glob(os.path.join(folder_srad_1, '*.tif')))
    # tif_files_srad_2 = sorted(glob.glob(os.path.join(folder_srad_2, '*.tif')))
    # tif_files_srad_3 = sorted(glob.glob(os.path.join(folder_srad_3, '*.tif')))
    # srad_length_tif = os.path.join(input_path, 'Best_preseason_length\Srad_preseason_length.tif')

    tif_files_cor = sorted(glob.glob(os.path.join(folder_cor, '*.tif')))
    tif_files_cor_pvalue = sorted(glob.glob(os.path.join(folder_cor_pvalue, '*.tif')))
    tif_files_sm = sorted(glob.glob(os.path.join(folder_sm, '*.tif')))
    tif_files_vpd = sorted(glob.glob(os.path.join(folder_vpd, '*.tif')))
    tif_files_ta = sorted(glob.glob(os.path.join(folder_ta, '*.tif')))
    tif_files_pre = sorted(glob.glob(os.path.join(folder_pre, '*.tif')))
    tif_files_srad = sorted(glob.glob(os.path.join(folder_srad, '*.tif')))
else:
    tif_files_cor = sorted(glob.glob(os.path.join(folder_cor, '*.tif')))
    tif_files_cor_pvalue = sorted(glob.glob(os.path.join(folder_cor_pvalue, '*.tif')))

    tif_files_sm_gpp = sorted(glob.glob(os.path.join(folder_sm_gpp, '*.tif')))
    tif_files_sm_12q2 = sorted(glob.glob(os.path.join(folder_sm_12q2, '*.tif')))

    tif_files_vpd_gpp = sorted(glob.glob(os.path.join(folder_vpd_gpp, '*.tif')))
    tif_files_vpd_12q2 = sorted(glob.glob(os.path.join(folder_vpd_12q2, '*.tif')))

    tif_files_ta_gpp = sorted(glob.glob(os.path.join(folder_ta_gpp, '*.tif')))
    tif_files_ta_12q2 = sorted(glob.glob(os.path.join(folder_ta_12q2, '*.tif')))

    tif_files_pre = sorted(glob.glob(os.path.join(folder_pre, '*.tif')))
    tif_files_srad = sorted(glob.glob(os.path.join(folder_srad, '*.tif')))

pos_origin_tif_files = sorted(glob.glob(os.path.join(pos_origin_folder, '*.tif')))
sos_origin_tif_files = sorted(glob.glob(os.path.join(sos_origin_folder, '*.tif')))
pos_detrend_tif_files = sorted(glob.glob(os.path.join(pos_detrend_folder, '*.tif')))
sos_detrend_tif_files = sorted(glob.glob(os.path.join(sos_detrend_folder, '*.tif')))

drought_year_tif_files_gpp = sorted(glob.glob(os.path.join(drought_path_gpp, '*.tif')))
spei_strength_year_tif_files_gpp = sorted(glob.glob(os.path.join(spei_strength_input_path_gpp, '*.tif')))

gppmax_abnormal_normal_tif_files = sorted(glob.glob(os.path.join(gppmax_abnormal_normal, '*.tif')))

drought_year_tif_files_12q2 = sorted(glob.glob(os.path.join(drought_path_12q2, '*.tif')))
spei_strength_year_tif_files_12q2 = sorted(glob.glob(os.path.join(spei_strength_input_path_12q2, '*.tif')))

###################################################
first_tif = tif_files_sm_12q2[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"无法打开 TIF 文件：{sample_tif}（驱动不支持或文件损坏）")

# Get geotransform, projection and pixel size
#Coordinates and projection: the data's spatial reference system
crs = sample_tif.GetProjectionRef()  # Get the input CRS automatically
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


########## SPEI 的gt和行列
# 原始 0.5° SPEI 网格 GeoTransform
# 以下数值来自你之前给出的 SPEI 栅格信息；若当前使用的源 SPEI 文件不同，请替换为它的真实 GeoTransform。
spei_gt = (
    -180.33679328699392,
     0.4940734062657368,
     0.0,
     84.486552471441,
     0.0,
    -0.4940734062657368
)
spei_rows, spei_cols = 171, 730

# 生成每个 11 km 像元中心的经纬度
row_grid, col_grid = np.indices((rows, cols))

lon11 = (
    gt[0]
    + (col_grid + 0.5) * gt[1]
    + (row_grid + 0.5) * gt[2]
)
lat11 = (
    gt[3]
    + (col_grid + 0.5) * gt[4]
    + (row_grid + 0.5) * gt[5]
)

# 将 11 km 像元中心映射到原始 SPEI 网格行列
spei_col = np.floor(
    (lon11 - spei_gt[0]) / spei_gt[1]
).astype(np.int32)

spei_row = np.floor(
    (lat11 - spei_gt[3]) / spei_gt[5]
).astype(np.int32)

# 每个原始 SPEI 网格生成唯一整数 ID
spei_grid_group = spei_row * spei_cols + spei_col

# 排除落在 SPEI 网格范围之外的像元
inside_spei = (
    (spei_row >= 0) & (spei_row < spei_rows) &
    (spei_col >= 0) & (spei_col < spei_cols)
)
spei_grid_group = np.where(inside_spei, spei_grid_group, -1)

############## 长期气候状态 ################
#### SM
sm_long_data = gdal.Open(sm_tif_file)
band = sm_long_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

sm_long_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    sm_long_data[sm_long_data == nodata] = np.nan

#### VPD
vpd_long_data = gdal.Open(vpd_tif_file)
band = vpd_long_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

vpd_long_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    vpd_long_data[vpd_long_data == nodata] = np.nan

#### Ta
ta_long_data = gdal.Open(ta_tif_file)
band = ta_long_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

ta_long_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    ta_long_data[ta_long_data == nodata] = np.nan

#### Pre
pre_long_data = gdal.Open(pre_tif_file)
band = pre_long_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

pre_long_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    pre_long_data[pre_long_data == nodata] = np.nan

#### Pre
srad_long_data = gdal.Open(srad_tif_file)
band = srad_long_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

srad_long_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    srad_long_data[srad_long_data == nodata] = np.nan

############# 3.2 Vegetation type ################
veg_type_data = gdal.Open(veg_type_file)
band = veg_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

veg_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    veg_type_data[veg_type_data == nodata] = np.nan

############### 3.4 AI index #################
ai_type_data = gdal.Open(ai_tif_file)
band = ai_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

ai_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    ai_type_data[ai_type_data == nodata] = np.nan

############### 3.4 Coupling gradient data #################
cor_mean_data = gdal.Open(cor_mean_file)
band = cor_mean_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

cor_mean_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    cor_mean_data[cor_mean_data == nodata] = np.nan


if cal_done == 'no':
    ############################################ 3 Time-Stacking ###################################################

    ## Stack data
    cor_stack = []

    cor_pvalue_stack = []

    sm_stack_gpp = []
    sm_stack_12q2 = []

    vpd_stack_gpp = []
    vpd_stack_12q2 = []

    ta_stack_gpp = []
    ta_stack_12q2 = []

    pre_stack = []

    srad_stack = []

    for tif_file in tif_files_cor:
        get_band(tif_file, cor_stack)
        # get_band_clip(tif_file, cor_stack, row_start, row_end)

    for tif_file in tif_files_cor_pvalue:
        get_band(tif_file, cor_pvalue_stack)
        # get_band_clip(tif_file, cor_pvalue_stack, row_start, row_end)

    for tif_file in tif_files_sm_gpp:
        get_band(tif_file, sm_stack_gpp)
        # get_band_clip(tif_file, sm_stack_gpp, row_start, row_end)

    for tif_file in tif_files_sm_12q2:
        get_band(tif_file, sm_stack_12q2)
        # get_band_clip(tif_file, sm_stack_12q2, row_start, row_end)

    for tif_file in tif_files_vpd_gpp:
        get_band(tif_file, vpd_stack_gpp)
        # get_band_clip(tif_file, vpd_stack_gpp, row_start, row_end)

    for tif_file in tif_files_vpd_12q2:
        get_band(tif_file, vpd_stack_12q2)
        # get_band_clip(tif_file, vpd_stack_12q2, row_start, row_end)

    for tif_file in tif_files_ta_gpp:
        get_band(tif_file, ta_stack_gpp)
        # get_band_clip(tif_file, ta_stack_gpp, row_start, row_end)

    for tif_file in tif_files_ta_12q2:
        get_band(tif_file, ta_stack_12q2)
        # get_band_clip(tif_file, ta_stack_12q2, row_start, row_end)

    for tif_file in tif_files_pre:
        get_band(tif_file, pre_stack)
        # get_band_clip(tif_file, pre_stack, row_start, row_end)

    for tif_file in tif_files_srad:
        get_band(tif_file, srad_stack)
        # get_band_clip(tif_file, srad_stack, row_start, row_end)

    cor_stack = np.stack(cor_stack, axis=0)
    cor_stack = cor_stack[:years_length, :, :]

    cor_pvalue_stack = np.stack(cor_pvalue_stack, axis=0)

    cor_pvalue_stack = cor_pvalue_stack[:years_length, :, :]

    sm_stack_gpp = np.stack(sm_stack_gpp, axis=0)
    sm_stack_12q2 = np.stack(sm_stack_12q2, axis=0)

    vpd_stack_gpp = np.stack(vpd_stack_gpp, axis=0)
    vpd_stack_12q2 = np.stack(vpd_stack_12q2, axis=0)

    ta_stack_gpp = np.stack(ta_stack_gpp, axis=0)
    ta_stack_12q2 = np.stack(ta_stack_12q2, axis=0)

    if data_type == 'MODIS_GPP_8d':
        sm_stack = sm_stack_gpp
        vpd_stack = vpd_stack_gpp
        ta_stack = ta_stack_gpp
    elif data_type == 'MCD12Q2':
        sm_stack = sm_stack_12q2
        vpd_stack = vpd_stack_12q2
        ta_stack = ta_stack_12q2

    pre_stack = np.stack(pre_stack, axis=0)

    srad_stack = np.stack(srad_stack, axis=0)

    pos_origin_stack = []
    sos_origin_stack = []

    pos_detrend_stack = []
    sos_detrend_stack = []

    drought_year_stack_gpp = []
    spei_strength_year_stack_gpp = []

    drought_year_stack_12q2 = []
    spei_strength_year_stack_12q2 = []

    gppmax_abnormal_normal_stack = []

    for tif_file in pos_origin_tif_files:
        get_band(tif_file, pos_origin_stack)

    for tif_file in sos_origin_tif_files:
        # get_band_clip(tif_file, sos_origin_stack, row_start, row_end)
        get_band(tif_file, sos_origin_stack)

    for tif_file in pos_detrend_tif_files:
        # get_band_clip(tif_file, pos_detrend_stack, row_start, row_end)
        get_band(tif_file, pos_detrend_stack)
    for tif_file in sos_detrend_tif_files:
        # get_band_clip(tif_file, sos_detrend_stack, row_start, row_end)
        get_band(tif_file, sos_detrend_stack)

    for tif_file in drought_year_tif_files_gpp:
        # get_band_clip(tif_file, drought_year_stack, row_start, row_end)
        get_band(tif_file, drought_year_stack_gpp)
    for tif_file in spei_strength_year_tif_files_gpp:
        # get_band_clip(tif_file, spei_strength_year_stack, row_start, row_end)
        get_band(tif_file, spei_strength_year_stack_gpp)

    for tif_file in drought_year_tif_files_12q2:
        # get_band_clip(tif_file, drought_year_stack_12q2, row_start, row_end)
        get_band(tif_file, drought_year_stack_12q2)
    for tif_file in spei_strength_year_tif_files_12q2:
        # get_band_clip(tif_file, spei_strength_year_stack_12q2, row_start, row_end)
        get_band(tif_file, spei_strength_year_stack_12q2)

    for tif_file in gppmax_abnormal_normal_tif_files:
        # get_band_clip(tif_file, spei_strength_year_stack_12q2, row_start, row_end)
        get_band(tif_file, gppmax_abnormal_normal_stack)


    print('Stack start')

    pos_origin_stack = np.stack(pos_origin_stack, axis=0)
    pos_origin_stack = pos_origin_stack[:years_length, :, :]
    pos_detrend_stack = np.stack(pos_detrend_stack, axis=0)
    pos_detrend_stack = pos_detrend_stack[:years_length, :, :]

    pos_detrend_stack = np.where((pos_detrend_stack > -50) & (pos_detrend_stack < 50) & np.isfinite(veg_type_data), pos_detrend_stack, np.nan)
    pos_origin_stack = np.where((pos_origin_stack > 90) & (pos_origin_stack < 300) & np.isfinite(veg_type_data), pos_origin_stack, np.nan)

    sos_origin_stack = np.stack(sos_origin_stack, axis=0)
    sos_origin_stack = sos_origin_stack[:years_length, :, :]
    sos_detrend_stack = np.stack(sos_detrend_stack, axis=0)
    sos_detrend_stack = sos_detrend_stack[:years_length, :, :]

    sos_detrend_stack = np.where((sos_detrend_stack > -50) & (sos_detrend_stack < 50), sos_detrend_stack, np.nan)
    sos_origin_stack = np.where((sos_origin_stack > 30), sos_origin_stack, np.nan)

    drought_year_stack_gpp = np.stack(drought_year_stack_gpp, axis=0)

    spei_strength_year_stack_gpp = np.stack(spei_strength_year_stack_gpp, axis=0)

    drought_year_stack_12q2 = np.stack(drought_year_stack_12q2, axis=0)

    spei_strength_year_stack_12q2 = np.stack(spei_strength_year_stack_12q2, axis=0)

    gppmax_abnormal_normal_stack = np.stack(gppmax_abnormal_normal_stack, axis=0)

    if data_type == 'MCD12Q2':
        drought_year_stack = drought_year_stack_12q2
        spei_strength_year_stack = spei_strength_year_stack_12q2
    elif data_type == 'MODIS_GPP_8d':
        drought_year_stack = drought_year_stack_gpp
        spei_strength_year_stack = spei_strength_year_stack_gpp

    print('Stack end!')

    time_length = drought_year_stack.shape[0]

    #########============== 5 逐像元Standardization处理 ===============###########
    ### 5.1 Remove outliers与否的mask

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
        if len(x_flatten) < years_length / 4:
            # Return all NaN if no valid data are available
            return x, i, j
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

            # if (len(np.isfinite(x_masked)) > (years_length / 2)) & (len(np.isfinite(x_masked)) <= years_length):
            return x_masked, i, j  # IQR, lower_range, upper_range,  minv, maxv
            # else:
            #     return np.full_like(x, np.nan), i, j

    if Outlier == 'Yes':
        outlier_pos_stack = np.full((years_length, rows, cols), np.nan)

        outlier_sos_stack = np.full((years_length, rows, cols), np.nan)
        outlier_cor_stack = np.full((years_length, rows, cols), np.nan)
        outlier_sm_stack = np.full((years_length, rows, cols), np.nan)
        outlier_vpd_stack = np.full((years_length, rows, cols), np.nan)
        outlier_ta_stack = np.full((years_length, rows, cols), np.nan)
        outlier_pre_stack = np.full((years_length, rows, cols), np.nan)
        outlier_srad_stack = np.full((years_length, rows, cols), np.nan)

        if Basedon == 'Based_on_detrendPheno':
            pos_input_stack = pos_detrend_stack
            sos_input_stack = sos_detrend_stack
        elif Basedon == 'Based_on_OriginPheno':
            pos_input_stack = pos_origin_stack
            sos_input_stack = sos_origin_stack

        ### IQR去 POS 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                pos_input_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_pos_stack[:, i, j] = data_mask

        ### IQR去 SOS 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                sos_input_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_sos_stack[:, i, j] = data_mask

        ### Remove Cor outliers using IQR
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                cor_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_cor_stack[:, i, j] = data_mask

        ### Remove SM outliers using IQR
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                sm_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_sm_stack[:, i, j] = data_mask

        ### Remove VPD outliers using IQR
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                vpd_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_vpd_stack[:, i, j] = data_mask

        ### IQR去 Ta 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                ta_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_ta_stack[:, i, j] = data_mask

        ### IQR去 Pre 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                pre_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_pre_stack[:, i, j] = data_mask

        ### IQR去 Srad 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                srad_stack[:, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_srad_stack[:, i, j] = data_mask

        ### Remove outliers后的mask
        mask = (np.isfinite(outlier_pos_stack) & np.isfinite(outlier_sos_stack) &
                np.isfinite(outlier_cor_stack) &
                np.isfinite(outlier_sm_stack) & np.isfinite(outlier_vpd_stack) &
                np.isfinite(outlier_ta_stack) & np.isfinite(outlier_pre_stack) & np.isfinite(outlier_srad_stack))


    elif Outlier == 'No':
        if Basedon == 'Based_on_detrendPheno':
            ##不Remove outliers的mask
            mask = (np.isfinite(pos_detrend_stack) &
                    np.isfinite(sos_detrend_stack) &
                    np.isfinite(cor_stack) &
                    np.isfinite(sm_stack) & np.isfinite(vpd_stack) &
                    np.isfinite(ta_stack) & np.isfinite(pre_stack) & np.isfinite(srad_stack))
        elif Basedon == 'Based_on_OriginPheno':
            ##不Remove outliers的mask
            mask = (np.isfinite(pos_origin_stack) &
                    np.isfinite(sos_origin_stack) &
                    np.isfinite(cor_stack) &
                    np.isfinite(sm_stack) & np.isfinite(vpd_stack) &
                    np.isfinite(ta_stack) & np.isfinite(pre_stack) & np.isfinite(srad_stack))

    if SigCorPvalue == 'Yes':
        mask = (cor_pvalue_stack <= 0.1) & np.isfinite(cor_pvalue_stack) & mask

    vaild_year_length = np.sum(mask, axis=0)

    space_mask = vaild_year_length > (years_length / 2)

    ### 4.2 有效的mask+求cor mean
    space_mask_3d = (space_mask & np.isfinite(cor_mean_data))[np.newaxis, :, :]  # (1, rows, cols)
    space_mask_3d = np.repeat(space_mask_3d, years_length, axis=0)  # (years_length, rows, cols)

    space_mask_3d = np.where(np.isfinite(space_mask_3d) & mask, space_mask_3d, np.nan)

    cor_stack_masked = np.where(space_mask_3d, cor_stack, np.nan)

    cor_mean = np.nanmean(cor_stack_masked, axis=0)  # 形状: (rows, cols)

    ### 4.3 Time序列Standardization
    drought_event = np.where(
        (drought_year_stack == 1) &
        # (sm_stack < 0) & #(vpd_stack > 0) & # (ta_stack > 0) &
        # (cor_stack < 0)&
        # (sm_stack_gpp < 0) & (vpd_stack_gpp > 0) & (ta_stack_gpp > 0) &
        # (sm_stack_12q2 < 0) & (vpd_stack_12q2 > 0) & (ta_stack_12q2 > 0) &
        (spei_strength_year_stack <= spei_strength) &
        (drought_year_stack_12q2 != 2) &
        (drought_year_stack_gpp != 2) &
        (gppmax_abnormal_normal_stack >= 0) &
        np.isfinite(space_mask_3d),
        drought_year_stack, np.nan)
        # gppmax_abnormal_normal_stack, np.nan)
    wet_event = np.where(
        (drought_year_stack == 2) &
        # (sm_stack > 0) & #(vpd_stack < 0) &
        # # (sm_stack_gpp > 0) & (vpd_stack_gpp < 0) &
        # # (sm_stack_12q2 > 0) & (vpd_stack_12q2 < 0) &
        (spei_strength_year_stack >= -spei_strength) &
        (drought_year_stack_12q2 != 1) &
        (drought_year_stack_gpp != 1) &
        (gppmax_abnormal_normal_stack <= 0) &
        np.isfinite(space_mask_3d),
        drought_year_stack, np.nan)
        # gppmax_abnormal_normal_stack, np.nan)


    normal_event = np.where(
        (drought_year_stack == 0) &
        (spei_strength_year_stack >= normal_threshlod_low) & (spei_strength_year_stack <= normal_threshlod_high) &
        # (drought_year_stack_12q2 == 0) & (drought_year_stack_gpp == 0) &
        (gppmax_abnormal_normal_stack == 0)&
        np.isfinite(space_mask_3d),
        drought_year_stack, np.nan)
        # gppmax_abnormal_normal_stack, np.nan)

    # # 每个像元发生湿润事件的次数（沿Time轴Statistics）
    # wet_event_pixel_num = np.sum(np.isfinite(wet_event), axis=0)
    #
    # # 至少发生过一次湿润事件的像元数量
    # wet_pixel_count = np.count_nonzero(wet_event_pixel_num > 0)
    #
    # print("至少发生过一次湿润事件的像元数量:", wet_pixel_count)
    # sys.exit()
    ###### !!! 对Regular年份下 POS不Regular的data做剔除 ############
    if Basedon == 'Based_on_detrendPheno':
        pos_stack = pos_detrend_stack
    elif Basedon == 'Based_on_OriginPheno':
        pos_stack = pos_origin_stack

    ### IQR去 POS 异常值
    ppt_normal_event = np.where(np.isfinite(normal_event), pos_stack, np.nan)

    results = Parallel(n_jobs=18, verbose=10)(
        delayed(Outlier_array_IQR)(ppt_normal_event[:, i, j], i, j, 25, 75)
        for i, j in zip(row_indices, col_indices)
    )
    ppt_normal_event_outlier = np.full((years_length, rows, cols), np.nan)
    for data_mask, i, j in results:
        ppt_normal_event_outlier[:, i, j] = data_mask

    ##### 4.1 Statistics发生次数和像元数
    drought_event_count = np.count_nonzero(drought_event == 1, axis=0)
    pixel_count = np.count_nonzero(drought_event_count >= 2)
    print(rf'干旱事件次数大于2的像元个数：{pixel_count}')

    wet_event_count = np.count_nonzero(wet_event == 2, axis=0)
    pixel_count = np.count_nonzero(wet_event_count >= 2)
    print(rf'湿润事件次数大于2的像元个数：{pixel_count}')

    # sys.exit()

    ##### 4.3
    def standardize_data(data_stack, mask_3d):
        """Temporal standardization"""

        def standardize_func(data_pixel, i, j):
            original_nan_mask = np.isnan(data_pixel)

            spatio_mean = np.nanmean(data_pixel)
            spatio_std = np.nanstd(data_pixel)

            standardized_pixel_data = (data_pixel - spatio_mean) / spatio_std

            standardized_pixel_data[original_nan_mask] = np.nan

            return standardized_pixel_data, i, j

        data_stack = np.where(mask_3d, data_stack, np.nan)

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


    if Outlier == 'Yes':
        pos_standardized = standardize_data(outlier_pos_stack, space_mask_3d)
        sos_standardized = standardize_data(outlier_sos_stack, space_mask_3d)
        cor_standardized = standardize_data(outlier_cor_stack, space_mask_3d)
        sm_standardized = standardize_data(outlier_sm_stack, space_mask_3d)
        vpd_standardized = standardize_data(outlier_vpd_stack, space_mask_3d)
        ta_standardized = standardize_data(outlier_ta_stack, space_mask_3d)
        pre_standardized = standardize_data(outlier_pre_stack, space_mask_3d)
        srad_standardized = standardize_data(outlier_srad_stack, space_mask_3d)
    if Outlier == 'No':
        if Basedon == 'Based_on_detrendPheno':
            pos_detrend_stack_1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)), pos_detrend_stack, np.nan)
            sos_detrend_stack_1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)), sos_detrend_stack, np.nan)

            pos_standardized = standardize_data(pos_detrend_stack_1, space_mask_3d)
            sos_standardized = standardize_data(sos_detrend_stack_1, space_mask_3d)
            # pos_standardized = pos_detrend_stack_1
            # sos_standardized = sos_detrend_stack_1
        elif Basedon == 'Based_on_OriginPheno':
            pos_origin_stack_1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)), pos_origin_stack, np.nan)
            sos_origin_stack_1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)), sos_origin_stack, np.nan)

            # pos_standardized = standardize_data(pos_origin_stack_1, space_mask_3d)
            # sos_standardized = standardize_data(sos_origin_stack_1, space_mask_3d)

            pos_standardized = pos_origin_stack_1
            sos_standardized = sos_origin_stack_1

        cor_stack1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)),
                                    cor_stack, np.nan)
        sm_stack1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)),
                                    sm_stack, np.nan)
        vpd_stack1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)),
                                    vpd_stack, np.nan)
        ta_stack1 = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)),
                                    ta_stack, np.nan)
        pre_stack = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)),
                                    pre_stack, np.nan)
        srad_stack = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier)),
                                    srad_stack, np.nan)

        cor_standardized = standardize_data(cor_stack1, space_mask_3d)
        sm_standardized = standardize_data(sm_stack1, space_mask_3d)
        vpd_standardized = standardize_data(vpd_stack1, space_mask_3d)
        ta_standardized = standardize_data(ta_stack1, space_mask_3d)
        pre_standardized = standardize_data(pre_stack, space_mask_3d)
        srad_standardized = standardize_data(srad_stack, space_mask_3d)

    # cor_standardized = cor_stack1
    # sm_standardized = sm_stack1
    # vpd_standardized = vpd_stack1
    # ta_standardized = ta_stack1
    # pre_standardized = pre_stack
    # srad_standardized = srad_stack



    ### 5.3 计算有效像元的mean/slope/std ###
    def drought_wet_normal_year_diff(data_stack, drought_wet_normal_year_stack):
        ######### 先分别筛选出干旱、湿润、Regular年份 ##########
        # if data_type == 'MODIS_GPP_8d':
        drought_year_mask = ((drought_wet_normal_year_stack == 1) & (spei_strength_year_stack <= spei_strength)&
                             # (drought_year_stack_12q2 != 2) & (drought_year_stack_gpp != 2) &
                             (gppmax_abnormal_normal_stack >= 0))
                             # (sm_stack < 0)  )#& (vpd_stack > 0) )#&  (ta_stack > 0))
                             #    (cor_stack < 0))
                             # (sm_stack_gpp < 0)  & (vpd_stack_gpp > 0) & (ta_stack_gpp > 0) &
                             # (sm_stack_12q2 < 0)  & (vpd_stack_12q2 > 0) & (ta_stack_12q2 > 0) )
        wet_year_mask = ((drought_wet_normal_year_stack == 2) & (spei_strength_year_stack >= -spei_strength)&
                         # (drought_year_stack_12q2 != 1) & (drought_year_stack_gpp != 1) &
                        (gppmax_abnormal_normal_stack <= 0))
                         # (sm_stack > 0) )#& (vpd_stack < 0))
                         # (sm_stack_gpp > 0) & (vpd_stack_gpp < 0) &
                         # (sm_stack_12q2 > 0) & (vpd_stack_12q2 < 0) )

        # elif data_type == 'MCD12Q2':
        #     drought_year_mask = ((drought_wet_normal_year_stack == 1) & (spei_strength_year_stack <= spei_strength) & (drought_year_stack_gpp != 2) &
        #                          (sm_stack < 0)  & (vpd_stack > 0) )
        #     wet_year_mask = ((drought_wet_normal_year_stack == 2) & (spei_strength_year_stack >= -spei_strength) & (drought_year_stack_gpp != 1) &
        #                      (sm_stack > 0) )

        normal_year_mask = ((drought_wet_normal_year_stack == 0) &
                            (spei_strength_year_stack >= normal_threshlod_low) & (spei_strength_year_stack <= normal_threshlod_high)&
                            # (drought_year_stack_12q2 == 0) & (drought_year_stack_gpp == 0) &
                            (gppmax_abnormal_normal_stack ==0))


        # valid_dought_pixel = np.count_nonzero(drought_year_mask, axis=0) >= 2
        # valid_wet_pixel = np.count_nonzero(wet_year_mask, axis=0) >= 1

        # valid_dought_pixel = np.nansum(drought_year_mask, axis=0) >= 4
        # valid_wet_pixel = np.nansum(wet_year_mask, axis=0) >= 1

        ######### 计算干旱、湿润、Regular年份data的均值后分别做差 #########
        data_in_drought_year = np.where(drought_year_mask, data_stack, np.nan)
        data_in_wet_year = np.where(wet_year_mask, data_stack, np.nan)
        data_in_normal_year = np.where(normal_year_mask, data_stack, np.nan)

        data_in_drought_mean = np.nanmean(data_in_drought_year, axis=0)
        data_in_wet_mean = np.nanmean(data_in_wet_year, axis=0)
        data_in_normal_mean = np.nanmean(data_in_normal_year, axis=0)

        ### DIFF: drought - normal
        data_diff_in_drought_noraml_year = data_in_drought_mean - data_in_normal_mean
        data_diff_in_wet_noraml_year = data_in_wet_mean - data_in_normal_mean

        return data_diff_in_drought_noraml_year, data_diff_in_wet_noraml_year

    ### POS:
    pos_origin_stack = np.where((np.isfinite(drought_event) | np.isfinite(ppt_normal_event_outlier) | np.isfinite(wet_event)) & np.isfinite(space_mask_3d), pos_origin_stack, np.nan)

    pos_diff_drought_normal_origin, pos_diff_wet_normal_origin = drought_wet_normal_year_diff(pos_origin_stack, drought_year_stack)
    if Basedon == 'Based_on_detrendPheno':
        pos_diff_drought_normal, pos_diff_wet_normal = drought_wet_normal_year_diff(pos_standardized, drought_year_stack)
        sos_diff_drought_normal, sos_diff_wet_normal = drought_wet_normal_year_diff(sos_standardized, drought_year_stack)
    elif Basedon == 'Based_on_OriginPheno':
        pos_diff_drought_normal, pos_diff_wet_normal = drought_wet_normal_year_diff(pos_standardized, drought_year_stack)
        sos_diff_drought_normal, sos_diff_wet_normal = drought_wet_normal_year_diff(sos_standardized, drought_year_stack)

    print(f'pos_diff_wet_normal_origin有效像元数量：{np.count_nonzero(np.isfinite(pos_diff_wet_normal_origin))}')

    ### Cor
    cor_diff_drought_normal, cor_diff_wet_normal = drought_wet_normal_year_diff(cor_standardized, drought_year_stack)

    ### SM
    sm_diff_drought_normal, sm_diff_wet_normal = drought_wet_normal_year_diff(sm_standardized, drought_year_stack)

    ### VPD
    vpd_diff_drought_normal, vpd_diff_wet_normal = drought_wet_normal_year_diff(vpd_standardized, drought_year_stack)

    ### Ta
    ta_diff_drought_normal, ta_diff_wet_normal = drought_wet_normal_year_diff(ta_standardized, drought_year_stack)

    ### Pre
    pre_diff_drought_normal, pre_diff_wet_normal = drought_wet_normal_year_diff(pre_standardized, drought_year_stack)

    ### Srad
    srad_diff_drought_normal, srad_diff_wet_normal = drought_wet_normal_year_diff(srad_standardized, drought_year_stack)

    del cor_standardized, sm_standardized, vpd_standardized, ta_standardized, pre_standardized, srad_standardized

    ## PPT 原始值差异
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_POS_diff_drought_normal_origin.tif")
    save_tif_gdal(output, pos_diff_drought_normal_origin, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_pos_diff_wet_normal_origin.tif")
    save_tif_gdal(output, pos_diff_wet_normal_origin, rows, cols, crs, gt)

    ## PPT Standardization值差异
    output = os.path.join(output_partial_max_var_r_tif_path,rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_pos_diff_drought_normal.tif")
    save_tif_gdal(output, pos_diff_drought_normal, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_sos_diff_drought_normal.tif")
    save_tif_gdal(output, sos_diff_drought_normal, rows, cols, crs, gt)

    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_cor_diff_drought_normal.tif")
    save_tif_gdal(output, cor_diff_drought_normal, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_sm_diff_drought_normal.tif")
    save_tif_gdal(output, sm_diff_drought_normal, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_vpd_diff_drought_normal.tif")
    save_tif_gdal(output, vpd_diff_drought_normal, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_ta_diff_drought_normal.tif")
    save_tif_gdal(output, ta_diff_drought_normal, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_pre_diff_drought_normal.tif")
    save_tif_gdal(output, pre_diff_drought_normal, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_srad_diff_drought_normal.tif")
    save_tif_gdal(output, srad_diff_drought_normal, rows, cols, crs, gt)

    output = os.path.join(output_partial_max_var_r_tif_path,rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_drought_event_count.tif")
    save_tif_gdal(output, drought_event_count, rows, cols, crs, gt)
    output = os.path.join(output_partial_max_var_r_tif_path, rf"SPEI{spei_length}\{Basedon}\Diff\SPEI{spei_strength}_wet_event_count.tif")
    save_tif_gdal(output, wet_event_count, rows, cols, crs, gt)



elif cal_done == 'yes':
    def read_diff_tif(filename):
        path = os.path.join(
            output_partial_max_var_r_tif_path,
            f"SPEI{spei_length}",
            str(Basedon),
            "Diff",
            filename
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到文件：{path}")

        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"无法打开文件：{path}")

        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()

        if nodata is not None:
            data[data == nodata] = np.nan
        data[~np.isfinite(data)] = np.nan

        band = None
        ds = None
        return data


    # PPT 原始值差异
    pos_diff_drought_normal_origin = read_diff_tif(f"SPEI{spei_strength}_POS_diff_drought_normal_origin.tif")
    pos_diff_wet_normal_origin = read_diff_tif(f"SPEI{spei_strength}_pos_diff_wet_normal_origin.tif")

    # Standardization值差异
    pos_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_pos_diff_drought_normal.tif")
    sos_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_sos_diff_drought_normal.tif")
    cor_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_cor_diff_drought_normal.tif")
    sm_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_sm_diff_drought_normal.tif")
    vpd_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_vpd_diff_drought_normal.tif")
    ta_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_ta_diff_drought_normal.tif")
    pre_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_pre_diff_drought_normal.tif")
    srad_diff_drought_normal = read_diff_tif(f"SPEI{spei_strength}_srad_diff_drought_normal.tif")

    drought_event_count = read_diff_tif(f"SPEI{spei_strength}_drought_event_count.tif")
    wet_event_count = read_diff_tif(f"SPEI{spei_strength}_wet_event_count.tif")


########################### 6 Preseason Pearson+Partial ################
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches


## a or b
def plot_cor_mean_or_slope_and_pvalue_forAllvegType(plot_data, colorbarmin, colorbarmax, drought_or_wet, name, ax):
    fig = ax.get_figure()
    if name == 'All':
        height = 0.18
        wspace = 0.05
    else:
        height = 0.25
        wspace = 0.15
    gs_inner = ax.get_subplotspec().subgridspec(2, 2,
                                                width_ratios=[5, 0.8],
                                                height_ratios=[5, 0.3],
                                                hspace=height, wspace=wspace)

    # Hide the parent ax, which is a placeholder
    ax.axis('off')

    plots = []  # Store the plot object for each subplot

    # Create the three inner axes
    ax1 = fig.add_subplot(gs_inner[0, 0])  # map
    ax2 = fig.add_subplot(gs_inner[0, 1])  # Latitude profile
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar

    plot_data_mean = np.nanmean(plot_data)
    valid_mask = np.isfinite(plot_data)
    plot_data_count_sum = np.sum(valid_mask)

    plot_data_gte0 = np.where(plot_data >= 0, plot_data, np.nan)
    plot_data_gte0_mean = np.nanmean(plot_data_gte0)
    plot_data_gte0_sum = np.sum(plot_data >= 0)

    plot_data_lt0 = np.where(plot_data < 0, plot_data, np.nan)
    plot_data_lt0_mean = np.nanmean(plot_data_lt0)
    plot_data_lt0_sum = np.sum(plot_data < 0)

    advance_percent = (plot_data_lt0_sum / plot_data_count_sum) * 100
    delay_percent = (plot_data_gte0_sum / plot_data_count_sum) * 100

    ########### Subplot 1: spatial distribution #################
    ax1.set_box_aspect(1)  #Make the map axis square and fill the grid height
    ax1.axis('off')
    ### Create the map
    m = Basemap(ax=ax1,
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
    color_list = ['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef',
                  '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221']  # PiYG / 8
    cmap = mpl.colors.ListedColormap(color_list)
    bins = np.linspace(colorbarmin, colorbarmax, 9)
    norm = mpl.colors.BoundaryNorm(bins, cmap.N)

    plot = m.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Avoid tearing near the pole

    plots.append(plot)  # Save the plot object

    ax1.set_frame_on(False)

    ### Plotboundaries
    terrence = r'D:/code/NH_terrence/NH_Terrence_30_90'
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
        edgecolor='black',  # color
        linewidth=1,
        clip_on=False,
        zorder=4  # Place on top
    )

    ax1.add_patch(boundary_circle)

    # if name == 'All':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(a)', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(b)', pad=10, fontweight='bold')
    # elif name == 'Forest' or name == 'Arid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(a) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(b)', pad=10, fontweight='bold')
    # elif name == 'Shrub' or name == 'Semi-arid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(d) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(e)', pad=10, fontweight='bold')
    # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(g) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(h)', pad=10, fontweight='bold')
    # elif name == 'Grass' or name == 'Humid':
    #     if drought_or_wet == 'drought':
    #         ax1.set_title(f'(j) {name}', pad=10, fontweight='bold')
    #     elif drought_or_wet == 'wet':
    #         ax1.set_title(f'(k)', pad=10, fontweight='bold')

    # ax1.set_title(f'({word}) {name}', pad=10, fontweight='bold')
    if name == 'All':
        h = 0.24
        v = 0.815
    else:
        h = 0.22
        v = 0.8
    ax1.text(h, v,
             f'Mean = {plot_data_mean:.1f}\n'
             f'Advance = {plot_data_lt0_mean:.1f} ({advance_percent:.1f}%)\n'
             f'Delay = {plot_data_gte0_mean:.1f} ({delay_percent:.1f}%)',
             transform=ax1.transAxes,  # Use relative coordinates for positioning
             multialignment='center',  # Center vertically
             fontsize=6)

    ########### Subplot 2: variation by latitude

    # Use actual latitude values on the y-axis
    lat_centers = lats[:, 0]

    plot_data_lat = np.nanmean(plot_data, axis=1)

    # if drought_or_wet == 'drought':
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    # elif drought_or_wet == 'wet':
    #     ax2.axvline(x=-0.2, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    ax2.tick_params(axis='both', length=2, pad=3)  #,which='major', )

    # if drought_or_wet == 'drought':
    ax2.set_xlim(-10, 10)
    ax2.set_xticks(np.arange(-8, 8.1, 8))
    ax2.set_xticklabels(['-8 ', '0', '8'])  # Set labels manually
    # elif drought_or_wet == 'wet':
    #     ax2.set_xlim(-0.5, 0)
    #     ax2.set_xticks(np.arange(-0.4, 0.01, 0.2))
    #     ax2.set_xticklabels(['-0.4', '-0.2', '0'], rotation=45)  # Set labels manually

    ax2.set_ylim(30, 90)
    ax2.set_yticks(np.arange(30, 91, 10))
    ax2.set_yticklabels(f'{int(x)}°' for x in np.arange(30, 91, 10))

    ########### Subplot 3：Colorbar
    ### Create the colorbar in the final position
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

            cbar.set_ticks(bins)

            cbar.set_label('PPT difference (days)', labelpad=3)
            cbar.set_ticklabels([f'{int(x)}' for x in bins])

    plt.tight_layout()

    # Current lower-left corner of ax1
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            pos3 = ax3.get_position()

    # Reposition ax1
    if name == 'All':
        xpos = 0.06
        ax1.set_position([
            pos1.x0 - xpos,  # Keep the left edge unchanged
            pos2.y0,  # Align the bottom with ax2
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        xpos = 0.02
        ax1.set_position([
            pos1.x0 - xpos,  # 仅左移
            pos1.y0,  # 保持原bottom
            pos1.width,  # 保持原width
            pos1.height  # 保持原height
        ])

    pos1_new = ax1.get_position()
    pos2_new = ax2.get_position()
    # print(f'ax1 height = {pos1_new.height}\n'
    #       f'ax1 width = {pos1_new.width}\n'
    #       f'ax2 height = {pos2_new.height}\n')
    if drought_or_wet == 'drought':
        if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
            pos1_new = ax1.get_position()
            ax3.set_position([
                pos1_new.x0 + 0.11,
                pos3.y0 - 0.01,
                pos2.x1 - pos1_new.x0,
                pos3.height])

    # plt.tight_layout()

    # plt.show()


## c
def plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(drought_data, wet_data, colorbarmin, colorbarmax,
                                                                       name, ax):
    bins = np.arange(colorbarmin, colorbarmax + 1, 1)
    count_drought_data, _ = np.histogram(drought_data, bins=bins)
    count_wet_data, _ = np.histogram(wet_data, bins=bins)

    ### plot
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[5, 0.3],
        hspace=0.15
    )

    ax1 = fig.add_subplot(gs_inner[0])

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
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Set bar positions
    bin_centers = (bins[:-1] + bins[1:]) / 2
    print(f'bin_centers:{bin_centers}')

    total_width = 1  # Total bar width per tick
    n = 2  # Number of categories
    width = total_width / n  # Width of each bar

    ax1.bar(bin_centers - width / 2, count_drought_data, width=width, color='#c51b7d', alpha=0.5,
            label='drought - normal')
    ax1.bar(bin_centers + width / 2, count_wet_data, width=width, color='#4d9221', alpha=0.5, label='wet - normal')

    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

    ax1.tick_params(axis='both', length=2, pad=3)  #,which='major', )

    ticks = np.arange(colorbarmin, colorbarmax + 0.0001, 4)
    ax1.set_xlim(colorbarmin, colorbarmax)
    ax1.set_xticks(ticks)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_xlabel('PPT difference (days)', labelpad=5)  #Control spacing between labels and ticks

    if name == 'All':
        if data_type == 'MODIS_GPP_8d':
            # if spei_strength == -1.5:
            #     ax1.set_ylim(0,10000)
            #     ax1.set_yticks(np.arange(0, 10000.1, 2000))
            #     ax1.set_yticklabels(f'{int(x * 0.001)}' for x in np.arange(0, 12000, 2000))
            # elif spei_strength == -1:
            ax1.set_ylim(0, 15000)
            ax1.set_yticks(np.arange(0, 18000, 3000))
            ax1.set_yticklabels(f'{int(x * 0.001)}' for x in np.arange(0, 18000, 3000))
        elif data_type == 'MCD12Q2':
            ax1.set_ylim(0, 25000)
            ax1.set_yticks(np.arange(0, 30000, 5000))
            ax1.set_yticklabels(f'{int(x * 0.001)}' for x in np.arange(0, 30000, 5000))

    # elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
    else:
        ax1.set_ylim(0, 6000)
        ax1.set_yticks(np.arange(0, 6000.1, 1000))
        ax1.set_yticklabels(f'{int(x * 0.001)}' for x in np.arange(0, 7000, 1000))
    # else:
    #     ax1.set_ylim(0, 700)
    #     ax1.set_yticks(np.arange(0, 700.1, 200))
    #     ax1.set_yticklabels(f'{int(x * 0.01)}' for x in np.arange(0, 700.1, 200))

    if name == 'All':
        x_pos = -0.15
        y_pos = 1.15
    else:
        x_pos = -0.15
        y_pos = 1.12

    ax1.text(
        x_pos,  # x = Tick position in data coordinates
        y_pos,  # y = Slightly lower in axes coordinates
        r'$×10^{3}$',  # Text to display
        transform=ax1.transAxes,
        ha='left',  # Align rightward to avoid compression
        va='top',
        rotation=0,
        clip_on=False
    )

    # elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
    #     ax.set_ylim(0, 2500)
    #     ax.set_yticks(np.arange(0, 2500.1, 500))

    # Colorbar
    if name == 'All':
        c = 0.5
    else:
        c = 0.45
    if name in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
        ax1.legend(
            loc='lower center',
            bbox_to_anchor=(c, -0.45),
            ncol=2,
            columnspacing=0.4,
            frameon=False,  # Control legend frame visibility
            handlelength=0.8,
            handleheight=0.8,
            fontsize=8
        )

    ax1.set_ylabel('Frequency', labelpad=3)  # Control spacing between labels and ticks

    # if name == 'Forest' or name == 'Arid' or name == 'All':
    #     word = 'c'
    # elif name == 'Shrub' or name == 'Semi-arid':
    #     word = 'f'
    # elif name == 'Savanna' or name == 'Dry sub-humid':
    #     word = 'i'
    # elif name == 'Grass' or name == 'Humid':
    #     word = 'l'
    # ax.set_title(f'({word})', pad=10, fontweight='bold')

    # elif data_type == 'cor mean':
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #
    #     total_width = 0.2  # Total bar width per tick
    #     n = 2  # Number of categories
    #     width = total_width / n  # Width of each bar
    #
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #
    #     color_list = ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272',
    #                   '#fcbba1', '#fee5d9', '#9ecae1']
    #
    #     # Plot bars individually to preserve their colors
    #     for j in range(len(count_mean)):
    #         ax.bar(bin_centers[j], count_mean[j], width=width,
    #                 color=color_list[j], edgecolor='none')
    #
    #     ax.set_xlim(colorbarmin, colorbarmax)
    #     ax.set_xticks(np.arange(colorbarmin, colorbarmax, 0.2))
    #
    #     ax.set_xlabel('VPD-SM coupling', labelpad=3) #Control spacing between labels and ticks
    #     if  name in ['Forest', 'Shrub', 'Savanna', 'Grass']:
    #         ax.set_ylim(0, 2500)
    #         ax.set_yticks(np.arange(0, 2500.1, 500))
    #     elif name in ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']:
    #         ax.set_ylim(0, 4000)
    #         ax.set_yticks(np.arange(0, 4000.1, 1000))
    #
    #     # ax.axs[0].set_yticks([5000, 6000, 7500])
    #     # ax.axs[1].set_yticks([0, 1000, 2000, 3500])
    #
    #     ax.set_ylabel('Frequency', labelpad=3)  # Control spacing between labels and ticks

    # plt.show()


# d & e & S26-28: Driving factors of Drought/Wet and normal diff
# -*- coding: utf-8 -*-
# 替换辅助函数和4个分析函数；须预先读取pre_long_data和srad_long_data。
def _read_extra_long_climate_raster(path):
    """读取已计算好的长期状态，保留单位和数值，不额外除以年数。"""
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(f'无法打开长期气候栅格：{path}')
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        if np.isnan(nodata):
            values[~np.isfinite(values)] = np.nan
        else:
            values[values == nodata] = np.nan
    values[~np.isfinite(values)] = np.nan
    band = None
    dataset = None
    return values


LONG_CLIMATE_FEATURES = ['SM long-term mean', 'VPD long-term mean', 'Ta long-term mean', 'Pre long-term state', 'Srad long-term state']


def _resolve_long_climate(reference, sm_long_data=None, vpd_long_data=None, ta_long_data=None, pre_long_data=None, srad_long_data=None):
    """保持原单位；只核对数组形状，不做重采样、不做干旱/Regular相减。"""
    result = {}
    for label, global_name, supplied in zip(
            LONG_CLIMATE_FEATURES,
            ['sm_long_data', 'vpd_long_data', 'ta_long_data', 'pre_long_data', 'srad_long_data'],
            [sm_long_data, vpd_long_data, ta_long_data, pre_long_data, srad_long_data]):
        value = globals().get(global_name) if supplied is None else supplied
        if value is None:
            raise ValueError(f'缺少 {global_name}；请先读取长期气候栅格或通过同名关键字传入。')
        value = np.asarray(value, dtype=float)
        if value.shape != np.asarray(reference).shape:
            raise ValueError(f'{global_name} 的形状 {value.shape} 与差值栅格 '
                             f'{np.asarray(reference).shape} 不一致，请先检查栅格对齐。')
        result[label] = value
    return result


def _pcorr_with_long_climate(df, y, x_list, background_cols):
    """普通 Pearson 偏相关：只检验差值变量，额外控制长期背景。

    所有检验使用同一组完整案例。临时缩放仅用于数值稳定，不更改 RF 数据。
    p 值仍是普通独立样本假设下的 p 值，并未作空间相关性校正。
    """
    import warnings
    columns = list(dict.fromkeys([y, *x_list, *background_cols]))
    data = df.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    constant = [c for c in columns if data[c].nunique() <= 1]
    invalid = [c for c in [y, *x_list] if c in constant]
    if invalid:
        raise ValueError(f'偏相关目标或待检验差值变量没有有效变异：{invalid}')
    active_background = [c for c in background_cols if c not in constant]
    if any(c in constant for c in background_cols):
        warnings.warn(f'以下长期控制变量在当前分组内为常数，已移除：{constant}')
    active_cols = [y, *x_list, *active_background]
    data = data[active_cols]
    data = (data - data.mean()) / data.std(ddof=0)
    records = []
    for variable in x_list:
        controls = [c for c in x_list if c != variable] + active_background
        if len(data) <= len(controls) + 2:
            raise ValueError(f'偏相关完整样本数 {len(data)} 不足，控制变量数={len(controls)}。')
        design = np.column_stack([np.ones(len(data)), data[controls].to_numpy()])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError('偏相关控制变量完全共线，请检查长期气候与差值变量。')
        # 如果目标/待检验变量完全由控制变量决定，偏相关没有定义。
        pair = data[[y, variable]].to_numpy()
        residual = pair - design @ np.linalg.lstsq(design, pair, rcond=None)[0]
        if np.any(np.linalg.norm(residual, axis=0) < 1e-10 * np.sqrt(len(data))):
            raise ValueError(f'{variable} 或 {y} 被控制变量完全解释，偏相关没有定义。')
        result = pg.partial_corr(data=data, y=y, x=variable, covar=controls, method='pearson')
        result['var'] = variable
        result['n_pcorr'] = len(data)
        result['covariates'] = '; '.join(controls)
        records.append(result)
    return pd.concat(records, axis=0)


VEGETATION_CODES = {1: 'Forest', 2: 'Shrub', 3: 'Savanna', 4: 'Grass'}


AI_CATEGORY_CODES = (1, 2, 3, 4, 5, 6)


ML_CATEGORY_GROUPS = {
    'Vegetation type': [f'Vegetation: {name}' for name in VEGETATION_CODES.values()],
    'AI type': [f'AI category: {code}' for code in AI_CATEGORY_CODES],
}


def _resolve_categorical_background(reference, vegetation, ai_classes):
    """保留原分类；缺失/非预设编码变NaN，不把缺失值编码为全部0。"""
    import warnings
    result = {}
    settings = [
        ('veg_type_data', vegetation, tuple(VEGETATION_CODES), ML_CATEGORY_GROUPS['Vegetation type']),
        ('ai_type_data', ai_classes, AI_CATEGORY_CODES, ML_CATEGORY_GROUPS['AI type']),
    ]
    for name, array, codes, labels in settings:
        if array is None:
            raise ValueError(f'缺少{name}，请读取分类栅格或显式传入。')
        array = np.asarray(array, dtype=float)
        if array.shape != np.asarray(reference).shape:
            raise ValueError(f'{name}形状{array.shape}与差值栅格{np.asarray(reference).shape}不一致。')
        valid = np.isfinite(array) & np.isin(array, codes)
        unknown = np.isfinite(array) & ~np.isin(array, codes)
        if unknown.any():
            warnings.warn(f'{name}有{int(unknown.sum())}个非预设分类像元，将从机器学习中排除。'
                          f'非预设编码示例：{np.unique(array[unknown])[:10]}')
        for code, label in zip(codes, labels):
            result[label] = np.where(valid, (array == code).astype(float), np.nan)
    return result


def _group_categorical_shap(shap_values_df, importance):
    """以编码列训练，以两个类别组展示。

    树重要性：组内各列求和。
    SHAP：先在每个样本内对组内有符号贡献求和，再求mean(abs(sum))。
    保留底层编码值及SHAP列，以便核查；展示列用NaN使类别组蜂群点显示灰色。
    """
    data = shap_values_df.copy()
    grouped_rows = []
    members = {c for cols in ML_CATEGORY_GROUPS.values() for c in cols}
    for _, row in importance.iterrows():
        if row['feature'] not in members:
            grouped_rows.append(row.to_dict())
    for group, columns in ML_CATEGORY_GROUPS.items():
        present = [c for c in columns if c in importance['feature'].values]
        if not present:
            continue
        contributions = data[[f'shap_{c}' for c in present]].sum(axis=1)
        data[f'shap_{group}'] = contributions
        # 类别编码没有连续大小，不能映射成红=高、蓝=低。
        data[group] = np.nan
        rows = importance.loc[importance['feature'].isin(present)]
        row = rows.iloc[0].to_dict()
        row.update(feature=group, importance=float(rows['importance'].sum()),
                   shapvalue=float(contributions.abs().mean()))
        grouped_rows.append(row)
    return data, pd.DataFrame(grouped_rows).reset_index(drop=True)


def _export_encoded_background(detail, grade_by, group_name, target, variant):
    directory = os.path.join(output_partial_max_var_r_tif_path,
                             f'SPEI{spei_length}', str(Basedon))
    os.makedirs(directory, exist_ok=True)
    stem = (f'{variant}_CatBackground_5Climate_{grade_by}_{group_name}_{target}'
            f'_SPEI{spei_strength}_{coupling_method}_lag{lag_day}_{months_before_pos}'
            f'_{ML_model_in_spatio}_TS{test_size}_EncodedImportance.csv')
    detail.to_csv(os.path.join(directory, stem), index=False, encoding='utf-8-sig')


def _plot_long_climate_pcorr(ax, df_pcorr, feature_order, control_long):
    """用空行保持与14因素 ML/SHAP 的行对应；空行不代表偏相关系数为零。"""
    table = df_pcorr.set_index('feature').reindex(feature_order)
    positions = np.arange(len(feature_order))
    valid = table['r_pcorr'].notna().to_numpy()
    colors = np.where(table['r_pcorr'].to_numpy() < 0, '#4393c3', '#d6604d')
    ax.barh(positions[valid], table['Abs_Coefficient'].to_numpy()[valid],
            color=colors[valid], height=0.4, alpha=0.8)
    labels = []
    short = {'SOS': 'SOS', 'Cor': 'Coupling', 'SM': 'SM', 'VPD': 'VPD',
             'Ta': 'Ta', 'Pre': 'Pre', 'Srad': 'Srad'}
    for i, feature in enumerate(feature_order):
        label = (feature.replace(' long-term mean', ' (clim)').replace(' long-term state', ' (clim)')
                 if feature in LONG_CLIMATE_FEATURES else short.get(feature.split()[0], feature))
        labels.append(label)
        if valid[i]:
            sig = table.iloc[i]['Significance']
            if isinstance(sig, str) and sig:
                ax.text(float(table.iloc[i]['Abs_Coefficient']) + 0.015, i, sig, va='center')
        elif feature in LONG_CLIMATE_FEATURES:
            ax.text(0.02, i, 'control only' if control_long else 'RF only',
                    transform=ax.get_yaxis_transform(), va='center', color='0.45', fontsize=7)
        elif feature in ML_CATEGORY_GROUPS:
            ax.text(0.02, i, 'ML only', transform=ax.get_yaxis_transform(),
                    va='center', color='0.45', fontsize=7)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(feature_order) - 0.5, -0.5)
    ax.tick_params(axis='both', length=2, pad=3)


def _export_long_climate_results(pcorr, importance, grade_by, group_name, target, variant):
    """额外导出可追溯结果；不覆盖原来的7变量分析结果。"""
    directory = os.path.join(output_partial_max_var_r_tif_path,
                             f'SPEI{spei_length}')
    os.makedirs(directory, exist_ok=True)
    stem = f'_SPEI{spei_strength}_{grade_by}({group_name})'
    pcorr.to_csv(os.path.join(directory, stem + '_Partial.csv'), index=False, encoding='utf-8-sig')
    importance.to_csv(os.path.join(directory, stem + '_Importance_SHAP.csv'),
                      index=False, encoding='utf-8-sig')


def Partial_ML_calculate_and_plot0(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data,
                                  grade_by, veg_type, ax, *, sm_long_data=None, vpd_long_data=None, ta_long_data=None, vegetation_data=None, ai_class_data=None, pre_long_data=None, srad_long_data=None):
    # LONG_CLIMATE: 默认读取同名全局数组，也支持显式关键字传入。
    long_arrays = _resolve_long_climate(pos_data, sm_long_data, vpd_long_data, ta_long_data, pre_long_data, srad_long_data)
    long_cols = list(long_arrays)

    # CATEGORICAL_BACKGROUND: 两个栅格均为分类，固定独热编码后拼入ML表。
    veg_type_data = globals().get('veg_type_data') if vegetation_data is None else vegetation_data
    ai_type_data = globals().get('ai_type_data') if ai_class_data is None else ai_class_data
    category_arrays = _resolve_categorical_background(pos_data, veg_type_data, ai_type_data)
    category_cols = list(category_arrays)

    os.makedirs(os.path.join(output_partial_max_var_r_tif_path, f'SPEI{spei_length}', str(Basedon)), exist_ok=True)
    for folder in ['All', 'Veg', 'AI', 'Cor mean']:
        os.makedirs(os.path.join(output_partial_max_var_png_path, folder), exist_ok=True)
    var_list = ['POS mean', 'POS slope', 'POS std', 'POS drought normal diff', 'POS wet normal diff',
                'POS corhigh corlow diff',

                'SOS mean', 'SOS slope', 'SOS std', 'SOS drought normal diff', 'SOS wet normal diff',
                'SOS corhigh corlow diff',

                'Cor mean', 'Cor slope', 'Cor std', 'Cor drought normal diff', 'Cor wet normal diff',
                'Cor corhigh corlow diff',

                'SM mean', 'SM slope', 'SM std', 'SM drought normal diff', 'SM wet normal diff',
                'SM corhigh corlow diff',

                'VPD mean', 'VPD slope', 'VPD std', 'VPD drought normal diff', 'VPD wet normal diff',
                'VPD corhigh corlow diff',

                'Ta mean', 'Ta slope', 'Ta std', 'Ta drought normal diff', 'Ta wet normal diff',
                'Ta corhigh corlow diff',

                'Pre mean', 'Pre slope', 'Pre std', 'Pre drought normal diff', 'Pre wet normal diff',
                'Pre corhigh corlow diff',

                'Srad mean', 'Srad slope', 'Srad std', 'Srad drought normal diff', 'Srad wet normal diff',
                'Srad corhigh corlow diff']  ##修改此处顺序可以修改子图一的顺序

    if grade_by != 'All':
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

    ### 划分Vegetation type/AI梯度
    var_mask = (np.isfinite(pos_data) & np.isfinite(sos_data) &
                np.isfinite(cor_data) & np.isfinite(sm_data) & np.isfinite(vpd_data) &
                np.isfinite(ta_data) & np.isfinite(pre_data) & np.isfinite(srad_data)
                )

    if grade_by == 'All':
        types = ['All']
        codes = [1]

        fig = ax.figure
        gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                    width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                    height_ratios=[1],  # Reserve the last slot for the colorbar
                                                    hspace=0.3, wspace=0.2)

        ax.axis('off')

    elif grade_by == 'Veg':
        if veg_type == 'All':
            types = ['Forest', 'Shrub', 'Savanna', 'Grass']
            codes = [1, 2, 3, 4]

            fig = plt.figure(figsize=(8.2, 14))
            gs = gridspec.GridSpec(4, 4,
                                   width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                   height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                                   hspace=0.3, wspace=0.2)
        elif veg_type == 'Grass':
            types = ['Grass']
            codes = [4]

            fig = ax.figure
            gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                        width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                        height_ratios=[1],  # Reserve the last slot for the colorbar
                                                        hspace=0.3, wspace=0.2)
            ax.axis('off')

    elif grade_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig = plt.figure(figsize=(8.2, 14))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    elif grade_by == 'Cor mean':
        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']
        codes = [0, 1, 2, 3, 4]

        fig = plt.figure(figsize=(8.2, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    for i, (type, code) in enumerate(zip(types, codes)):
        if grade_by == 'All':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        elif grade_by == 'Veg' and veg_type == 'Grass':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        else:
            ax1 = plt.subplot(gs[i, 0])
            ax2 = plt.subplot(gs[i, 1])
            ax3 = plt.subplot(gs[i, 2])
            ax4 = plt.subplot(gs[i, 3])

        if grade_by == 'Veg':
            mask = (veg_type_data == code) & var_mask

        elif grade_by == 'AI':
            if code == 3:
                mask = ((ai_type_data == code) | (ai_type_data == 4)) & var_mask

            else:
                mask = (ai_type_data == code) & var_mask


        elif grade_by == 'Cor mean':
            if code == 4:
                mask = (
                        (cor_mean_data <= -0.4) & var_mask)
            else:
                mask = (
                        (cor_mean_data <= -(code * 0.1)) & (cor_mean_data > -((code + 1) * 0.1)) & var_mask)

        elif grade_by == 'All':
            mask = var_mask

        valid_indices = np.where(mask.flatten())[0]

        ########## 空间尺度
        ## StandardizationPrepare data：drought/wet/normal year diff ###
        # standardized_data_drought = pd.DataFrame({
        #     'POS drought normal diff': pos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SOS drought normal diff': sos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Cor drought normal diff': cor_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SM drought normal diff': sm_diff_drought_normal.flatten()[valid_indices],
        #
        #     'VPD drought normal diff': vpd_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Ta drought normal diff': ta_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Pre drought normal diff': pre_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Srad drought normal diff': srad_diff_drought_normal.flatten()[valid_indices]
        # })
        standardized_data_drought = pd.DataFrame({
            'POS drought normal diff': pos_data.flatten()[valid_indices],

            'SOS drought normal diff': sos_data.flatten()[valid_indices],

            'Cor drought normal diff': cor_data.flatten()[valid_indices],

            'SM drought normal diff': sm_data.flatten()[valid_indices],

            'VPD drought normal diff': vpd_data.flatten()[valid_indices],

            'Ta drought normal diff': ta_data.flatten()[valid_indices],

            'Pre drought normal diff': pre_data.flatten()[valid_indices],

            'Srad drought normal diff': srad_data.flatten()[valid_indices]
        })

        standardized_data_wet = pd.DataFrame({
            'POS wet normal diff': pos_diff_wet_normal.flatten()[valid_indices],

            'SOS wet normal diff': sos_diff_wet_normal.flatten()[valid_indices],

            'Cor wet normal diff': cor_diff_wet_normal.flatten()[valid_indices],

            'SM wet normal diff': sm_diff_wet_normal.flatten()[valid_indices],

            'VPD wet normal diff': vpd_diff_wet_normal.flatten()[valid_indices],

            'Ta wet normal diff': ta_diff_wet_normal.flatten()[valid_indices],

            'Pre wet normal diff': pre_diff_wet_normal.flatten()[valid_indices],

            'Srad wet normal diff': srad_diff_wet_normal.flatten()[valid_indices],
        })

        # standardized_data_corhigh_corlow= pd.DataFrame({
        #     'POS corhigh corlow diff': pos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SOS corhigh corlow diff': sos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Cor corhigh corlow diff': cor_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SM corhigh corlow diff': sm_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'VPD corhigh corlow diff': vpd_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Ta corhigh corlow diff': ta_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Pre corhigh corlow diff': pre_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Srad corhigh corlow diff': srad_diff_corhigh_corlow.flatten()[valid_indices]
        # })

        # print(f'standardized_data drought的有效data量：{len(standardized_data_drought.dropna())}')
        # print(f'standardized_data wet的有效data量：{len(standardized_data_wet.dropna())}')

        # LONG_CLIMATE: RF另建完整案例表，原偏相关表不加长期变量。
        rf_data_drought = standardized_data_drought.copy()
        rf_data_wet = standardized_data_wet.copy()
        for column, values in long_arrays.items():
            rf_data_drought[column] = values.ravel()[valid_indices]
            rf_data_wet[column] = values.ravel()[valid_indices]
        # CATEGORICAL_BACKGROUND: 类别缺失只影响ML表，不改变偏相关样本。
        for column, values in category_arrays.items():
            rf_data_drought[column] = values.ravel()[valid_indices]
            rf_data_wet[column] = values.ravel()[valid_indices]
        rf_data_drought = rf_data_drought.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        rf_data_wet = rf_data_wet.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        standardized_data_drought = standardized_data_drought.dropna().reset_index()
        standardized_data_wet = standardized_data_wet.dropna().reset_index()



        # standardized_data_corhigh_corlow = standardized_data_corhigh_corlow.dropna().reset_index()

        # print('standardized_data_drought:\n', standardized_data_drought.iloc[:20, :])
        # print('standardized_data_wet:\n', standardized_data_wet.iloc[:20, :])

        # sys.exit()

        ######## Partial correlation & ML ###########
        def seq_pcorr(df, y, x_list):
            """
            Perform sequential partial correlation analysis.

            Parameters:
            - df (DataFrame): Input DataFrame containing the variables.
            - y (str): Dependent variable.
            - x_list (list): List of independent variables.

            Returns:
            - result_df (DataFrame): DataFrame containing the sequential partial correlation results.
            """
            res_list = []
            # print('df type:', df.dtypes)
            # print('y:', y)
            # print('x_list:', x_list)
            for var in x_list:
                # Perform partial correlation for each variable with other variables
                res = pg.partial_corr(df, y=y, x=var, covar=[item for item in x_list if item != var])
                res['var'] = var
                res_list.append(res)
            result_df = pd.concat(res_list, axis=0)
            return result_df

        results_dict = {}  # 用于存储每个y的结果

        for y in y_list:


            ### ============= 1. 偏相关分析 ============== ###
            print('Start partial analyze!')
            # if y == 'POS mean' :
            #     # x_list = ['SOS mean', 'SOS std', 'Cor mean', 'Cor std',
            #     #           'SM mean', 'SM std', 'VPD mean', 'VPD std', 'Ta mean', 'Ta std', 'Pre mean', 'Pre std', 'Srad mean', 'Srad std']
            #     x_list = ['SOS mean', 'Cor mean',
            #               'SM mean', 'VPD mean', 'Ta mean', 'Pre mean', 'Srad mean']
            #     df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_drought
            #
            # elif y == 'POS std':
            #     x_list = ['SOS std', 'Cor std',
            #               'SM std', 'VPD std', 'Ta std', 'Pre std', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data
            #
            #
            # elif y == 'POS slope':
            #     x_list = ['SOS slope', 'SOS std',
            #               'Cor slope', 'Cor std',
            #               'SM slope', 'SM std', 'VPD slope', 'VPD std',
            #               'Ta slope', 'Ta std', 'Pre slope', 'Pre std',
            #               'Srad slope', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data

            if y == 'POS drought normal diff':
                x_list = ['SOS drought normal diff',
                          'Cor drought normal diff',
                          'SM drought normal diff',
                          'VPD drought normal diff',
                          'Ta drought normal diff',
                          'Pre drought normal diff',
                          'Srad drought normal diff']
                df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(
                    drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            elif y == 'POS wet normal diff':
                x_list = ['SOS wet normal diff',
                          'Cor wet normal diff',
                          'SM wet normal diff',
                          'VPD wet normal diff',
                          'Ta wet normal diff',
                          'Pre wet normal diff',
                          'Srad wet normal diff']
                df_pcorr = seq_pcorr(standardized_data_wet, y, x_list)[['var', 'r', 'p-val']].reset_index(
                    drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            # elif y == 'POS corhigh corlow diff':
            #     x_list = ['SOS corhigh corlow diff',
            #               'Cor corhigh corlow diff',
            #               'SM corhigh corlow diff',
            #               'VPD corhigh corlow diff',
            #               'Ta corhigh corlow diff',
            #               'Pre corhigh corlow diff',
            #               'Srad corhigh corlow diff']
            #     df_pcorr = seq_pcorr(standardized_data_corhigh_corlow, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_corhigh_corlow

            df_pcorr = df_pcorr.rename(columns={'var': 'feature'})
            df_pcorr['Significance'] = ''

            for x in x_list:
                p_value = df_pcorr[df_pcorr['feature'] == x]['p_pcorr'].iloc[0]
                if p_value < 0.01:
                    significance = '**'  # p < 0.01 标记为 **
                elif p_value < 0.05:
                    significance = '*'  # 0.01 <= p < 0.05 标记为 *
                else:
                    significance = ''  # p >= 0.05 不标记

                df_pcorr.loc[df_pcorr['feature'] == x, 'Significance'] = significance

            df_pcorr['Abs_Coefficient'] = df_pcorr['r_pcorr'].abs()

            print(f"{type} df_pcorr:\n", df_pcorr)

            # sys.exit()
            # # ==================== VIF计算 ====================
            # from statsmodels.stats.outliers_influence import variance_inflation_factor
            #
            # # 只取干旱情景的自变量，并清除缺失值和无穷值
            # vif_df = (
            #     standardized_data_drought.loc[:, x_list]
            #     .replace([np.inf, -np.inf], np.nan)
            #     .dropna()
            #     .astype(np.float64)
            # )
            #
            # if len(vif_df) <= len(x_list):
            #     raise ValueError(
            #         f"计算 VIF 的有效样本数不足：{len(vif_df)} 行，"
            #         f"自变量有 {len(x_list)} 个。"
            #     )
            #
            # constant_features = [
            #     col for col in x_list
            #     if vif_df[col].nunique() <= 1
            # ]
            # if constant_features:
            #     raise ValueError(f"以下自变量是常数，无法计算 VIF：{constant_features}")
            #
            # # 转成 NumPy 后手动添加截距列，确保第 0 列是常数项
            # X = vif_df.to_numpy(dtype=np.float64)
            # vif_design = np.column_stack([np.ones(X.shape[0]), X])
            #
            # vif_results = []
            # for i, feature in enumerate(x_list, start=1):
            #     vif_value = variance_inflation_factor(vif_design, i)
            #     vif_results.append({
            #         "feature": feature,
            #         "VIF": vif_value,
            #     })
            #
            # vif_df_result = pd.DataFrame(vif_results)
            #
            # df_pcorr = df_pcorr.merge(vif_df_result, on="feature", how="left")
            # print("df_pcorr:\n", df_pcorr)
            #
            # def plot_posdiff_relationships(
            #         df,
            #         y_col,
            #         comparison_name,
            # ):
            #     # 除目标POS diff之外的变量作为自变量
            #     x_cols = [
            #         col for col in df.columns
            #         if col != y_col and col != "index"
            #     ]
            #
            #     # 去掉缺失值和无穷值
            #     plot_df = (
            #         df[[y_col] + x_cols]
            #         .replace([np.inf, -np.inf], np.nan)
            #         .dropna()
            #     )
            #
            #     if plot_df.empty:
            #         print(f"{comparison_name} 没有可用于绘图的有效data。")
            #         return
            #
            #     ncols = 3
            #     nrows = int(np.ceil(len(x_cols) / ncols))
            #
            #     # 画布缩小；字号通过下面各项单独调大
            #     fig, axes = plt.subplots(
            #         nrows,
            #         ncols,
            #         figsize=(11, 3.2 * nrows),
            #         constrained_layout=True,
            #     )
            #     axes = np.asarray(axes).reshape(-1)
            #
            #     last_hexbin = None
            #
            #     for ax_i, x_col in enumerate(x_cols):
            #         ax = axes[ax_i]
            #
            #         x = plot_df[x_col].to_numpy(dtype=float)
            #         y = plot_df[y_col].to_numpy(dtype=float)
            #
            #         # 六边形密度图，适合大量像元
            #         last_hexbin = ax.hexbin(
            #             x,
            #             y,
            #             gridsize=45,
            #             mincnt=1,
            #             bins="log",
            #             cmap="viridis",
            #         )
            #
            #         # Pearson相关系数和线性趋势线
            #         if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            #             r = np.corrcoef(x, y)[0, 1]
            #
            #             slope, intercept = np.polyfit(x, y, 1)
            #             x_line = np.linspace(np.min(x), np.max(x), 100)
            #             y_line = slope * x_line + intercept
            #
            #             ax.plot(
            #                 x_line,
            #                 y_line,
            #                 color="red",
            #                 linewidth=2.2,
            #                 label="Linear trend",
            #                 zorder=5,
            #             )
            #             ax.legend(fontsize=10, frameon=True)
            #         else:
            #             r = np.nan
            #
            #         ax.set_title(
            #             f"{x_col}\nPearson r = {r:.2f}",
            #             fontsize=12,
            #         )
            #         ax.set_xlabel(x_col, fontsize=11)
            #         ax.set_ylabel(y_col, fontsize=11)
            #         ax.tick_params(axis="both", labelsize=10)
            #
            #         ax.axhline(
            #             0,
            #             color="gray",
            #             linewidth=0.8,
            #             linestyle="--",
            #             zorder=3,
            #         )
            #         ax.axvline(
            #             0,
            #             color="gray",
            #             linewidth=0.8,
            #             linestyle="--",
            #             zorder=3,
            #         )
            #         ax.grid(alpha=0.2)
            #
            #     # 隐藏未使用的空子图
            #     for ax in axes[len(x_cols):]:
            #         ax.set_visible(False)
            #
            #     fig.suptitle(
            #         f"{comparison_name}: variables vs {y_col}",
            #         fontsize=15,
            #     )
            #
            #     if last_hexbin is not None:
            #         cbar = fig.colorbar(
            #             last_hexbin,
            #             ax=axes[:len(x_cols)].tolist(),
            #             label="Pixel count (log scale)",
            #             shrink=0.85,
            #         )
            #         cbar.ax.tick_params(labelsize=10)
            #         cbar.set_label("Pixel count (log scale)", fontsize=11)
            #
            #     print(f"有效像元数：{len(plot_df)}")
            #     plt.show()
            #
            # plot_posdiff_relationships(
            #     standardized_data_drought,
            #     y_col="POS drought normal diff",
            #     comparison_name=f"{grade_by}_{type}_Drought_vs_Normal"
            # )
            #
            # import statsmodels.api as sm
            # from scipy.stats import pearsonr
            #
            # x = "VPD drought normal diff"
            # controls = [v for v in x_list if v != x]
            # cols = ["POS drought normal diff", x] + controls
            # d = standardized_data_drought[cols].replace(
            #     [np.inf, -np.inf], np.nan
            # ).dropna()
            #
            # Z = sm.add_constant(d[controls])
            # resid_y = sm.OLS(d["POS drought normal diff"], Z).fit().resid
            # resid_x = sm.OLS(d[x], Z).fit().resid
            #
            # print("残差相关:", pearsonr(resid_x, resid_y))
            # print("Pingouin 偏相关:",
            #       pg.partial_corr(data=d, y="POS drought normal diff",
            #                       x=x, covar=controls))
            #
            # sys.exit()



            ##### ================= 2. ML =================== #######
            if y == 'POS drought normal diff':
                ml_data = standardized_data_drought
            elif y == 'POS wet normal diff':
                ml_data = standardized_data_wet
            # elif y == 'POS corhigh corlow diff':
            #     ml_data = standardized_data_corhigh_corlow

            # LONG_CLIMATE: 偏相关列表与RF列表分开；ML使用7差值+5气候态+类别独热编码。
            pcorr_x_list = x_list.copy()
            if ML_model_in_spatio in ('RF', 'XGBoost'):
                # CATEGORICAL_BACKGROUND: 7差值+5气候态+4植被列+6AI列。
                x_list = pcorr_x_list + long_cols + category_cols
                ml_data = rf_data_drought if y == 'POS drought normal diff' else rf_data_wet
            print(f'{type} {y}: model features={x_list}; valid rows={len(ml_data)}')


            X = ml_data[x_list]
            Y = ml_data[y]

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            print("test split done")

            def calculate_shap(xgb_model, x_train, y_train, x_test, y_test):

                # Step 1: Calculate MSE and R2 for train and test data
                mse_train = mean_squared_error(y_train, xgb_model.predict(x_train))
                r2_train = xgb_model.score(x_train, y_train)
                print(f'Train R2: {r2_train}, MSE: {mse_train}')

                y_train_pred = xgb_model.predict(x_test)
                r2_test = r2_score(y_test, y_train_pred)
                mse_test = mean_squared_error(y_test, y_train_pred)
                print(f'Test R2: {r2_test}, MSE: {mse_test}')

                # Step 2: Calculate SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                print("SHAP explainer done")
                shap_values = explainer(x_train)
                print("SHAP values calculation done")

                # Convert SHAP values into a DataFrame
                shap_values_df = pd.DataFrame(shap_values.values, columns=x_train.columns, index=x_train.index)
                shap_values_df.columns = ['shap_' + col for col in x_train.columns]

                shap_values_df = pd.concat(
                    [x_train.reset_index(drop=True),
                     pd.DataFrame(shap_values.values,
                                  columns=[f'shap_{c}' for c in x_train.columns])],
                    axis=1
                )

                # Step 3: Consolidate feature importance and metrics
                df_xgb_model = pd.DataFrame(xgb_model.feature_importances_, index=x_list,
                                            columns=["importance"]).reset_index().rename(
                    columns={'index': 'var'})
                print("Feature importances calculated")
                print(df_xgb_model)

                df_xgb_model["model_mse"] = mse_train
                df_xgb_model["model_r2"] = r2_train
                df_xgb_model["model_test_mse"] = mse_test
                df_xgb_model["model_test_r2"] = r2_test

                # Step 4: Calculate mean absolute SHAP values
                mean_abs_shap_values = abs(shap_values.values).mean(0)
                print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                df_shap = pd.DataFrame([x_list, mean_abs_shap_values], index=["var", "shapvalue"]).T
                print("SHAP values calculated")

                pos_range = y_train.max() - y_train.min()
                shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                print("POS_mean spatial range (days):", pos_range)
                print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                # Step 5: Merge SHAP values and feature importance
                df_merge = pd.merge(df_shap, df_xgb_model, on="var")
                # print('df_merge1:\n', df_merge)
                df_merge = df_merge.rename(columns={'var': 'feature'})
                # df_merge['importance'] = df_merge['importance'] / df_merge['importance'].sum()  # Normalize feature importance
                # df_merge['shapvalue'] = df_merge['shapvalue'] / df_merge['shapvalue'].sum()  # Normalize SHAP importance
                # df_merge['mean'] = (df_merge['importance'] + df_merge['shapvalue']) / 2  # Calculate mean importance

                print('df_merge:\n', df_merge)

                # # Step 6: Sort features by SHAP value
                # df_merge = df_merge.sort_values(by='shapvalue', ascending=False)

                # CATEGORICAL_BACKGROUND: 编码列明细单独保存，图表汇总为14个因素。
                _export_encoded_background(df_merge, grade_by, type, y, 'Partial_ML_calculate_and_plot0')
                shap_values_df, df_merge = _group_categorical_shap(shap_values_df, df_merge)
                return shap_values_df, df_merge, r2_test

            # sys.exit()

            ############################### XGBoost ########################################
            if ML_model_in_spatio == 'XGBoost':
                ## K-fold和使用网格搜索寻找最佳参数 ##
                xgb_model = xgb.XGBRegressor(random_state=42)
                # 定义参数网格
                param_grid = {'n_estimators': [100, 150, 200, 250, 300, 350],
                              'max_depth': [5, 10, 15],
                              'learning_rate': [0.05, 0.1],
                              'subsample': [0.7, 0.8, 0.9, 1],
                              'colsample_bytree': [0.7, 0.8, 0.9, 1]}
                # # 定义 K 折交叉验证 (K-Fold)
                kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                # 使用网格搜索寻找最佳参数
                grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                           verbose=10, n_jobs=15, error_score='raise')

                # 拟合模型
                grid_search.fit(x_train, y_train)

                print(f'XGboost model done')

                # Output最优参数和最优得分
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                Best_parameters.to_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                #
                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # xgboost = grid_search.best_estimator_

                ######################################################
                # #
                Best_parameters = pd.read_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                colsample_bytree = float(Best_parameters['colsample_bytree'][0])
                subsample = float(Best_parameters['subsample'][0])
                learning_rate = float(Best_parameters['learning_rate'][0])

                xgboost = XGBRegressor(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       colsample_bytree=colsample_bytree,  #每次分裂会考虑所有特征中 80% 的特征子集
                                       subsample=subsample,  # 每棵树的随机采样比例
                                       learning_rate=learning_rate,
                                       random_state=42,
                                       n_jobs=20)

                xgboost.fit(x_train, y_train)
                print(f"xgb_model:", xgboost)

                shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(xgboost, x_train, y_train,
                                                                                            x_test, y_test)


                # CATEGORICAL_BACKGROUND: 接通版本0的XGBoost结果Output。
                shap_values_model = shap_values_df
                shap_importance_model = ModelImportance_and_ShapValueMean

            ############################ RF #########################################
            elif ML_model_in_spatio == 'RF':
                ## K-fold和使用网格搜索寻找最佳参数 ##
                rf_model = RandomForestRegressor(random_state=42, n_jobs=15)
                # if grade_by == 'All':
                # 定义参数网格
                # param_grid = {
                #     'n_estimators': [300, 400, 500],#, 600, 700, 800, 900, 1000],
                #     'max_depth': [5, 10, 20],
                #     'min_samples_split': [5, 10, 15],
                #     'min_samples_leaf': [5, 10, 20],
                #     'max_features': [0.5, 0.7]  # 如果是回归任务，建议包含 0.5, 0.7, 1.0
                # }
                param_grid = {
                    'n_estimators': [100, 300, 500, 700],  # , 600, 700, 800, 900, 1000],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 5, 10, 20],
                    'max_features': [0.7, 0.9]  # 如果是回归任务，建议包含 0.5, 0.7, 1.0
                }
                    # param_grid = {
                    #     'n_estimators': [200, 300, 500, 800],
                    #     'max_depth': [None, 15, 20, 25],          # 上限 25，不要 30
                    #     'min_samples_split': [5, 10, 20, 40],
                    #     'min_samples_leaf': [5, 10, 20, 50],      # 下限 5，不要 1、2
                    #     'max_features': ['sqrt', 0.3, 0.5],       # 上限 0.5，不要 0.7、0.8
                    # }
                # else:
                #     param_grid = {
                #         'n_estimators': [150, 200, 250, 300],
                #         'max_depth': [15, 20, 30],
                #         'min_samples_split': [2, 5, 10],
                #         'min_samples_leaf': [1, 2, 4],
                #         'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.8]
                #     }
                # # 定义 K 折交叉验证 (K-Fold)
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                # 使用网格搜索寻找最佳参数
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                           verbose=10, n_jobs=15, error_score='raise')
                # 拟合模型
                grid_search.fit(x_train, y_train)
                # grid_search.fit(X, Y)

                print(f'{type} RF model done')

                # Output最优参数和最优得分
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                # if analyze_by == 'All':
                Best_parameters.to_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif analyze_by != 'All':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPara_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # rf = grid_search.best_estimator_

                ######################################################
                #
                # if analyze_by == 'All':
                Best_parameters = pd.read_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif analyze_by != 'All':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                min_samples_split = int(Best_parameters['min_samples_split'][0])
                min_samples_leaf = int(Best_parameters['min_samples_leaf'][0])
                max_features = Best_parameters['max_features'][0]
                # 如果保存时是数字，才需要转换
                if isinstance(max_features, str):
                    max_features = max_features  # 保持字符串
                else:
                    max_features = float(max_features)

                # if data_type == 'MODIS_GPP_8d' and lag_day ==0 and spei_strength == -1:
                #     # 最优
                # n_estimators = 400
                # max_depth = 40
                # min_samples_split = 5
                # min_samples_leaf = 5
                # max_features = 0.7
                # elif data_type == 'MODIS_GPP_8d' and lag_day ==0 and spei_strength == -1.5:
                #     n_estimators = 200
                #     max_depth = 40
                #     min_samples_split = 2  # 每次分裂会考虑所有特征中 80% 的特征子集
                #     min_samples_leaf = 1 # 每棵树的随机采样比例
                #     max_features = 0.8
                # elif data_type == 'MCD12Q2' and lag_day ==0 and spei_strength == -1:
                #     n_estimators = 400
                #     max_depth = 20
                #     min_samples_split = 2  # 每次分裂会考虑所有特征中 80% 的特征子集
                #     min_samples_leaf = 1  # 每棵树的随机采样比例
                #     max_features = 0.3

                # 转成一维数组，确保 OOB 误差和训练样本一一对应
                y_train_array = np.asarray(y_train).reshape(-1)

                # ==================== 模型1：使用全部训练样本 ====================
                rf_model1 = RandomForestRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,  # 每次分裂会考虑所有特征中 80% 的特征子集
                                           min_samples_leaf=min_samples_leaf,  # 每棵树的随机采样比例
                                           max_features=max_features,
                                           bootstrap=True,
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=20)

                rf_model1.fit(x_train, y_train)
                print(f"rf_model1:", rf_model1)
                print('Train R²:', rf_model1.score(x_train, y_train))
                print('Test  R²:', rf_model1.score(x_test, y_test))
                print('OOB   R²:', rf_model1.oob_score_)
                print('feature importances:', rf_model1.feature_importances_)

                # 每条训练样本的折外预测误差
                oob_pred_model1 = np.asarray(rf_model1.oob_prediction_).reshape(-1)

                valid_oob = (np.isfinite(y_train_array) & np.isfinite(oob_pred_model1))

                oob_abs_error_model1 = np.abs(
                    y_train_array - oob_pred_model1
                )

                oob_mean_model1 = float(np.mean(oob_abs_error_model1))
                oob_median_model1 = float(np.median(oob_abs_error_model1))
                oob_threshold = float(np.quantile(oob_abs_error_model1, 0.90))

                # 模型1 SHAP：使用原训练集和同一独立测试集
                shap_values_model1, shap_importance_model1, shap_r2_model1 = calculate_shap(
                    rf_model1,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )

                Best_parameters['r2_test'] = shap_r2_model1
                shap_importance_model = shap_importance_model1
                shap_values_model = shap_values_model1

                if shap_r2_model1 < 0.4:


                    # ==================== 根据模型1 OOB误差筛样本 ====================
                    # 误差高于90%分位数的训练样本被剔除
                    keep_mask = oob_abs_error_model1 <= oob_threshold
                    n_removed = int(np.sum(~keep_mask))
                    n_kept = int(np.sum(keep_mask))

                    print(f"模型1训练样本数：{len(keep_mask)}")
                    print(f"剔除样本数：{n_removed}")
                    print(f"保留样本数：{n_kept}")

                    # 同时兼容 pandas DataFrame/Series 和 NumPy 数组
                    if hasattr(x_train, "iloc"):
                        x_train_model2 = x_train.iloc[np.flatnonzero(keep_mask)].copy()
                    else:
                        x_train_model2 = x_train[keep_mask]

                    if hasattr(y_train, "iloc"):
                        y_train_model2 = y_train.iloc[np.flatnonzero(keep_mask)].copy()
                    else:
                        y_train_model2 = y_train[keep_mask]

                    # ==================== 模型2：用筛选后的训练样本 ====================
                    rf_model2 = RandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,  # 每次分裂会考虑所有特征中 80% 的特征子集
                                               min_samples_leaf=min_samples_leaf,  # 每棵树的随机采样比例
                                               max_features=max_features,
                                               bootstrap=True,
                                               oob_score=True,
                                               random_state=42,
                                               n_jobs=20)
                    rf_model2.fit(x_train_model2, y_train_model2)
                    print(f"rf_model2:", rf_model2)
                    print('Train R²:', rf_model2.score(x_train, y_train))
                    print('Test  R²:', rf_model2.score(x_test, y_test))
                    print('OOB   R²:', rf_model2.oob_score_)
                    print('feature importances:', rf_model2.feature_importances_)

                    y_train_model2_array = np.asarray(y_train_model2).reshape(-1)
                    oob_pred_model2 = np.asarray(
                        rf_model2.oob_prediction_
                    ).reshape(-1)

                    valid_oob_model2 = (
                            np.isfinite(y_train_model2_array)
                            & np.isfinite(oob_pred_model2)
                    )

                    if not np.all(valid_oob_model2):
                        raise ValueError("模型2存在无效OOB预测，请检查data或增加树的数量。")

                    oob_abs_error_model2 = np.abs(
                        y_train_model2_array - oob_pred_model2
                    )

                    oob_mean_model2 = float(np.mean(oob_abs_error_model2))
                    oob_median_model2 = float(np.median(oob_abs_error_model2))

                    print(f"模型2 OOB绝对误差均值：{oob_mean_model2:.4f}")
                    print(f"模型2 OOB绝对误差中位数：{oob_median_model2:.4f}")

                    # 模型2 SHAP：训练集是筛选后的样本，测试集仍与模型1相同
                    shap_values_model2, shap_importance_model2, shap_r2_model2 = calculate_shap(
                        rf_model2,
                        x_train_model2,
                        y_train_model2,
                        x_test,
                        y_test,
                    )

                    # 汇总两个模型的 OOB 与测试集结果
                    summary = pd.DataFrame([
                        {
                            "model": "model1",
                            "training_samples": len(y_train_array),
                            "removed_samples": 0,
                            "oob_R2": rf_model1.oob_score_,
                            "oob_MAE_mean": oob_mean_model1,
                            "oob_MAE_median": oob_median_model1,
                            "test_R2": rf_model1.score(x_test, y_test),
                            "SHAP_test_R2": shap_r2_model1,
                        },
                        {
                            "model": "model2",
                            "training_samples": n_kept,
                            "removed_samples": n_removed,
                            "oob_R2": rf_model2.oob_score_,
                            "oob_MAE_mean": oob_mean_model2,
                            "oob_MAE_median": oob_median_model2,
                            "test_R2": rf_model2.score(x_test, y_test),
                            "SHAP_test_R2": shap_r2_model2,
                        },
                    ])

                    print('summary\n'
                          rf'{summary}')

                    summary.to_csv(
                        fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TS{test_size}_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_Summary_LongClimate_CatBackground_5Climate.csv')

                    Best_parameters['r2_test'] = shap_r2_model2
                    shap_importance_model = shap_importance_model2
                    shap_values_model = shap_values_model2

            # if analyze_by == 'All':
            Best_parameters.to_csv(
                fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
            # elif analyze_by != 'All':
            #     Best_parameters.to_csv(
            #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

            # LONG_CLIMATE: 保存偏相关及RF/SHAP结果，文件名含函数版本。
            _export_long_climate_results(df_pcorr, shap_importance_model,
                                         grade_by, type, y, 'Partial_ML_calculate_and_plot0')

            # 将每个y的结果存储到字典中
            results_dict[y] = {
                'df_pcorr': df_pcorr,
                'importance_and_shap_values_mean': shap_importance_model,
                'shap_values_df': shap_values_model,
                # 训练用22列，图表用12个连续因素+2个类别组。
                'x_list': shap_importance_model['feature'].tolist(),
                'y': y
            }

            # # srad_std_shap = shap_values_df.values[:, -1]
            # print(f"SHAP值Statistics:")
            # print(f"  最小值: {np.nanmin(shap_values_df.values):.1f}")
            # print(f"  最大值: {np.nanmax(shap_values_df.values):.1f}")
            # print(f"  平均值: {np.nanmean(shap_values_df.values):.1f}")
            #
            # # Statistics-10到10范围内的占比
            # in_range_mask = (shap_values_df >= -1) & (shap_values_df <= 1) & (~np.isnan(shap_values_df))
            # total_valid = np.sum(~np.isnan(shap_values_df))
            # in_range_count = np.sum(in_range_mask)

            # print(f"[-1, 1]区间Statistics:\n")
            # print(f"  区间内数量: {in_range_count}")
            # print(f"  有效总数: {total_valid}")
            # print(f"  占比: {(in_range_count / total_valid) * 100:.2f}%")
            #
            # sys.exit()

        ######## ================= 3. plot ====================== #######

        #### 驱动分析结果图Plot ####
        for y, results in results_dict.items():
            # 使用存储的结果进行绘图
            df_pcorr = results['df_pcorr']
            ModelImportance_and_ShapValueMean = results['importance_and_shap_values_mean']
            shap_values_df = results['shap_values_df']
            x_list = results['x_list']

            # ========== 1. 定义“唯一的 y 轴顺序”（以 SHAP / importance 为准） ==========
            feature_order = ModelImportance_and_ShapValueMean['feature'].tolist()
            # feature_order = ['SOS drought normal diff', 'Cor drought normal diff', 'SM drought normal diff', 'VPD drought normal diff',
            #                  'Ta drought normal diff', 'Pre drought normal diff', 'Srad drought normal diff']
            print(f'feature_order:{feature_order}')

            # # ========== 2. Create the figure ==========
            # fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=False)
            # ax1, ax2, ax3 = axes
            # plt.subplots_adjust(wspace=0.05)

            # ========== 3. 子图 1：Partial correlation ==========
            # LONG_CLIMATE: 背景行只作控制说明，不Plot伪造的偏相关条形。
            _plot_long_climate_pcorr(ax1, df_pcorr, feature_order,
                                     False)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'a'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            #     elif type == 'Arid':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'd'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'g'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'j'
            #     ytitle = 1.07
            # ax1.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax1.set_xlabel(f'Partial correlation')
            else:
                ax1.set_xlabel('')

            if data_type == 'MODIS_GPP_8d':
                ax1.set_xlim(0, 0.3)
                ax1.set_xticks(np.arange(0, 0.4, 0.1))
                ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.4, 0.1)])
            elif data_type == 'MCD12Q2':
                ax1.set_xlim(0, 0.8)
                ax1.set_xticks(np.arange(0, 0.81, 0.2))
                ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.81, 0.2)])

            pcorr_bound = min(1.0, max(ax1.get_xlim()[1], float(df_pcorr['Abs_Coefficient'].max()) * 1.08))
            ax1.set_xlim(0, pcorr_bound)
            ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
            ax1.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax1.set_ylabel('')
            # LONG_CLIMATE: y标签已由_plot_long_climate_pcorr按实际变量数设置。
            # ax1.set_yticklabels(['Cor', 'SM', 'VPD', 'Ta', 'Pre', 'Srad'])
            ax1.tick_params(axis='y', length=0)

            # legend_handles = [
            #     mpatches.Patch(color='#4393c3', label='Negative', alpha=0.8),
            #     mpatches.Patch(color='#d6604d', label='Positive', alpha=0.8)
            # ]
            # ax1.legend(handles=legend_handles, frameon=False, fontsize=8, loc='lower right')
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#4393c3',
                       markeredgecolor='none', markersize=6, label='Negative', alpha=0.8),
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#d6604d',
                       markeredgecolor='none', markersize=6, label='Positive', alpha=0.8)
            ]

            ax1.legend(handles=legend_handles, frameon=False, loc='lower right', fontsize=8,
                       handletextpad=0.5, handlelength=1.0)  # 缩短句柄长度使其看起来更像纯方块

            # ========== 4. 子图 2：ML feature importance ==========
            importance_plot = ModelImportance_and_ShapValueMean.copy()

            importance_plot['feature'] = pd.Categorical(
                importance_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            importance_plot = importance_plot.sort_values('feature')
            print(f'importance_plot:{importance_plot}')

            sns.barplot(
                x='importance',
                y='feature',
                data=importance_plot,
                color='#4393c3',
                alpha=0.8,
                width=0.4,
                ax=ax2
            )

            ax2.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'b'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'e'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'h'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'k'
            #     ytitle = 1.07
            # ax2.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax2.set_xlabel(f'{ML_model_in_spatio} importance')
            else:
                ax2.set_xlabel('')

            if grade_by == 'All':
                x_min = 0
                x_max = 0.45
                interval = 0.1
            else:
                x_min = 0
                x_max = 0.2
                interval = 0.1
            x_max = max(x_max, float(importance_plot['importance'].max()) * 1.1)
            ax2.set_xlim(x_min, x_max)
            ax2.set_xticks(np.arange(x_min, x_max + 0.01, interval))
            ax2.set_xticklabels(
                [f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(x_min, x_max + 0.01, interval)])

            ax2.set_ylabel('')
            ax2.tick_params(axis='y', length=0, labelright=False, labelleft=False)

            # ========== 5. 子图 3：SHAP beeswarm（关键：用 order 控制顺序） ==========
            plt.sca(ax3)
            ### 1 Shap value sactter
            # —— 构造 SHAP Explanation（按 x_list 顺序）——
            shap_values = shap.Explanation(
                values=shap_values_df[[f'shap_{c}' for c in x_list]].values,
                data=shap_values_df[x_list].values,
                feature_names=x_list
            )

            # —— 将 feature_order 映射为 SHAP 的索引顺序 ——
            order_idx = [x_list.index(f) for f in feature_order if f in x_list]
            print(f'order_idx:{order_idx}')

            shap.plots.beeswarm(
                shap_values,
                order=order_idx,
                max_display=len(order_idx),
                plot_size=None,  # 由外部 ax 控制
                show=False,
                color_bar=False
            )

            # 更改点的color
            new_cmap = plt.get_cmap('coolwarm')

            for coll in ax3.collections:
                coll.set_sizes([5])
                # 将现有的color数组映射到新的 cmap 上
                # SHAP 内部通常将 feature value 映射在 [0, 1] 之间
                coll.set_cmap(new_cmap)

            # 点大小
            for coll in ax3.collections:
                coll.set_sizes([5])

            # x 轴范围
            # if y == 'POS mean':
            #     ax3.set_xlim(-20, 20)
            # elif y == 'POS slope':
            #     ax3.set_xlim(-1, 1)
            # elif y == 'POS std':
            #     ax3.set_xlim(-5, 5)
            # else:
            ax3.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            fp = ax2.get_xticklabels()[0].get_fontproperties()
            shap_bound = max(1.5, float(np.nanmax(np.abs(shap_values.values))) * 1.05)
            ax3.set_xlim(-shap_bound, shap_bound)
            ax3.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))


            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'c'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'f'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'i'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'l'
            #     ytitle = 1.07
            # ax3.set_title(f'({word}) {type}', y=ytitle)
            fp = ax2.xaxis.label.get_fontproperties()
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax3.set_xlabel(f'{ML_model_in_spatio} SHAP value', fontproperties=fp)  #, labelpad=4)
            else:
                ax3.set_xlabel('')
            ax3.tick_params(axis='y', labelleft=False)

            ### 2 Shap abs mean

            ax3_twin = ax3.twiny()
            ax3_twin.set_ylim(ax1.get_ylim())

            shapvalue_mean = ModelImportance_and_ShapValueMean.copy()

            shapvalue_mean['feature'] = pd.Categorical(
                shapvalue_mean['feature'],
                categories=feature_order[::-1],
                ordered=True
            )
            shapvalue_mean = shapvalue_mean.sort_values('feature')
            print(f'shapvalue_mean:{shapvalue_mean}')
            sns.barplot(
                x='shapvalue',
                y='feature',
                data=shapvalue_mean,
                color='gray',
                width=0.4,
                alpha=0.3,
                ax=ax3_twin,
                zorder=0
            )

            # 设置顶层 X 轴的范围和标签
            shap_mean_bound = max(0.3, float(shapvalue_mean['shapvalue'].max()) * 1.1)
            ax3_twin.set_xlim(0, shap_mean_bound)  # 给条形图留出余量
            ax3_twin.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
            ax3_twin.tick_params(axis='x', length=2, pad=1)
            if type in ['All', 'Forest', 'Arid', 'Cor(-0.1~0)']:
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            elif veg_type == 'Grass' and grade_by == 'Veg':
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            else:
                ax3_twin.set_xlabel('')

            # # 隐藏右侧和多余的 Y 轴标签
            # ax3_twin.spines['right'].set_visible(False)
            # ax3.spines['right'].set_visible(False)
            # ax3_twin.get_yaxis().set_visible(False)  # 隐藏 twin 轴的 Y 标签，防止重叠

            # for ax, name in zip([ax2, ax3], ['ax2', 'ax3']):
            #     t = ax.get_xticklabels()[0]
            #     print(name)
            #     print("family:", t.get_fontfamily())
            #     print("size:", t.get_fontsize())
            #     print("weight:", t.get_fontweight())
            #     print("style:", t.get_fontstyle())
            #     print("name:", t.get_fontproperties().get_name())

            ### ================ 6. 子图 4：Subplot 3的color ================ ###
            norm = mpl.colors.Normalize(vmin=0, vmax=1)

            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
                cax=ax4,  # 确保使用预留的 ax4
                orientation='vertical'
            )

            cb.ax.tick_params(
                axis='y',  # 针对 y 轴（Colorbar 的纵向轴）
                length=0,  # 刻度线长度设为 0（隐藏刻度线）
                pad=2  # 标签与色条的间距，可根据需要微调
            )
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label('Feature value', labelpad=-7)
            cb.outline.set_visible(False)  # 去掉 colorbar 的黑色边框显得更高级

            pos = ax4.get_position()

            new_width = pos.width * 0.5

            ax4.set_position([pos.x0 - 0.01, pos.y0, new_width, pos.height])

            # ========== 6. 统一边框风格 ==========
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1)
                    spine.set_edgecolor('black')

            # ---------------------- 统一对齐 y 轴 ----------------------
            # 1. 以 ax1 为基准
            y_min, y_max = ax1.get_ylim()
            y_lim = (min(y_min, y_max), max(y_min, y_max))  # 确保正向

            # 2. 获取 ax1 的 ticks
            yticks = ax1.get_yticks()

            # 3. 循环应用到 ax2 和 ax3
            for ax in [ax3_twin]:
                # 设置相同的 ylim（正向保证）
                ax.set_ylim(y_lim)

                # 设置相同的 ticks
                ax.set_yticks(yticks)
                # ax.invert_yaxis()

                # 可选：保持 tick 标签与 ax1 一致
                # ticklabel 可以使用字符串或数值
                # yticklabels = [str(round(t, 2)) for t in yticks]  # 保留两位小数
                # ax.set_yticklabels(yticklabels)

            # base_ylim = ax1.get_ylim()
            #
            # # 2. 强制覆盖所有相关轴
            # for target_ax in [ax2, ax3, ax3_twin]:
            #     target_ax.set_ylim(base_ylim)
            #
            # # 3. 检查反转状态（关键！）
            # # 如果 ax1 是Regular的（0在下，N在上），而 SHAP 点在 ax3 里还是反的
            # # 那么我们只需要反转 ax3 和它的 twin 轴
            # if ax3.yaxis_inverted() != ax1.yaxis_inverted():
            #     ax3.invert_yaxis()
            #     ax3_twin.invert_yaxis()  # 必须两个一起反转
            # ax3_twin.get_yaxis().set_visible(False)

    # ========== 7. 保存 ==========

    if grade_by in ['Veg', 'AI', 'Cor mean']:
        # plt.tight_layout()
        # if analyze_by == 'All':
        fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor({coupling_method}_lag={lag_day}){months_before_pos}_{grade_by}_{ML_model_in_spatio}_TestSize{test_size}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        # elif analyze_by == 'advance PPT':
        #     fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{cor_test_number}_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        # elif analyze_by == 'delay PPT':
        #     fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{cor_test_number}_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        plt.savefig(
            fig_path,
            dpi=300, bbox_inches='tight')
        print(f'Fig保存到：{fig_path}')


def Partial_ML_calculate_and_plot1(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data,
                                  grade_by, veg_type, ax, *, sm_long_data=None, vpd_long_data=None, ta_long_data=None, vegetation_data=None, ai_class_data=None, pre_long_data=None, srad_long_data=None):
    # LONG_CLIMATE: 默认读取同名全局数组，也支持显式关键字传入。
    long_arrays = _resolve_long_climate(pos_data, sm_long_data, vpd_long_data, ta_long_data, pre_long_data, srad_long_data)
    long_cols = list(long_arrays)

    # CATEGORICAL_BACKGROUND: 两个栅格均为分类，固定独热编码后拼入ML表。
    veg_type_data = globals().get('veg_type_data') if vegetation_data is None else vegetation_data
    ai_type_data = globals().get('ai_type_data') if ai_class_data is None else ai_class_data
    category_arrays = _resolve_categorical_background(pos_data, veg_type_data, ai_type_data)
    category_cols = list(category_arrays)

    os.makedirs(os.path.join(output_partial_max_var_r_tif_path, f'SPEI{spei_length}', str(Basedon)), exist_ok=True)
    for folder in ['All', 'Veg', 'AI', 'Cor mean']:
        os.makedirs(os.path.join(output_partial_max_var_png_path, folder), exist_ok=True)
    var_list = ['POS mean', 'POS slope', 'POS std', 'POS drought normal diff', 'POS wet normal diff',
                'POS corhigh corlow diff',

                'SOS mean', 'SOS slope', 'SOS std', 'SOS drought normal diff', 'SOS wet normal diff',
                'SOS corhigh corlow diff',

                'Cor mean', 'Cor slope', 'Cor std', 'Cor drought normal diff', 'Cor wet normal diff',
                'Cor corhigh corlow diff',

                'SM mean', 'SM slope', 'SM std', 'SM drought normal diff', 'SM wet normal diff',
                'SM corhigh corlow diff',

                'VPD mean', 'VPD slope', 'VPD std', 'VPD drought normal diff', 'VPD wet normal diff',
                'VPD corhigh corlow diff',

                'Ta mean', 'Ta slope', 'Ta std', 'Ta drought normal diff', 'Ta wet normal diff',
                'Ta corhigh corlow diff',

                'Pre mean', 'Pre slope', 'Pre std', 'Pre drought normal diff', 'Pre wet normal diff',
                'Pre corhigh corlow diff',

                'Srad mean', 'Srad slope', 'Srad std', 'Srad drought normal diff', 'Srad wet normal diff',
                'Srad corhigh corlow diff']  ##修改此处顺序可以修改子图一的顺序

    if grade_by != 'All':
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

    ### 11 km 像元按类别筛选后，按 SPEI 原始网格取算术均值
    def _aggregate_to_spei_grid(arrays, group_grid, selection_mask):
        """每个 SPEI 网格Output一行；组内 11 km 像元等权算术平均。"""
        group_grid = np.asarray(group_grid)
        selection_mask = np.asarray(selection_mask, dtype=bool)
        if group_grid.shape != selection_mask.shape:
            raise ValueError(
                f"spei_grid_group shape {group_grid.shape} 与 mask shape "
                f"{selection_mask.shape} 不一致。"
            )

        flat_group = group_grid.ravel()
        valid = selection_mask.ravel() & np.isfinite(flat_group) & (flat_group >= 0)
        flat_arrays = {}
        for name, array in arrays.items():
            values = np.asarray(array, dtype=float)
            if values.shape != group_grid.shape:
                raise ValueError(
                    f"{name} shape {values.shape} 与 spei_grid_group shape "
                    f"{group_grid.shape} 不一致。"
                )
            values = values.ravel()
            flat_arrays[name] = values
            valid &= np.isfinite(values)

        if not np.any(valid):
            return pd.DataFrame(
                columns=["SPEI_group", *arrays.keys(), "n_11km_pixels"]
            )

        pixel_df = pd.DataFrame({
            "SPEI_group": flat_group[valid].astype(np.int64),
            **{name: values[valid] for name, values in flat_arrays.items()},
        })
        value_columns = list(arrays.keys())
        aggregated = (
            pixel_df.groupby("SPEI_group", sort=False)[value_columns]
            .mean()
            .reset_index()
        )
        counts = pixel_df.groupby("SPEI_group", sort=False).size()
        aggregated["n_11km_pixels"] = (
            aggregated["SPEI_group"].map(counts).astype(int)
        )
        return aggregated

    ### 类别掩膜与data有效性掩膜分开；干旱、湿润样本分别检查完整值

    if grade_by == 'All':
        types = ['All']
        codes = [1]

        fig = ax.figure
        gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                    width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                    height_ratios=[1],  # Reserve the last slot for the colorbar
                                                    hspace=0.3, wspace=0.2)

        ax.axis('off')

    elif grade_by == 'Veg':
        if veg_type == 'All':
            types = ['Forest', 'Shrub', 'Savanna', 'Grass']
            codes = [1, 2, 3, 4]

            fig = plt.figure(figsize=(8.2, 14))
            gs = gridspec.GridSpec(4, 4,
                                   width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                   height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                                   hspace=0.3, wspace=0.2)
        elif veg_type == 'Grass':
            types = ['Grass']
            codes = [4]

            fig = ax.figure
            gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                        width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                        height_ratios=[1],  # Reserve the last slot for the colorbar
                                                        hspace=0.3, wspace=0.2)
            ax.axis('off')

    elif grade_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig = plt.figure(figsize=(8.2, 14))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    elif grade_by == 'Cor mean':
        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']
        codes = [0, 1, 2, 3, 4]

        fig = plt.figure(figsize=(8.2, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    for i, (type, code) in enumerate(zip(types, codes)):
        if grade_by == 'All':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        elif grade_by == 'Veg' and veg_type == 'Grass':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        else:
            ax1 = plt.subplot(gs[i, 0])
            ax2 = plt.subplot(gs[i, 1])
            ax3 = plt.subplot(gs[i, 2])
            ax4 = plt.subplot(gs[i, 3])

        if grade_by == 'Veg':
            class_mask = (veg_type_data == code)

        elif grade_by == 'AI':
            if code == 3:
                class_mask = ((ai_type_data == code) | (ai_type_data == 4))

            else:
                class_mask = (ai_type_data == code)


        elif grade_by == 'Cor mean':
            if code == 4:
                class_mask = (cor_mean_data <= -0.4)
            else:
                class_mask = (
                    (cor_mean_data <= -(code * 0.1))
                    & (cor_mean_data > -((code + 1) * 0.1))
                )

        elif grade_by == 'All':
            class_mask = np.ones(np.asarray(pos_data).shape, dtype=bool)

        ########## 空间尺度
        ## StandardizationPrepare data：drought/wet/normal year diff ###
        # standardized_data_drought = pd.DataFrame({
        #     'POS drought normal diff': pos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SOS drought normal diff': sos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Cor drought normal diff': cor_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SM drought normal diff': sm_diff_drought_normal.flatten()[valid_indices],
        #
        #     'VPD drought normal diff': vpd_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Ta drought normal diff': ta_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Pre drought normal diff': pre_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Srad drought normal diff': srad_diff_drought_normal.flatten()[valid_indices]
        # })
        drought_arrays = {
            'POS drought normal diff': pos_data,
            'SOS drought normal diff': sos_data,
            'Cor drought normal diff': cor_data,
            'SM drought normal diff': sm_data,
            'VPD drought normal diff': vpd_data,
            'Ta drought normal diff': ta_data,
            'Pre drought normal diff': pre_data,
            'Srad drought normal diff': srad_data,
        }
        # wet_arrays = {
        #     'POS wet normal diff': pos_diff_wet_normal,
        #     'SOS wet normal diff': sos_diff_wet_normal,
        #     'Cor wet normal diff': cor_diff_wet_normal,
        #     'SM wet normal diff': sm_diff_wet_normal,
        #     'VPD wet normal diff': vpd_diff_wet_normal,
        #     'Ta wet normal diff': ta_diff_wet_normal,
        #     'Pre wet normal diff': pre_diff_wet_normal,
        #     'Srad wet normal diff': srad_diff_wet_normal,
        # }

        standardized_data_drought = _aggregate_to_spei_grid(
            drought_arrays, spei_grid_group, class_mask
        )
        # standardized_data_wet = _aggregate_to_spei_grid(
        #     wet_arrays, spei_grid_group, class_mask
        # )

        # LONG_CLIMATE: RF表同时检查全部差值和背景的完整性。
        # 聚合版本先在共同有效11km像元上筛选，再对所有列一起求55km均值。
        rf_data_drought = _aggregate_to_spei_grid(
            {**drought_arrays, **long_arrays, **category_arrays}, spei_grid_group, class_mask)
        # rf_data_wet = _aggregate_to_spei_grid(
        #     {**wet_arrays, **long_arrays, **category_arrays}, spei_grid_group, class_mask)

        print(
            f"{type}: 干旱有效 SPEI 网格数={len(standardized_data_drought)}, "
            # f"湿润有效 SPEI 网格数={len(standardized_data_wet)}"
        )

        # standardized_data_corhigh_corlow= pd.DataFrame({
        #     'POS corhigh corlow diff': pos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SOS corhigh corlow diff': sos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Cor corhigh corlow diff': cor_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SM corhigh corlow diff': sm_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'VPD corhigh corlow diff': vpd_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Ta corhigh corlow diff': ta_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Pre corhigh corlow diff': pre_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Srad corhigh corlow diff': srad_diff_corhigh_corlow.flatten()[valid_indices]
        # })

        # print(f'standardized_data drought的有效data量：{len(standardized_data_drought.dropna())}')
        # print(f'standardized_data wet的有效data量：{len(standardized_data_wet.dropna())}')

        # 聚合函数已按完整案例筛选；这里仅重置索引，不额外生成 index 列。
        standardized_data_drought = standardized_data_drought.reset_index(drop=True)
        standardized_data_wet = standardized_data_wet.reset_index(drop=True)



        # standardized_data_corhigh_corlow = standardized_data_corhigh_corlow.dropna().reset_index()

        # print('standardized_data_drought:\n', standardized_data_drought.iloc[:20, :])
        # print('standardized_data_wet:\n', standardized_data_wet.iloc[:20, :])

        # sys.exit()

        ######## Partial correlation & ML ###########
        def seq_pcorr(df, y, x_list):
            """
            Perform sequential partial correlation analysis.

            Parameters:
            - df (DataFrame): Input DataFrame containing the variables.
            - y (str): Dependent variable.
            - x_list (list): List of independent variables.

            Returns:
            - result_df (DataFrame): DataFrame containing the sequential partial correlation results.
            """
            res_list = []
            # print('df type:', df.dtypes)
            # print('y:', y)
            # print('x_list:', x_list)
            for var in x_list:
                # Perform partial correlation for each variable with other variables
                res = pg.partial_corr(df, y=y, x=var, covar=[item for item in x_list if item != var])
                res['var'] = var
                res_list.append(res)
            result_df = pd.concat(res_list, axis=0)
            return result_df

        results_dict = {}  # 用于存储每个y的结果

        for y in y_list:


            ### ============= 1. 偏相关分析 ============== ###
            print('Start partial analyze!')
            # if y == 'POS mean' :
            #     # x_list = ['SOS mean', 'SOS std', 'Cor mean', 'Cor std',
            #     #           'SM mean', 'SM std', 'VPD mean', 'VPD std', 'Ta mean', 'Ta std', 'Pre mean', 'Pre std', 'Srad mean', 'Srad std']
            #     x_list = ['SOS mean', 'Cor mean',
            #               'SM mean', 'VPD mean', 'Ta mean', 'Pre mean', 'Srad mean']
            #     df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_drought
            #
            # elif y == 'POS std':
            #     x_list = ['SOS std', 'Cor std',
            #               'SM std', 'VPD std', 'Ta std', 'Pre std', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data
            #
            #
            # elif y == 'POS slope':
            #     x_list = ['SOS slope', 'SOS std',
            #               'Cor slope', 'Cor std',
            #               'SM slope', 'SM std', 'VPD slope', 'VPD std',
            #               'Ta slope', 'Ta std', 'Pre slope', 'Pre std',
            #               'Srad slope', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data

            if y == 'POS drought normal diff':
                x_list = ['SOS drought normal diff',
                          'Cor drought normal diff',
                          'SM drought normal diff',
                          'VPD drought normal diff',
                          'Ta drought normal diff',
                          'Pre drought normal diff',
                          'Srad drought normal diff']
                df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(
                    drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            elif y == 'POS wet normal diff':
                x_list = ['SOS wet normal diff',
                          'Cor wet normal diff',
                          'SM wet normal diff',
                          'VPD wet normal diff',
                          'Ta wet normal diff',
                          'Pre wet normal diff',
                          'Srad wet normal diff']
                df_pcorr = seq_pcorr(standardized_data_wet, y, x_list)[['var', 'r', 'p-val']].reset_index(
                    drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            # elif y == 'POS corhigh corlow diff':
            #     x_list = ['SOS corhigh corlow diff',
            #               'Cor corhigh corlow diff',
            #               'SM corhigh corlow diff',
            #               'VPD corhigh corlow diff',
            #               'Ta corhigh corlow diff',
            #               'Pre corhigh corlow diff',
            #               'Srad corhigh corlow diff']
            #     df_pcorr = seq_pcorr(standardized_data_corhigh_corlow, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_corhigh_corlow

            df_pcorr = df_pcorr.rename(columns={'var': 'feature'})
            df_pcorr['Significance'] = ''

            for x in x_list:
                p_value = df_pcorr[df_pcorr['feature'] == x]['p_pcorr'].iloc[0]
                if p_value < 0.01:
                    significance = '**'  # p < 0.01 标记为 **
                elif p_value < 0.05:
                    significance = '*'  # 0.01 <= p < 0.05 标记为 *
                else:
                    significance = ''  # p >= 0.05 不标记

                df_pcorr.loc[df_pcorr['feature'] == x, 'Significance'] = significance

            df_pcorr['Abs_Coefficient'] = df_pcorr['r_pcorr'].abs()

            print(f"{type} df_pcorr:\n", df_pcorr)

            # sys.exit()
            # # ==================== VIF计算 ====================
            # from statsmodels.stats.outliers_influence import variance_inflation_factor
            #
            # # 只取干旱情景的自变量，并清除缺失值和无穷值
            # vif_df = (
            #     standardized_data_drought.loc[:, x_list]
            #     .replace([np.inf, -np.inf], np.nan)
            #     .dropna()
            #     .astype(np.float64)
            # )
            #
            # if len(vif_df) <= len(x_list):
            #     raise ValueError(
            #         f"计算 VIF 的有效样本数不足：{len(vif_df)} 行，"
            #         f"自变量有 {len(x_list)} 个。"
            #     )
            #
            # constant_features = [
            #     col for col in x_list
            #     if vif_df[col].nunique() <= 1
            # ]
            # if constant_features:
            #     raise ValueError(f"以下自变量是常数，无法计算 VIF：{constant_features}")
            #
            # # 转成 NumPy 后手动添加截距列，确保第 0 列是常数项
            # X = vif_df.to_numpy(dtype=np.float64)
            # vif_design = np.column_stack([np.ones(X.shape[0]), X])
            #
            # vif_results = []
            # for i, feature in enumerate(x_list, start=1):
            #     vif_value = variance_inflation_factor(vif_design, i)
            #     vif_results.append({
            #         "feature": feature,
            #         "VIF": vif_value,
            #     })
            #
            # vif_df_result = pd.DataFrame(vif_results)
            #
            # df_pcorr = df_pcorr.merge(vif_df_result, on="feature", how="left")
            # print("df_pcorr:\n", df_pcorr)
            #
            # def plot_posdiff_relationships(
            #         df,
            #         y_col,
            #         comparison_name,
            # ):
            #     # 除目标POS diff之外的变量作为自变量
            #     x_cols = [
            #         col for col in df.columns
            #         if col != y_col and col != "index"
            #     ]
            #
            #     # 去掉缺失值和无穷值
            #     plot_df = (
            #         df[[y_col] + x_cols]
            #         .replace([np.inf, -np.inf], np.nan)
            #         .dropna()
            #     )
            #
            #     if plot_df.empty:
            #         print(f"{comparison_name} 没有可用于绘图的有效data。")
            #         return
            #
            #     ncols = 3
            #     nrows = int(np.ceil(len(x_cols) / ncols))
            #
            #     # 画布缩小；字号通过下面各项单独调大
            #     fig, axes = plt.subplots(
            #         nrows,
            #         ncols,
            #         figsize=(11, 3.2 * nrows),
            #         constrained_layout=True,
            #     )
            #     axes = np.asarray(axes).reshape(-1)
            #
            #     last_hexbin = None
            #
            #     for ax_i, x_col in enumerate(x_cols):
            #         ax = axes[ax_i]
            #
            #         x = plot_df[x_col].to_numpy(dtype=float)
            #         y = plot_df[y_col].to_numpy(dtype=float)
            #
            #         # 六边形密度图，适合大量像元
            #         last_hexbin = ax.hexbin(
            #             x,
            #             y,
            #             gridsize=45,
            #             mincnt=1,
            #             bins="log",
            #             cmap="viridis",
            #         )
            #
            #         # Pearson相关系数和线性趋势线
            #         if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            #             r = np.corrcoef(x, y)[0, 1]
            #
            #             slope, intercept = np.polyfit(x, y, 1)
            #             x_line = np.linspace(np.min(x), np.max(x), 100)
            #             y_line = slope * x_line + intercept
            #
            #             ax.plot(
            #                 x_line,
            #                 y_line,
            #                 color="red",
            #                 linewidth=2.2,
            #                 label="Linear trend",
            #                 zorder=5,
            #             )
            #             ax.legend(fontsize=10, frameon=True)
            #         else:
            #             r = np.nan
            #
            #         ax.set_title(
            #             f"{x_col}\nPearson r = {r:.2f}",
            #             fontsize=12,
            #         )
            #         ax.set_xlabel(x_col, fontsize=11)
            #         ax.set_ylabel(y_col, fontsize=11)
            #         ax.tick_params(axis="both", labelsize=10)
            #
            #         ax.axhline(
            #             0,
            #             color="gray",
            #             linewidth=0.8,
            #             linestyle="--",
            #             zorder=3,
            #         )
            #         ax.axvline(
            #             0,
            #             color="gray",
            #             linewidth=0.8,
            #             linestyle="--",
            #             zorder=3,
            #         )
            #         ax.grid(alpha=0.2)
            #
            #     # 隐藏未使用的空子图
            #     for ax in axes[len(x_cols):]:
            #         ax.set_visible(False)
            #
            #     fig.suptitle(
            #         f"{comparison_name}: variables vs {y_col}",
            #         fontsize=15,
            #     )
            #
            #     if last_hexbin is not None:
            #         cbar = fig.colorbar(
            #             last_hexbin,
            #             ax=axes[:len(x_cols)].tolist(),
            #             label="Pixel count (log scale)",
            #             shrink=0.85,
            #         )
            #         cbar.ax.tick_params(labelsize=10)
            #         cbar.set_label("Pixel count (log scale)", fontsize=11)
            #
            #     print(f"有效像元数：{len(plot_df)}")
            #     plt.show()
            #
            # plot_posdiff_relationships(
            #     standardized_data_drought,
            #     y_col="POS drought normal diff",
            #     comparison_name=f"{grade_by}_{type}_Drought_vs_Normal"
            # )
            #
            # import statsmodels.api as sm
            # from scipy.stats import pearsonr
            #
            # x = "VPD drought normal diff"
            # controls = [v for v in x_list if v != x]
            # cols = ["POS drought normal diff", x] + controls
            # d = standardized_data_drought[cols].replace(
            #     [np.inf, -np.inf], np.nan
            # ).dropna()
            #
            # Z = sm.add_constant(d[controls])
            # resid_y = sm.OLS(d["POS drought normal diff"], Z).fit().resid
            # resid_x = sm.OLS(d[x], Z).fit().resid
            #
            # print("残差相关:", pearsonr(resid_x, resid_y))
            # print("Pingouin 偏相关:",
            #       pg.partial_corr(data=d, y="POS drought normal diff",
            #                       x=x, covar=controls))
            #
            # sys.exit()



            ##### ================= 2. ML =================== #######
            if y == 'POS drought normal diff':
                ml_data = standardized_data_drought
            elif y == 'POS wet normal diff':
                ml_data = standardized_data_wet
            # elif y == 'POS corhigh corlow diff':
            #     ml_data = standardized_data_corhigh_corlow

            # LONG_CLIMATE: 偏相关列表与RF列表分开；ML使用7差值+5气候态+类别独热编码。
            pcorr_x_list = x_list.copy()
            if ML_model_in_spatio in ('RF', 'XGBoost'):
                # CATEGORICAL_BACKGROUND: 7差值+5气候态+4植被列+6AI列。
                x_list = pcorr_x_list + long_cols + category_cols
                ml_data = rf_data_drought if y == 'POS drought normal diff' else rf_data_wet
            print(f'{type} {y}: model features={x_list}; valid rows={len(ml_data)}')


            # X = ml_data[x_list]
            # Y = ml_data[y]

            # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            # print("test split done")

            if ML_model_in_spatio == 'XGBoost':
                x_train, x_test, y_train, y_test = train_test_split(
                    ml_data[x_list], ml_data[y], test_size=test_size, random_state=42)

            def calculate_shap(xgb_model, x_train, y_train, x_test, y_test):

                # Step 1: Calculate MSE and R2 for train and test data
                mse_train = mean_squared_error(y_train, xgb_model.predict(x_train))
                r2_train = xgb_model.score(x_train, y_train)
                print(f'Train R2: {r2_train}, MSE: {mse_train}')

                y_train_pred = xgb_model.predict(x_test)
                r2_test = r2_score(y_test, y_train_pred)
                mse_test = mean_squared_error(y_test, y_train_pred)
                print(f'Test R2: {r2_test}, MSE: {mse_test}')

                # Step 2: Calculate SHAP values
                explainer = shap.TreeExplainer(xgb_model)
                print("SHAP explainer done")
                shap_values = explainer(x_train)
                print("SHAP values calculation done")

                # Convert SHAP values into a DataFrame
                shap_values_df = pd.DataFrame(shap_values.values, columns=x_train.columns, index=x_train.index)
                shap_values_df.columns = ['shap_' + col for col in x_train.columns]

                shap_values_df = pd.concat(
                    [x_train.reset_index(drop=True),
                     pd.DataFrame(shap_values.values,
                                  columns=[f'shap_{c}' for c in x_train.columns])],
                    axis=1
                )

                # Step 3: Consolidate feature importance and metrics
                df_xgb_model = pd.DataFrame(xgb_model.feature_importances_, index=x_list,
                                            columns=["importance"]).reset_index().rename(
                    columns={'index': 'var'})
                print("Feature importances calculated")
                print(df_xgb_model)

                df_xgb_model["model_mse"] = mse_train
                df_xgb_model["model_r2"] = r2_train
                df_xgb_model["model_test_mse"] = mse_test
                df_xgb_model["model_test_r2"] = r2_test

                # Step 4: Calculate mean absolute SHAP values
                mean_abs_shap_values = abs(shap_values.values).mean(0)
                print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                df_shap = pd.DataFrame([x_list, mean_abs_shap_values], index=["var", "shapvalue"]).T
                print("SHAP values calculated")

                pos_range = y_train.max() - y_train.min()
                shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                print("POS_mean spatial range (days):", pos_range)
                print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                # Step 5: Merge SHAP values and feature importance
                df_merge = pd.merge(df_shap, df_xgb_model, on="var")
                # print('df_merge1:\n', df_merge)
                df_merge = df_merge.rename(columns={'var': 'feature'})
                # df_merge['importance'] = df_merge['importance'] / df_merge['importance'].sum()  # Normalize feature importance
                # df_merge['shapvalue'] = df_merge['shapvalue'] / df_merge['shapvalue'].sum()  # Normalize SHAP importance
                # df_merge['mean'] = (df_merge['importance'] + df_merge['shapvalue']) / 2  # Calculate mean importance

                print('df_merge:\n', df_merge)

                # # Step 6: Sort features by SHAP value
                # df_merge = df_merge.sort_values(by='shapvalue', ascending=False)

                # CATEGORICAL_BACKGROUND: 编码列明细单独保存，图表汇总为14个因素。
                _export_encoded_background(df_merge, grade_by, type, y, 'Partial_ML_calculate_and_plot1')
                shap_values_df, df_merge = _group_categorical_shap(shap_values_df, df_merge)
                return shap_values_df, df_merge, r2_test

            # sys.exit()

            ############################### XGBoost ########################################
            if ML_model_in_spatio == 'XGBoost':
                ## K-fold和使用网格搜索寻找最佳参数 ##
                xgb_model = xgb.XGBRegressor(random_state=42)
                # 定义参数网格
                param_grid = {'n_estimators': [100, 150, 200, 250, 300, 350],
                              'max_depth': [5, 10, 15],
                              'learning_rate': [0.05, 0.1],
                              'subsample': [0.7, 0.8, 0.9, 1],
                              'colsample_bytree': [0.7, 0.8, 0.9, 1]}
                # # 定义 K 折交叉验证 (K-Fold)
                kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                # 使用网格搜索寻找最佳参数
                grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                           verbose=10, n_jobs=15, error_score='raise')

                # 拟合模型
                grid_search.fit(x_train, y_train)

                print(f'XGboost model done')

                # Output最优参数和最优得分
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                Best_parameters.to_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                #
                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # xgboost = grid_search.best_estimator_

                ######################################################
                # #
                Best_parameters = pd.read_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                colsample_bytree = float(Best_parameters['colsample_bytree'][0])
                subsample = float(Best_parameters['subsample'][0])
                learning_rate = float(Best_parameters['learning_rate'][0])

                xgboost = XGBRegressor(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       colsample_bytree=colsample_bytree,  #每次分裂会考虑所有特征中 80% 的特征子集
                                       subsample=subsample,  # 每棵树的随机采样比例
                                       learning_rate=learning_rate,
                                       random_state=42,
                                       n_jobs=20)

                xgboost.fit(x_train, y_train)
                print(f"xgb_model:", xgboost)

                shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(xgboost, x_train, y_train,
                                                                                            x_test, y_test)


            if ML_model_in_spatio == 'XGBoost':
                shap_values_model = shap_values_df
                shap_importance_model = ModelImportance_and_ShapValueMean

            ############################ RF #########################################
            elif ML_model_in_spatio == 'RF':
                param_grid = {
                    'n_estimators': [100, 300, 500, 700],  # , 600, 700, 800, 900, 1000],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 5, 10, 20],
                    'max_features': [0.7, 0.9]  # 如果是回归任务，建议包含 0.5, 0.7, 1.0
                }

                # 已经一行代表一个 SPEI 网格；SPEI_group 只保留为网格 ID，
                # 不再用于组内分层，也不作为模型解释变量。
                model_data = (
                    ml_data[x_list + [y, "SPEI_group", "n_11km_pixels"]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .reset_index(drop=True)
                )

                X = model_data[x_list]
                Y = model_data[y]

                print("有效样本数：", len(model_data))
                print("0.5° SPEI 网格数：", model_data["SPEI_group"].nunique())
                print("每个网格内纳入均值的 11 km 像元数范围：",
                      model_data["n_11km_pixels"].min(), "-",
                      model_data["n_11km_pixels"].max())

                if len(model_data) < 10:
                    raise ValueError(
                        f"聚合后只有 {len(model_data)} 个有效 SPEI 网格，"
                        "不足以稳定进行训练/测试拆分和交叉验证。"
                    )

                # 每个网格只有一条样本；随机拆分的是不同 SPEI 网格。
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    Y,
                    test_size=test_size,
                    random_state=42
                )

                print(f"训练集 SPEI 网格样本数：{len(X_train)}")
                print(f"测试集 SPEI 网格样本数：{len(X_test)}")

                # 聚合后网格 ID 基本唯一，不能再用 StratifiedKFold 按网格 ID 分层。
                # 普通 KFold 对训练集内的 SPEI 网格进行 5 折交叉验证。
                n_splits = min(5, len(X_train))
                inner_cv = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42
                )

                cv_splits = list(inner_cv.split(X_train, y_train))

                # 兼容后续原有模型代码使用的小写变量名
                x_train, x_test = X_train, X_test
                y_train = y_train.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)
                x_train = x_train.reset_index(drop=True)
                x_test = x_test.reset_index(drop=True)

                rf_model = RandomForestRegressor(
                    random_state=42,
                    n_jobs=1
                )

                grid_search = GridSearchCV(
                    estimator=rf_model,
                    param_grid=param_grid,
                    scoring="r2",
                    cv=cv_splits,
                    verbose=10,
                    n_jobs=15,
                    error_score="raise",
                    return_train_score=True,
                    refit=True
                )

                grid_search.fit(X_train, y_train)

                print("最佳参数：", grid_search.best_params_)
                print("训练集内部交叉验证平均 R²：", grid_search.best_score_)

                # GridSearchCV 已用最佳参数在整个训练集上重新拟合
                rf_model1 = grid_search.best_estimator_

                print("Train R²:", rf_model1.score(X_train, y_train))
                print("Test R²:", rf_model1.score(X_test, y_test))
                print("Feature importances:", rf_model1.feature_importances_)

                # 最优参数对应的每折 R²
                best_idx = grid_search.best_index_
                fold_scores = [
                    grid_search.cv_results_[f"split{i}_test_score"][best_idx]
                    for i in range(n_splits)
                ]
                print("各折 R²：", fold_scores)
                print("各折平均 R²：", np.mean(fold_scores))
                print("各折 R² 标准差：", np.std(fold_scores, ddof=1))

                # LONG_CLIMATE: 取消原调试退出，直接使用最佳聚合RFOutput结果。
                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                Best_parameters['mean R2 score'] = grid_search.best_score_
                shap_values_model, shap_importance_model, r2_test = calculate_shap(
                    rf_model1, x_train, y_train, x_test, y_test)
                Best_parameters['r2_test'] = r2_test

            # if analyze_by == 'All':
            Best_parameters.to_csv(
                fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
            # elif analyze_by != 'All':
            #     Best_parameters.to_csv(
            #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

            # LONG_CLIMATE: 保存偏相关及RF/SHAP结果，文件名含函数版本。
            _export_long_climate_results(df_pcorr, shap_importance_model,
                                         grade_by, type, y, 'Partial_ML_calculate_and_plot1')

            # 将每个y的结果存储到字典中
            results_dict[y] = {
                'df_pcorr': df_pcorr,
                'importance_and_shap_values_mean': shap_importance_model,
                'shap_values_df': shap_values_model,
                # 训练用22列，图表用12个连续因素+2个类别组。
                'x_list': shap_importance_model['feature'].tolist(),
                'y': y
            }

            # # srad_std_shap = shap_values_df.values[:, -1]
            # print(f"SHAP值Statistics:")
            # print(f"  最小值: {np.nanmin(shap_values_df.values):.1f}")
            # print(f"  最大值: {np.nanmax(shap_values_df.values):.1f}")
            # print(f"  平均值: {np.nanmean(shap_values_df.values):.1f}")
            #
            # # Statistics-10到10范围内的占比
            # in_range_mask = (shap_values_df >= -1) & (shap_values_df <= 1) & (~np.isnan(shap_values_df))
            # total_valid = np.sum(~np.isnan(shap_values_df))
            # in_range_count = np.sum(in_range_mask)

            # print(f"[-1, 1]区间Statistics:\n")
            # print(f"  区间内数量: {in_range_count}")
            # print(f"  有效总数: {total_valid}")
            # print(f"  占比: {(in_range_count / total_valid) * 100:.2f}%")
            #
            # sys.exit()

        ######## ================= 3. plot ====================== #######

        #### 驱动分析结果图Plot ####
        for y, results in results_dict.items():
            # 使用存储的结果进行绘图
            df_pcorr = results['df_pcorr']
            ModelImportance_and_ShapValueMean = results['importance_and_shap_values_mean']
            shap_values_df = results['shap_values_df']
            x_list = results['x_list']

            # ========== 1. 定义“唯一的 y 轴顺序”（以 SHAP / importance 为准） ==========
            feature_order = ModelImportance_and_ShapValueMean['feature'].tolist()
            # feature_order = ['SOS drought normal diff', 'Cor drought normal diff', 'SM drought normal diff', 'VPD drought normal diff',
            #                  'Ta drought normal diff', 'Pre drought normal diff', 'Srad drought normal diff']
            print(f'feature_order:{feature_order}')

            # # ========== 2. Create the figure ==========
            # fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=False)
            # ax1, ax2, ax3 = axes
            # plt.subplots_adjust(wspace=0.05)

            # ========== 3. 子图 1：Partial correlation ==========
            # LONG_CLIMATE: 背景行只作控制说明，不Plot伪造的偏相关条形。
            _plot_long_climate_pcorr(ax1, df_pcorr, feature_order,
                                     False)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'a'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            #     elif type == 'Arid':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'd'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'g'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'j'
            #     ytitle = 1.07
            # ax1.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax1.set_xlabel(f'Partial correlation')
            else:
                ax1.set_xlabel('')

            if data_type == 'MODIS_GPP_8d':
                ax1.set_xlim(0, 0.6)
                ax1.set_xticks(np.arange(0, 0.601, 0.2))
                ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.601, 0.2)])
            elif data_type == 'MCD12Q2':
                ax1.set_xlim(0, 0.8)
                ax1.set_xticks(np.arange(0, 0.81, 0.2))
                ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.81, 0.2)])

            pcorr_bound = min(1.0, max(ax1.get_xlim()[1], float(df_pcorr['Abs_Coefficient'].max()) * 1.08))
            ax1.set_xlim(0, pcorr_bound)
            ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
            ax1.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax1.set_ylabel('')
            # LONG_CLIMATE: y标签已由_plot_long_climate_pcorr按实际变量数设置。
            # ax1.set_yticklabels(['Cor', 'SM', 'VPD', 'Ta', 'Pre', 'Srad'])
            ax1.tick_params(axis='y', length=0)

            # legend_handles = [
            #     mpatches.Patch(color='#4393c3', label='Negative', alpha=0.8),
            #     mpatches.Patch(color='#d6604d', label='Positive', alpha=0.8)
            # ]
            # ax1.legend(handles=legend_handles, frameon=False, fontsize=8, loc='lower right')
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#4393c3',
                       markeredgecolor='none', markersize=6, label='Negative', alpha=0.8),
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#d6604d',
                       markeredgecolor='none', markersize=6, label='Positive', alpha=0.8)
            ]

            ax1.legend(handles=legend_handles, frameon=False, loc='lower right', fontsize=8,
                       handletextpad=0.5, handlelength=1.0)  # 缩短句柄长度使其看起来更像纯方块

            # ========== 4. 子图 2：ML feature importance ==========
            importance_plot = ModelImportance_and_ShapValueMean.copy()

            importance_plot['feature'] = pd.Categorical(
                importance_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            importance_plot = importance_plot.sort_values('feature')
            print(f'importance_plot:{importance_plot}')

            sns.barplot(
                x='importance',
                y='feature',
                data=importance_plot,
                color='#4393c3',
                alpha=0.8,
                width=0.4,
                ax=ax2
            )

            ax2.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'b'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'e'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'h'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'k'
            #     ytitle = 1.07
            # ax2.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax2.set_xlabel(f'{ML_model_in_spatio} importance')
            else:
                ax2.set_xlabel('')

            if grade_by == 'All':
                x_min = 0
                x_max = 0.45
                interval = 0.1
            else:
                x_min = 0
                x_max = 0.8
                interval = 0.2
            x_max = max(x_max, float(importance_plot['importance'].max()) * 1.1)
            ax2.set_xlim(x_min, x_max)
            ax2.set_xticks(np.arange(x_min, x_max + 0.01, interval))
            ax2.set_xticklabels(
                [f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(x_min, x_max + 0.01, interval)])

            ax2.set_ylabel('')
            ax2.tick_params(axis='y', length=0, labelright=False, labelleft=False)

            # ========== 5. 子图 3：SHAP beeswarm（关键：用 order 控制顺序） ==========
            plt.sca(ax3)
            ### 1 Shap value sactter
            # —— 构造 SHAP Explanation（按 x_list 顺序）——
            shap_values = shap.Explanation(
                values=shap_values_df[[f'shap_{c}' for c in x_list]].values,
                data=shap_values_df[x_list].values,
                feature_names=x_list
            )

            # —— 将 feature_order 映射为 SHAP 的索引顺序 ——
            order_idx = [x_list.index(f) for f in feature_order if f in x_list]
            print(f'order_idx:{order_idx}')

            shap.plots.beeswarm(
                shap_values,
                order=order_idx,
                max_display=len(order_idx),
                plot_size=None,  # 由外部 ax 控制
                show=False,
                color_bar=False
            )

            # 更改点的color
            new_cmap = plt.get_cmap('coolwarm')

            for coll in ax3.collections:
                coll.set_sizes([5])
                # 将现有的color数组映射到新的 cmap 上
                # SHAP 内部通常将 feature value 映射在 [0, 1] 之间
                coll.set_cmap(new_cmap)

            # 点大小
            for coll in ax3.collections:
                coll.set_sizes([5])

            # x 轴范围
            # if y == 'POS mean':
            #     ax3.set_xlim(-20, 20)
            # elif y == 'POS slope':
            #     ax3.set_xlim(-1, 1)
            # elif y == 'POS std':
            #     ax3.set_xlim(-5, 5)
            # else:
            ax3.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            fp = ax2.get_xticklabels()[0].get_fontproperties()
            shap_bound = max(1.5, float(np.nanmax(np.abs(shap_values.values))) * 1.05)
            ax3.set_xlim(-shap_bound, shap_bound)
            ax3.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))


            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'c'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'f'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'i'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'l'
            #     ytitle = 1.07
            # ax3.set_title(f'({word}) {type}', y=ytitle)
            fp = ax2.xaxis.label.get_fontproperties()
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax3.set_xlabel(f'{ML_model_in_spatio} SHAP value', fontproperties=fp)  #, labelpad=4)
            else:
                ax3.set_xlabel('')
            ax3.tick_params(axis='y', labelleft=False)

            ### 2 Shap abs mean

            ax3_twin = ax3.twiny()
            ax3_twin.set_ylim(ax1.get_ylim())

            shapvalue_mean = ModelImportance_and_ShapValueMean.copy()

            shapvalue_mean['feature'] = pd.Categorical(
                shapvalue_mean['feature'],
                categories=feature_order[::-1],
                ordered=True
            )
            shapvalue_mean = shapvalue_mean.sort_values('feature')
            print(f'shapvalue_mean:{shapvalue_mean}')
            sns.barplot(
                x='shapvalue',
                y='feature',
                data=shapvalue_mean,
                color='gray',
                width=0.4,
                alpha=0.3,
                ax=ax3_twin,
                zorder=0
            )

            # 设置顶层 X 轴的范围和标签
            shap_mean_bound = max(0.3, float(shapvalue_mean['shapvalue'].max()) * 1.1)
            ax3_twin.set_xlim(0, shap_mean_bound)  # 给条形图留出余量
            ax3_twin.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
            ax3_twin.tick_params(axis='x', length=2, pad=1)
            if type in ['All', 'Forest', 'Arid', 'Cor(-0.1~0)']:
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            elif veg_type == 'Grass' and grade_by == 'Veg':
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            else:
                ax3_twin.set_xlabel('')

            # # 隐藏右侧和多余的 Y 轴标签
            # ax3_twin.spines['right'].set_visible(False)
            # ax3.spines['right'].set_visible(False)
            # ax3_twin.get_yaxis().set_visible(False)  # 隐藏 twin 轴的 Y 标签，防止重叠

            # for ax, name in zip([ax2, ax3], ['ax2', 'ax3']):
            #     t = ax.get_xticklabels()[0]
            #     print(name)
            #     print("family:", t.get_fontfamily())
            #     print("size:", t.get_fontsize())
            #     print("weight:", t.get_fontweight())
            #     print("style:", t.get_fontstyle())
            #     print("name:", t.get_fontproperties().get_name())

            ### ================ 6. 子图 4：Subplot 3的color ================ ###
            norm = mpl.colors.Normalize(vmin=0, vmax=1)

            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
                cax=ax4,  # 确保使用预留的 ax4
                orientation='vertical'
            )

            cb.ax.tick_params(
                axis='y',  # 针对 y 轴（Colorbar 的纵向轴）
                length=0,  # 刻度线长度设为 0（隐藏刻度线）
                pad=2  # 标签与色条的间距，可根据需要微调
            )
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label('Feature value', labelpad=-7)
            cb.outline.set_visible(False)  # 去掉 colorbar 的黑色边框显得更高级

            pos = ax4.get_position()

            new_width = pos.width * 0.5

            ax4.set_position([pos.x0 - 0.01, pos.y0, new_width, pos.height])

            # ========== 6. 统一边框风格 ==========
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1)
                    spine.set_edgecolor('black')

            # ---------------------- 统一对齐 y 轴 ----------------------
            # 1. 以 ax1 为基准
            y_min, y_max = ax1.get_ylim()
            y_lim = (min(y_min, y_max), max(y_min, y_max))  # 确保正向

            # 2. 获取 ax1 的 ticks
            yticks = ax1.get_yticks()

            # 3. 循环应用到 ax2 和 ax3
            for ax in [ax3_twin]:
                # 设置相同的 ylim（正向保证）
                ax.set_ylim(y_lim)

                # 设置相同的 ticks
                ax.set_yticks(yticks)
                # ax.invert_yaxis()

                # 可选：保持 tick 标签与 ax1 一致
                # ticklabel 可以使用字符串或数值
                # yticklabels = [str(round(t, 2)) for t in yticks]  # 保留两位小数
                # ax.set_yticklabels(yticklabels)

            # base_ylim = ax1.get_ylim()
            #
            # # 2. 强制覆盖所有相关轴
            # for target_ax in [ax2, ax3, ax3_twin]:
            #     target_ax.set_ylim(base_ylim)
            #
            # # 3. 检查反转状态（关键！）
            # # 如果 ax1 是Regular的（0在下，N在上），而 SHAP 点在 ax3 里还是反的
            # # 那么我们只需要反转 ax3 和它的 twin 轴
            # if ax3.yaxis_inverted() != ax1.yaxis_inverted():
            #     ax3.invert_yaxis()
            #     ax3_twin.invert_yaxis()  # 必须两个一起反转
            # ax3_twin.get_yaxis().set_visible(False)

    # ========== 7. 保存 ==========

    if grade_by in ['Veg', 'AI', 'Cor mean']:
        # plt.tight_layout()
        # if analyze_by == 'All':
        fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor({coupling_method}_lag={lag_day}){months_before_pos}_{grade_by}_{ML_model_in_spatio}_TestSize{test_size}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        # elif analyze_by == 'advance PPT':
        #     fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{cor_test_number}_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        # elif analyze_by == 'delay PPT':
        #     fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{cor_test_number}_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        plt.savefig(
            fig_path,
            dpi=300, bbox_inches='tight')
        print(f'Fig保存到：{fig_path}')


def Partial_ML_calculate_and_plot2(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data,
                                  grade_by, veg_type, ax, *, sm_long_data=None, vpd_long_data=None, ta_long_data=None, vegetation_data=None, ai_class_data=None, pre_long_data=None, srad_long_data=None):
    # LONG_CLIMATE: 默认读取同名全局数组，也支持显式关键字传入。
    long_arrays = _resolve_long_climate(pos_data, sm_long_data, vpd_long_data, ta_long_data, pre_long_data, srad_long_data)
    long_cols = list(long_arrays)

    # CATEGORICAL_BACKGROUND: 两个栅格均为分类，固定独热编码后拼入ML表。
    veg_type_data = globals().get('veg_type_data') if vegetation_data is None else vegetation_data
    ai_type_data = globals().get('ai_type_data') if ai_class_data is None else ai_class_data
    category_arrays = _resolve_categorical_background(pos_data, veg_type_data, ai_type_data)
    category_cols = list(category_arrays)

    os.makedirs(os.path.join(output_partial_max_var_r_tif_path, f'SPEI{spei_length}', str(Basedon)), exist_ok=True)
    for folder in ['All', 'Veg', 'AI', 'Cor mean']:
        os.makedirs(os.path.join(output_partial_max_var_png_path, folder), exist_ok=True)
    var_list = ['POS mean', 'POS slope', 'POS std', 'POS drought normal diff', 'POS wet normal diff',
                'POS corhigh corlow diff',

                'SOS mean', 'SOS slope', 'SOS std', 'SOS drought normal diff', 'SOS wet normal diff',
                'SOS corhigh corlow diff',

                'Cor mean', 'Cor slope', 'Cor std', 'Cor drought normal diff', 'Cor wet normal diff',
                'Cor corhigh corlow diff',

                'SM mean', 'SM slope', 'SM std', 'SM drought normal diff', 'SM wet normal diff',
                'SM corhigh corlow diff',

                'VPD mean', 'VPD slope', 'VPD std', 'VPD drought normal diff', 'VPD wet normal diff',
                'VPD corhigh corlow diff',

                'Ta mean', 'Ta slope', 'Ta std', 'Ta drought normal diff', 'Ta wet normal diff',
                'Ta corhigh corlow diff',

                'Pre mean', 'Pre slope', 'Pre std', 'Pre drought normal diff', 'Pre wet normal diff',
                'Pre corhigh corlow diff',

                'Srad mean', 'Srad slope', 'Srad std', 'Srad drought normal diff', 'Srad wet normal diff',
                'Srad corhigh corlow diff']  ##修改此处顺序可以修改子图一的顺序

    if grade_by != 'All':
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

    ### 分析使用 11 km 像元；SPEI_group 只用于 RF 按 55 km 网格分层抽样

    if grade_by == 'All':
        types = ['All']
        codes = [1]

        fig = ax.figure
        gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                    width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                    height_ratios=[1],  # Reserve the last slot for the colorbar
                                                    hspace=0.3, wspace=0.2)

        ax.axis('off')

    elif grade_by == 'Veg':
        if veg_type == 'All':
            types = ['Forest', 'Shrub', 'Savanna', 'Grass']
            codes = [1, 2, 3, 4]

            fig = plt.figure(figsize=(8.2, 14))
            gs = gridspec.GridSpec(4, 4,
                                   width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                   height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                                   hspace=0.3, wspace=0.2)
        elif veg_type == 'Grass':
            types = ['Grass']
            codes = [4]

            fig = ax.figure
            gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                        width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                        height_ratios=[1],  # Reserve the last slot for the colorbar
                                                        hspace=0.3, wspace=0.2)
            ax.axis('off')

    elif grade_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig = plt.figure(figsize=(8.2, 14))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    elif grade_by == 'Cor mean':
        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']
        codes = [0, 1, 2, 3, 4]

        fig = plt.figure(figsize=(8.2, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    def _make_pixel_dataframe(arrays, group_grid, class_mask):
        """按类别筛选完整案例 11 km 像元；不对像元做空间聚合。"""
        group_grid = np.asarray(group_grid)
        class_mask = np.asarray(class_mask, dtype=bool)
        if group_grid.shape != class_mask.shape:
            raise ValueError(
                f"spei_grid_group shape {group_grid.shape} 与类别掩膜 "
                f"shape {class_mask.shape} 不一致。"
            )

        flat_group = group_grid.ravel()
        valid = class_mask.ravel() & np.isfinite(flat_group) & (flat_group >= 0)
        flat_arrays = {}
        for name, array in arrays.items():
            values = np.asarray(array, dtype=float)
            if values.shape != group_grid.shape:
                raise ValueError(
                    f"{name} shape {values.shape} 与 spei_grid_group "
                    f"shape {group_grid.shape} 不一致。"
                )
            values = values.ravel()
            flat_arrays[name] = values
            valid &= np.isfinite(values)

        if not np.any(valid):
            return pd.DataFrame(columns=["SPEI_group", *arrays.keys()])

        return pd.DataFrame({
            "SPEI_group": flat_group[valid].astype(np.int64),
            **{name: values[valid] for name, values in flat_arrays.items()},
        }).reset_index(drop=True)

    def _split_rf_within_grid_by_pos_quantiles(
        data, target, test_fraction, random_state=42
    ):
        """每个 SPEI 网格内按目标 POS 的 P25/P75 分层后分别抽训练/测试样本。

        小于 P25：极端负异常；P25 至 P75（含边界）：中间范围；
        大于 P75：极端正异常。某层只有一个样本时，将其留在训练集。
        """
        if not 0 < test_fraction < 1:
            raise ValueError("test_size 必须在 0 和 1 之间。")

        data = data.reset_index(drop=True).copy()
        data["_pos_stratum"] = -1
        for _, indices in data.groupby("SPEI_group", sort=False).groups.items():
            indices = np.asarray(list(indices), dtype=int)
            values = data.loc[indices, target].to_numpy(dtype=float)
            q25, q75 = np.quantile(values, [0.25, 0.75])
            labels = np.where(values < q25, 0, np.where(values > q75, 2, 1))
            data.loc[indices, "_pos_stratum"] = labels

        rng = np.random.default_rng(random_state)
        train_indices, test_indices = [], []
        for _, stratum in data.groupby(
            ["SPEI_group", "_pos_stratum"], sort=False
        ):
            indices = stratum.index.to_numpy().copy()
            rng.shuffle(indices)
            n = len(indices)
            if n < 2:
                train_indices.extend(indices.tolist())
                continue

            n_test = min(max(int(round(n * test_fraction)), 1), n - 1)
            test_indices.extend(indices[:n_test].tolist())
            train_indices.extend(indices[n_test:].tolist())

        if not train_indices or not test_indices:
            raise ValueError(
                "按 SPEI 网格内 POS 分位层抽样后训练集或测试集为空；"
                "请检查有效像元数量或调整 test_size。"
            )

        train_df = data.loc[train_indices].drop(columns="_pos_stratum")
        test_df = data.loc[test_indices].drop(columns="_pos_stratum")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    for i, (type, code) in enumerate(zip(types, codes)):
        if grade_by == 'All':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        elif grade_by == 'Veg' and veg_type == 'Grass':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        else:
            ax1 = plt.subplot(gs[i, 0])
            ax2 = plt.subplot(gs[i, 1])
            ax3 = plt.subplot(gs[i, 2])
            ax4 = plt.subplot(gs[i, 3])

        if grade_by == 'Veg':
            class_mask = (veg_type_data == code)

        elif grade_by == 'AI':
            if code == 3:
                class_mask = ((ai_type_data == code) | (ai_type_data == 4))

            else:
                class_mask = (ai_type_data == code)


        elif grade_by == 'Cor mean':
            if code == 4:
                class_mask = (cor_mean_data <= -0.4)
            else:
                class_mask = (
                    (cor_mean_data <= -(code * 0.1))
                    & (cor_mean_data > -((code + 1) * 0.1))
                )

        elif grade_by == 'All':
            class_mask = np.ones(np.asarray(pos_data).shape, dtype=bool)

        drought_arrays = {
            'POS drought normal diff': pos_data,
            'SOS drought normal diff': sos_data,
            'Cor drought normal diff': cor_data,
            'SM drought normal diff': sm_data,
            'VPD drought normal diff': vpd_data,
            'Ta drought normal diff': ta_data,
            'Pre drought normal diff': pre_data,
            'Srad drought normal diff': srad_data,
        }
        # wet_arrays = {
        #     'POS wet normal diff': pos_diff_wet_normal,
        #     'SOS wet normal diff': sos_diff_wet_normal,
        #     'Cor wet normal diff': cor_diff_wet_normal,
        #     'SM wet normal diff': sm_diff_wet_normal,
        #     'VPD wet normal diff': vpd_diff_wet_normal,
        #     'Ta wet normal diff': ta_diff_wet_normal,
        #     'Pre wet normal diff': pre_diff_wet_normal,
        #     'Srad wet normal diff': srad_diff_wet_normal,
        # }

        ########## 空间尺度
        ## StandardizationPrepare data：drought/wet/normal year diff ###
        # standardized_data_drought = pd.DataFrame({
        #     'POS drought normal diff': pos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SOS drought normal diff': sos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Cor drought normal diff': cor_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SM drought normal diff': sm_diff_drought_normal.flatten()[valid_indices],
        #
        #     'VPD drought normal diff': vpd_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Ta drought normal diff': ta_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Pre drought normal diff': pre_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Srad drought normal diff': srad_diff_drought_normal.flatten()[valid_indices]
        # })
        standardized_data_drought = _make_pixel_dataframe(
            drought_arrays, spei_grid_group, class_mask
        )
        # standardized_data_wet = _make_pixel_dataframe(
        #     wet_arrays, spei_grid_group, class_mask
        # )

        # standardized_data_corhigh_corlow= pd.DataFrame({
        #     'POS corhigh corlow diff': pos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SOS corhigh corlow diff': sos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Cor corhigh corlow diff': cor_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SM corhigh corlow diff': sm_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'VPD corhigh corlow diff': vpd_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Ta corhigh corlow diff': ta_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Pre corhigh corlow diff': pre_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Srad corhigh corlow diff': srad_diff_corhigh_corlow.flatten()[valid_indices]
        # })

        # print(f'standardized_data drought的有效data量：{len(standardized_data_drought.dropna())}')
        # print(f'standardized_data wet的有效data量：{len(standardized_data_wet.dropna())}')

        # LONG_CLIMATE: RF表同时检查全部差值和背景的完整性。
        # 聚合版本先在共同有效11km像元上筛选，再对所有列一起求55km均值。
        rf_data_drought = _make_pixel_dataframe(
            {**drought_arrays, **long_arrays, **category_arrays}, spei_grid_group, class_mask)
        # rf_data_wet = _make_pixel_dataframe(
        #     {**wet_arrays, **long_arrays, **category_arrays}, spei_grid_group, class_mask)

        standardized_data_drought = standardized_data_drought.dropna().reset_index()
        standardized_data_wet = standardized_data_wet.dropna().reset_index()



        # standardized_data_corhigh_corlow = standardized_data_corhigh_corlow.dropna().reset_index()

        # print('standardized_data_drought:\n', standardized_data_drought.iloc[:20, :])
        # print('standardized_data_wet:\n', standardized_data_wet.iloc[:20, :])

        # sys.exit()

        ######## Partial correlation & ML ###########
        def seq_pcorr(df, y, x_list):
            """
            Perform sequential partial correlation analysis.

            Parameters:
            - df (DataFrame): Input DataFrame containing the variables.
            - y (str): Dependent variable.
            - x_list (list): List of independent variables.

            Returns:
            - result_df (DataFrame): DataFrame containing the sequential partial correlation results.
            """
            res_list = []
            # print('df type:', df.dtypes)
            # print('y:', y)
            # print('x_list:', x_list)
            for var in x_list:
                # Perform partial correlation for each variable with other variables
                res = pg.partial_corr(df, y=y, x=var, covar=[item for item in x_list if item != var])
                res['var'] = var
                res_list.append(res)
            result_df = pd.concat(res_list, axis=0)
            return result_df

        results_dict = {}  # 用于存储每个y的结果

        for y in y_list:


            ### ============= 1. 偏相关分析 ============== ###
            print('Start partial analyze!')
            # if y == 'POS mean' :
            #     # x_list = ['SOS mean', 'SOS std', 'Cor mean', 'Cor std',
            #     #           'SM mean', 'SM std', 'VPD mean', 'VPD std', 'Ta mean', 'Ta std', 'Pre mean', 'Pre std', 'Srad mean', 'Srad std']
            #     x_list = ['SOS mean', 'Cor mean',
            #               'SM mean', 'VPD mean', 'Ta mean', 'Pre mean', 'Srad mean']
            #     df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_drought
            #
            # elif y == 'POS std':
            #     x_list = ['SOS std', 'Cor std',
            #               'SM std', 'VPD std', 'Ta std', 'Pre std', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data
            #
            #
            # elif y == 'POS slope':
            #     x_list = ['SOS slope', 'SOS std',
            #               'Cor slope', 'Cor std',
            #               'SM slope', 'SM std', 'VPD slope', 'VPD std',
            #               'Ta slope', 'Ta std', 'Pre slope', 'Pre std',
            #               'Srad slope', 'Srad std']
            #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = outlier_standardized_data

            if y == 'POS drought normal diff':
                x_list = ['SOS drought normal diff',
                          'Cor drought normal diff',
                          'SM drought normal diff',
                          'VPD drought normal diff',
                          'Ta drought normal diff',
                          'Pre drought normal diff',
                          'Srad drought normal diff']
                df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val']].reset_index(
                    drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            elif y == 'POS wet normal diff':
                x_list = ['SOS wet normal diff',
                          'Cor wet normal diff',
                          'SM wet normal diff',
                          'VPD wet normal diff',
                          'Ta wet normal diff',
                          'Pre wet normal diff',
                          'Srad wet normal diff']
                df_pcorr = seq_pcorr(standardized_data_wet, y, x_list)[['var', 'r', 'p-val']].reset_index(
                    drop=True).rename(
                    columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

            # elif y == 'POS corhigh corlow diff':
            #     x_list = ['SOS corhigh corlow diff',
            #               'Cor corhigh corlow diff',
            #               'SM corhigh corlow diff',
            #               'VPD corhigh corlow diff',
            #               'Ta corhigh corlow diff',
            #               'Pre corhigh corlow diff',
            #               'Srad corhigh corlow diff']
            #     df_pcorr = seq_pcorr(standardized_data_corhigh_corlow, y, x_list)[['var', 'r', 'p-val']].reset_index(drop=True).rename(
            #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
            #     ml_data = standardized_data_corhigh_corlow

            df_pcorr = df_pcorr.rename(columns={'var': 'feature'})
            df_pcorr['Significance'] = ''

            for x in x_list:
                p_value = df_pcorr[df_pcorr['feature'] == x]['p_pcorr'].iloc[0]
                if p_value < 0.01:
                    significance = '**'  # p < 0.01 标记为 **
                elif p_value < 0.05:
                    significance = '*'  # 0.01 <= p < 0.05 标记为 *
                else:
                    significance = ''  # p >= 0.05 不标记

                df_pcorr.loc[df_pcorr['feature'] == x, 'Significance'] = significance

            df_pcorr['Abs_Coefficient'] = df_pcorr['r_pcorr'].abs()

            print(f"{type} df_pcorr:\n", df_pcorr)

            # sys.exit()
            # # ==================== VIF计算 ====================
            # from statsmodels.stats.outliers_influence import variance_inflation_factor
            #
            # # 只取干旱情景的自变量，并清除缺失值和无穷值
            # vif_df = (
            #     standardized_data_drought.loc[:, x_list]
            #     .replace([np.inf, -np.inf], np.nan)
            #     .dropna()
            #     .astype(np.float64)
            # )
            #
            # if len(vif_df) <= len(x_list):
            #     raise ValueError(
            #         f"计算 VIF 的有效样本数不足：{len(vif_df)} 行，"
            #         f"自变量有 {len(x_list)} 个。"
            #     )
            #
            # constant_features = [
            #     col for col in x_list
            #     if vif_df[col].nunique() <= 1
            # ]
            # if constant_features:
            #     raise ValueError(f"以下自变量是常数，无法计算 VIF：{constant_features}")
            #
            # # 转成 NumPy 后手动添加截距列，确保第 0 列是常数项
            # X = vif_df.to_numpy(dtype=np.float64)
            # vif_design = np.column_stack([np.ones(X.shape[0]), X])
            #
            # vif_results = []
            # for i, feature in enumerate(x_list, start=1):
            #     vif_value = variance_inflation_factor(vif_design, i)
            #     vif_results.append({
            #         "feature": feature,
            #         "VIF": vif_value,
            #     })
            #
            # vif_df_result = pd.DataFrame(vif_results)
            #
            # df_pcorr = df_pcorr.merge(vif_df_result, on="feature", how="left")
            # print("df_pcorr:\n", df_pcorr)
            #
            # def plot_posdiff_relationships(
            #         df,
            #         y_col,
            #         comparison_name,
            # ):
            #     # 除目标POS diff之外的变量作为自变量
            #     x_cols = [
            #         col for col in df.columns
            #         if col != y_col and col != "index"
            #     ]
            #
            #     # 去掉缺失值和无穷值
            #     plot_df = (
            #         df[[y_col] + x_cols]
            #         .replace([np.inf, -np.inf], np.nan)
            #         .dropna()
            #     )
            #
            #     if plot_df.empty:
            #         print(f"{comparison_name} 没有可用于绘图的有效data。")
            #         return
            #
            #     ncols = 3
            #     nrows = int(np.ceil(len(x_cols) / ncols))
            #
            #     # 画布缩小；字号通过下面各项单独调大
            #     fig, axes = plt.subplots(
            #         nrows,
            #         ncols,
            #         figsize=(11, 3.2 * nrows),
            #         constrained_layout=True,
            #     )
            #     axes = np.asarray(axes).reshape(-1)
            #
            #     last_hexbin = None
            #
            #     for ax_i, x_col in enumerate(x_cols):
            #         ax = axes[ax_i]
            #
            #         x = plot_df[x_col].to_numpy(dtype=float)
            #         y = plot_df[y_col].to_numpy(dtype=float)
            #
            #         # 六边形密度图，适合大量像元
            #         last_hexbin = ax.hexbin(
            #             x,
            #             y,
            #             gridsize=45,
            #             mincnt=1,
            #             bins="log",
            #             cmap="viridis",
            #         )
            #
            #         # Pearson相关系数和线性趋势线
            #         if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
            #             r = np.corrcoef(x, y)[0, 1]
            #
            #             slope, intercept = np.polyfit(x, y, 1)
            #             x_line = np.linspace(np.min(x), np.max(x), 100)
            #             y_line = slope * x_line + intercept
            #
            #             ax.plot(
            #                 x_line,
            #                 y_line,
            #                 color="red",
            #                 linewidth=2.2,
            #                 label="Linear trend",
            #                 zorder=5,
            #             )
            #             ax.legend(fontsize=10, frameon=True)
            #         else:
            #             r = np.nan
            #
            #         ax.set_title(
            #             f"{x_col}\nPearson r = {r:.2f}",
            #             fontsize=12,
            #         )
            #         ax.set_xlabel(x_col, fontsize=11)
            #         ax.set_ylabel(y_col, fontsize=11)
            #         ax.tick_params(axis="both", labelsize=10)
            #
            #         ax.axhline(
            #             0,
            #             color="gray",
            #             linewidth=0.8,
            #             linestyle="--",
            #             zorder=3,
            #         )
            #         ax.axvline(
            #             0,
            #             color="gray",
            #             linewidth=0.8,
            #             linestyle="--",
            #             zorder=3,
            #         )
            #         ax.grid(alpha=0.2)
            #
            #     # 隐藏未使用的空子图
            #     for ax in axes[len(x_cols):]:
            #         ax.set_visible(False)
            #
            #     fig.suptitle(
            #         f"{comparison_name}: variables vs {y_col}",
            #         fontsize=15,
            #     )
            #
            #     if last_hexbin is not None:
            #         cbar = fig.colorbar(
            #             last_hexbin,
            #             ax=axes[:len(x_cols)].tolist(),
            #             label="Pixel count (log scale)",
            #             shrink=0.85,
            #         )
            #         cbar.ax.tick_params(labelsize=10)
            #         cbar.set_label("Pixel count (log scale)", fontsize=11)
            #
            #     print(f"有效像元数：{len(plot_df)}")
            #     plt.show()
            #
            # plot_posdiff_relationships(
            #     standardized_data_drought,
            #     y_col="POS drought normal diff",
            #     comparison_name=f"{grade_by}_{type}_Drought_vs_Normal"
            # )
            #
            # import statsmodels.api as sm
            # from scipy.stats import pearsonr
            #
            # x = "VPD drought normal diff"
            # controls = [v for v in x_list if v != x]
            # cols = ["POS drought normal diff", x] + controls
            # d = standardized_data_drought[cols].replace(
            #     [np.inf, -np.inf], np.nan
            # ).dropna()
            #
            # Z = sm.add_constant(d[controls])
            # resid_y = sm.OLS(d["POS drought normal diff"], Z).fit().resid
            # resid_x = sm.OLS(d[x], Z).fit().resid
            #
            # print("残差相关:", pearsonr(resid_x, resid_y))
            # print("Pingouin 偏相关:",
            #       pg.partial_corr(data=d, y="POS drought normal diff",
            #                       x=x, covar=controls))
            #
            # sys.exit()



            ##### ================= 2. ML =================== #######
            if y == 'POS drought normal diff':
                ml_data = standardized_data_drought
            elif y == 'POS wet normal diff':
                ml_data = standardized_data_wet
            # elif y == 'POS corhigh corlow diff':
            #     ml_data = standardized_data_corhigh_corlow

            # LONG_CLIMATE: 偏相关列表与RF列表分开；ML使用7差值+5气候态+类别独热编码。
            pcorr_x_list = x_list.copy()
            if ML_model_in_spatio in ('RF', 'XGBoost'):
                # CATEGORICAL_BACKGROUND: 7差值+5气候态+4植被列+6AI列。
                x_list = pcorr_x_list + long_cols + category_cols
                ml_data = rf_data_drought if y == 'POS drought normal diff' else rf_data_wet
            print(f'{type} {y}: model features={x_list}; valid rows={len(ml_data)}')


            X = ml_data[x_list]
            Y = ml_data[y]

            if ML_model_in_spatio == 'RF':
                # RF 外层拆分在每个 SPEI 网格内按 POS 的 P25/P75 分层抽样。
                rf_split_data = (
                    ml_data[x_list + [y, 'SPEI_group']]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .reset_index(drop=True)
                )
                if len(rf_split_data) < 10:
                    raise ValueError(
                        f"RF 有效 11 km 样本仅 {len(rf_split_data)} 个，"
                        "不足以进行稳定的训练/测试划分。"
                    )
                train_df, test_df = _split_rf_within_grid_by_pos_quantiles(
                    rf_split_data,
                    target=y,
                    test_fraction=test_size,
                    random_state=42,
                )
                x_train, y_train = train_df[x_list], train_df[y]
                x_test, y_test = test_df[x_list], test_df[y]
                print(
                    f"RF 分层拆分完成：训练像元={len(train_df)}，"
                    f"测试像元={len(test_df)}；训练网格="
                    f"{train_df['SPEI_group'].nunique()}，测试网格="
                    f"{test_df['SPEI_group'].nunique()}"
                )
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    X, Y, test_size=test_size, random_state=42
                )
                print("test split done")

            def calculate_shap(xgb_model, x_train, y_train, x_test, y_test):

                # Step 1: Calculate MSE and R2 for train and test data
                mse_train = mean_squared_error(y_train, xgb_model.predict(x_train))
                r2_train = xgb_model.score(x_train, y_train)
                print(f'Train R2: {r2_train}, MSE: {mse_train}')

                y_train_pred = xgb_model.predict(x_test)
                r2_test = r2_score(y_test, y_train_pred)
                mse_test = mean_squared_error(y_test, y_train_pred)
                print(f'Test R2: {r2_test}, MSE: {mse_test}')

                # Step 2: Calculate SHAP values on a reproducible training subset
                # 可按内存/绘图速度调整；若训练样本少于此数则全量计算。
                if grade_by == 'All':
                    max_shap_samples = 10000
                else:
                    max_shap_samples = 5000
                n_shap = min(max_shap_samples, len(x_train))
                x_shap = x_train.sample(n=n_shap, random_state=42).reset_index(drop=True)
                print(f"SHAP 使用训练样本：{n_shap}/{len(x_train)}")

                explainer = shap.TreeExplainer(xgb_model)
                print("SHAP explainer done")
                shap_values = explainer(x_shap, check_additivity=False)
                print("SHAP values calculation done")

                shap_values_df = pd.concat(
                    [x_shap,
                     pd.DataFrame(shap_values.values,
                                  columns=[f'shap_{c}' for c in x_shap.columns])],
                    axis=1
                )

                # Step 3: Consolidate feature importance and metrics
                df_xgb_model = pd.DataFrame(xgb_model.feature_importances_, index=x_list,
                                            columns=["importance"]).reset_index().rename(
                    columns={'index': 'var'})
                print("Feature importances calculated")
                print(df_xgb_model)

                df_xgb_model["model_mse"] = mse_train
                df_xgb_model["model_r2"] = r2_train
                df_xgb_model["model_test_mse"] = mse_test
                df_xgb_model["model_test_r2"] = r2_test

                # Step 4: Calculate mean absolute SHAP values
                mean_abs_shap_values = abs(shap_values.values).mean(0)
                print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                df_shap = pd.DataFrame([x_list, mean_abs_shap_values], index=["var", "shapvalue"]).T
                print("SHAP values calculated")

                pos_range = y_train.max() - y_train.min()
                shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                print("POS_mean spatial range (days):", pos_range)
                print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                # Step 5: Merge SHAP values and feature importance
                df_merge = pd.merge(df_shap, df_xgb_model, on="var")
                # print('df_merge1:\n', df_merge)
                df_merge = df_merge.rename(columns={'var': 'feature'})
                # df_merge['importance'] = df_merge['importance'] / df_merge['importance'].sum()  # Normalize feature importance
                # df_merge['shapvalue'] = df_merge['shapvalue'] / df_merge['shapvalue'].sum()  # Normalize SHAP importance
                # df_merge['mean'] = (df_merge['importance'] + df_merge['shapvalue']) / 2  # Calculate mean importance

                print('df_merge:\n', df_merge)

                # # Step 6: Sort features by SHAP value
                # df_merge = df_merge.sort_values(by='shapvalue', ascending=False)

                # CATEGORICAL_BACKGROUND: 编码列明细单独保存，图表汇总为14个因素。
                _export_encoded_background(df_merge, grade_by, type, y, 'Partial_ML_calculate_and_plot2')
                shap_values_df, df_merge = _group_categorical_shap(shap_values_df, df_merge)
                return shap_values_df, df_merge, r2_test

            # sys.exit()

            ############################### XGBoost ########################################
            if ML_model_in_spatio == 'XGBoost':
                ## K-fold和使用网格搜索寻找最佳参数 ##
                xgb_model = xgb.XGBRegressor(random_state=42)
                # 定义参数网格
                param_grid = {'n_estimators': [100, 150, 200, 250, 300, 350],
                              'max_depth': [5, 10, 15],
                              'learning_rate': [0.05, 0.1],
                              'subsample': [0.7, 0.8, 0.9, 1],
                              'colsample_bytree': [0.7, 0.8, 0.9, 1]}
                # # 定义 K 折交叉验证 (K-Fold)
                kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                # 使用网格搜索寻找最佳参数
                grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                           verbose=10, n_jobs=15, error_score='raise')

                # 拟合模型
                grid_search.fit(x_train, y_train)

                print(f'XGboost model done')

                # Output最优参数和最优得分
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                Best_parameters.to_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                #
                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # xgboost = grid_search.best_estimator_

                ######################################################
                # #
                Best_parameters = pd.read_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                colsample_bytree = float(Best_parameters['colsample_bytree'][0])
                subsample = float(Best_parameters['subsample'][0])
                learning_rate = float(Best_parameters['learning_rate'][0])

                xgboost = XGBRegressor(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       colsample_bytree=colsample_bytree,  #每次分裂会考虑所有特征中 80% 的特征子集
                                       subsample=subsample,  # 每棵树的随机采样比例
                                       learning_rate=learning_rate,
                                       random_state=42,
                                       n_jobs=20)

                xgboost.fit(x_train, y_train)
                print(f"xgb_model:", xgboost)

                shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(xgboost, x_train, y_train,
                                                                                            x_test, y_test)


            if ML_model_in_spatio == 'XGBoost':
                shap_values_model = shap_values_df
                shap_importance_model = ModelImportance_and_ShapValueMean

            ############################ RF #########################################
            elif ML_model_in_spatio == 'RF':
                ## K-fold和使用网格搜索寻找最佳参数 ##
                rf_model = RandomForestRegressor(random_state=42, n_jobs=15)
                # if grade_by == 'All':
                # 定义参数网格
                param_grid = {
                    'n_estimators': [500],#, 600, 700, 800, 900, 1000],
                    'max_depth': [10],
                    'min_samples_split': [10],
                    'min_samples_leaf': [10],
                    'max_features': [0.7]  # 如果是回归任务，建议包含 0.5, 0.7, 1.0
                }
                    # param_grid = {
                    #     'n_estimators': [200, 300, 500, 800],
                    #     'max_depth': [None, 15, 20, 25],          # 上限 25，不要 30
                    #     'min_samples_split': [5, 10, 20, 40],
                    #     'min_samples_leaf': [5, 10, 20, 50],      # 下限 5，不要 1、2
                    #     'max_features': ['sqrt', 0.3, 0.5],       # 上限 0.5，不要 0.7、0.8
                    # }
                # else:
                #     param_grid = {
                #         'n_estimators': [150, 200, 250, 300],
                #         'max_depth': [15, 20, 30],
                #         'min_samples_split': [2, 5, 10],
                #         'min_samples_leaf': [1, 2, 4],
                #         'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.8]
                #     }
                # # 定义 K 折交叉验证 (K-Fold)
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                # 使用网格搜索寻找最佳参数
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                           verbose=10, n_jobs=15, error_score='raise')
                # 拟合模型
                grid_search.fit(x_train, y_train)
                # grid_search.fit(X, Y)

                print(f'{type} RF model done')

                # Output最优参数和最优得分
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best R2 score: {grid_search.best_score_}")

                Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                print(f'Best_parameters:{Best_parameters}')
                Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                # if analyze_by == 'All':
                Best_parameters.to_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif analyze_by != 'All':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPara_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

                # if grade_by == 'Veg':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                print(f'Best_parameters done')

                # ## 使用最优参数训练模型 ##
                # rf = grid_search.best_estimator_

                ######################################################
                #
                # if analyze_by == 'All':
                Best_parameters = pd.read_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif analyze_by != 'All':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

                # if grade_by == 'Veg':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'AI':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'Cor mean':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif grade_by == 'All':
                #     Best_parameters = pd.read_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')

                n_estimators = int(Best_parameters['n_estimators'][0])
                max_depth = int(Best_parameters['max_depth'][0])
                min_samples_split = int(Best_parameters['min_samples_split'][0])
                min_samples_leaf = int(Best_parameters['min_samples_leaf'][0])
                max_features = Best_parameters['max_features'][0]
                # 如果保存时是数字，才需要转换
                if isinstance(max_features, str):
                    max_features = max_features  # 保持字符串
                else:
                    max_features = float(max_features)

                # if data_type == 'MODIS_GPP_8d' and lag_day ==0 and spei_strength == -1:
                #     # 最优
                # n_estimators = 400
                # max_depth = 40
                # min_samples_split = 5
                # min_samples_leaf = 5
                # max_features = 0.7
                # elif data_type == 'MODIS_GPP_8d' and lag_day ==0 and spei_strength == -1.5:
                #     n_estimators = 200
                #     max_depth = 40
                #     min_samples_split = 2  # 每次分裂会考虑所有特征中 80% 的特征子集
                #     min_samples_leaf = 1 # 每棵树的随机采样比例
                #     max_features = 0.8
                # elif data_type == 'MCD12Q2' and lag_day ==0 and spei_strength == -1:
                #     n_estimators = 400
                #     max_depth = 20
                #     min_samples_split = 2  # 每次分裂会考虑所有特征中 80% 的特征子集
                #     min_samples_leaf = 1  # 每棵树的随机采样比例
                #     max_features = 0.3

                # 转成一维数组，确保 OOB 误差和训练样本一一对应
                y_train_array = np.asarray(y_train).reshape(-1)

                # ==================== 模型1：使用全部训练样本 ====================
                rf_model1 = RandomForestRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,  # 每次分裂会考虑所有特征中 80% 的特征子集
                                           min_samples_leaf=min_samples_leaf,  # 每棵树的随机采样比例
                                           max_features=max_features,
                                           bootstrap=True,
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=20)

                rf_model1.fit(x_train, y_train)
                print(f"rf_model1:", rf_model1)
                print('Train R²:', rf_model1.score(x_train, y_train))
                print('Test  R²:', rf_model1.score(x_test, y_test))
                print('OOB   R²:', rf_model1.oob_score_)
                print('feature importances:', rf_model1.feature_importances_)

                # 每条训练样本的折外预测误差
                oob_pred_model1 = np.asarray(rf_model1.oob_prediction_).reshape(-1)

                valid_oob = (np.isfinite(y_train_array) & np.isfinite(oob_pred_model1))

                oob_abs_error_model1 = np.abs(
                    y_train_array - oob_pred_model1
                )

                oob_mean_model1 = float(np.mean(oob_abs_error_model1))
                oob_median_model1 = float(np.median(oob_abs_error_model1))
                oob_threshold = float(np.quantile(oob_abs_error_model1, 0.90))

                # 模型1 SHAP：使用原训练集和同一独立测试集
                shap_values_model1, shap_importance_model1, shap_r2_model1 = calculate_shap(
                    rf_model1,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )

                Best_parameters['r2_test'] = shap_r2_model1

                shap_importance_model = shap_importance_model1
                shap_values_model = shap_values_model1

                if shap_r2_model1 < 0.4:


                    # ==================== 根据模型1 OOB误差筛样本 ====================
                    # 误差高于90%分位数的训练样本被剔除
                    keep_mask = oob_abs_error_model1 <= oob_threshold
                    n_removed = int(np.sum(~keep_mask))
                    n_kept = int(np.sum(keep_mask))

                    print(f"模型1训练样本数：{len(keep_mask)}")
                    print(f"剔除样本数：{n_removed}")
                    print(f"保留样本数：{n_kept}")

                    # 同时兼容 pandas DataFrame/Series 和 NumPy 数组
                    if hasattr(x_train, "iloc"):
                        x_train_model2 = x_train.iloc[np.flatnonzero(keep_mask)].copy()
                    else:
                        x_train_model2 = x_train[keep_mask]

                    if hasattr(y_train, "iloc"):
                        y_train_model2 = y_train.iloc[np.flatnonzero(keep_mask)].copy()
                    else:
                        y_train_model2 = y_train[keep_mask]

                    # ==================== 模型2：用筛选后的训练样本 ====================
                    rf_model2 = RandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,  # 每次分裂会考虑所有特征中 80% 的特征子集
                                               min_samples_leaf=min_samples_leaf,  # 每棵树的随机采样比例
                                               max_features=max_features,
                                               bootstrap=True,
                                               oob_score=True,
                                               random_state=42,
                                               n_jobs=20)
                    rf_model2.fit(x_train_model2, y_train_model2)
                    print(f"rf_model2:", rf_model2)
                    print('Train R²:', rf_model2.score(x_train, y_train))
                    print('Test  R²:', rf_model2.score(x_test, y_test))
                    print('OOB   R²:', rf_model2.oob_score_)
                    print('feature importances:', rf_model2.feature_importances_)

                    y_train_model2_array = np.asarray(y_train_model2).reshape(-1)
                    oob_pred_model2 = np.asarray(
                        rf_model2.oob_prediction_
                    ).reshape(-1)

                    valid_oob_model2 = (
                            np.isfinite(y_train_model2_array)
                            & np.isfinite(oob_pred_model2)
                    )

                    if not np.all(valid_oob_model2):
                        raise ValueError("模型2存在无效OOB预测，请检查data或增加树的数量。")

                    oob_abs_error_model2 = np.abs(
                        y_train_model2_array - oob_pred_model2
                    )

                    oob_mean_model2 = float(np.mean(oob_abs_error_model2))
                    oob_median_model2 = float(np.median(oob_abs_error_model2))

                    print(f"模型2 OOB绝对误差均值：{oob_mean_model2:.4f}")
                    print(f"模型2 OOB绝对误差中位数：{oob_median_model2:.4f}")

                    # 模型2 SHAP：训练集是筛选后的样本，测试集仍与模型1相同
                    shap_values_model2, shap_importance_model2, shap_r2_model2 = calculate_shap(
                        rf_model2,
                        x_train_model2,
                        y_train_model2,
                        x_test,
                        y_test,
                    )

                    # 汇总两个模型的 OOB 与测试集结果
                    summary = pd.DataFrame([
                        {
                            "model": "model1",
                            "training_samples": len(y_train_array),
                            "removed_samples": 0,
                            "oob_R2": rf_model1.oob_score_,
                            "oob_MAE_mean": oob_mean_model1,
                            "oob_MAE_median": oob_median_model1,
                            "test_R2": rf_model1.score(x_test, y_test),
                            "SHAP_test_R2": shap_r2_model1,
                        },
                        {
                            "model": "model2",
                            "training_samples": n_kept,
                            "removed_samples": n_removed,
                            "oob_R2": rf_model2.oob_score_,
                            "oob_MAE_mean": oob_mean_model2,
                            "oob_MAE_median": oob_median_model2,
                            "test_R2": rf_model2.score(x_test, y_test),
                            "SHAP_test_R2": shap_r2_model2,
                        },
                    ])

                    print('summary\n'
                          rf'{summary}')

                    summary.to_csv(
                        fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TS{test_size}_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_Summary_LongClimate_CatBackground_5Climate.csv')

                    Best_parameters['r2_test'] = shap_r2_model2

                    shap_importance_model = shap_importance_model2
                    shap_values_model = shap_values_model2

            # if analyze_by == 'All':
            Best_parameters.to_csv(
                fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
            # elif analyze_by != 'All':
            #     Best_parameters.to_csv(
            #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

            # LONG_CLIMATE: 保存偏相关及RF/SHAP结果，文件名含函数版本。
            _export_long_climate_results(df_pcorr, shap_importance_model,
                                         grade_by, type, y, 'Partial_ML_calculate_and_plot2')

            # 将每个y的结果存储到字典中
            results_dict[y] = {
                'df_pcorr': df_pcorr,
                'importance_and_shap_values_mean': shap_importance_model,
                'shap_values_df': shap_values_model,
                # 训练用22列，图表用12个连续因素+2个类别组。
                'x_list': shap_importance_model['feature'].tolist(),
                'y': y
            }

            # # srad_std_shap = shap_values_df.values[:, -1]
            # print(f"SHAP值Statistics:")
            # print(f"  最小值: {np.nanmin(shap_values_df.values):.1f}")
            # print(f"  最大值: {np.nanmax(shap_values_df.values):.1f}")
            # print(f"  平均值: {np.nanmean(shap_values_df.values):.1f}")
            #
            # # Statistics-10到10范围内的占比
            # in_range_mask = (shap_values_df >= -1) & (shap_values_df <= 1) & (~np.isnan(shap_values_df))
            # total_valid = np.sum(~np.isnan(shap_values_df))
            # in_range_count = np.sum(in_range_mask)

            # print(f"[-1, 1]区间Statistics:\n")
            # print(f"  区间内数量: {in_range_count}")
            # print(f"  有效总数: {total_valid}")
            # print(f"  占比: {(in_range_count / total_valid) * 100:.2f}%")
            #
            # sys.exit()

        ######## ================= 3. plot ====================== #######

        #### 驱动分析结果图Plot ####
        for y, results in results_dict.items():
            # 使用存储的结果进行绘图
            df_pcorr = results['df_pcorr']
            ModelImportance_and_ShapValueMean = results['importance_and_shap_values_mean']
            shap_values_df = results['shap_values_df']
            x_list = results['x_list']

            # ========== 1. 定义“唯一的 y 轴顺序”（以 SHAP / importance 为准） ==========
            feature_order = ModelImportance_and_ShapValueMean['feature'].tolist()
            # feature_order = ['SOS drought normal diff', 'Cor drought normal diff', 'SM drought normal diff', 'VPD drought normal diff',
            #                  'Ta drought normal diff', 'Pre drought normal diff', 'Srad drought normal diff']
            print(f'feature_order:{feature_order}')

            # # ========== 2. Create the figure ==========
            # fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=False)
            # ax1, ax2, ax3 = axes
            # plt.subplots_adjust(wspace=0.05)

            # ========== 3. 子图 1：Partial correlation ==========
            # LONG_CLIMATE: 背景行只作控制说明，不Plot伪造的偏相关条形。
            _plot_long_climate_pcorr(ax1, df_pcorr, feature_order,
                                     False)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'a'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            #     elif type == 'Arid':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'd'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'g'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'j'
            #     ytitle = 1.07
            # ax1.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax1.set_xlabel(f'Partial correlation')
            else:
                ax1.set_xlabel('')

            if data_type == 'MODIS_GPP_8d':
                ax1.set_xlim(0, 0.6)
                ax1.set_xticks(np.arange(0, 0.601, 0.2))
                ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.601, 0.2)])
            elif data_type == 'MCD12Q2':
                ax1.set_xlim(0, 0.8)
                ax1.set_xticks(np.arange(0, 0.81, 0.2))
                ax1.set_xticklabels([f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(0, 0.81, 0.2)])

            pcorr_bound = min(1.0, max(ax1.get_xlim()[1], float(df_pcorr['Abs_Coefficient'].max()) * 1.08))
            ax1.set_xlim(0, pcorr_bound)
            ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
            ax1.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax1.set_ylabel('')
            # LONG_CLIMATE: y标签已由_plot_long_climate_pcorr按实际变量数设置。
            # ax1.set_yticklabels(['Cor', 'SM', 'VPD', 'Ta', 'Pre', 'Srad'])
            ax1.tick_params(axis='y', length=0)

            # legend_handles = [
            #     mpatches.Patch(color='#4393c3', label='Negative', alpha=0.8),
            #     mpatches.Patch(color='#d6604d', label='Positive', alpha=0.8)
            # ]
            # ax1.legend(handles=legend_handles, frameon=False, fontsize=8, loc='lower right')
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#4393c3',
                       markeredgecolor='none', markersize=6, label='Negative', alpha=0.8),
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#d6604d',
                       markeredgecolor='none', markersize=6, label='Positive', alpha=0.8)
            ]

            ax1.legend(handles=legend_handles, frameon=False, loc='lower right', fontsize=8,
                       handletextpad=0.5, handlelength=1.0)  # 缩短句柄长度使其看起来更像纯方块

            # ========== 4. 子图 2：ML feature importance ==========
            importance_plot = ModelImportance_and_ShapValueMean.copy()

            importance_plot['feature'] = pd.Categorical(
                importance_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            importance_plot = importance_plot.sort_values('feature')
            print(f'importance_plot:{importance_plot}')

            sns.barplot(
                x='importance',
                y='feature',
                data=importance_plot,
                color='#4393c3',
                alpha=0.8,
                width=0.4,
                ax=ax2
            )

            ax2.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'b'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'e'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'h'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'k'
            #     ytitle = 1.07
            # ax2.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax2.set_xlabel(f'{ML_model_in_spatio} importance')
            else:
                ax2.set_xlabel('')

            if grade_by == 'All':
                x_min = 0
                x_max = 0.45
                interval = 0.1
            else:
                x_min = 0
                x_max = 0.8
                interval = 0.2
            x_max = max(x_max, float(importance_plot['importance'].max()) * 1.1)
            ax2.set_xlim(x_min, x_max)
            ax2.set_xticks(np.arange(x_min, x_max + 0.01, interval))
            ax2.set_xticklabels(
                [f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(x_min, x_max + 0.01, interval)])

            ax2.set_ylabel('')
            ax2.tick_params(axis='y', length=0, labelright=False, labelleft=False)

            # ========== 5. 子图 3：SHAP beeswarm（关键：用 order 控制顺序） ==========
            plt.sca(ax3)
            ### 1 Shap value sactter
            # —— 构造 SHAP Explanation（按 x_list 顺序）——
            shap_values = shap.Explanation(
                values=shap_values_df[[f'shap_{c}' for c in x_list]].values,
                data=shap_values_df[x_list].values,
                feature_names=x_list
            )

            # —— 将 feature_order 映射为 SHAP 的索引顺序 ——
            order_idx = [x_list.index(f) for f in feature_order if f in x_list]
            print(f'order_idx:{order_idx}')

            shap.plots.beeswarm(
                shap_values,
                order=order_idx,
                max_display=len(order_idx),
                plot_size=None,  # 由外部 ax 控制
                show=False,
                color_bar=False
            )

            # 更改点的color
            new_cmap = plt.get_cmap('coolwarm')

            for coll in ax3.collections:
                coll.set_sizes([5])
                # 将现有的color数组映射到新的 cmap 上
                # SHAP 内部通常将 feature value 映射在 [0, 1] 之间
                coll.set_cmap(new_cmap)

            # 点大小
            for coll in ax3.collections:
                coll.set_sizes([5])

            # x 轴范围
            # if y == 'POS mean':
            #     ax3.set_xlim(-20, 20)
            # elif y == 'POS slope':
            #     ax3.set_xlim(-1, 1)
            # elif y == 'POS std':
            #     ax3.set_xlim(-5, 5)
            # else:
            ax3.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            fp = ax2.get_xticklabels()[0].get_fontproperties()
            shap_bound = max(1.5, float(np.nanmax(np.abs(shap_values.values))) * 1.05)
            ax3.set_xlim(-shap_bound, shap_bound)
            ax3.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))


            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'c'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'f'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'i'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'l'
            #     ytitle = 1.07
            # ax3.set_title(f'({word}) {type}', y=ytitle)
            fp = ax2.xaxis.label.get_fontproperties()
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax3.set_xlabel(f'{ML_model_in_spatio} SHAP value', fontproperties=fp)  #, labelpad=4)
            else:
                ax3.set_xlabel('')
            ax3.tick_params(axis='y', labelleft=False)

            ### 2 Shap abs mean

            ax3_twin = ax3.twiny()
            ax3_twin.set_ylim(ax1.get_ylim())

            shapvalue_mean = ModelImportance_and_ShapValueMean.copy()

            shapvalue_mean['feature'] = pd.Categorical(
                shapvalue_mean['feature'],
                categories=feature_order[::-1],
                ordered=True
            )
            shapvalue_mean = shapvalue_mean.sort_values('feature')
            print(f'shapvalue_mean:{shapvalue_mean}')
            sns.barplot(
                x='shapvalue',
                y='feature',
                data=shapvalue_mean,
                color='gray',
                width=0.4,
                alpha=0.3,
                ax=ax3_twin,
                zorder=0
            )

            # 设置顶层 X 轴的范围和标签
            shap_mean_bound = max(0.3, float(shapvalue_mean['shapvalue'].max()) * 1.1)
            ax3_twin.set_xlim(0, shap_mean_bound)  # 给条形图留出余量
            ax3_twin.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
            ax3_twin.tick_params(axis='x', length=2, pad=1)
            if type in ['All', 'Forest', 'Arid', 'Cor(-0.1~0)']:
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            elif veg_type == 'Grass' and grade_by == 'Veg':
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            else:
                ax3_twin.set_xlabel('')

            # # 隐藏右侧和多余的 Y 轴标签
            # ax3_twin.spines['right'].set_visible(False)
            # ax3.spines['right'].set_visible(False)
            # ax3_twin.get_yaxis().set_visible(False)  # 隐藏 twin 轴的 Y 标签，防止重叠

            # for ax, name in zip([ax2, ax3], ['ax2', 'ax3']):
            #     t = ax.get_xticklabels()[0]
            #     print(name)
            #     print("family:", t.get_fontfamily())
            #     print("size:", t.get_fontsize())
            #     print("weight:", t.get_fontweight())
            #     print("style:", t.get_fontstyle())
            #     print("name:", t.get_fontproperties().get_name())

            ### ================ 6. 子图 4：Subplot 3的color ================ ###
            norm = mpl.colors.Normalize(vmin=0, vmax=1)

            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
                cax=ax4,  # 确保使用预留的 ax4
                orientation='vertical'
            )

            cb.ax.tick_params(
                axis='y',  # 针对 y 轴（Colorbar 的纵向轴）
                length=0,  # 刻度线长度设为 0（隐藏刻度线）
                pad=2  # 标签与色条的间距，可根据需要微调
            )
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label('Feature value', labelpad=-7)
            cb.outline.set_visible(False)  # 去掉 colorbar 的黑色边框显得更高级

            pos = ax4.get_position()

            new_width = pos.width * 0.5

            ax4.set_position([pos.x0 - 0.01, pos.y0, new_width, pos.height])

            # ========== 6. 统一边框风格 ==========
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1)
                    spine.set_edgecolor('black')

            # ---------------------- 统一对齐 y 轴 ----------------------
            # 1. 以 ax1 为基准
            y_min, y_max = ax1.get_ylim()
            y_lim = (min(y_min, y_max), max(y_min, y_max))  # 确保正向

            # 2. 获取 ax1 的 ticks
            yticks = ax1.get_yticks()

            # 3. 循环应用到 ax2 和 ax3
            for ax in [ax3_twin]:
                # 设置相同的 ylim（正向保证）
                ax.set_ylim(y_lim)

                # 设置相同的 ticks
                ax.set_yticks(yticks)
                # ax.invert_yaxis()

                # 可选：保持 tick 标签与 ax1 一致
                # ticklabel 可以使用字符串或数值
                # yticklabels = [str(round(t, 2)) for t in yticks]  # 保留两位小数
                # ax.set_yticklabels(yticklabels)

            # base_ylim = ax1.get_ylim()
            #
            # # 2. 强制覆盖所有相关轴
            # for target_ax in [ax2, ax3, ax3_twin]:
            #     target_ax.set_ylim(base_ylim)
            #
            # # 3. 检查反转状态（关键！）
            # # 如果 ax1 是Regular的（0在下，N在上），而 SHAP 点在 ax3 里还是反的
            # # 那么我们只需要反转 ax3 和它的 twin 轴
            # if ax3.yaxis_inverted() != ax1.yaxis_inverted():
            #     ax3.invert_yaxis()
            #     ax3_twin.invert_yaxis()  # 必须两个一起反转
            # ax3_twin.get_yaxis().set_visible(False)

    # ========== 7. 保存 ==========

    if grade_by in ['Veg', 'AI', 'Cor mean']:
        # plt.tight_layout()
        # if analyze_by == 'All':
        fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor({coupling_method}_lag={lag_day}){months_before_pos}_{grade_by}_{ML_model_in_spatio}_TestSize{test_size}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        # elif analyze_by == 'advance PPT':
        #     fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{cor_test_number}_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        # elif analyze_by == 'delay PPT':
        #     fig_path = rf'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{cor_test_number}_{grade_by}_{ML_model_in_spatio}_TestSize({test_size})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.png'
        plt.savefig(
            fig_path,
            dpi=300, bbox_inches='tight')
        print(f'Fig保存到：{fig_path}')


def Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data,
                                  grade_by, veg_type, ax, rf_max_per_grid=18,
                                  rf_random_state=42, *, sm_long_data=None, vpd_long_data=None, ta_long_data=None, pcorr_control_long=True, vegetation_data=None, ai_class_data=None, pre_long_data=None, srad_long_data=None, cached_repeat=None):
    # LONG_CLIMATE: 默认读取同名全局数组，也支持显式关键字传入。
    long_arrays = _resolve_long_climate(pos_data, sm_long_data, vpd_long_data, ta_long_data, pre_long_data, srad_long_data)
    long_cols = list(long_arrays)

    # CATEGORICAL_BACKGROUND: 两个栅格均为分类，固定独热编码后拼入ML表。
    veg_type_data = globals().get('veg_type_data') if vegetation_data is None else vegetation_data
    ai_type_data = globals().get('ai_type_data') if ai_class_data is None else ai_class_data
    category_arrays = _resolve_categorical_background(pos_data, veg_type_data, ai_type_data)
    category_cols = list(category_arrays)

    os.makedirs(os.path.join(output_partial_max_var_r_tif_path, f'SPEI{spei_length}', str(Basedon)), exist_ok=True)
    for folder in ['All', 'Veg', 'AI', 'Cor mean']:
        os.makedirs(os.path.join(output_partial_max_var_png_path, folder), exist_ok=True)
    var_list = ['POS mean', 'POS slope', 'POS std', 'POS drought normal diff', 'POS wet normal diff',
                'POS corhigh corlow diff',

                'SOS mean', 'SOS slope', 'SOS std', 'SOS drought normal diff', 'SOS wet normal diff',
                'SOS corhigh corlow diff',

                'Cor mean', 'Cor slope', 'Cor std', 'Cor drought normal diff', 'Cor wet normal diff',
                'Cor corhigh corlow diff',

                'SM mean', 'SM slope', 'SM std', 'SM drought normal diff', 'SM wet normal diff',
                'SM corhigh corlow diff',

                'VPD mean', 'VPD slope', 'VPD std', 'VPD drought normal diff', 'VPD wet normal diff',
                'VPD corhigh corlow diff',

                'Ta mean', 'Ta slope', 'Ta std', 'Ta drought normal diff', 'Ta wet normal diff',
                'Ta corhigh corlow diff',

                'Pre mean', 'Pre slope', 'Pre std', 'Pre drought normal diff', 'Pre wet normal diff',
                'Pre corhigh corlow diff',

                'Srad mean', 'Srad slope', 'Srad std', 'Srad drought normal diff', 'Srad wet normal diff',
                'Srad corhigh corlow diff']  ##修改此处顺序可以修改子图一的顺序

    if grade_by != 'All':
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

    ### 分析使用 11 km 像元；SPEI_group 只用于 RF 按 55 km 网格分层抽样

    if grade_by == 'All':
        types = ['All']
        codes = [1]

        fig = ax.figure
        gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                    width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                    height_ratios=[1],  # Reserve the last slot for the colorbar
                                                    hspace=0.3, wspace=0.2)

        ax.axis('off')

    elif grade_by == 'Veg':
        if veg_type == 'All':
            types = ['Forest', 'Shrub', 'Savanna', 'Grass']
            codes = [1, 2, 3, 4]

            fig = plt.figure(figsize=(8.2, 14))
            gs = fig.add_gridspec(4, 4,
                                   width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                   height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                                   hspace=0.3, wspace=0.2)
        elif veg_type == 'Grass':
            types = ['Grass']
            codes = [4]

            fig = ax.figure
            gs_inner = ax.get_subplotspec().subgridspec(1, 4,
                                                        width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                                                        height_ratios=[1],  # Reserve the last slot for the colorbar
                                                        hspace=0.3, wspace=0.2)
            ax.axis('off')

    elif grade_by == 'AI':
        types = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']
        codes = [2, 3, 5, 6]

        fig = plt.figure(figsize=(8.2, 14))
        gs = fig.add_gridspec(4, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    elif grade_by == 'Cor mean':
        types = ['Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']
        codes = [0, 1, 2, 3, 4]

        fig = plt.figure(figsize=(8.2, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[1, 1, 1, 0.1],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)

    def _make_pixel_dataframe(arrays, group_grid, class_mask):
        """按类别筛选完整案例 11 km 像元；不对像元做空间聚合。"""
        group_grid = np.asarray(group_grid)
        class_mask = np.asarray(class_mask, dtype=bool)
        if group_grid.shape != class_mask.shape:
            raise ValueError(
                f"spei_grid_group shape {group_grid.shape} 与类别掩膜 "
                f"shape {class_mask.shape} 不一致。"
            )

        flat_group = group_grid.ravel()
        valid = class_mask.ravel() & np.isfinite(flat_group) & (flat_group >= 0)
        flat_arrays = {}
        for name, array in arrays.items():
            values = np.asarray(array, dtype=float)
            if values.shape != group_grid.shape:
                raise ValueError(
                    f"{name} shape {values.shape} 与 spei_grid_group "
                    f"shape {group_grid.shape} 不一致。"
                )
            values = values.ravel()
            flat_arrays[name] = values
            valid &= np.isfinite(values)

        if not np.any(valid):
            return pd.DataFrame(columns=["SPEI_group", *arrays.keys()])

        return pd.DataFrame({
            "SPEI_group": flat_group[valid].astype(np.int64),
            **{name: values[valid] for name, values in flat_arrays.items()},
        }).reset_index(drop=True)

    def _sample_and_split_rf_within_grid(
            data, target, test_fraction=test_size, max_per_grid=6, random_state=42
    ):
        """
        按每个55 km网格内POS diff的低/中/高三分位层抽样。

        每格最多抽 max_per_grid 个：
        - 少于6个：全部抽取，随机1个作为测试，其余作为训练
        - 6个及以上：随机分层抽6个，再按test_fraction拆分
        """
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction 必须在 0 和 1 之间。")
        if max_per_grid < 2:
            raise ValueError("max_per_grid 至少为2。")

        data = data.reset_index(drop=True).copy()
        rng = np.random.default_rng(random_state)

        train_indices = []
        test_indices = []
        sampled_indices = []
        sampled_strata = np.zeros(3, dtype=int)
        n_singleton_grids = 0

        for _, grid_df in data.groupby("SPEI_group", sort=False):
            indices = grid_df.index.to_numpy()
            n_available_grid = len(indices)

            if n_available_grid == 0:
                continue

            n_take = min(max_per_grid, n_available_grid)
            pos_values = grid_df[target].to_numpy(dtype=float)

            # 该55 km网格内部按POS diff三分位数分低、中、高
            q_low, q_high = np.quantile(pos_values, [1 / 3, 2 / 3])
            labels = np.where(
                pos_values <= q_low,
                0,  # 低
                np.where(pos_values <= q_high, 1, 2),  # 中、高
            )

            strata_indices = [indices[labels == k] for k in range(3)]
            stratum_counts = np.array(
                [len(stratum) for stratum in strata_indices],
                dtype=int,
            )

            # 尽量均衡地从低、中、高层抽取。
            # 某层不足时，其余名额会分配给仍有候选像元的层。
            quotas = np.zeros(3, dtype=int)

            for _ in range(n_take):
                capacity = stratum_counts - quotas
                candidates = np.flatnonzero(capacity > 0)

                # 优先给当前抽样数最少的层分配名额
                min_quota = quotas[candidates].min()
                candidates = candidates[quotas[candidates] == min_quota]
                chosen_stratum = int(rng.choice(candidates))
                quotas[chosen_stratum] += 1

            selected = []
            for k, (stratum, quota) in enumerate(zip(strata_indices, quotas)):
                if quota > 0:
                    chosen = rng.choice(
                        stratum,
                        size=int(quota),
                        replace=False,
                    )
                    selected.extend(chosen.tolist())
                    sampled_strata[k] += int(quota)

            selected = np.asarray(selected, dtype=int)
            rng.shuffle(selected)
            sampled_indices.extend(selected.tolist())

            # 不足max_per_grid个的网格：全部选中，随机1个测试，其余训练。
            # 达到max_per_grid个：抽max_per_grid个后按test_fraction拆分。
            if n_take == 1:
                n_test = 1
                n_singleton_grids += 1
            elif n_take < max_per_grid:
                n_test = 1
            else:
                n_test = min(
                    max(int(round(n_take * test_fraction)), 1),
                    n_take - 1,
                )

            test_indices.extend(selected[:n_test].tolist())
            train_indices.extend(selected[n_test:].tolist())

        if not train_indices or not test_indices:
            raise ValueError(
                "抽样后训练集或测试集为空。请检查各网格有效像元数量。"
            )

        train_df = data.loc[train_indices].reset_index(drop=True)
        test_df = data.loc[test_indices].reset_index(drop=True)

        sample_info = {
            "n_available": len(data),
            "n_sampled": len(sampled_indices),
            "n_unused": len(data) - len(sampled_indices),
            "n_singleton_grids": n_singleton_grids,
            "n_sampled_low": int(sampled_strata[0]),
            "n_sampled_middle": int(sampled_strata[1]),
            "n_sampled_high": int(sampled_strata[2]),
        }

        return train_df, test_df, sample_info

    for i, (type, code) in enumerate(zip(types, codes)):
        if grade_by == 'All':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        elif grade_by == 'Veg' and veg_type == 'Grass':
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax4 = fig.add_subplot(gs_inner[0, 3])
        else:
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            ax3 = fig.add_subplot(gs[i, 2])
            ax4 = fig.add_subplot(gs[i, 3])

        if grade_by == 'Veg':
            class_mask = (veg_type_data == code)

        elif grade_by == 'AI':
            if code == 3:
                class_mask = ((ai_type_data == code) | (ai_type_data == 4))

            else:
                class_mask = (ai_type_data == code)


        elif grade_by == 'Cor mean':
            if code == 4:
                class_mask = (cor_mean_data <= -0.4)
            else:
                class_mask = (
                    (cor_mean_data <= -(code * 0.1))
                    & (cor_mean_data > -((code + 1) * 0.1))
                )

        elif grade_by == 'All':
            class_mask = np.ones(np.asarray(pos_data).shape, dtype=bool)

        drought_arrays = {
            'POS drought normal diff': pos_data,
            'SOS drought normal diff': sos_data,
            'Cor drought normal diff': cor_data,
            'SM drought normal diff': sm_data,
            'VPD drought normal diff': vpd_data,
            'Ta drought normal diff': ta_data,
            'Pre drought normal diff': pre_data,
            'Srad drought normal diff': srad_data,
        }
        # wet_arrays = {
        #     'POS wet normal diff': pos_diff_wet_normal,
        #     'SOS wet normal diff': sos_diff_wet_normal,
        #     'Cor wet normal diff': cor_diff_wet_normal,
        #     'SM wet normal diff': sm_diff_wet_normal,
        #     'VPD wet normal diff': vpd_diff_wet_normal,
        #     'Ta wet normal diff': ta_diff_wet_normal,
        #     'Pre wet normal diff': pre_diff_wet_normal,
        #     'Srad wet normal diff': srad_diff_wet_normal,
        # }

        ########## 空间尺度
        ## StandardizationPrepare data：drought/wet/normal year diff ###
        # standardized_data_drought = pd.DataFrame({
        #     'POS drought normal diff': pos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SOS drought normal diff': sos_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Cor drought normal diff': cor_diff_drought_normal.flatten()[valid_indices],
        #
        #     'SM drought normal diff': sm_diff_drought_normal.flatten()[valid_indices],
        #
        #     'VPD drought normal diff': vpd_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Ta drought normal diff': ta_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Pre drought normal diff': pre_diff_drought_normal.flatten()[valid_indices],
        #
        #     'Srad drought normal diff': srad_diff_drought_normal.flatten()[valid_indices]
        # })
        # LONG_CLIMATE: 同一像元position加入5个背景列；干旱/湿润分别取完整案例。
        drought_arrays.update(long_arrays)
        # wet_arrays.update(long_arrays)
        standardized_data_drought = _make_pixel_dataframe(
            drought_arrays, spei_grid_group, class_mask
        )
        # standardized_data_wet = _make_pixel_dataframe(
        #     wet_arrays, spei_grid_group, class_mask
        # )

        # standardized_data_corhigh_corlow= pd.DataFrame({
        #     'POS corhigh corlow diff': pos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SOS corhigh corlow diff': sos_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Cor corhigh corlow diff': cor_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'SM corhigh corlow diff': sm_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'VPD corhigh corlow diff': vpd_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Ta corhigh corlow diff': ta_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Pre corhigh corlow diff': pre_diff_corhigh_corlow.flatten()[valid_indices],
        #
        #     'Srad corhigh corlow diff': srad_diff_corhigh_corlow.flatten()[valid_indices]
        # })

        # print(f'standardized_data drought的有效data量：{len(standardized_data_drought.dropna())}')
        # print(f'standardized_data wet的有效data量：{len(standardized_data_wet.dropna())}')

        # CATEGORICAL_BACKGROUND: ML单独筛选完整类别，偏相关仍用原表。
        rf_data_drought = _make_pixel_dataframe(
            {**drought_arrays, **category_arrays}, spei_grid_group, class_mask)
        # rf_data_wet = _make_pixel_dataframe(
        #     {**wet_arrays, **category_arrays}, spei_grid_group, class_mask)
        standardized_data_drought = standardized_data_drought.dropna().reset_index()
        # standardized_data_wet = standardized_data_wet.dropna().reset_index()



        # standardized_data_corhigh_corlow = standardized_data_corhigh_corlow.dropna().reset_index()

        # print('standardized_data_drought:\n', standardized_data_drought.iloc[:20, :])
        # print('standardized_data_wet:\n', standardized_data_wet.iloc[:20, :])

        # sys.exit()

        # 原Fig6仅替换结果来源，下面原绘图代码保持不变。
        if cached_repeat is not None:
            results_dict = {}
            for y in y_list:
                cached = (cached_repeat[type][y] if type in cached_repeat else cached_repeat[y])
                table = standardized_data_drought if y == 'POS drought normal diff' else standardized_data_wet
                diff_columns = [f'{name} ' + ('drought normal diff' if y == 'POS drought normal diff' else 'wet normal diff')
                                for name in ['SOS','Cor','SM','VPD','Ta','Pre','Srad']]
                result = _pcorr_with_long_climate(table,y,diff_columns,long_cols if pcorr_control_long else [])
                result = result.rename(columns={'var':'feature','r':'r_pcorr','p-val':'p_pcorr'})
                result['Abs_Coefficient'] = result['r_pcorr'].abs()
                result['Significance'] = np.where(result['p_pcorr']<.01,'**',np.where(result['p_pcorr']<.05,'*',''))
                results_dict[y] = dict(df_pcorr=result,
                    importance_and_shap_values_mean=cached['importance'],
                    shap_values_df=cached['shap'],x_list=cached['importance']['feature'].tolist(),y=y)
        else:
            ######## Partial correlation & ML ###########
            def seq_pcorr(df, y, x_list):
                # LONG_CLIMATE: x_list仍是7个差值，只把5个气候态放入covar。
                controls = long_cols if pcorr_control_long else []
                return _pcorr_with_long_climate(df, y, x_list, controls)

            results_dict = {}  # 用于存储每个y的结果

            for y in y_list:


                ### ============= 1. 偏相关分析 ============== ###
                print('Start partial analyze!')
                # if y == 'POS mean' :
                #     # x_list = ['SOS mean', 'SOS std', 'Cor mean', 'Cor std',
                #     #           'SM mean', 'SM std', 'VPD mean', 'VPD std', 'Ta mean', 'Ta std', 'Pre mean', 'Pre std', 'Srad mean', 'Srad std']
                #     x_list = ['SOS mean', 'Cor mean',
                #               'SM mean', 'VPD mean', 'Ta mean', 'Pre mean', 'Srad mean']
                #     df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val', 'n_pcorr', 'covariates']].reset_index(drop=True).rename(
                #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
                #     ml_data = standardized_data_drought
                #
                # elif y == 'POS std':
                #     x_list = ['SOS std', 'Cor std',
                #               'SM std', 'VPD std', 'Ta std', 'Pre std', 'Srad std']
                #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val', 'n_pcorr', 'covariates']].reset_index(drop=True).rename(
                #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
                #     ml_data = outlier_standardized_data
                #
                #
                # elif y == 'POS slope':
                #     x_list = ['SOS slope', 'SOS std',
                #               'Cor slope', 'Cor std',
                #               'SM slope', 'SM std', 'VPD slope', 'VPD std',
                #               'Ta slope', 'Ta std', 'Pre slope', 'Pre std',
                #               'Srad slope', 'Srad std']
                #     df_pcorr = seq_pcorr(outlier_standardized_data, y, x_list)[['var', 'r', 'p-val', 'n_pcorr', 'covariates']].reset_index(drop=True).rename(
                #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
                #     ml_data = outlier_standardized_data

                if y == 'POS drought normal diff':
                    x_list = ['SOS drought normal diff',
                              'Cor drought normal diff',
                              'SM drought normal diff',
                              'VPD drought normal diff',
                              'Ta drought normal diff',
                              'Pre drought normal diff',
                              'Srad drought normal diff']
                    df_pcorr = seq_pcorr(standardized_data_drought, y, x_list)[['var', 'r', 'p-val', 'n_pcorr', 'covariates']].reset_index(
                        drop=True).rename(
                        columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

                elif y == 'POS wet normal diff':
                    x_list = ['SOS wet normal diff',
                              'Cor wet normal diff',
                              'SM wet normal diff',
                              'VPD wet normal diff',
                              'Ta wet normal diff',
                              'Pre wet normal diff',
                              'Srad wet normal diff']
                    df_pcorr = seq_pcorr(standardized_data_wet, y, x_list)[['var', 'r', 'p-val', 'n_pcorr', 'covariates']].reset_index(
                        drop=True).rename(
                        columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})

                # elif y == 'POS corhigh corlow diff':
                #     x_list = ['SOS corhigh corlow diff',
                #               'Cor corhigh corlow diff',
                #               'SM corhigh corlow diff',
                #               'VPD corhigh corlow diff',
                #               'Ta corhigh corlow diff',
                #               'Pre corhigh corlow diff',
                #               'Srad corhigh corlow diff']
                #     df_pcorr = seq_pcorr(standardized_data_corhigh_corlow, y, x_list)[['var', 'r', 'p-val', 'n_pcorr', 'covariates']].reset_index(drop=True).rename(
                #         columns={'r': 'r_pcorr', 'p-val': 'p_pcorr'})
                #     ml_data = standardized_data_corhigh_corlow

                df_pcorr = df_pcorr.rename(columns={'var': 'feature'})
                df_pcorr['Significance'] = ''

                for x in x_list:
                    p_value = df_pcorr[df_pcorr['feature'] == x]['p_pcorr'].iloc[0]
                    if p_value < 0.01:
                        significance = '**'  # p < 0.01 标记为 **
                    elif p_value < 0.05:
                        significance = '*'  # 0.01 <= p < 0.05 标记为 *
                    else:
                        significance = ''  # p >= 0.05 不标记

                    df_pcorr.loc[df_pcorr['feature'] == x, 'Significance'] = significance

                df_pcorr['Abs_Coefficient'] = df_pcorr['r_pcorr'].abs()

                print(f"{type} df_pcorr:\n", df_pcorr)

                # sys.exit()
                # # ==================== VIF计算 ====================
                # from statsmodels.stats.outliers_influence import variance_inflation_factor
                #
                # # 只取干旱情景的自变量，并清除缺失值和无穷值
                # vif_df = (
                #     standardized_data_drought.loc[:, x_list]
                #     .replace([np.inf, -np.inf], np.nan)
                #     .dropna()
                #     .astype(np.float64)
                # )
                #
                # if len(vif_df) <= len(x_list):
                #     raise ValueError(
                #         f"计算 VIF 的有效样本数不足：{len(vif_df)} 行，"
                #         f"自变量有 {len(x_list)} 个。"
                #     )
                #
                # constant_features = [
                #     col for col in x_list
                #     if vif_df[col].nunique() <= 1
                # ]
                # if constant_features:
                #     raise ValueError(f"以下自变量是常数，无法计算 VIF：{constant_features}")
                #
                # # 转成 NumPy 后手动添加截距列，确保第 0 列是常数项
                # X = vif_df.to_numpy(dtype=np.float64)
                # vif_design = np.column_stack([np.ones(X.shape[0]), X])
                #
                # vif_results = []
                # for i, feature in enumerate(x_list, start=1):
                #     vif_value = variance_inflation_factor(vif_design, i)
                #     vif_results.append({
                #         "feature": feature,
                #         "VIF": vif_value,
                #     })
                #
                # vif_df_result = pd.DataFrame(vif_results)
                #
                # df_pcorr = df_pcorr.merge(vif_df_result, on="feature", how="left")
                # print("df_pcorr:\n", df_pcorr)
                #
                # def plot_posdiff_relationships(
                #         df,
                #         y_col,
                #         comparison_name,
                # ):
                #     # 除目标POS diff之外的变量作为自变量
                #     x_cols = [
                #         col for col in df.columns
                #         if col != y_col and col != "index"
                #     ]
                #
                #     # 去掉缺失值和无穷值
                #     plot_df = (
                #         df[[y_col] + x_cols]
                #         .replace([np.inf, -np.inf], np.nan)
                #         .dropna()
                #     )
                #
                #     if plot_df.empty:
                #         print(f"{comparison_name} 没有可用于绘图的有效data。")
                #         return
                #
                #     ncols = 3
                #     nrows = int(np.ceil(len(x_cols) / ncols))
                #
                #     # 画布缩小；字号通过下面各项单独调大
                #     fig, axes = plt.subplots(
                #         nrows,
                #         ncols,
                #         figsize=(11, 3.2 * nrows),
                #         constrained_layout=True,
                #     )
                #     axes = np.asarray(axes).reshape(-1)
                #
                #     last_hexbin = None
                #
                #     for ax_i, x_col in enumerate(x_cols):
                #         ax = axes[ax_i]
                #
                #         x = plot_df[x_col].to_numpy(dtype=float)
                #         y = plot_df[y_col].to_numpy(dtype=float)
                #
                #         # 六边形密度图，适合大量像元
                #         last_hexbin = ax.hexbin(
                #             x,
                #             y,
                #             gridsize=45,
                #             mincnt=1,
                #             bins="log",
                #             cmap="viridis",
                #         )
                #
                #         # Pearson相关系数和线性趋势线
                #         if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
                #             r = np.corrcoef(x, y)[0, 1]
                #
                #             slope, intercept = np.polyfit(x, y, 1)
                #             x_line = np.linspace(np.min(x), np.max(x), 100)
                #             y_line = slope * x_line + intercept
                #
                #             ax.plot(
                #                 x_line,
                #                 y_line,
                #                 color="red",
                #                 linewidth=2.2,
                #                 label="Linear trend",
                #                 zorder=5,
                #             )
                #             ax.legend(fontsize=10, frameon=True)
                #         else:
                #             r = np.nan
                #
                #         ax.set_title(
                #             f"{x_col}\nPearson r = {r:.2f}",
                #             fontsize=12,
                #         )
                #         ax.set_xlabel(x_col, fontsize=11)
                #         ax.set_ylabel(y_col, fontsize=11)
                #         ax.tick_params(axis="both", labelsize=10)
                #
                #         ax.axhline(
                #             0,
                #             color="gray",
                #             linewidth=0.8,
                #             linestyle="--",
                #             zorder=3,
                #         )
                #         ax.axvline(
                #             0,
                #             color="gray",
                #             linewidth=0.8,
                #             linestyle="--",
                #             zorder=3,
                #         )
                #         ax.grid(alpha=0.2)
                #
                #     # 隐藏未使用的空子图
                #     for ax in axes[len(x_cols):]:
                #         ax.set_visible(False)
                #
                #     fig.suptitle(
                #         f"{comparison_name}: variables vs {y_col}",
                #         fontsize=15,
                #     )
                #
                #     if last_hexbin is not None:
                #         cbar = fig.colorbar(
                #             last_hexbin,
                #             ax=axes[:len(x_cols)].tolist(),
                #             label="Pixel count (log scale)",
                #             shrink=0.85,
                #         )
                #         cbar.ax.tick_params(labelsize=10)
                #         cbar.set_label("Pixel count (log scale)", fontsize=11)
                #
                #     print(f"有效像元数：{len(plot_df)}")
                #     plt.show()
                #
                # plot_posdiff_relationships(
                #     standardized_data_drought,
                #     y_col="POS drought normal diff",
                #     comparison_name=f"{grade_by}_{type}_Drought_vs_Normal"
                # )
                #
                # import statsmodels.api as sm
                # from scipy.stats import pearsonr
                #
                # x = "VPD drought normal diff"
                # controls = [v for v in x_list if v != x]
                # cols = ["POS drought normal diff", x] + controls
                # d = standardized_data_drought[cols].replace(
                #     [np.inf, -np.inf], np.nan
                # ).dropna()
                #
                # Z = sm.add_constant(d[controls])
                # resid_y = sm.OLS(d["POS drought normal diff"], Z).fit().resid
                # resid_x = sm.OLS(d[x], Z).fit().resid
                #
                # print("残差相关:", pearsonr(resid_x, resid_y))
                # print("Pingouin 偏相关:",
                #       pg.partial_corr(data=d, y="POS drought normal diff",
                #                       x=x, covar=controls))
                #
                # sys.exit()



                ##### ================= 2. ML =================== #######
                if y == 'POS drought normal diff':
                    ml_data = standardized_data_drought
                elif y == 'POS wet normal diff':
                    ml_data = standardized_data_wet
                # elif y == 'POS corhigh corlow diff':
                #     ml_data = standardized_data_corhigh_corlow

                # LONG_CLIMATE: 偏相关列表与RF列表分开；ML使用7差值+5气候态+类别独热编码。
                pcorr_x_list = x_list.copy()
                if ML_model_in_spatio in ('RF', 'XGBoost'):
                    # CATEGORICAL_BACKGROUND: 7差值+5气候态+4植被列+6AI列。
                    x_list = pcorr_x_list + long_cols + category_cols
                    ml_data = rf_data_drought if y == 'POS drought normal diff' else rf_data_wet
                print(f'{type} {y}: model features={x_list}; valid rows={len(ml_data)}')


                X = ml_data[x_list]
                Y = ml_data[y]

                if ML_model_in_spatio == 'RF':
                    # 先在每个55 km SPEI网格内按目标POS低/中/高分位层的像元比例抽样，
                    # 再在该网格抽中的样本中拆分训练集/测试集；其余像元不进入RF。
                    rf_split_data = (
                        ml_data[x_list + [y, 'SPEI_group']]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                        .reset_index(drop=True)
                    )
                    if len(rf_split_data) < 10:
                        raise ValueError(
                            f"RF 有效 11 km 样本仅 {len(rf_split_data)} 个，"
                            "不足以进行稳定的训练/测试划分。"
                        )
                    train_df, test_df, sample_info = _sample_and_split_rf_within_grid(
                        rf_split_data,
                        target=y,
                        test_fraction=test_size,
                        max_per_grid=rf_max_per_grid,
                        random_state=rf_random_state,
                    )
                    x_train, y_train = train_df[x_list], train_df[y]
                    x_test, y_test = test_df[x_list], test_df[y]
                    print(
                        f"RF 网格内分位数抽样拆分完成：每格最多抽取={rf_max_per_grid}，"
                        f"候选像元={sample_info['n_available']}，"
                        f"抽中={sample_info['n_sampled']}，"
                        f"未使用={sample_info['n_unused']}，"
                        f"低/中/高层抽样数={sample_info['n_sampled_low']}/"
                        f"{sample_info['n_sampled_middle']}/"
                        f"{sample_info['n_sampled_high']}，"
                        f"仅进入测试集的单像元网格={sample_info['n_singleton_grids']}；"
                        f"训练像元={len(train_df)}，测试像元={len(test_df)}；训练网格="
                        f"{train_df['SPEI_group'].nunique()}，测试网格="
                        f"{test_df['SPEI_group'].nunique()}"
                    )
                else:
                    x_train, x_test, y_train, y_test = train_test_split(
                        X, Y, test_size=test_size, random_state=42
                    )
                    print("test split done")

                def calculate_shap(xgb_model, x_train, y_train, x_test, y_test):

                    # Step 1: Calculate MSE and R2 for train and test data
                    mse_train = mean_squared_error(y_train, xgb_model.predict(x_train))
                    r2_train = xgb_model.score(x_train, y_train)
                    print(f'Train R2: {r2_train}, MSE: {mse_train}')

                    y_train_pred = xgb_model.predict(x_test)
                    r2_test = r2_score(y_test, y_train_pred)
                    mse_test = mean_squared_error(y_test, y_train_pred)
                    print(f'Test R2: {r2_test}, MSE: {mse_test}')

                    # Step 2: Calculate SHAP values on a reproducible training subset
                    # 可按内存/绘图速度调整；若训练样本少于此数则全量计算。
                    if grade_by == 'All':
                        max_shap_samples = 10000
                    else:
                        max_shap_samples = 5000
                    n_shap = min(max_shap_samples, len(x_train))
                    x_shap = x_train.sample(n=n_shap, random_state=42).reset_index(drop=True)
                    print(f"SHAP 使用训练样本：{n_shap}/{len(x_train)}")

                    explainer = shap.TreeExplainer(xgb_model)
                    print("SHAP explainer done")
                    shap_values = explainer(x_shap, check_additivity=False)
                    print("SHAP values calculation done")

                    shap_values_df = pd.concat(
                        [x_shap,
                         pd.DataFrame(shap_values.values,
                                      columns=[f'shap_{c}' for c in x_shap.columns])],
                        axis=1
                    )

                    # Step 3: Consolidate feature importance and metrics
                    df_xgb_model = pd.DataFrame(xgb_model.feature_importances_, index=x_list,
                                                columns=["importance"]).reset_index().rename(
                        columns={'index': 'var'})
                    print("Feature importances calculated")
                    print(df_xgb_model)

                    df_xgb_model["model_mse"] = mse_train
                    df_xgb_model["model_r2"] = r2_train
                    df_xgb_model["model_test_mse"] = mse_test
                    df_xgb_model["model_test_r2"] = r2_test

                    # Step 4: Calculate mean absolute SHAP values
                    mean_abs_shap_values = abs(shap_values.values).mean(0)
                    print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                    df_shap = pd.DataFrame([x_list, mean_abs_shap_values], index=["var", "shapvalue"]).T
                    print("SHAP values calculated")

                    pos_range = y_train.max() - y_train.min()
                    shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                    print("POS_mean spatial range (days):", pos_range)
                    print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                    # Step 5: Merge SHAP values and feature importance
                    df_merge = pd.merge(df_shap, df_xgb_model, on="var")
                    # print('df_merge1:\n', df_merge)
                    df_merge = df_merge.rename(columns={'var': 'feature'})
                    # df_merge['importance'] = df_merge['importance'] / df_merge['importance'].sum()  # Normalize feature importance
                    # df_merge['shapvalue'] = df_merge['shapvalue'] / df_merge['shapvalue'].sum()  # Normalize SHAP importance
                    # df_merge['mean'] = (df_merge['importance'] + df_merge['shapvalue']) / 2  # Calculate mean importance

                    print('df_merge:\n', df_merge)

                    # # Step 6: Sort features by SHAP value
                    # df_merge = df_merge.sort_values(by='shapvalue', ascending=False)

                    # CATEGORICAL_BACKGROUND: 编码列明细单独保存，图表汇总为14个因素。
                    _export_encoded_background(df_merge, grade_by, type, y, 'Partial_ML_calculate_and_plot')
                    shap_values_df, df_merge = _group_categorical_shap(shap_values_df, df_merge)
                    return shap_values_df, df_merge, r2_test

                # sys.exit()

                ############################### XGBoost ########################################
                if ML_model_in_spatio == 'XGBoost':
                    ## K-fold和使用网格搜索寻找最佳参数 ##
                    xgb_model = xgb.XGBRegressor(random_state=42)
                    # 定义参数网格
                    param_grid = {'n_estimators': [100, 150, 200, 250, 300, 350],
                                  'max_depth': [5, 10, 15],
                                  'learning_rate': [0.05, 0.1],
                                  'subsample': [0.7, 0.8, 0.9, 1],
                                  'colsample_bytree': [0.7, 0.8, 0.9, 1]}
                    # # 定义 K 折交叉验证 (K-Fold)
                    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                    # 使用网格搜索寻找最佳参数
                    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                               verbose=10, n_jobs=15, error_score='raise')

                    # 拟合模型
                    grid_search.fit(x_train, y_train)

                    print(f'XGboost model done')

                    # Output最优参数和最优得分
                    print(f"Best parameters: {grid_search.best_params_}")
                    print(f"Best R2 score: {grid_search.best_score_}")

                    Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                    print(f'Best_parameters:{Best_parameters}')
                    Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                    Best_parameters.to_csv(
                        fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    #
                    # if grade_by == 'Veg':
                    #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'AI':
                    #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'Cor mean':
                    #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'All':
                    #     Best_parameters.to_csv(fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    print(f'Best_parameters done')

                    # ## 使用最优参数训练模型 ##
                    # xgboost = grid_search.best_estimator_

                    ######################################################
                    # #
                    Best_parameters = pd.read_csv(
                        fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_Best_para_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # if grade_by == 'Veg':
                    #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'AI':
                    #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'Cor mean':
                    #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'All':
                    #     Best_parameters = pd.read_csv(fr'D:/code/{Basedon}/SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_XGBoost_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')

                    n_estimators = int(Best_parameters['n_estimators'][0])
                    max_depth = int(Best_parameters['max_depth'][0])
                    colsample_bytree = float(Best_parameters['colsample_bytree'][0])
                    subsample = float(Best_parameters['subsample'][0])
                    learning_rate = float(Best_parameters['learning_rate'][0])

                    xgboost = XGBRegressor(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           colsample_bytree=colsample_bytree,  #每次分裂会考虑所有特征中 80% 的特征子集
                                           subsample=subsample,  # 每棵树的随机采样比例
                                           learning_rate=learning_rate,
                                           random_state=42,
                                           n_jobs=20)

                    xgboost.fit(x_train, y_train)
                    print(f"xgb_model:", xgboost)

                    shap_values_df, ModelImportance_and_ShapValueMean, r2_test = calculate_shap(xgboost, x_train, y_train,
                                                                                                x_test, y_test)


                if ML_model_in_spatio == 'XGBoost':
                    shap_values_model = shap_values_df
                    shap_importance_model = ModelImportance_and_ShapValueMean

                ############################ RF #########################################
                elif ML_model_in_spatio == 'RF':
                    ## K-fold和使用网格搜索寻找最佳参数 ##
                    rf_model = RandomForestRegressor(random_state=42, n_jobs=15)
                    # if grade_by == 'All':
                    # 定义参数网格
                    param_grid = {
                        'n_estimators': [300, 500],#, 600, 700, 800, 900, 1000],
                        'max_depth': [20],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 5],
                        'max_features': [0.7, 0.9]  # 如果是回归任务，建议包含 0.5, 0.7, 1.0
                    }
                        # param_grid = {
                        #     'n_estimators': [200, 300, 500, 800],
                        #     'max_depth': [None, 15, 20, 25],          # 上限 25，不要 30
                        #     'min_samples_split': [5, 10, 20, 40],
                        #     'min_samples_leaf': [5, 10, 20, 50],      # 下限 5，不要 1、2
                        #     'max_features': ['sqrt', 0.3, 0.5],       # 上限 0.5，不要 0.7、0.8
                        # }
                    # else:
                    #     param_grid = {
                    #         'n_estimators': [150, 200, 250, 300],
                    #         'max_depth': [15, 20, 30],
                    #         'min_samples_split': [2, 5, 10],
                    #         'min_samples_leaf': [1, 2, 4],
                    #         'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.8]
                    #     }
                    # # 定义 K 折交叉验证 (K-Fold)
                    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                    # 使用网格搜索寻找最佳参数
                    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='r2', cv=kfold,
                                               verbose=10, n_jobs=15, error_score='raise')
                    # 拟合模型
                    grid_search.fit(x_train, y_train)
                    # grid_search.fit(X, Y)

                    print(f'{type} RF model done')

                    # Output最优参数和最优得分
                    print(f"Best parameters: {grid_search.best_params_}")
                    print(f"Best R2 score: {grid_search.best_score_}")

                    Best_parameters = pd.DataFrame(grid_search.best_params_, index=[0])
                    print(f'Best_parameters:{Best_parameters}')
                    Best_parameters['mean R2 score'] = round(grid_search.best_score_, 4)
                    # if analyze_by == 'All':
                    Best_parameters.to_csv(
                        fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif analyze_by != 'All':
                    #     Best_parameters.to_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPara_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

                    # if grade_by == 'Veg':
                    #     Best_parameters.to_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'AI':
                    #     Best_parameters.to_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'Cor mean':
                    #     Best_parameters.to_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'All':
                    #     Best_parameters.to_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    print(f'Best_parameters done')

                    # ## 使用最优参数训练模型 ##
                    # rf = grid_search.best_estimator_

                    ######################################################
                    #
                    # if analyze_by == 'All':
                    Best_parameters = pd.read_csv(
                        fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif analyze_by != 'All':
                    #     Best_parameters = pd.read_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

                    # if grade_by == 'Veg':
                    #     Best_parameters = pd.read_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_VegType({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'AI':
                    #     Best_parameters = pd.read_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_AI({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'Cor mean':
                    #     Best_parameters = pd.read_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_Cormean({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                    # elif grade_by == 'All':
                    #     Best_parameters = pd.read_csv(
                    #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEIstrength{spei_strength}_SM_VPD_Cor{cor_test_number}_RF_Best_parameters_All_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')

                    n_estimators = int(Best_parameters['n_estimators'][0])
                    max_depth = int(Best_parameters['max_depth'][0])
                    min_samples_split = int(Best_parameters['min_samples_split'][0])
                    min_samples_leaf = int(Best_parameters['min_samples_leaf'][0])
                    max_features = Best_parameters['max_features'][0]
                    # 如果保存时是数字，才需要转换
                    if isinstance(max_features, str):
                        max_features = max_features  # 保持字符串
                    else:
                        max_features = float(max_features)

                    # if data_type == 'MODIS_GPP_8d' and lag_day ==0 and spei_strength == -1:
                    #     # 最优
                    # n_estimators = 400
                    # max_depth = 40
                    # min_samples_split = 5
                    # min_samples_leaf = 5
                    # max_features = 0.7
                    # elif data_type == 'MODIS_GPP_8d' and lag_day ==0 and spei_strength == -1.5:
                    #     n_estimators = 200
                    #     max_depth = 40
                    #     min_samples_split = 2  # 每次分裂会考虑所有特征中 80% 的特征子集
                    #     min_samples_leaf = 1 # 每棵树的随机采样比例
                    #     max_features = 0.8
                    # elif data_type == 'MCD12Q2' and lag_day ==0 and spei_strength == -1:
                    #     n_estimators = 400
                    #     max_depth = 20
                    #     min_samples_split = 2  # 每次分裂会考虑所有特征中 80% 的特征子集
                    #     min_samples_leaf = 1  # 每棵树的随机采样比例
                    #     max_features = 0.3

                    # 转成一维数组，确保 OOB 误差和训练样本一一对应
                    y_train_array = np.asarray(y_train).reshape(-1)

                    # ==================== 模型1：使用全部训练样本 ====================
                    rf_model1 = RandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,  # 每次分裂会考虑所有特征中 80% 的特征子集
                                               min_samples_leaf=min_samples_leaf,  # 每棵树的随机采样比例
                                               max_features=max_features,
                                               bootstrap=True,
                                               oob_score=True,
                                               random_state=42,
                                               n_jobs=20)

                    rf_model1.fit(x_train, y_train)
                    print(f"rf_model1:", rf_model1)
                    print('Train R²:', rf_model1.score(x_train, y_train))
                    print('Test  R²:', rf_model1.score(x_test, y_test))
                    print('OOB   R²:', rf_model1.oob_score_)
                    print('feature importances:', rf_model1.feature_importances_)

                    # 每条训练样本的折外预测误差
                    oob_pred_model1 = np.asarray(rf_model1.oob_prediction_).reshape(-1)

                    valid_oob = (np.isfinite(y_train_array) & np.isfinite(oob_pred_model1))

                    oob_abs_error_model1 = np.abs(
                        y_train_array - oob_pred_model1
                    )

                    oob_mean_model1 = float(np.mean(oob_abs_error_model1))
                    oob_median_model1 = float(np.median(oob_abs_error_model1))
                    oob_threshold = float(np.quantile(oob_abs_error_model1, 0.90))

                    # 模型1 SHAP：使用原训练集和同一独立测试集
                    shap_values_model1, shap_importance_model1, shap_r2_model1 = calculate_shap(
                        rf_model1,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                    )

                    Best_parameters['r2_test'] = shap_r2_model1

                    shap_importance_model = shap_importance_model1
                    shap_values_model = shap_values_model1

                    if shap_r2_model1 < 0.4:


                        # ==================== 根据模型1 OOB误差筛样本 ====================
                        # 误差高于90%分位数的训练样本被剔除
                        keep_mask = oob_abs_error_model1 <= oob_threshold
                        n_removed = int(np.sum(~keep_mask))
                        n_kept = int(np.sum(keep_mask))

                        print(f"模型1训练样本数：{len(keep_mask)}")
                        print(f"剔除样本数：{n_removed}")
                        print(f"保留样本数：{n_kept}")

                        # 同时兼容 pandas DataFrame/Series 和 NumPy 数组
                        if hasattr(x_train, "iloc"):
                            x_train_model2 = x_train.iloc[np.flatnonzero(keep_mask)].copy()
                        else:
                            x_train_model2 = x_train[keep_mask]

                        if hasattr(y_train, "iloc"):
                            y_train_model2 = y_train.iloc[np.flatnonzero(keep_mask)].copy()
                        else:
                            y_train_model2 = y_train[keep_mask]

                        # ==================== 模型2：用筛选后的训练样本 ====================
                        rf_model2 = RandomForestRegressor(n_estimators=n_estimators,
                                                   max_depth=max_depth,
                                                   min_samples_split=min_samples_split,  # 每次分裂会考虑所有特征中 80% 的特征子集
                                                   min_samples_leaf=min_samples_leaf,  # 每棵树的随机采样比例
                                                   max_features=max_features,
                                                   bootstrap=True,
                                                   oob_score=True,
                                                   random_state=42,
                                                   n_jobs=20)
                        rf_model2.fit(x_train_model2, y_train_model2)
                        print(f"rf_model2:", rf_model2)
                        print('Train R²:', rf_model2.score(x_train, y_train))
                        print('Test  R²:', rf_model2.score(x_test, y_test))
                        print('OOB   R²:', rf_model2.oob_score_)
                        print('feature importances:', rf_model2.feature_importances_)

                        y_train_model2_array = np.asarray(y_train_model2).reshape(-1)
                        oob_pred_model2 = np.asarray(
                            rf_model2.oob_prediction_
                        ).reshape(-1)

                        valid_oob_model2 = (
                                np.isfinite(y_train_model2_array)
                                & np.isfinite(oob_pred_model2)
                        )

                        if not np.all(valid_oob_model2):
                            raise ValueError("模型2存在无效OOB预测，请检查data或增加树的数量。")

                        oob_abs_error_model2 = np.abs(
                            y_train_model2_array - oob_pred_model2
                        )

                        oob_mean_model2 = float(np.mean(oob_abs_error_model2))
                        oob_median_model2 = float(np.median(oob_abs_error_model2))

                        print(f"模型2 OOB绝对误差均值：{oob_mean_model2:.4f}")
                        print(f"模型2 OOB绝对误差中位数：{oob_median_model2:.4f}")

                        # 模型2 SHAP：训练集是筛选后的样本，测试集仍与模型1相同
                        shap_values_model2, shap_importance_model2, shap_r2_model2 = calculate_shap(
                            rf_model2,
                            x_train_model2,
                            y_train_model2,
                            x_test,
                            y_test,
                        )

                        # 汇总两个模型的 OOB 与测试集结果
                        summary = pd.DataFrame([
                            {
                                "model": "model1",
                                "training_samples": len(y_train_array),
                                "removed_samples": 0,
                                "oob_R2": rf_model1.oob_score_,
                                "oob_MAE_mean": oob_mean_model1,
                                "oob_MAE_median": oob_median_model1,
                                "test_R2": rf_model1.score(x_test, y_test),
                                "SHAP_test_R2": shap_r2_model1,
                            },
                            {
                                "model": "model2",
                                "training_samples": n_kept,
                                "removed_samples": n_removed,
                                "oob_R2": rf_model2.oob_score_,
                                "oob_MAE_mean": oob_mean_model2,
                                "oob_MAE_median": oob_median_model2,
                                "test_R2": rf_model2.score(x_test, y_test),
                                "SHAP_test_R2": shap_r2_model2,
                            },
                        ])

                        print('summary\n'
                              rf'{summary}')

                        summary.to_csv(
                            fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag={lag_day}){months_before_pos}_{ML_model_in_spatio}_TS{test_size}_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_Summary_LongClimate_CatBackground_5Climate.csv')

                        Best_parameters['r2_test'] = shap_r2_model2

                        shap_importance_model = shap_importance_model2
                        shap_values_model = shap_values_model2

                # if analyze_by == 'All':
                Best_parameters.to_csv(
                    fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\{Basedon}\SPEI{spei_strength}_Cor({coupling_method}_lag{lag_day}){months_before_pos}_{ML_model_in_spatio}_TestSize{test_size}_BestPa_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_LongClimate_CatBackground_5Climate.csv')
                # elif analyze_by != 'All':
                #     Best_parameters.to_csv(
                #         fr'{output_partial_max_var_r_tif_path}\SPEI{spei_length}\Climate{climate_test_number}\{Basedon}\SPEI{spei_strength}_Cor{cor_test_number}_{ML_model_in_spatio}_TestSize{test_size}_BestParameters_{grade_by}({type})_{y}_Outlier({Outlier})_CorSig({SigCorPvalue})_analyzeBy({analyze_by})_LongClimate_CatBackground_5Climate.csv')

                # LONG_CLIMATE: 保存偏相关及RF/SHAP结果，文件名含函数版本。
                _export_long_climate_results(df_pcorr, shap_importance_model,
                                             grade_by, type, y, 'Partial_ML_calculate_and_plot')

                # 将每个y的结果存储到字典中
                results_dict[y] = {
                    'df_pcorr': df_pcorr,
                    'importance_and_shap_values_mean': shap_importance_model,
                    'shap_values_df': shap_values_model,
                    # 训练用22列，图表用12个连续因素+2个类别组。
                    'x_list': shap_importance_model['feature'].tolist(),
                    'y': y
                }

                # # srad_std_shap = shap_values_df.values[:, -1]
                # print(f"SHAP值Statistics:")
                # print(f"  最小值: {np.nanmin(shap_values_df.values):.1f}")
                # print(f"  最大值: {np.nanmax(shap_values_df.values):.1f}")
                # print(f"  平均值: {np.nanmean(shap_values_df.values):.1f}")
                #
                # # Statistics-10到10范围内的占比
                # in_range_mask = (shap_values_df >= -1) & (shap_values_df <= 1) & (~np.isnan(shap_values_df))
                # total_valid = np.sum(~np.isnan(shap_values_df))
                # in_range_count = np.sum(in_range_mask)

                # print(f"[-1, 1]区间Statistics:\n")
                # print(f"  区间内数量: {in_range_count}")
                # print(f"  有效总数: {total_valid}")
                # print(f"  占比: {(in_range_count / total_valid) * 100:.2f}%")
                #
                # sys.exit()

        ######## ================= 3. plot ====================== #######

        #### 驱动分析结果图Plot ####
        for y, results in results_dict.items():
            # 使用存储的结果进行绘图
            df_pcorr = results['df_pcorr']
            ModelImportance_and_ShapValueMean = results['importance_and_shap_values_mean']
            shap_values_df = results['shap_values_df']
            x_list = results['x_list']

            # ========== 1. 定义“唯一的 y 轴顺序”（以 SHAP / importance 为准） ==========
            # 仅筛选绘图data：保留原模型全部Input、原偏相关控制和原重要性数值。
            suffix = 'drought normal diff' if y == 'POS drought normal diff' else 'wet normal diff'
            feature_order = [f'{name} {suffix}' for name in ['SOS','Cor','SM','VPD','Ta','Pre','Srad']]
            missing = [name for name in feature_order
                       if name not in ModelImportance_and_ShapValueMean['feature'].values
                       or name not in shap_values_df.columns or f'shap_{name}' not in shap_values_df.columns]
            if missing:
                raise ValueError(f'七变量绘图所需列缺失：{missing}')
            df_pcorr = df_pcorr[df_pcorr['feature'].isin(feature_order)].copy()
            ModelImportance_and_ShapValueMean = (
                ModelImportance_and_ShapValueMean.set_index('feature').loc[feature_order].reset_index())
            # 必须同时裁剪SHAP值及对应原始特征data，避免生成“其他变量之和”行。
            shap_values_df = shap_values_df[
                feature_order + [f'shap_{name}' for name in feature_order]].copy()
            x_list = feature_order.copy()
            # feature_order = ['SOS drought normal diff', 'Cor drought normal diff', 'SM drought normal diff', 'VPD drought normal diff',
            #                  'Ta drought normal diff', 'Pre drought normal diff', 'Srad drought normal diff']
            print(f'feature_order:{feature_order}')

            # # ========== 2. Create the figure ==========
            # fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=False)
            # ax1, ax2, ax3 = axes
            # plt.subplots_adjust(wspace=0.05)

            # ========== 3. 子图 1：Partial correlation ==========
            # LONG_CLIMATE: 背景行只作控制说明，不Plot伪造的偏相关条形。
            _plot_long_climate_pcorr(ax1, df_pcorr, feature_order,
                                     pcorr_control_long)

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'a'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            #     elif type == 'Arid':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'd'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'g'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'j'
            #     ytitle = 1.07
            # ax1.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax1.set_xlabel(f'Partial correlation')
            else:
                ax1.set_xlabel('')

            if data_type == 'MODIS_GPP_8d':
                ticks = np.arange(0, 0.31, 0.1)
                ax1.set_xlim(0, 0.3)
            elif data_type == 'MCD12Q2':
                ticks = np.arange(0, 0.8, 0.2)
                ax1.set_xlim(0, 0.6)
            else:
                raise ValueError(f'未配置的data类型：{data_type}')

            ax1.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
            ax1.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, pos: f'{x:g}')
            )

            ax1.set_ylabel('')
            ax1.tick_params(axis='y', length=0)

            # legend_handles = [
            #     mpatches.Patch(color='#4393c3', label='Negative', alpha=0.8),
            #     mpatches.Patch(color='#d6604d', label='Positive', alpha=0.8)
            # ]
            # ax1.legend(handles=legend_handles, frameon=False, fontsize=8, loc='lower right')
            from matplotlib.lines import Line2D

            legend_handles = [
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#4393c3',
                       markeredgecolor='none', markersize=6, label='Negative', alpha=0.8),
                Line2D([0], [0], linestyle='none', marker='s', markerfacecolor='#d6604d',
                       markeredgecolor='none', markersize=6, label='Positive', alpha=0.8)
            ]

            ax1.legend(handles=legend_handles, frameon=False, loc='lower right', fontsize=8,
                       handletextpad=0.5, handlelength=1.0)  # 缩短句柄长度使其看起来更像纯方块

            # ========== 4. 子图 2：ML feature importance ==========
            importance_plot = ModelImportance_and_ShapValueMean.copy()

            importance_plot['feature'] = pd.Categorical(
                importance_plot['feature'],
                categories=feature_order,
                ordered=True
            )
            importance_plot = importance_plot.sort_values('feature')
            print(f'importance_plot:{importance_plot}')

            sns.barplot(
                x='importance',
                y='feature',
                data=importance_plot,
                color='#4393c3',
                alpha=0.8,
                width=0.4,
                ax=ax2
            )

            ax2.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'b'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'e'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'h'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'k'
            #     ytitle = 1.07
            # ax2.set_title(f'({word}) {type}', y=ytitle)
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax2.set_xlabel(f'{ML_model_in_spatio} importance')
            else:
                ax2.set_xlabel('')


            if data_type == 'MODIS_GPP_8d':
                x_min = 0
                x_max = 0.2
                interval = 0.1
            else:
                x_min = 0
                x_max = 0.4
                interval = 0.2
            # x_max = max(x_max, float(importance_plot['importance'].max()) * 1.1)
            ax2.set_xlim(x_min, x_max)
            ax2.set_xticks(np.arange(x_min, x_max + 0.01, interval))
            ax2.set_xticklabels(
                [f"{int(x)}" if x.is_integer() else f"{x:.1f}" for x in np.arange(x_min, x_max + 0.01, interval)])

            ax2.set_ylabel('')
            ax2.tick_params(axis='y', length=0, labelright=False, labelleft=False)

            # ========== 5. 子图 3：SHAP beeswarm（关键：用 order 控制顺序） ==========
            plt.sca(ax3)
            ### 1 Shap value sactter
            # —— 构造 SHAP Explanation（按 x_list 顺序）——
            shap_values = shap.Explanation(
                values=shap_values_df[[f'shap_{c}' for c in x_list]].values,
                data=shap_values_df[x_list].values,
                feature_names=x_list
            )

            # —— 将 feature_order 映射为 SHAP 的索引顺序 ——
            order_idx = [x_list.index(f) for f in feature_order if f in x_list]
            print(f'order_idx:{order_idx}')

            shap.plots.beeswarm(
                shap_values,
                order=order_idx,
                max_display=len(order_idx),
                plot_size=None,  # 由外部 ax 控制
                show=False,
                color_bar=False
            )

            # 更改点的color
            new_cmap = plt.get_cmap('coolwarm')

            for coll in ax3.collections:
                coll.set_sizes([5])
                # 将现有的color数组映射到新的 cmap 上
                # SHAP 内部通常将 feature value 映射在 [0, 1] 之间
                coll.set_cmap(new_cmap)

            # 点大小
            for coll in ax3.collections:
                coll.set_sizes([5])

            # x 轴范围
            # if y == 'POS mean':
            #     ax3.set_xlim(-20, 20)
            # elif y == 'POS slope':
            #     ax3.set_xlim(-1, 1)
            # elif y == 'POS std':
            #     ax3.set_xlim(-5, 5)
            # else:
            ax3.tick_params(axis='both', length=2, pad=3)  # ,which='major', )

            fp = ax2.get_xticklabels()[0].get_fontproperties()
            # shap_bound = max(1.5, float(np.nanmax(np.abs(shap_values.values))) * 1.05)
            shap_bound = 1.5
            ax3.set_xlim(-shap_bound, shap_bound)
            ticks = [-1, -0.5, 0, 0.5, 1]
            ax3.set_xticks(ticks)
            ax3.set_xticklabels([f'{x:g}' for x in ticks])

            # 应用到SHAP横轴刻度
            for label in ax3.get_xticklabels():
                label.set_fontproperties(fp)


            # if type == 'Forest' or type == 'Arid' or type == 'All':
            #     word = 'c'
            #     if type == 'All':
            #         ytitle = 1.1
            #     elif type == 'Forest':
            #         ytitle = 1.13
            # elif type == 'Shrub' or type == 'Semi-arid':
            #     word = 'f'
            #     ytitle = 1.07
            # elif type == 'Savanna' or type == 'Dry sub-humid':
            #     word = 'i'
            #     ytitle = 1.07
            # elif type == 'Grass' or type == 'Humid':
            #     word = 'l'
            #     ytitle = 1.07
            # ax3.set_title(f'({word}) {type}', y=ytitle)
            fp = ax2.xaxis.label.get_fontproperties()
            if type in ['All', 'Grass', 'Humid', 'Cor(lt-0.4)']:
                ax3.set_xlabel(f'{ML_model_in_spatio} SHAP value', fontproperties=fp)  #, labelpad=4)
            else:
                ax3.set_xlabel('')
            ax3.tick_params(axis='y', labelleft=False)

            ### 2 Shap abs mean

            ax3_twin = ax3.twiny()
            ax3_twin.set_ylim(ax1.get_ylim())

            shapvalue_mean = ModelImportance_and_ShapValueMean.copy()

            shapvalue_mean['feature'] = pd.Categorical(
                shapvalue_mean['feature'],
                categories=feature_order[::-1],
                ordered=True
            )
            shapvalue_mean = shapvalue_mean.sort_values('feature')
            print(f'shapvalue_mean:{shapvalue_mean}')
            sns.barplot(
                x='shapvalue',
                y='feature',
                data=shapvalue_mean,
                color='gray',
                width=0.4,
                alpha=0.3,
                ax=ax3_twin,
                zorder=0
            )

            # 设置顶层 X 轴的范围和标签
            # shap_mean_bound = max(0.3, float(shapvalue_mean['shapvalue'].max()) * 1.1)
            if data_type == 'MODIS_GPP_8d':
                shap_mean_bound = 0.2
                ax3_twin.xaxis.set_major_locator(mpl.ticker.FixedLocator([0, 0.1, 0.2]))
            elif data_type == 'MCD12Q2':
                shap_mean_bound = 0.4
                ax3_twin.xaxis.set_major_locator(mpl.ticker.FixedLocator([0, 0.2, 0.4]))

            ax3_twin.set_xlim(0, shap_mean_bound)
            ax3_twin.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
            ax3_twin.tick_params(axis='x', length=2, pad=1)
            if type in ['All', 'Forest', 'Arid', 'Cor(-0.1~0)']:
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            elif veg_type == 'Grass' and grade_by == 'Veg':
                ax3_twin.set_xlabel(f'mean |{ML_model_in_spatio} SHAP value|', labelpad=6)
            else:
                ax3_twin.set_xlabel('')

            # # 隐藏右侧和多余的 Y 轴标签
            # ax3_twin.spines['right'].set_visible(False)
            # ax3.spines['right'].set_visible(False)
            # ax3_twin.get_yaxis().set_visible(False)  # 隐藏 twin 轴的 Y 标签，防止重叠

            # for ax, name in zip([ax2, ax3], ['ax2', 'ax3']):
            #     t = ax.get_xticklabels()[0]
            #     print(name)
            #     print("family:", t.get_fontfamily())
            #     print("size:", t.get_fontsize())
            #     print("weight:", t.get_fontweight())
            #     print("style:", t.get_fontstyle())
            #     print("name:", t.get_fontproperties().get_name())

            ### ================ 6. 子图 4：Subplot 3的color ================ ###
            norm = mpl.colors.Normalize(vmin=0, vmax=1)

            cb = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
                cax=ax4,  # 确保使用预留的 ax4
                orientation='vertical'
            )

            cb.ax.tick_params(
                axis='y',  # 针对 y 轴（Colorbar 的纵向轴）
                length=0,  # 刻度线长度设为 0（隐藏刻度线）
                pad=2  # 标签与色条的间距，可根据需要微调
            )
            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Low', 'High'])
            cb.set_label('Feature value', labelpad=-7)
            cb.outline.set_visible(False)  # 去掉 colorbar 的黑色边框显得更高级

            pos = ax4.get_position()

            new_width = pos.width * 0.5

            ax4.set_position([pos.x0 - 0.01, pos.y0, new_width, pos.height])

            # ========== 6. 统一边框风格 ==========
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1)
                    spine.set_edgecolor('black')

            # ---------------------- 统一对齐 y 轴 ----------------------
            # 1. 以 ax1 为基准
            y_min, y_max = ax1.get_ylim()
            y_lim = (min(y_min, y_max), max(y_min, y_max))  # 确保正向

            # 2. 获取 ax1 的 ticks
            yticks = ax1.get_yticks()

            # 3. 循环应用到 ax2 和 ax3
            for ax in [ax3_twin]:
                # 设置相同的 ylim（正向保证）
                ax.set_ylim(y_lim)

                # 设置相同的 ticks
                ax.set_yticks(yticks)
                # ax.invert_yaxis()

                # 可选：保持 tick 标签与 ax1 一致
                # ticklabel 可以使用字符串或数值
                # yticklabels = [str(round(t, 2)) for t in yticks]  # 保留两位小数
                # ax.set_yticklabels(yticklabels)

            # base_ylim = ax1.get_ylim()
            #
            # # 2. 强制覆盖所有相关轴
            # for target_ax in [ax2, ax3, ax3_twin]:
            #     target_ax.set_ylim(base_ylim)
            #
            # # 3. 检查反转状态（关键！）
            # # 如果 ax1 是Regular的（0在下，N在上），而 SHAP 点在 ax3 里还是反的
            # # 那么我们只需要反转 ax3 和它的 twin 轴
            # if ax3.yaxis_inverted() != ax1.yaxis_inverted():
            #     ax3.invert_yaxis()
            #     ax3_twin.invert_yaxis()  # 必须两个一起反转
            # ax3_twin.get_yaxis().set_visible(False)

    # ========== 7. 保存 ==========

    if grade_by in ['Veg', 'AI', 'Cor mean']:
        if cached_repeat is not None:
            # 分组缓存调用者负责短文件名和明确的Figure保存，避免plt当前图切换。
            return fig
        fig_path = Path(output_partial_max_var_png_path) / grade_by / f'SPEI{spei_strength}_{grade_by}.png'
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Fig保存到：{fig_path}')
    return fig




## S23-25: Distrition of Drought/Wet and normal diff


def plot_pos(drought_difference_data, wet_difference_data, colorbarmin, colorbarmax, grade_by, ax):
    if grade_by != 'All':
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

    if grade_by == 'All':
        fig = ax.figure
        gs = ax.get_subplotspec().subgridspec(1, 3,
                                              width_ratios=[5, 5, 4],  # Width ratios for the three columns
                                              height_ratios=[1],  # Reserve the last slot for the colorbar
                                              wspace=0.25)
        ax.axis('off')

        drought_data_list = [drought_difference_data]
        wet_data_list = [wet_difference_data]
        labels = ['All']

    elif grade_by == 'Veg' or grade_by == 'AI':

        fig = plt.figure(figsize=(8.6, 12.5))
        gs = gridspec.GridSpec(4, 4,
                               width_ratios=[5, 5, 0.04, 4],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.25, wspace=0.25)

        if grade_by == 'Veg':
            drought_data_list = [np.where(veg_type_data == i, drought_difference_data, np.nan) for i in [1, 2, 3, 4]]
            wet_data_list = [np.where(veg_type_data == i, wet_difference_data, np.nan) for i in [1, 2, 3, 4]]
            labels = ['Forest', 'Shrub', 'Savanna', 'Grass']
        elif grade_by == 'AI':
            drought_data_list = [
                np.where(ai_type_data == 2, drought_difference_data, np.nan),  # Arid
                np.where((ai_type_data == 3) | (ai_type_data == 4), drought_difference_data, np.nan),
                # Semi-Arid (Combine 3 and 4)
                np.where(ai_type_data == 5, drought_difference_data, np.nan),  # Dry sub-humid
                np.where(ai_type_data == 6, drought_difference_data, np.nan)  # Humid
            ]
            wet_data_list = [
                np.where(ai_type_data == 2, wet_difference_data, np.nan),  # Arid
                np.where((ai_type_data == 3) | (ai_type_data == 4), wet_difference_data, np.nan),
                # Semi-Arid (Combine 3 and 4)
                np.where(ai_type_data == 5, wet_difference_data, np.nan),  # Dry sub-humid
                np.where(ai_type_data == 6, wet_difference_data, np.nan)  # Humid
            ]
            labels = ['Arid', 'Semi-arid', 'Dry sub-humid', 'Humid']

    elif grade_by == 'Cor mean':

        fig = plt.figure(figsize=(8.6, 15.5))
        gs = gridspec.GridSpec(5, 4,
                               width_ratios=[5, 5, 0.03, 4],  # Width ratios for the three columns
                               height_ratios=[1, 1, 1, 1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.25, wspace=0.25)

        drought_data_list = [
            np.where((cor_mean_data >= -0.1) & (cor_mean_data < 0), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.2) & (cor_mean_data < -0.1), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.3) & (cor_mean_data < -0.2), drought_difference_data, np.nan),
            np.where((cor_mean_data >= -0.4) & (cor_mean_data < -0.3), drought_difference_data, np.nan),
            np.where((cor_mean_data < -0.4), drought_difference_data, np.nan)
        ]
        wet_data_list = [
            np.where((cor_mean_data >= -0.1) & (cor_mean_data < 0), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.2) & (cor_mean_data < -0.1), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.3) & (cor_mean_data < -0.2), wet_difference_data, np.nan),
            np.where((cor_mean_data >= -0.4) & (cor_mean_data < -0.3), wet_difference_data, np.nan),
            np.where((cor_mean_data < -0.4), wet_difference_data, np.nan)
        ]

        labels = [  # 'Cor(<-0.5)',
            'Cor(-0.1~0)', 'Cor(-0.2~-0.1)', 'Cor(-0.3~-0.2)',
            'Cor(-0.4~-0.3)', 'Cor(lt-0.4)']

    plots = []  # Store the plot object for each subplot

    for i, (drought_data, wet_data, name) in enumerate(zip(drought_data_list, wet_data_list, labels)):
        if grade_by == 'All':
            pos_ax1 = fig.add_subplot(gs[0, 0])  # map
            pos_ax2 = fig.add_subplot(gs[0, 1])  # Latitude profile
            pos_ax3 = fig.add_subplot(gs[0, 2])  # Colorbar spans both columns
        else:
            pos_ax1 = plt.subplot(gs[i, 0])
            pos_ax2 = plt.subplot(gs[i, 1])
            pos_ax3 = plt.subplot(gs[i, 3])

        print(f'{name}--pos_diff_drought_normal有效像元数量：{np.count_nonzero(np.isfinite(drought_data))}')
        # print(f'{name}--pos_diff_drought_normal有效像元数量占比：{(np.count_nonzero(np.isfinite(drought_data)) / np.count_nonzero(np.isfinite(pos_diff_drought_normal))) * 100}%')
        #
        ###### 子图1：干旱与RegularPOS差异空间分布
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(drought_data, colorbarmin, colorbarmax, 'drought', name,
                                                        pos_ax1)
        plot_cor_mean_or_slope_and_pvalue_forAllvegType(wet_data, colorbarmin, colorbarmax, 'wet', name, pos_ax2)
        plot_cor_mean_or_slope_lt0_sig_and_lt0_nosig_and_gt0_forAllvegType(drought_data, wet_data, colorbarmin, colorbarmax, name,
                                                                           pos_ax3)

    if grade_by != 'All':
        # plt.tight_layout()
        # plt.savefig(fr'D:/code/{grade_by}/POSmean_difference(drought and normal)_way3.png', dpi=600, bbox_inches='tight')
        # if analyze_by == 'All':
        plt.savefig(
            fr'{output_partial_max_var_png_path}\{grade_by}\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor({coupling_method}_{partial_for}_lag={lag_day}){months_before_pos}_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
            dpi=300, bbox_inches='tight')
        # elif analyze_by == 'advance PPT':
        #     plt.savefig(
        #         fr'{output_partial_max_var_png_path}\{grade_by}\RF_by_advancedPPT_drought_event\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png',
        #         dpi=300, bbox_inches='tight')
        # elif analyze_by == 'delay PPT':
        #     plt.savefig(
        #         fr'{output_partial_max_var_png_path}\{grade_by}\RF_by_delay PPT_drought_event\DroughtEventTimes({drought_or_wet_times})_SPEIstrength({spei_strength})_POSmean_difference(drought and normal)_way3_CorSig({SigCorPvalue}).png',
        #         dpi=300, bbox_inches='tight')

    # plt.show()


### Fig 6
def plot_fig6(drought_data, wet_data,
              colorbarmin, colorbarmax,
              pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, cached_repeat=None, save_path=None, show=True):
    with plt.ioff():
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

        fig = plt.figure(figsize=(8.4, 7))
        gs = fig.add_gridspec(2, 3,
                               width_ratios=[0.5, 4, 0.5],  # Width ratios for the three columns
                               height_ratios=[1, 1],  # Reserve the last slot for the colorbar
                               hspace=0.7, wspace=0.15)

        ### a Drought and normal diff
        combine_ax1 = fig.add_subplot(gs[0, :])
        plot_pos(drought_data, wet_data, colorbarmin, colorbarmax, 'All', combine_ax1)

        # plt.show()
        # #
        # sys.exit()

        ## d Driving factor analysis of All pixels
        combine_ax2 = fig.add_subplot(gs[1, 1])
        Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, 'All',
                                      'All', combine_ax2, cached_repeat=cached_repeat)

        # ### d Driving factor analysis of only grassland
        # combine_ax3 = plt.subplot(gs[1, 1])
        # Partial_ML_calculate_and_plot(pos_data, sos_data, cor_data, sm_data, vpd_data, ta_data, pre_data, srad_data, 'Veg', 'Grass', combine_ax3)
        # sys.exit()
        # plt.savefig(fr'D:/code/Based_on_detrendPheno/POSdiff_combined_SM_VPD_Cor17_8_{months_before_pos}_Global_All_XGBoost_POS drought normal diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
        # if analyze_by == 'All':
        # 显式保存组合图对象，避免plt当前Figure被SHAP/IDE切换。
        # 只将最终完整Fig6保存到用户指定目录；RF缓存保存position保持不变。
        save_dir = fr'D:/code/All'
        fraction_tag = os.path.basename(os.path.dirname(os.path.dirname(os.fspath(save_path)))) if save_path else 'all'
        final_name = f'SPEI{spei_strength}_N{REPEAT_RF_N}_{fraction_tag}.png'
        final_path = os.path.join(save_dir, final_name)
        final_path = os.path.abspath(os.fspath(final_path))
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        fig.canvas.draw()
        fig.savefig(final_path, dpi=300, bbox_inches='tight')
        print(f'完整Fig6已保存：{final_path}（{os.path.getsize(final_path)} bytes）', flush=True)
        # elif analyze_by == 'advance PPT':
        #     plt.savefig(fr'{output_partial_max_var_png_path}\All\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{months_before_pos}_All_{ML_model_in_spatio}_TestSize{test_size}_drought_normal_diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
        #                 dpi=300, bbox_inches='tight')
        # elif analyze_by == 'delay PPT':
        #     plt.savefig(fr'{output_partial_max_var_png_path}\All\DroughtTimes({drought_or_wet_times})_SPEI({spei_strength})_Cor{months_before_pos}_All_{ML_model_in_spatio}_TestSize{test_size}_drought_normal_diff_Outlier({Outlier})_CorSig({SigCorPvalue}).png',
        #                 dpi=300, bbox_inches='tight')

        print('Fig 6 plot done!')


    if show:
        plt.show()


"""55 km网格有放回抽样、原始11 km像元无交叉划分、重复RF与汇总。"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as _RepeatRF
from sklearn.model_selection import train_test_split as _repeat_split
from sklearn.metrics import r2_score as _repeat_r2, mean_squared_error as _repeat_mse


def prepare_repeated_rf_data(target_array, difference_arrays, group_grid,
                             sm_long_data, vpd_long_data, ta_long_data,
                             pre_long_data, srad_long_data,
                             vegetation_data, ai_class_data, selection_mask=None):
    """一行一个原始11km像元；保留像元ID，全部22个Input共同筛选完整案例。"""
    target_array = np.asarray(target_array, dtype=float)
    shape = target_array.shape
    if len(difference_arrays) != 7:
        raise ValueError('difference_arrays必须包含7个差值自变量，不含目标。')
    if selection_mask is None:
        selection_mask = np.ones(shape, dtype=bool)
    selection_mask = np.asarray(selection_mask, dtype=bool)
    grid = np.asarray(group_grid, dtype=float)
    if grid.shape != shape or selection_mask.shape != shape:
        raise ValueError('目标、网格ID和selection_mask形状必须一致。')
    long_arrays = _resolve_long_climate(target_array, sm_long_data, vpd_long_data,
                                       ta_long_data, pre_long_data, srad_long_data)
    categories = _resolve_categorical_background(target_array, vegetation_data, ai_class_data)
    arrays = {**difference_arrays, **long_arrays, **categories}
    if len(arrays) != 22:
        raise ValueError('解释变量名有重复或输入数量不正确，预期22列。')
    flat = {}
    valid = (selection_mask.ravel() & np.isfinite(target_array.ravel())
             & np.isfinite(grid.ravel()) & (grid.ravel() >= 0))
    if np.any(np.isfinite(grid) & (grid >= 0) & (grid != np.floor(grid))):
        raise ValueError('SPEI_group必须是整数网格ID。')
    for name, array in arrays.items():
        values = np.asarray(array, dtype=float)
        if values.shape != shape:
            raise ValueError(f'{name}形状{values.shape}与目标{shape}不一致。')
        flat[name] = values.ravel()
        valid &= np.isfinite(flat[name])
    ids = np.flatnonzero(valid)
    frame = pd.DataFrame({'pixel_id': ids,
                          'SPEI_group': grid.ravel()[valid].astype(np.int64),
                          'target': target_array.ravel()[valid],
                          **{name: values[valid] for name, values in flat.items()}})
    return frame, list(arrays)


def _draw_and_split_repeat(data, grid_fraction, validation_fraction, draw_seed, split_seed,
                            split_unit='pixel'):
    """重复网格计数保留；同一原始像元的全部副本固定在同一侧。"""
    grids = np.sort(data['SPEI_group'].unique())
    n_draw = max(1, int(math.ceil(len(grids) * grid_fraction)))
    drawn = np.random.default_rng(draw_seed).choice(grids, size=n_draw, replace=True)
    counts = pd.Series(drawn).value_counts(sort=False)
    multiplicity = data['SPEI_group'].map(counts).fillna(0).to_numpy(dtype=int)
    selected = np.flatnonzero(multiplicity > 0)
    if split_unit == 'pixel':
        if len(selected) < 4:
            raise ValueError('本次抽中原始像元少于4个，无法稳定划分。')
        train, validation = _repeat_split(selected, test_size=validation_fraction, random_state=split_seed)
    elif split_unit == 'grid':
        selected_grids = np.sort(counts.index.to_numpy())
        if len(selected_grids) < 2:
            raise ValueError('网格划分至少需要抽中2个不同网格。')
        train_grids, validation_grids = _repeat_split(selected_grids, test_size=validation_fraction,
                                                     random_state=split_seed)
        train = selected[np.isin(data.iloc[selected]['SPEI_group'], train_grids)]
        validation = selected[np.isin(data.iloc[selected]['SPEI_group'], validation_grids)]
    else:
        raise ValueError("split_unit只能是'pixel'或'grid'。")
    if len(train) < 2 or len(validation) < 2:
        raise ValueError('本次训练或验证原始像元少于2个。')
    overlap = np.intersect1d(data.iloc[train]['pixel_id'], data.iloc[validation]['pixel_id'])
    if len(overlap):
        raise AssertionError('原始像元进入训练和验证两侧。')
    # 真正展开训练重复副本，保留RF内部bootstrap对重复行的正确抽样行为。
    train_expanded = np.repeat(train, multiplicity[train])
    train_counts = data.iloc[train].groupby('SPEI_group').size()
    val_counts = data.iloc[validation].groupby('SPEI_group').size()
    ledger = pd.DataFrame({'SPEI_group': counts.index, 'draw_count': counts.to_numpy()})
    ledger['train_unique_pixels'] = ledger['SPEI_group'].map(train_counts).fillna(0).astype(int)
    ledger['validation_unique_pixels'] = ledger['SPEI_group'].map(val_counts).fillna(0).astype(int)
    return train, validation, train_expanded, multiplicity, ledger


def _label_repeat_r2(value, status='ok'):
    if status != 'ok' or not np.isfinite(value):
        return 'failed_or_undefined'
    if value < 0:
        return '<0'
    if value <= 0.2:
        return '0<=R2<=0.2'
    if value <= 0.4:
        return '0.2<R2<=0.4'
    return '>0.4'


def _summarize_repeat_results(runs, importance, trim_fraction):
    runs = runs.copy()
    runs['retained'] = False
    summaries, frequencies = [], []
    bins = ['<0', '0<=R2<=0.2', '0.2<R2<=0.4', '>0.4', 'failed_or_undefined']
    for fraction, group in runs.groupby('grid_fraction', sort=True):
        valid = group[(group['status'] == 'ok') & np.isfinite(group['validation_r2'])]
        valid = valid.sort_values(['validation_r2', 'iteration'], kind='stable')
        n_drop = int(math.floor(len(valid) * trim_fraction))
        kept = valid.iloc[n_drop:]
        runs.loc[kept.index, 'retained'] = True
        summaries.append({'grid_fraction': fraction, 'n_attempted': len(group), 'n_valid': len(valid),
                          'n_failed': len(group)-len(valid), 'n_dropped': n_drop, 'n_retained': len(kept),
                          'all_r2_mean': valid['validation_r2'].mean(),
                          'all_r2_std': valid['validation_r2'].std(ddof=1),
                          'all_r2_median': valid['validation_r2'].median(),
                          'retained_r2_mean': kept['validation_r2'].mean(),
                          'retained_r2_std': kept['validation_r2'].std(ddof=1),
                          'retained_min_r2': kept['validation_r2'].min()})
        labels = group.apply(lambda row: _label_repeat_r2(row['validation_r2'], row['status']), axis=1)
        for label in bins:
            count = int((labels == label).sum())
            frequencies.append({'grid_fraction': fraction, 'bin': label, 'count': count,
                                'frequency_all_attempts': count / len(group),
                                'frequency_valid_runs': count / len(valid) if len(valid) and label != 'failed_or_undefined' else np.nan})
    if importance.empty:
        return runs, pd.DataFrame(summaries), pd.DataFrame(frequencies), pd.DataFrame(), importance
    detail = importance.merge(runs[['grid_fraction','iteration','retained']],
                              on=['grid_fraction','iteration'], validate='many_to_one')
    means = []
    for scope, subset in [('all_valid', detail), ('retained_top80', detail[detail['retained']])]:
        if subset.empty:
            continue
        table = subset.groupby(['grid_fraction','level','feature'])['importance'].agg(
            importance_mean='mean', importance_std='std', n_models='count').reset_index()
        table['scope'] = scope
        means.append(table)
    return runs, pd.DataFrame(summaries), pd.DataFrame(frequencies), pd.concat(means, ignore_index=True), detail


def _plot_repeat_results(frequencies, means, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    colors = ['#999999','#d73027','#fdae61','#1a9850','#333333']
    fractions = sorted(frequencies['grid_fraction'].unique())
    bins = ['<0', '0<=R2<=0.2', '0.2<R2<=0.4', '>0.4', 'failed_or_undefined']
    for i, fraction in enumerate(fractions):
        subset = frequencies[frequencies['grid_fraction'] == fraction].set_index('bin')
        bottom = 0.
        for label, color in zip(bins, colors):
            value = float(subset.loc[label, 'frequency_all_attempts'])
            axes[0].bar(i, value, bottom=bottom, color=color, label=label if i == 0 else None)
            if value > 0:
                axes[0].text(i, bottom+value/2, f'{value:.0%}', ha='center', va='center', fontsize=9)
            bottom += value
    axes[0].set_xticks(range(len(fractions)), [f'{f:.0%} grid draws' for f in fractions])
    axes[0].set_ylabel('Frequency across all attempted runs')
    axes[0].set_ylim(0,1.02)
    axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(0,1.2), ncol=2)
    if not means.empty:
        selected = means[(means['level'] == 'grouped') & (means['scope'] == 'retained_top80')]
        order = selected.groupby('feature')['importance_mean'].mean().sort_values().index.tolist()
        height = 0.8 / max(len(fractions), 1)
        for i, fraction in enumerate(fractions):
            subset = selected[selected['grid_fraction'] == fraction].set_index('feature').reindex(order)
            axes[1].barh(np.arange(len(order))+(i-(len(fractions)-1)/2)*height,
                         subset['importance_mean'], xerr=subset['importance_std'].fillna(0),
                         height=height, label=f'{fraction:.0%}', capsize=2)
        axes[1].set_yticks(range(len(order)), order)
        axes[1].set_xlabel('Mean RF impurity importance (retained runs)')
        axes[1].legend(title='Grid draw fraction')
    fig.savefig(Path(output_dir) / 'R2_frequency_and_retained_importance.png', dpi=200)
    plt.close(fig)


def calculate_best_repeat_shap(data, feature_names, best, output_dir, max_samples=10000):
    import shap
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not isinstance(max_samples,int) or max_samples < 1:
        raise ValueError('SHAP最大样本数必须是正整数。')
    pool = np.asarray(best['train_indices'],dtype=int)
    seed = int(best['model_seed'])
    selected = np.random.default_rng(seed).choice(pool,size=min(max_samples,len(pool)),replace=False)
    x = data.iloc[selected][feature_names].reset_index(drop=True)
    values = shap.TreeExplainer(best['model'])(x,check_additivity=False).values
    encoded = pd.concat([x,pd.DataFrame(values,columns=[f'shap_{c}' for c in feature_names])],axis=1)
    encoded.insert(0,'pixel_id',data.iloc[selected]['pixel_id'].to_numpy())
    encoded.insert(1,'SPEI_group',data.iloc[selected]['SPEI_group'].to_numpy())
    detail = pd.DataFrame({'feature':feature_names,'importance':best['model'].feature_importances_,
                            'shapvalue':np.abs(values).mean(axis=0)})
    grouped_data,grouped_importance = _group_categorical_shap(encoded,detail)
    encoded.to_csv(directory/'best_run_SHAP_encoded.csv',index=False,encoding='utf-8-sig')
    grouped_data.to_csv(directory/'best_run_SHAP_grouped.csv',index=False,encoding='utf-8-sig')
    grouped_importance.to_csv(directory/'best_run_importance_SHAP.csv',index=False,encoding='utf-8-sig')
    metadata = {k:v for k,v in best.items() if k not in ('model','train_indices','validation_indices')}
    metadata.update(shap_sample_count=len(x),shap_population='unique training pixels of best run',
                    selection='highest validation R2; ties keep first iteration',
                    shap_model='the actual fitted best-run model; not refitted')
    (directory/'best_run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return grouped_data,grouped_importance



def draw_repeat_d_panels(anchor, pcorr, means, shap_data, shap_importance, feature_order,
                          best, n_retained, n_valid, retained_r2_mean):
    import shap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    figure=anchor.figure
    grid=anchor.get_subplotspec().subgridspec(1,4,width_ratios=[1,1,1.25,.06],wspace=.18)
    anchor.axis('off')
    ax1,ax2,ax3,cax=[figure.add_subplot(grid[0,i]) for i in range(4)]
    n=len(feature_order)
    _plot_long_climate_pcorr(ax1,pcorr,feature_order,True)
    ax1.set_xlim(0,min(1.,max(.6,float(pcorr['Abs_Coefficient'].max())*1.08)))
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax1.set_xlabel('Partial correlation')
    ax1.set_title('(d1) Partial correlation',fontsize=9,loc='left',pad=28)
    handles=[Line2D([0],[0],marker='s',linestyle='none',color=c,label=l)
             for c,l in [('#4393c3','Negative'),('#d6604d','Positive')]]
    ax1.legend(handles=handles,loc='lower right',fontsize=7,frameon=False)
    rows=means.set_index('feature').reindex(feature_order)
    ax2.barh(np.arange(n),rows['importance_mean'],xerr=rows['importance_std'].fillna(0),
              color='#4393c3',alpha=.85,height=.45,capsize=2,error_kw={'elinewidth':.7})
    ax2.set_ylim(n-.5,-.5)
    ax2.set_yticks(np.arange(n))
    ax2.tick_params(axis='y',labelleft=False,length=0)
    ax2.set_xlim(left=0)
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.set_xlabel('Mean RF importance')
    ax2.set_title(f'(d2) Top {n_retained}/{n_valid}: mean ± SD\nMean validation $R^2$={retained_r2_mean:.3f}',
                  fontsize=9,loc='left',pad=17)
    explanation=shap.Explanation(
        values=shap_data[[f'shap_{c}' for c in feature_order]].to_numpy(dtype=float),
        data=shap_data[feature_order].to_numpy(dtype=float),feature_names=feature_order)
    plt.sca(ax3)
    shap.plots.beeswarm(explanation,order=np.arange(n),max_display=n,plot_size=None,
                        show=False,color_bar=False)
    for collection in ax3.collections:
        collection.set_sizes([5])
        collection.set_cmap('coolwarm')
    ax3.set_ylim(-.5,n-.5)
    ax3.tick_params(axis='y',labelleft=False,length=0)
    ax3.set_ylabel('')
    ax3.set_xlabel('Best-run SHAP value')
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax3.set_title(f'(d3) Best run #{best["iteration"]}\nValidation $R^2$={best["validation_r2"]:.3f}',
                  fontsize=9,loc='left',pad=45)
    twin=ax3.twiny()
    shap_mean=shap_importance.set_index('feature').reindex(feature_order)['shapvalue'].to_numpy(dtype=float)
    twin.barh(np.arange(n),shap_mean[::-1],color='gray',alpha=.2,height=.45)
    twin.set_ylim(-.5,n-.5)
    twin.set_xlim(0,max(.01,float(shap_mean.max())*1.1))
    twin.xaxis.set_major_locator(MaxNLocator(3))
    twin.set_xlabel('Best-run mean |SHAP|',fontsize=8)
    twin.tick_params(axis='both',labelsize=8)
    ax3.set_zorder(twin.get_zorder()+1)
    ax3.patch.set_visible(False)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cb=figure.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='coolwarm'),cax=cax)
    cb.set_ticks([0,1],labels=['Low','High'])
    cb.set_label('Feature value',fontsize=8)
    cb.ax.tick_params(labelsize=8,length=0)
    cb.outline.set_visible(False)
    for ax in [ax1,ax2,ax3]:
        ax.tick_params(labelsize=8)
    return ax1,ax2,ax3,twin



def plot_cached_repeat_fig6(result,data,features,directory,target_name,drought_data,wet_data,show=True):
    """只读取已经保存的前80%重要性和最佳SHAP，不重新拟合模型/计算SHAP。"""
    directory=Path(directory)
    members={c for cols in ML_CATEGORY_GROUPS.values() for c in cols}
    order=[c for c in features if c not in members]+list(ML_CATEGORY_GROUPS)
    partial=_pcorr_with_long_climate(data.rename(columns={'target':target_name}),
        target_name,features[:7],LONG_CLIMATE_FEATURES)
    partial=partial.rename(columns={'var':'feature','r':'r_pcorr','p-val':'p_pcorr'})
    partial['Abs_Coefficient']=partial['r_pcorr'].abs()
    partial['Significance']=np.where(partial['p_pcorr']<.01,'**',np.where(partial['p_pcorr']<.05,'*',''))
    figures=[]
    for fraction in sorted(result['runs']['grid_fraction'].unique()):
        folder=directory/f'p{fraction*100:g}'
        best_dir=folder/'best'
        if not (best_dir/'best_run_metadata.json').is_file():
            print(f'No valid best model, skip Fig6: {folder}',flush=True)
            continue
        means=pd.read_csv(folder/'summary'/'top80_mean_importance.csv')
        means=means[means['level']=='grouped']
        shap_data=pd.read_csv(best_dir/'best_run_SHAP_grouped.csv')
        shap_importance=pd.read_csv(best_dir/'best_run_importance_SHAP.csv')
        best=json.loads((best_dir/'best_run_metadata.json').read_text(encoding='utf-8'))
        summary=result['summary'].set_index('grid_fraction').loc[fraction]
        partial.to_csv(folder/'summary'/'partial_correlation.csv',index=False,encoding='utf-8-sig')
        output=folder/'figures'
        output.mkdir(exist_ok=True)
        # 独立d面板：便于直接使用和核查。
        fig=plt.figure(figsize=(12.5,7.5))
        draw_repeat_d_panels(fig.add_subplot(111),partial,means,shap_data,shap_importance,
            order,best,int(summary['n_retained']),int(summary['n_valid']),float(summary['retained_r2_mean']))
        fig.subplots_adjust(left=.14,right=.96,bottom=.11,top=.80)
        for ext in ['png','pdf']:
            fig.savefig(output/f'Fig6_d.{ext}',dpi=300,bbox_inches='tight')
        figures.append(fig)
        # 完整Fig6，顶部继续调用你的原plot_pos。
        fig=plt.figure(figsize=(11.8,10.5))
        grid=fig.add_gridspec(2,1,height_ratios=[1,1.65],hspace=.65)
        plot_pos(drought_data,wet_data,-20,20,'All',fig.add_subplot(grid[0]))
        draw_repeat_d_panels(fig.add_subplot(grid[1]),partial,means,shap_data,shap_importance,
            order,best,int(summary['n_retained']),int(summary['n_valid']),float(summary['retained_r2_mean']))
        fig.subplots_adjust(left=.13,right=.96,bottom=.07,top=.94)
        for ext in ['png','pdf']:
            fig.savefig(output/f'Fig6_full.{ext}',dpi=300,bbox_inches='tight')
        figures.append(fig)
        print(f'Fig6 saved: {output}; d2=top {int(summary["n_retained"])}/{int(summary["n_valid"])} mean, '
              f'd3=best run #{best["iteration"]}, R2={best["validation_r2"]:.4f}',flush=True)
    if show:
        plt.show()
    for figure in figures:
        plt.close(figure)


def _save_requested_repeat_layout(directory,data,features,runs,summary,frequency,
                                   means,detail,ledger,best_models):
    import joblib
    directory=Path(directory)
    for fraction in sorted(runs['grid_fraction'].unique()):
        folder=directory/f'p{fraction*100:g}'
        summary_dir=folder/'summary'
        summary_dir.mkdir(parents=True,exist_ok=True)
        for table,filename in [(runs,'runs.csv'),(detail,'importance_each_run.csv'),
                                (ledger,'grid_draw_counts.csv')]:
            if not table.empty:
                table[table['grid_fraction']==fraction].to_csv(folder/filename,index=False,encoding='utf-8-sig')
        for table,filename in [(summary,'r2_summary.csv'),(frequency,'r2_frequency.csv')]:
            table[table['grid_fraction']==fraction].to_csv(summary_dir/filename,index=False,encoding='utf-8-sig')
        if not means.empty:
            top=means[(means['grid_fraction']==fraction)&(means['scope']=='retained_top80')]
            top.to_csv(summary_dir/'top80_mean_importance.csv',index=False,encoding='utf-8-sig')
        if fraction not in best_models:
            continue
        best=best_models[fraction]
        best_dir=folder/'best'
        best_dir.mkdir(exist_ok=True)
        calculate_best_repeat_shap(data,features,best,best_dir,10000)
        joblib.dump(best['model'],best_dir/'model.joblib',compress=3)
        for split,indices in [('train',best['train_indices']),('validation',best['validation_indices'])]:
            data.iloc[indices][['pixel_id','SPEI_group','target']].to_csv(
                best_dir/f'{split}_pixels.csv',index=False,encoding='utf-8-sig')


def run_repeated_grid_rf(data, feature_names, output_dir, grid_fractions=(0.25, 0.50),
                         n_repeats=50, validation_fraction=0.2, trim_fraction=0.2,
                         rf_params=None, random_state=42, split_unit='pixel', make_plot=True,
                         progress=True):
    """每个比例独立重复50次；按验证R2剔除最低10次后汇总。

    默认8:2按原始像元划分，不是空间独立网格验证。重复网格的所有像元均纳入，
    重复次数在训练行和验证权重中保留，不存在同一像元重复副本跨训练/验证。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = data.reset_index(drop=True).copy()
    # 单比例写作(0.25)时兼容浮点Input，不改变抽样比例。
    if np.isscalar(grid_fractions):
        grid_fractions = (float(grid_fractions),)
    fractions = tuple(float(f) for f in grid_fractions)
    if not fractions or len(set(fractions)) != len(fractions) or any(not 0 < f <= 1 for f in fractions):
        raise ValueError('grid_fractions应是不重复的(0,1]内比例。')
    if not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError('n_repeats必须是正整数。')
    if not 0 < validation_fraction < 1 or not 0 <= trim_fraction < 1:
        raise ValueError('验证比例应在(0,1)，剔除比例应在[0,1)。')
    if split_unit not in ('pixel','grid'):
        raise ValueError('split_unit只能为pixel或grid。')
    if data['pixel_id'].duplicated().any() or len(data) < 4:
        raise ValueError('data必须一行一个原始像元且至少4行。')
    if data['SPEI_group'].isna().any() or data['SPEI_group'].nunique() < 2:
        raise ValueError('需要至少2个有效55km网格，且网格ID不能缺失。')
    if len(set(feature_names)) != len(feature_names) or set(feature_names) & {'pixel_id','SPEI_group','target'}:
        raise ValueError('解释变量必须唯一且不能包含ID或目标列。')
    if not np.isfinite(data[feature_names+['target']].to_numpy(dtype=float)).all():
        raise ValueError('输入必须是全部解释变量及目标均有效的完整案例。')
    params = dict(n_estimators=500, max_depth=20, min_samples_split=5,
                  min_samples_leaf=5, max_features=0.7, n_jobs=-1, bootstrap=True)
    if rf_params:
        if 'random_state' in rf_params:
            raise ValueError('请通过函数random_state设置随机种子，不放入rf_params。')
        params.update(rf_params)
    config = dict(grid_fractions=fractions, n_repeats=n_repeats,
                  validation_fraction=validation_fraction, trim_fraction=trim_fraction,
                  random_state=random_state, split_unit=split_unit, rf_params=params,
                  n_available_pixels=len(data), n_available_grids=int(data['SPEI_group'].nunique()),
                  feature_names=list(feature_names), replacement=True,
                  split_rule='unique original pixels; all copies remain on same side',
                  r2_primary='validation multiplicity-weighted R2',
                  trim_rule='floor(trim_fraction*n_valid); ties broken by iteration',
                  importance='RF impurity importance; no SHAP; no OOB-error sample pruning')
    (output_dir/'configuration.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    seeds = np.random.SeedSequence(random_state).spawn(len(fractions)*n_repeats)
    runs, importance_rows, ledgers = [], [], []
    best_models = {}  # 仅缓存每个比例的最佳已拟合模型
    for fraction_idx, fraction in enumerate(fractions):
        for iteration in range(1,n_repeats+1):
            draw_seed, split_seed, model_seed = [int(v) for v in seeds[fraction_idx*n_repeats+iteration-1].generate_state(3)]
            row = dict(grid_fraction=fraction, iteration=iteration, status='failed', error='',
                       validation_r2=np.nan, draw_seed=draw_seed, split_seed=split_seed, model_seed=model_seed)
            try:
                train, val, expanded, multiplicity, ledger = _draw_and_split_repeat(
                    data, fraction, validation_fraction, draw_seed, split_seed, split_unit)
                ledger['grid_fraction'], ledger['iteration'] = fraction, iteration
                ledgers.append(ledger)
                model = _RepeatRF(**params, random_state=model_seed)
                model.fit(data.iloc[expanded][feature_names], data.iloc[expanded]['target'])
                y_val = data.iloc[val]['target'].to_numpy()
                weights = multiplicity[val]
                if np.ptp(y_val) == 0:
                    raise ValueError('验证目标为常数，R2无定义。')
                prediction = model.predict(data.iloc[val][feature_names])
                score = _repeat_r2(y_val, prediction, sample_weight=weights, force_finite=False)
                if not np.isfinite(score):
                    raise ValueError('本次验证R2非有限值。')
                train_prediction = model.predict(data.iloc[train][feature_names])
                row.update(status='ok', validation_r2=float(score),
                    validation_r2_unique=float(_repeat_r2(y_val,prediction,force_finite=False)),
                    validation_rmse=float(np.sqrt(_repeat_mse(y_val,prediction,sample_weight=weights))),
                    train_r2=float(_repeat_r2(data.iloc[train]['target'],train_prediction,sample_weight=multiplicity[train])),
                    n_grid_draws=int(ledger['draw_count'].sum()), n_unique_grids=len(ledger),
                    n_selected_unique_pixels=len(train)+len(val), n_train_unique=len(train),
                    n_validation_unique=len(val), n_train_with_repeats=len(expanded),
                    n_validation_with_repeats=int(weights.sum()), duplicate_pixel_overlap=0,
                    n_grid_overlap=len(set(data.iloc[train]['SPEI_group']) & set(data.iloc[val]['SPEI_group'])))
                # 只保存模型引用和原始像元索引，不重新训练、不改变本轮结果。
                if fraction not in best_models or score > best_models[fraction]['validation_r2']:
                    best_models[fraction] = dict(model=model,train_indices=train.copy(),
                        validation_indices=val.copy(),grid_fraction=float(fraction),iteration=iteration,
                        validation_r2=float(score),draw_seed=draw_seed,split_seed=split_seed,
                        model_seed=model_seed,n_train_unique=len(train),n_validation_unique=len(val))
                raw = dict(zip(feature_names, model.feature_importances_))
                for feature, value in raw.items():
                    importance_rows.append(dict(grid_fraction=fraction, iteration=iteration,
                        level='encoded', feature=feature, importance=float(value)))
                members = {c for cols in ML_CATEGORY_GROUPS.values() for c in cols}
                grouped = {c:v for c,v in raw.items() if c not in members}
                grouped.update({group:sum(raw.get(c,0.) for c in cols)
                                for group,cols in ML_CATEGORY_GROUPS.items() if any(c in raw for c in cols)})
                for feature,value in grouped.items():
                    importance_rows.append(dict(grid_fraction=fraction,iteration=iteration,
                        level='grouped',feature=feature,importance=float(value)))
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
            runs.append(row)
            if progress:
                print(f'Grid fraction={fraction:.0%} repeat={iteration}/{n_repeats} '
                      f'validation R2={row["validation_r2"]:.4f} status={row["status"]}', flush=True)
            # 每轮检查点：长计算中断时仍保留已完成结果。
            pd.DataFrame(runs).to_csv(output_dir/'runs_checkpoint.csv',index=False,encoding='utf-8-sig')
    raw_importance = pd.DataFrame(importance_rows, columns=['grid_fraction','iteration','level','feature','importance'])
    runs, summary, frequency, means, detail = _summarize_repeat_results(pd.DataFrame(runs), raw_importance, trim_fraction)
    files = {'runs_all.csv':runs, 'r2_summary.csv':summary, 'r2_frequency.csv':frequency,
             'importance_each_run.csv':detail, 'importance_means_all_and_retained.csv':means,
             'grid_draw_counts.csv':pd.concat(ledgers,ignore_index=True) if ledgers else pd.DataFrame()}
    # 保存布局变更：每次记录、前80%均值和最佳模型分别保存。
    _save_requested_repeat_layout(output_dir,data,feature_names,runs,summary,frequency,
                                  means,detail,files['grid_draw_counts.csv'],best_models)
    if make_plot:
        _plot_repeat_results(frequency,means,output_dir)
    if summary['n_valid'].sum() == 0:
        raise RuntimeError(f'所有重复均失败；请检查{output_dir}/p*/runs.csv中的error。')
    return dict(runs=runs,summary=summary,frequency=frequency,importance_each=detail,
                importance_means=means,grid_draws=files['grid_draw_counts.csv'])


### 控制 干旱事件次数 + 干旱强度
pos_mean_difference_indrought_Nodrought = np.where((drought_event_count >= drought_or_wet_times) &
                                                   np.isfinite(mcd12q2_mask) & np.isfinite(gpp_mask) ,
                                                   # np.isfinite(r2_mcd12q2_ppt_mask) & np.isfinite(r2_mcd12q2_sos_mask),
                                                   pos_diff_drought_normal_origin, np.nan)
pos_mean_difference_inwet_Nowet = np.where((wet_event_count >= drought_or_wet_times) &
                                           np.isfinite(mcd12q2_mask) & np.isfinite(gpp_mask) ,
                                           # np.isfinite(r2_mcd12q2_ppt_mask) & np.isfinite(r2_mcd12q2_sos_mask),
                                           pos_diff_wet_normal_origin, np.nan)

print(f'pos_mean_difference_indrought_Nodrought有效像元数量：{np.count_nonzero(np.isfinite(pos_mean_difference_indrought_Nodrought))}')
print(f'pos_mean_difference_inwet_Nowet有效像元数量：{np.count_nonzero(np.isfinite(pos_mean_difference_inwet_Nowet))}')

if analyze_by == 'All':
    mask = drought_event_count >= drought_or_wet_times
elif analyze_by == 'advance PPT':
    mask = (drought_event_count >= drought_or_wet_times) & (pos_mean_difference_indrought_Nodrought < -3)
elif analyze_by == 'delay PPT':
    mask = (drought_event_count >= drought_or_wet_times) & (pos_mean_difference_indrought_Nodrought > 0)

mask = (mask & np.isfinite(mcd12q2_mask) & np.isfinite(gpp_mask) & #np.isfinite(r2_mcd12q2_ppt_mask) & np.isfinite(r2_mcd12q2_sos_mask)&
                np.isfinite(veg_type_data) & (veg_type_data < 5) &  # Forest:1 Shurb:2 Savnna:3 Grass:4
                np.isfinite(ai_type_data) & (ai_type_data < 7)) #Arid:2  Semi-arid:3+4  Dry sub-humid:5  Humid:6
                # np.isfinite(cor_mean_data) & (cor_mean_data < 0.3)) # -0.6 ~ 0


pos_diff_drought_normal = np.where(mask, pos_diff_drought_normal, np.nan)
sos_diff_drought_normal = np.where(mask, sos_diff_drought_normal, np.nan)
cor_diff_drought_normal = np.where(mask, cor_diff_drought_normal, np.nan)
sm_diff_drought_normal = np.where(mask, sm_diff_drought_normal, np.nan)
vpd_diff_drought_normal = np.where(mask, vpd_diff_drought_normal, np.nan)
ta_diff_drought_normal = np.where(mask, ta_diff_drought_normal, np.nan)
pre_diff_drought_normal = np.where(mask, pre_diff_drought_normal, np.nan)
srad_diff_drought_normal = np.where(mask, srad_diff_drought_normal, np.nan)

print(f'pos_diff_drought_normal有效像元数量：{np.count_nonzero(np.isfinite(pos_diff_drought_normal))}')
print(f'cor_diff_drought_normal有效像元数量：{np.count_nonzero(np.isfinite(cor_diff_drought_normal))}')

# sys.exit()
#
# def Outlier_2D_IQR(data, qmin=25, qmax=75, k=1.5):
#     """
#     对 2D 空间图做全局 IQR Remove outliers。
#     Returns: Remove outliers后的 2D 数组（异常值 -> NaN）
#     """
#     data = np.asarray(data, dtype=np.float32)
#     valid = data[np.isfinite(data)]
#
#     if valid.size == 0:
#         return np.full_like(data, np.nan)
#
#     q_low, q_high = np.percentile(valid, [qmin, qmax])
#     iqr = q_high - q_low
#
#     lower = q_low - k * iqr
#     upper = q_high + k * iqr
#
#     out = np.where((data >= lower) & (data <= upper), data, np.nan)
#     return out
#
# pos_diff_drought_normal  = Outlier_2D_IQR(pos_diff_drought_normal,  25, 75, 1.5)
# sos_diff_drought_normal  = Outlier_2D_IQR(sos_diff_drought_normal,  25, 75, 1.5)
# cor_diff_drought_normal  = Outlier_2D_IQR(cor_diff_drought_normal,  25, 75, 1.5)
# sm_diff_drought_normal   = Outlier_2D_IQR(sm_diff_drought_normal,   25, 75, 1.5)
# vpd_diff_drought_normal  = Outlier_2D_IQR(vpd_diff_drought_normal,  25, 75, 1.5)
# ta_diff_drought_normal   = Outlier_2D_IQR(ta_diff_drought_normal,   25, 75, 1.5)
# pre_diff_drought_normal  = Outlier_2D_IQR(pre_diff_drought_normal,  25, 75, 1.5)
# srad_diff_drought_normal = Outlier_2D_IQR(srad_diff_drought_normal, 25, 75, 1.5)

# ## Fig 6
# REPEATED_GRID_RF: 前面的data读取与筛选照旧；新流程不进行每格18像元上限抽样。
def load_original_fig6_cache(folder):
    folder=Path(folder)
    mean=pd.read_csv(folder/'summary'/'top80_mean_importance.csv')
    mean=mean[mean['level']=='grouped'][['feature','importance_mean']].rename(columns={'importance_mean':'importance'})
    best=pd.read_csv(folder/'best'/'best_run_importance_SHAP.csv')
    merged=best.drop(columns=['importance']).merge(mean,on='feature',validate='one_to_one')
    return dict(importance=merged,shap=pd.read_csv(folder/'best'/'best_run_SHAP_grouped.csv'))




######## 分Vegetation type、AI梯度
def repeated_analysis_groups():
    shape = np.asarray(pos_diff_drought_normal).shape
    # 开启任一分组时仅运行分组；两者均关闭时才运行全植被。
    groups = []
    if not RUN_GROUPED_VEG_RF and not RUN_GROUPED_AI_RF:
        groups.append(('All', 'All', np.ones(shape, dtype=bool)))
    if RUN_GROUPED_VEG_RF:
        groups.extend(('Veg', name, veg_type_data == code)
                      for name, code in [('Forest',1), ('Shrub',2), ('Savanna',3), ('Grass',4)])
    if RUN_GROUPED_AI_RF:
        groups.extend(('AI', name, np.isin(ai_type_data, codes))
                      for name, codes in [('Arid',[2]), ('Semi-arid',[3,4]),
                                          ('Dry sub-humid',[5]), ('Humid',[6])])
    return groups


def run_or_plot_all_repeat_groups():
    # 与原全植被保持相同的目标、参数、抽样、划分、排名及SHAP算法。
    if list(y_list) != ['POS drought normal diff']:
        raise ValueError('本流程仅分析POS drought normal diff；湿润data仅用于Fig6顶部差异图。')
    target = y_list[0]
    differences = {f'{name} drought normal diff': array for name, array in zip(
        ['SOS','Cor','SM','VPD','Ta','Pre','Srad'],
        [sos_diff_drought_normal,cor_diff_drought_normal,sm_diff_drought_normal,
         vpd_diff_drought_normal,ta_diff_drought_normal,pre_diff_drought_normal,srad_diff_drought_normal])}
    fractions = ((float(REPEAT_GRID_FRACTIONS),) if np.isscalar(REPEAT_GRID_FRACTIONS)
                 else tuple(float(x) for x in REPEAT_GRID_FRACTIONS))
    if not fractions or any(not 0 < x <= 1 for x in fractions):
        raise ValueError('REPEAT_GRID_FRACTIONS必须在(0,1]内。')
    root = Path(output_partial_max_var_r_tif_path)/'GridRF'/f'D{float(spei_strength):g}_N{REPEAT_RF_N}'
    groups = repeated_analysis_groups()
    cached_groups = {fraction: {} for fraction in fractions}
    for grade, name, mask in groups:
        directory = root/'dry' if grade == 'All' else root/'dry'/grade/name.replace(' ', '_')
        if RUN_REPEATED_GRID_RF:
            data, features = prepare_repeated_rf_data(
                pos_diff_drought_normal, differences, spei_grid_group,
                sm_long_data,vpd_long_data,ta_long_data,pre_long_data,srad_long_data,
                veg_type_data,ai_type_data,selection_mask=mask)
            print(f'开始重复RF：{grade}/{name}，{len(data)}个有效像元，{data.SPEI_group.nunique()}个55km网格', flush=True)
            result = run_repeated_grid_rf(data,features,directory,
                grid_fractions=fractions,n_repeats=REPEAT_RF_N,
                validation_fraction=REPEAT_RF_VALIDATION_FRACTION,
                trim_fraction=REPEAT_RF_TRIM_FRACTION,rf_params=REPEAT_RF_PARAMS,
                random_state=REPEAT_RF_RANDOM_STATE,split_unit=REPEAT_RF_SPLIT_UNIT)
            print(result['summary'].to_string(index=False))
        for fraction in fractions:
            folder = directory/f'p{fraction*100:g}'
            required = [folder/'summary'/'top80_mean_importance.csv',
                        folder/'best'/'best_run_SHAP_grouped.csv',
                        folder/'best'/'best_run_importance_SHAP.csv']
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise FileNotFoundError(f'{grade}/{name}缺少缓存，不自动补训：\n'+'\n'.join(missing))
            cached_groups[fraction][(grade,name)] = {target:load_original_fig6_cache(folder)}
            if grade == 'All':
                output = folder/'figures'
                output.mkdir(exist_ok=True)
                # 保留原Fig6，包括顶部干旱/湿润PPT差异；底部只分析干旱。
                plot_fig6(pos_mean_difference_indrought_Nodrought,pos_mean_difference_inwet_Nowet,-20,20,
                    pos_diff_drought_normal,sos_diff_drought_normal,cor_diff_drought_normal,
                    sm_diff_drought_normal,vpd_diff_drought_normal,ta_diff_drought_normal,
                    pre_diff_drought_normal,srad_diff_drought_normal,
                    cached_repeat=cached_groups[fraction][(grade,name)],
                    save_path=str(output/'Fig6_original.png'), show=False)
    for fraction in fractions:
        for grade, enabled in [('Veg',RUN_GROUPED_VEG_RF),('AI',RUN_GROUPED_AI_RF)]:
            if not enabled:
                continue
            caches = {name:value for (kind,name),value in cached_groups[fraction].items() if kind == grade}
            with plt.ioff():
                fig = Partial_ML_calculate_and_plot(
                    pos_diff_drought_normal,sos_diff_drought_normal,cor_diff_drought_normal,
                    sm_diff_drought_normal,vpd_diff_drought_normal,ta_diff_drought_normal,
                    pre_diff_drought_normal,srad_diff_drought_normal,grade,'All',None,
                    cached_repeat=caches)
                save_dir = Path(fr'D:/code/Fig 6 Driver of POS difference in drought event')/grade
                save_dir.mkdir(parents=True,exist_ok=True)
                final_path = save_dir/f'SPEI{spei_strength}_N{REPEAT_RF_N}_p{fraction*100:g}_{grade}.png'
                fig.canvas.draw()
                fig.savefig(final_path,dpi=300,bbox_inches='tight')
                print(f'分组Fig6已保存：{final_path}',flush=True)
    # 所有分组计算及图片保存完成后，再统一显示，避免首张图阻塞后续流程。
    print('本次启用的分析均已完成计算和保存。', flush=True)
    plt.show()


run_or_plot_all_repeat_groups()
# sys.exit()  # 全植被及分组流程均完成后退出，禁止落入旧的单次RF入口。



# plot_fig6(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet,
#           -20, 20,
#           pos_diff_drought_normal, sos_diff_drought_normal,
#           cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#           ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal)
#
# print('Fig6 or S22 plot done!')

# sys.exit()

# # # # Fig S 23 - 25
# ax = plt.figure(figsize=(17, 11))
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -20, 20, 'Veg', ax)
# print('S23 plot done!')
# plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -20, 20, 'AI', ax)
# print('S24 plot done!')
# # plot_pos(pos_mean_difference_indrought_Nodrought, pos_mean_difference_inwet_Nowet, -12, 12, 'Cor mean', ax)
# # print('S25 plot done!')
#
sys.exit()


# Fig S 26 - 27
ax = plt.figure(figsize=(17, 11))
Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
                              cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
                              ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
                              'Veg', 'All', ax)
print('S26 plot done!')

Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
                              cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
                              ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
                              'AI', 'All', ax)
print('S27 plot done!')
#
# Partial_ML_calculate_and_plot(pos_diff_drought_normal, sos_diff_drought_normal,
#                               cor_diff_drought_normal, sm_diff_drought_normal, vpd_diff_drought_normal,
#                               ta_diff_drought_normal, pre_diff_drought_normal, srad_diff_drought_normal,
#                               'Cor mean', 'All', ax)
# print('S28 plot done!')

sys.exit()

