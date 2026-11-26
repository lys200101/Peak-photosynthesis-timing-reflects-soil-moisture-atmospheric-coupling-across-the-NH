
import os
import glob
import datetime
import sys

# import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from duckdb.experimental.spark.sql.functions import isnan
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import pingouin as pg
from scipy.stats import pearsonr
from joblib import Parallel, delayed, parallel_backend
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
import pymannkendall as mk
from xgboost import XGBRegressor
import shap
import gc
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
from matplotlib.patches import Circle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import math
import random



######### Function #########
### Read SM and VPD bands
def get_band(tif_file, stack):
    tif = gdal.Open(tif_file)
    data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
    stack.append(data)

def clip_by_latitude(gt, rows, cols, lat_min, lat_max):
    """
    根据纬度范围裁剪数据
    返回: (row_start, row_end, new_gt)
    """
    # Calculate the center latitude of each row
    row_centers = np.arange(rows) * gt[5] + gt[3] + gt[5] / 2

    # Find rows with latitudes between 30 and 90 degrees
    valid_rows = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(valid_rows):
        raise ValueError(f"在纬度范围 {lat_min}-{lat_max} 内没有找到有效data")

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

def get_band_clip(tif_file, stack, row_start, row_end):
    tif = gdal.Open(tif_file)
    climate_data = tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    climate_data_cropped = climate_data[row_start:row_end, :]

    if scale == 11 and Basedon == 'Based_on_OriginPheno':
        ## MCD12Q2
        # climate_data_cropped = np.where(climate_data_cropped != 0, climate_data_cropped, np.nan)
        ## MOD17A2HGF / GIMMS NDVI
        mask = (climate_data_cropped > 90) & (climate_data_cropped < 300)
        climate_data_cropped = np.where(mask, climate_data_cropped, np.nan)

    stack.append(climate_data_cropped)

    tif = None  # Release resources promptly


def save_tif_gdal(output_path, data, rows, cols, crs, transform):

        driver = gdal.GetDriverByName("GTiff")

        # 创建Outputdata集（覆盖模式）
        output_ds = driver.Create(
            output_path,
            int(cols),  # 宽度（列数）
            int(rows),  # 高度（行数）
            1,  # 波段数
            gdal.GDT_Float32  # 默认data类型（可根据需求修改）
        )
        if not output_ds:
            raise RuntimeError(f"Cannot create output file: {output_path}")

        output_band = output_ds.GetRasterBand(1)  # 单波段索引为 1
        output_band.WriteArray(data, 0, 0)  # 写入data（0,0 表示左上角起始）
        output_band.SetNoDataValue(0)

        output_ds.SetProjection(crs)

        output_ds.SetGeoTransform(transform)

        output_ds = None  # 关闭data集（必须！否则文件可能损坏）
        return True


# def cal_partial(i, j, vars, y, pos_matrix, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):
def cal_partial(i, j, vars, y, pos_matrix, sos_matrix, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):
# def cal_partial(i, j, vars, y, pos_stack, cor_matrix, sm_matrix, vpd_matrix):

    # mask = (~np.isnan(pos_stack[:, i, j]) &
    #         ~np.isnan(cor_matrix[:, i, j]) &
    #         ~np.isnan(sm_matrix[:, i, j]) &
    #         ~np.isnan(vpd_matrix[:, i, j]) &
    #         ~np.isnan(ta_matrix[:, i, j]) &
    #         ~np.isnan(pre_matrix[:, i, j]) &
    #         ~np.isnan(srad_matrix[:, i, j]))\

    all_vars = np.stack([
        pos_matrix[:, i, j],
        sos_matrix[:, i, j],
        cor_matrix[:, i, j],
        sm_matrix[:, i, j],
        vpd_matrix[:, i, j],
        ta_matrix[:, i, j],
        pre_matrix[:, i, j],
        srad_matrix[:, i, j]
    ], axis=0)
    # mask: 哪些Time点所有变量都不是 NaN

    all_year = drought_year_stack[analyzed_start:analyzed_end, i, j]

    mask = np.all((np.isfinite(all_vars)) & (all_year != 2), axis=0)   #如果np前加~就是返回了无效值
    # mask = np.all(np.isfinite(all_vars), axis=0)   #如果np前加~就是返回了无效值

    if mask.sum() >= (years_length/2): #Statistics有效值数量

        # pos = np.array(pos_matrix[:, i, j][mask])
        #
        # sos_time_series = np.array(sos_matrix[:, i, j][mask])
        # cor_time_series = np.array(cor_matrix[:, i, j][mask])
        # sm_time_series = np.array(sm_matrix[:, i, j][mask])
        # vpd_time_series = np.array(vpd_matrix[:, i, j][mask])
        # ta_time_series = np.array(ta_matrix[:, i, j][mask])
        # pre_time_series = np.array(pre_matrix[:, i, j][mask])
        # srad_time_series = np.array(srad_matrix[:, i, j][mask])

        pos = np.where(mask, pos_matrix[:, i, j], np.nan)

        sos_time_series = np.where(mask, sos_matrix[:, i, j], np.nan)
        cor_time_series = np.where(mask, cor_matrix[:, i, j], np.nan)
        sm_time_series = np.where(mask, sm_matrix[:, i, j], np.nan)
        vpd_time_series = np.where(mask, vpd_matrix[:, i, j], np.nan)
        ta_time_series = np.where(mask, ta_matrix[:, i, j], np.nan)
        pre_time_series = np.where(mask, pre_matrix[:, i, j], np.nan)
        srad_time_series = np.where(mask, srad_matrix[:, i, j], np.nan)

        if variable_type == 'sos and climate':
            pixel_data = pd.DataFrame({
                'POS': pos,
                'SOS': sos_time_series,
                'Coupling': cor_time_series,   #cor
                'SM': sm_time_series,     #sm
                'VPD': vpd_time_series,    #vpd
                'Ta': ta_time_series,   #ta
                'Pre': pre_time_series, #pre
                'Srad': srad_time_series #srad
                })

        elif variable_type == 'only climate':
            pixel_data = pd.DataFrame({
                'POS': pos,
                'Coupling': cor_time_series,  # cor
                'SM': sm_time_series,  # sm
                'VPD': vpd_time_series,  # vpd
                'Ta': ta_time_series,  # ta
                'Pre': pre_time_series,  # pre
                'Srad': srad_time_series  # srad
            })

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

        # 【修改1】仅有效偏相关参与排名，不筛选显著性。
        top_vars_unique = (
            result_df.loc[np.isfinite(result_df['r'])]
            .sort_values(
                by='r',
                key=lambda values: values.abs(),
                ascending=False,
                kind='stable',
            )
            .drop_duplicates(subset='var')
            .head(3)
            .reset_index(drop=True)
        )

        # 【修改2】预先设置缺失结果。
        # 如果不足3个有效偏相关，剩余名次保持缺失。
        names = ['nan', 'nan', 'nan']
        r_values = [np.nan, np.nan, np.nan]
        p_values = [np.nan, np.nan, np.nan]

        # 【修改3】使用iloc按排序后的position取值。
        for rank in range(len(top_vars_unique)):
            row = top_vars_unique.iloc[rank]

            names[rank] = row['var']
            r_values[rank] = row['r']
            p_values[rank] = row['p-val']

        first_varname, second_varname, third_varname = names

        first_r, second_r, third_r = r_values

        first_p, second_p, third_p = p_values

        if variable_type == 'sos and climate':
            sos_pcor = result_df.loc[result_df['var'] == 'SOS', 'r'].iloc[0]
        elif variable_type == 'only climate':
            sos_pcor = np.nan
        sm_pcor = result_df.loc[result_df['var'] == 'SM', 'r'].iloc[0]
        vpd_pcor = result_df.loc[result_df['var'] == 'VPD', 'r'].iloc[0]
        cor_pcor = result_df.loc[result_df['var'] == 'Coupling', 'r'].iloc[0]
        ta_pcor = result_df.loc[result_df['var'] == 'Ta', 'r'].iloc[0]
        pre_pcor = result_df.loc[result_df['var'] == 'Pre', 'r'].iloc[0]
        srad_pcor = result_df.loc[result_df['var'] == 'Srad', 'r'].iloc[0]


    else:
        first_varname = 'nan'
        second_varname = 'nan'
        third_varname = 'nan'

        first_r = np.nan
        second_r = np.nan
        third_r = np.nan

        first_p = np.nan
        second_p = np.nan
        third_p = np.nan

        sos_pcor = np.nan
        sm_pcor = np.nan
        vpd_pcor = np.nan
        cor_pcor = np.nan
        ta_pcor = np.nan
        pre_pcor = np.nan
        srad_pcor = np.nan

    return i, j, first_varname, second_varname, third_varname, first_r, second_r, third_r, first_p, second_p, third_p, sos_pcor, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor
    # return i, j, first_varname, second_varname, first_r, second_r, first_p, second_p, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor


def cal_ML(i, j, pos_matrix, sos_matrix, cor_matrix, sm_matrix, vpd_matrix, ta_matrix, pre_matrix, srad_matrix):

    if np.sum(np.isfinite(pos_matrix[:, i, j])) > 0:

        ### 1 获取3×3的网格
        if grid_size == '3*3':
            r_start, r_end = max(0, i - 1), min(rows, i + 2)
            c_start, c_end = max(0, j - 1), min(cols, j + 2)
        elif grid_size == '5*5':
            r_start, r_end = max(0, i - 2), min(rows, i + 3)
            c_start, c_end = max(0, j - 2), min(cols, j + 3)
        elif grid_size == '10*10':
            r_start, r_end = max(0, i - 5), min(rows, i + 5)
            c_start, c_end = max(0, j - 5), min(cols, j + 5)

        if variable_type == 'sos and climate':
            all_vars = np.stack([
                pos_matrix[:, r_start:r_end, c_start:c_end],
                sos_matrix[:, r_start:r_end, c_start:c_end],
                cor_matrix[:, r_start:r_end, c_start:c_end],
                sm_matrix[:, r_start:r_end, c_start:c_end],
                vpd_matrix[:, r_start:r_end, c_start:c_end],
                ta_matrix[:, r_start:r_end, c_start:c_end],
                pre_matrix[:, r_start:r_end, c_start:c_end],
                srad_matrix[:, r_start:r_end, c_start:c_end]
            ], axis=0)
        elif variable_type == 'only climate':
            all_vars = np.stack([
                pos_matrix[:, r_start:r_end, c_start:c_end],
                cor_matrix[:, r_start:r_end, c_start:c_end],
                sm_matrix[:, r_start:r_end, c_start:c_end],
                vpd_matrix[:, r_start:r_end, c_start:c_end],
                ta_matrix[:, r_start:r_end, c_start:c_end],
                pre_matrix[:, r_start:r_end, c_start:c_end],
                srad_matrix[:, r_start:r_end, c_start:c_end]
            ], axis=0)

        ### 2 对网格内data进行Standardization
        normal_year = drought_year_stack[analyzed_start:analyzed_end, r_start:r_end, c_start:c_end]

        mask = np.all(np.isfinite(all_vars), axis=0) & (normal_year != 2)  # 如果np前加~就是返回了无效值

        all_vars_std = np.full_like(all_vars, np.nan, dtype=np.float32)

        for v in range(all_vars.shape[0]):

            var_data = all_vars[v]

            mean_val = np.nanmean(var_data)
            std_val = np.nanstd(var_data)

            if np.isfinite(std_val) and std_val > 0:
                all_vars_std[v] = (var_data - mean_val) / std_val


        ### 获取中心像元
        rel_i = i - r_start
        rel_j = j - c_start

        # val_mask = mask[:, rel_i, rel_j]
        # val_data = all_vars_std[:, val_mask, rel_i, rel_j]

        actual_rows = all_vars_std.shape[2]
        actual_cols = all_vars_std.shape[3]

        all_samples_list = []  # 3×3全部
        center_index = None
        # neighbor_samples_list = []  # 3×3中除去中心像元

        # if analyze_years_length <= 12:
        #     valid_years = 8
        # elif analyze_years_length > 13:
        #     valid_years = analyze_years_length / 2
        valid_years = 12

        for r_idx in range(actual_rows):
            for c_idx in range(actual_cols):
                m = mask[:, r_idx, c_idx]
                if np.sum(m) > valid_years:
                    pixel_data = all_vars_std[:, m, r_idx, c_idx]
                    all_samples_list.append(pixel_data)
                    if r_idx == rel_i and c_idx == rel_j:
                        center_index = len(all_samples_list) - 1
        if center_index is None:
            return (i, j,
                    'nan', 'nan', 'nan',
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        ### =========== 构建ML ============= ###
        if variable_type == 'sos and climate':
            feature_names = ['SOS', 'Coupling', 'SM', 'VPD', 'Ta', 'Pre', 'Srad']
        elif variable_type == 'only climate':
            feature_names = ['Coupling', 'SM', 'VPD', 'Ta', 'Pre', 'Srad']

        importances_df = None
        first_varname = second_varname = third_varname = 'nan'
        first_imp = second_imp = third_imp = np.nan
        sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan

        ### 划分训练、测试
        # val_mask = mask[:, rel_i, rel_j]
        if grid_size == '3*3':
            min_pixels = 6
        elif grid_size == '5*5':
            min_pixels = 16
        else:
            min_pixels = 66

        if len(all_samples_list) >= min_pixels:

            if the_most_important_var_method == 'RF':

                ## 以所有data进行训练，然后计算内部OOB R2
                train_data = np.concatenate(all_samples_list, axis=1)

            elif the_most_important_var_method == 'XGBoost':

                ## 训练data：all中取90%的data，且要求包含中心像元
                ## 测试集：all中取10%的data，且要求不包含中心像元

                n_samples = len(all_samples_list)

                # 所有非中心像元编号
                candidate_idx = [k for k in range(n_samples) if k != center_index]

                random.seed(42)
                random.shuffle(candidate_idx)  # 随机打乱编号的顺序

                n_test = max(1, int(0.1 * n_samples))  #

                test_idx = candidate_idx[:n_test]

                train_idx = [k for k in range(n_samples) if k not in test_idx]

                train_data = np.concatenate(
                    [all_samples_list[k] for k in train_idx],
                    axis=1
                )

                test_data = np.concatenate(
                    [all_samples_list[k] for k in test_idx],
                    axis=1
                )
                X_test = test_data[1:, :].T
                pos_test = test_data[0, :]

            X_train, pos_train = train_data[1:, :].T, train_data[0, :]

            ### 解释模型的变量重要性
            # def calculate_shap(model, x_train, y_train, x_test, y_test):
            def calculate_shap(model, x_train, y_train):
                # # Step 1: Calculate MSE and R2 for train and test data
                # mse_train = mean_squared_error(y_train, model.predict(x_train))
                # r2_train = model.score(x_train, y_train)
                # print(f'Train R2: {r2_train}, MSE: {mse_train}')
                #
                # y_train_pred = model.predict(x_test)
                # r2_test = r2_score(y_test, y_train_pred)
                # mse_test = mean_squared_error(y_test, y_train_pred)
                # print(f'Test R2: {r2_test}, MSE: {mse_test}')

                # Step 2: Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                # print("SHAP explainer done")
                shap_values = explainer(x_train)
                # print("SHAP values calculation done")

                # Step 3: Consolidate feature importance and metrics
                df_model = pd.DataFrame(model.feature_importances_, index=feature_names,
                                        columns=["importance"]).reset_index().rename(
                    columns={'index': 'var'})
                # print("Feature importances calculated")
                # print(df_model)

                # Step 4: Calculate mean absolute SHAP values
                mean_abs_shap_values = abs(shap_values.values).mean(0)
                # print(f'Mean absolute SHAP values: {mean_abs_shap_values}')

                df_shap = pd.DataFrame([feature_names, mean_abs_shap_values], index=["var", "shapvalue"]).T
                # print("SHAP values calculated")

                pos_range = y_train.max() - y_train.min()
                shap_sum = shap_values.values.sum(axis=1)  # sum of SHAP per pixel

                # print("POS_mean spatial range (days):", pos_range)
                # print("SHAP sum range (days):", shap_sum.max() - shap_sum.min())

                # Step 5: Merge SHAP values and feature importance
                df_merge = pd.merge(df_shap, df_model, on="var")
                # df_merge = df_merge.rename(columns={'var': 'feature'})

                # print('df_merge:\n', df_merge)

                return df_merge

            ##### RF ####
            if the_most_important_var_method == 'RF':
                # val_mask = mask[:, rel_i, rel_j]
                # if len(neighbor_samples_list) >= 5 and np.sum(val_mask) > 0:
                #     # train_data_all = np.concatenate(all_samples_list, axis=1)
                #     train_data_all = np.concatenate(neighbor_samples_list, axis=1)
                #     X_train, pos_train = train_data_all[1:, :].T, train_data_all[0, :]
                #
                #     val_data = all_vars_std[:, val_mask, rel_i, rel_j]
                #     X_val, pos_val = val_data[1:, :].T, val_data[0, :]

                rf_model = RandomForestRegressor(n_estimators=100, max_features=0.3,
                                                 bootstrap=True, oob_score=True, random_state=42)
                rf_model.fit(X_train, pos_train)

                # test_r2 = rf_model.score(X_test, pos_test)

                if rf_model.oob_score_ >= 0.2:  ### Xtrain 来自all_samples_list
                # if test_r2 >= 0.2:
                    # ### 模型预测的变量重要性
                    # importances_df = pd.DataFrame({'var': feature_names, 'importance': rf_model.feature_importances_})

                    # ### 模型解释的变量重要性
                    importances_df = calculate_shap(rf_model, X_train, pos_train)
            ##### XGBoost ####
            elif the_most_important_var_method == 'XGBoost':
                # val_mask = mask[:, rel_i, rel_j]
                # # 必须满足：有足够邻居训练 且 中心点有data验证
                # if len(neighbor_samples_list) >= 5 and np.sum(val_mask) > 0:
                #
                #     train_data_all = np.concatenate(neighbor_samples_list, axis=1)
                #     X_train, pos_train = train_data_all[1:, :].T, train_data_all[0, :]
                #
                #     val_data = all_vars_std[:, val_mask, rel_i, rel_j]
                #     X_val, pos_val = val_data[1:, :].T, val_data[0, :]

                xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                xgb_model.fit(X_train, pos_train)

                test_r2 = xgb_model.score(X_test, pos_test)
                if test_r2 >= 0.2:
                    # ### 模型预测的变量重要性
                    # importances_df = pd.DataFrame({'var': feature_names, 'importance': xgb_model.feature_importances_})

                    # ### 模型解释的变量重要性
                    importances_df = calculate_shap(xgb_model, X_train, pos_train)

            if importances_df is not None:
                # 排序并去重：按相关系数绝对值从大到小排序
                # top_vars = importances_df.sort_values(by='importance', ascending=False)
                top_vars = importances_df.sort_values(by='shapvalue', ascending=False)
                # print('top_vars:', top_vars)

                first_varname = top_vars.iloc[0]['var']
                second_varname = top_vars.iloc[1]['var']
                third_varname = top_vars.iloc[2]['var']
                # print(f'top var:{first_varname}\n'
                #       f'secong var:{second_varname}')

                first_imp = top_vars.iloc[0]['shapvalue']
                second_imp = top_vars.iloc[1]['shapvalue']
                third_imp = top_vars.iloc[2]['shapvalue']

                if variable_type == 'sos and climate':
                    sos_imp = importances_df.loc[importances_df['var'] == 'SOS', 'shapvalue'].iloc[0]
                elif variable_type == 'only climate':
                    sos_imp = np.nan
                cor_imp = importances_df.loc[importances_df['var'] == 'Coupling', 'shapvalue'].iloc[0]
                sm_imp = importances_df.loc[importances_df['var'] == 'SM', 'shapvalue'].iloc[0]
                vpd_imp = importances_df.loc[importances_df['var'] == 'VPD', 'shapvalue'].iloc[0]
                ta_imp = importances_df.loc[importances_df['var'] == 'Ta', 'shapvalue'].iloc[0]
                pre_imp = importances_df.loc[importances_df['var'] == 'Pre', 'shapvalue'].iloc[0]
                srad_imp = importances_df.loc[importances_df['var'] == 'Srad', 'shapvalue'].iloc[0]

            # else:
            #     first_varname = second_varname = third_varname = 'nan'
            #     first_imp = second_imp = third_imp = np.nan
            #     sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan

        else:
            first_varname = second_varname = third_varname = 'nan'
            first_imp = second_imp = third_imp = np.nan
            sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan

    else:
        first_varname = second_varname = third_varname = 'nan'
        first_imp = second_imp = third_imp = np.nan
        sos_imp = cor_imp = sm_imp = vpd_imp = ta_imp = pre_imp = srad_imp = np.nan


    return (i, j, first_varname, second_varname, third_varname,
            first_imp, second_imp, third_imp,
            sos_imp, cor_imp, sm_imp, vpd_imp, ta_imp, pre_imp, srad_imp)


###################################### 1 Input and output设定 ################################################
###########################  ==== 1.1 Input settings ==== #################################
start_year = 2001
end_year = 2024
years_length = end_year - start_year + 1
print('years_length:', years_length)

analyzed_start_year = 2001
analyzed_end_year = 2024
analyze_years_length = analyzed_end_year - analyzed_start_year +1
print('analyze_years_length:', analyze_years_length)
analyzed_start = analyzed_start_year - 2001
analyzed_end = analyzed_end_year - 2001 + 1

cal_done = 'yes'  # 'yes' means already calculated; 'no' means not yet calculated

data_type = 'MODIS_GPP_8d' # GIMMS_NDVI_16d / MODIS_GPP_16d / MODIS_GPP_8d / MODIS_EVI2_8d / MCD12Q2

mean_or_yearPOS = ''  #  '' uses annual POS; '_meanPOS' uses long-term mean POS

coupling_method = 'Pearson'  ### 'Pearson'/'Partial'
lag_day = 0

partial_for = 'SMlag_VPDlag'  ###SM and lagged VPD: SM_VPDlag; SM and VPD: SMlag_VPDlag

veg_threshold = 25

cor_test_number = '17_8'  ######### 请注意这里必须要按照要求修改好
climate_test_number = '9'
pheno = 'pos'  ##### 如果是POS就是'pos'，如果是Mat-Sen就是'length of pos'

scale = 11

Outlier = 'Yes'  # Yes / No
OutnosigCor = 'No'  # Yes / No

Basedon = 'Based_on_detrendPheno'  ### Based_on_detrendPheno uses detrended SOS and POS for partial correlation
                                   ### Based_on_OriginPheno uses original SOS and POS for partial correlation  差Pearson0

the_most_important_var_method = 'RF'  ### RF / XGBoost / Partial

## ! 注意：要是使用XGBoost or RF 需要选择
grid_size = '3*3' # '3*3' / '5*5' / '10*10'

variable_type = 'sos and climate'  ### sos and climate / only climate

#### InputClimate的tif path
input_path = rf'D:/code/Data'

if coupling_method == 'Pearson':
    input_path2 = Rf'D:/code/Cor_Pearson_SM_t_VPD_t{mean_or_yearPOS}'

elif coupling_method == 'Partial' and partial_for == 'SM_VPDlag':
    input_path2 = Rf'D:/code/Cor_Pcor_SM_lag{lag_day}_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}'

elif coupling_method == 'Partial' and partial_for == 'SMlag_VPDlag':
    input_path2 = Rf'D:/code/Cor_Pcor_SM_t_VPD_t_ctrl_VPD_lag{lag_day}{mean_or_yearPOS}'

folder_cor_1 = fr'{input_path2}\Cor_preseason_mean'  #(POS-30) - POS : Correlation(SM_VPD_pearson)17_8_1
# folder_cor_2 = fr'{input_path}\Climate\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_2'  #(POS-60) - POS
# folder_cor_3 = fr'{input_path}\Climate\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_3'  #(POS-90) - POS

folder_cor_pvalue_1 = fr'{input_path2}\Cor_preseason_mean\Pvalue'  #(POS-30) - POS
# folder_cor_pvalue_2 = fr'{input_path}\Climate\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_2\Pvalue'  #(POS-60) - POS
# folder_cor_pvalue_3 = fr'{input_path}\Climate\Correlation(SM_VPD_{coupling_method}_{partial_for}_lag={lag_day})_3\Pvalue'  #(POS-90) - POS

folder_sm_1 = fr'{input_path2}\SM_preseason_mean'  #(POS-30) - POS
# folder_sm_2 = fr'{input_path}\Climate\SM_preseason_mean2'  #(POS-60) - POS
# folder_sm_3 = fr'{input_path}\Climate\SM_preseason_mean3'  #(POS-90) - POS

folder_vpd_1 = fr'{input_path2}\VPD_preseason_mean'  #(POS-30) - POS
# folder_vpd_2 = fr'{input_path}\Climate\VPD_preseason_mean2'  #(POS-60) - POS
# folder_vpd_3 = fr'{input_path}\Climate\VPD_preseason_mean3'  #(POS-90) - POS

folder_ta_1 = fr'{input_path2}\Ta_preseason_mean'
# folder_ta_2 = fr'{input_path}\Climate\Ta_preseason_mean2'
# folder_ta_3 = fr'{input_path}\Climate\Ta_preseason_mean3'

folder_pre_1 = fr'{input_path2}\Pre_preseason_sum'
# folder_pre_2 = fr'{input_path}\Climate\Pre_preseason_sum2'
# folder_pre_3 = fr'{input_path}\Climate\Pre_preseason_sum3'

folder_srad_1 = fr'{input_path2}\Srad_preseason_sum'
# folder_srad_2 = fr'{input_path}\Climate\Srad_preseason_sum2'
# folder_srad_3 = fr'{input_path}\Climate\Srad_preseason_sum3'


#### InputPOS的tif path
if Basedon == 'Based_on_detrendPheno':
    pos_folder = fr'{input_path}\Pheno\POSdetrend_{scale}km'  #start
    sos_folder = fr'{input_path}\Pheno\SOSdetrend_{scale}km'  #start
elif Basedon == 'Based_on_OriginPheno':
    pos_folder = fr'{input_path}\Pheno\POS_{scale}km'  # start
    sos_folder = fr'{input_path}\Pheno\SOS_{scale}km'  # start
if pheno == 'length of pos':
    pos_folder2 = fr'{input_path}\Pheno\Sen'  #end


#### InputSPEI识别的干旱年份的tif path
drought_path = fr'{input_path}\SPEI\Drought_Event\drought_event(POS_SPEI3_threshold10%_way3)'

#### InputAI的tif path
AI = 'Yes'  ###Yes or No
ai_tif_file = rf'{input_path}\Classify\AI\NH30_84_AI(graident)_{scale}km.tif'

#### Input vegetation type TIF path
veg_type_need = 'Yes'
veg_type_file = rf'{input_path}\Classify\Veg_type\NH_permanent_veg_type_{scale}km_gte{veg_threshold}.tif'

### Input coupling gradient TIF path
cor_mean_file = fr'D:/code/mean/Cor(Pearson_SMlag_VPDlag_lag0)_mean_11km_SOS_All.tif'  #SOS - POS

###########################  ==== 1.2 Output settings ==== #################################
output_partial_max_var_png_path = fr'D:/code/6Partial'

output_partial_max_var_r_tif_path = fr'D:/code/6Partial'


####################################### 2 data读取 #################################################
tif_files_cor_1 = sorted(glob.glob(os.path.join(folder_cor_1, '*.tif')))
# tif_files_cor_2 = sorted(glob.glob(os.path.join(folder_cor_2, '*.tif')))
# tif_files_cor_3 = sorted(glob.glob(os.path.join(folder_cor_3, '*.tif')))
# cor_length_tif = os.path.join(input_path, r'Best_preseason_length\17_8_1\Cor_preseason_length.tif')

tif_files_cor_pvalue_1 = sorted(glob.glob(os.path.join(folder_cor_pvalue_1, '*.tif')))
# tif_files_cor_pvalue_2 = sorted(glob.glob(os.path.join(folder_cor_pvalue_2, '*.tif')))
# tif_files_cor_pvalue_3 = sorted(glob.glob(os.path.join(folder_cor_pvalue_3, '*.tif')))


tif_files_sm_1 = sorted(glob.glob(os.path.join(folder_sm_1, '*.tif')))
# tif_files_sm_2 = sorted(glob.glob(os.path.join(folder_sm_2, '*.tif')))
# tif_files_sm_3 = sorted(glob.glob(os.path.join(folder_sm_3, '*.tif')))
# sm_length_tif = os.path.join(input_path, r'Best_preseason_length\17_8_1\SM_preseason_length.tif')


tif_files_vpd_1 = sorted(glob.glob(os.path.join(folder_vpd_1, '*.tif')))
# tif_files_vpd_2 = sorted(glob.glob(os.path.join(folder_vpd_2, '*.tif')))
# tif_files_vpd_3 = sorted(glob.glob(os.path.join(folder_vpd_3, '*.tif')))
# vpd_length_tif = os.path.join(input_path, r'Best_preseason_length\17_8_1\VPD_preseason_length.tif')


tif_files_ta_1 = sorted(glob.glob(os.path.join(folder_ta_1, '*.tif')))
# tif_files_ta_2 = sorted(glob.glob(os.path.join(folder_ta_2, '*.tif')))
# tif_files_ta_3 = sorted(glob.glob(os.path.join(folder_ta_3, '*.tif')))
# ta_length_tif = os.path.join(input_path, r'Best_preseason_length\17_8_1\Ta_preseason_length.tif')

tif_files_pre_1 = sorted(glob.glob(os.path.join(folder_pre_1, '*.tif')))
# tif_files_pre_2 = sorted(glob.glob(os.path.join(folder_pre_2, '*.tif')))
# tif_files_pre_3 = sorted(glob.glob(os.path.join(folder_pre_3, '*.tif')))
# pre_length_tif = os.path.join(input_path, r'Best_preseason_length\17_8_1\Pre_preseason_length.tif')

tif_files_srad_1 = sorted(glob.glob(os.path.join(folder_srad_1, '*.tif')))
# tif_files_srad_2 = sorted(glob.glob(os.path.join(folder_srad_2, '*.tif')))
# tif_files_srad_3 = sorted(glob.glob(os.path.join(folder_srad_3, '*.tif')))
# srad_length_tif = os.path.join(input_path, r'Best_preseason_length\17_8_1\Srad_preseason_length.tif')

pos_tif_files = sorted(glob.glob(os.path.join(pos_folder, '*.tif')))
sos_tif_files = sorted(glob.glob(os.path.join(sos_folder, '*.tif')))

drought_year_tif_files = sorted(glob.glob(os.path.join(drought_path, '*.tif')))

if not tif_files_sm_1:
    raise FileNotFoundError("未找到任何 tif_files_sm_1 TIF 文件！")
# if not tif_files_sm_2:
#     raise FileNotFoundError("No tif_files2 TIF files found!")
# if not tif_files_sm_3:
#     raise FileNotFoundError("未找到任何 tif_files3 TIF 文件！")
if not pos_tif_files:
    raise FileNotFoundError("No POS TIF files found!")


###################################################
first_tif = tif_files_sm_1[0]
sample_tif = gdal.Open(first_tif)

if sample_tif is None:
    raise RuntimeError(f"无法打开 TIF 文件：{sample_tif}（驱动不支持或文件损坏）")

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

row_indices = np.repeat(np.arange(rows), cols)
col_indices = np.tile(np.arange(cols), rows)

# Calculate coordinate bounds with negative pixel height
lon_min = top_left_x
lon_max = top_left_x + cols * pixel_width  # Right longitude bound
lat_min = top_left_y + rows * pixel_height  # Lower latitude bound (southernmost)
lat_max = top_left_y  # Upper latitude bound (northernmost)
print(f"Longitude range: {lon_min:.6f} -> {lon_max:.6f}")
print(f"Latitude range: {lat_min:.6f} -> {lat_max:.6f}")


############# 3.2 Vegetation type ################
veg_type_data = gdal.Open(veg_type_file)
band = veg_type_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

veg_type_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    veg_type_data[veg_type_data == nodata] = np.nan
print('veg_type_data shape:', veg_type_data.shape)

############### 3.4 AI index #################
ai_data = gdal.Open(ai_tif_file)
band = ai_data.GetRasterBand(1)
nodata = band.GetNoDataValue()

ai_data = band.ReadAsArray().astype(np.float32)
if nodata is not None:
    ai_data[ai_data == nodata] = np.nan


cor_mean_tif = gdal.Open(cor_mean_file)
cor_mean_data = cor_mean_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
print('cor_mean_data shape:', cor_mean_data.shape)

if cal_done == 'no':
    ############################################ 3 Stacking ###################################################
    ## Stack data
    cor_stack_1 = []
    # cor_stack_2 = []
    # cor_stack_3 = []

    cor_pvalue_stack_1 = []
    # cor_pvalue_stack_2 = []
    # cor_pvalue_stack_3 = []

    sm_stack_1 = []
    # sm_stack_2 = []
    # sm_stack_3 = []

    vpd_stack_1 = []
    # vpd_stack_2 = []
    # vpd_stack_3 = []

    ta_stack_1 = []
    # ta_stack_2 = []
    # ta_stack_3 = []

    pre_stack_1 = []
    # pre_stack_2 = []
    # pre_stack_3 = []

    srad_stack_1 = []
    # srad_stack_2 = []
    # srad_stack_3 = []

    pos_stack = []
    sos_stack = []

    drought_year_stack = []

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

    for tif_file in tif_files_sm_1:
        get_band(tif_file, sm_stack_1)
    # for tif_file in tif_files_sm_2:
    #     get_band(tif_file, sm_stack_2)
    # for tif_file in tif_files_sm_3:
    #     get_band(tif_file, sm_stack_3)
    # sm_length_tif = gdal.Open(sm_length_tif)
    # sm_length = sm_length_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    for tif_file in tif_files_vpd_1:
        get_band(tif_file, vpd_stack_1)
    # for tif_file in tif_files_vpd_2:
    #     get_band(tif_file, vpd_stack_2)
    # for tif_file in tif_files_vpd_3:
    #     get_band(tif_file, vpd_stack_3)
    # vpd_length_tif = gdal.Open(vpd_length_tif)
    # vpd_length = vpd_length_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    for tif_file in tif_files_ta_1:
        get_band(tif_file, ta_stack_1)
    # for tif_file in tif_files_ta_2:
    #     get_band(tif_file, ta_stack_2)
    # for tif_file in tif_files_ta_3:
    #     get_band(tif_file, ta_stack_3)
    # ta_length_tif = gdal.Open(ta_length_tif)
    # ta_length = ta_length_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    for tif_file in tif_files_pre_1:
        get_band(tif_file, pre_stack_1)
    # for tif_file in tif_files_pre_2:
    #     get_band(tif_file, pre_stack_2)
    # for tif_file in tif_files_pre_3:
    #     get_band(tif_file, pre_stack_3)
    # pre_length_tif = gdal.Open(pre_length_tif)
    # pre_length = pre_length_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    for tif_file in tif_files_srad_1:
        get_band(tif_file, srad_stack_1)
    # for tif_file in tif_files_srad_2:
    #     get_band(tif_file, srad_stack_2)
    # for tif_file in tif_files_srad_3:
    #     get_band(tif_file, srad_stack_3)
    # srad_length_tif = gdal.Open(srad_length_tif)
    # srad_length = srad_length_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)

    for tif_file in pos_tif_files:
        get_band(tif_file, pos_stack)
    for tif_file in sos_tif_files:
        get_band(tif_file, sos_stack)


    for tif_file in drought_year_tif_files:
        get_band(tif_file, drought_year_stack)



    print('Stack start')
    cor_stack_1 = np.stack(cor_stack_1, axis=0)#[:, 505:510, 505:510]
    cor_pre_stack = cor_stack_1
    # cor_stack_2 = np.stack(cor_stack_2, axis=0)
    # cor_stack_3 = np.stack(cor_stack_3, axis=0)
    # cor_stack = np.stack([cor_stack_1[:years_length, :, :], cor_stack_2[:years_length, :, :], cor_stack_3[:years_length, :, :]], axis=0)

    cor_pvalue_stack_1 = np.stack(cor_pvalue_stack_1, axis=0)#[:, 505:510, 505:510]
    cor_pvalue_pre_stack = cor_pvalue_stack_1
    # cor_pvalue_stack_2 = np.stack(cor_pvalue_stack_2, axis=0)
    # cor_pvalue_stack_3 = np.stack(cor_pvalue_stack_3, axis=0)
    # cor_pvalue_stack = np.stack([cor_pvalue_stack_1[:years_length, :, :], cor_pvalue_stack_2[:years_length, :, :], cor_pvalue_stack_3[:years_length, :, :]], axis=0)

    sm_stack_1 = np.stack(sm_stack_1, axis=0)
    sm_pre_stack = sm_stack_1
    # sm_stack_2 = np.stack(sm_stack_2, axis=0)
    # sm_stack_3 = np.stack(sm_stack_3, axis=0)
    # sm_stack = np.stack([sm_stack_1[:years_length, :, :], sm_stack_2[:years_length, :, :], sm_stack_3[:years_length, :, :]], axis=0)
    # print('sm_stack shape:\n', sm_stack.shape)
    # print('sm_stack:\n', sm_stack[:,90:100,90:100])
    vpd_stack_1 = np.stack(vpd_stack_1, axis=0)
    vpd_pre_stack = vpd_stack_1
    # vpd_stack_2 = np.stack(vpd_stack_2, axis=0)
    # vpd_stack_3 = np.stack(vpd_stack_3, axis=0)
    # vpd_stack = np.stack([vpd_stack_1[:years_length, :, :], vpd_stack_2[:years_length, :, :], vpd_stack_3[:years_length, :, :]], axis=0)

    ta_stack_1 = np.stack(ta_stack_1, axis=0)
    ta_pre_stack = ta_stack_1
    # ta_stack_2 = np.stack(ta_stack_2, axis=0)
    # ta_stack_3 = np.stack(ta_stack_3, axis=0)
    # ta_stack = np.stack([ta_stack_1[:years_length, :, :], ta_stack_2[:years_length, :, :], ta_stack_3[:years_length, :, :]], axis=0)

    pre_stack_1 = np.stack(pre_stack_1, axis=0)
    pre_pre_stack = pre_stack_1
    # pre_stack_2 = np.stack(pre_stack_2, axis=0)
    # pre_stack_3 = np.stack(pre_stack_3, axis=0)
    # pre_stack = np.stack([pre_stack_1[:years_length, :, :], pre_stack_2[:years_length, :, :], pre_stack_3[:years_length, :, :]], axis=0)

    srad_stack_1 = np.stack(srad_stack_1, axis=0)
    srad_pre_stack = srad_stack_1
    # srad_stack_2 = np.stack(srad_stack_2, axis=0)
    # srad_stack_3 = np.stack(srad_stack_3, axis=0)
    # srad_stack = np.stack([srad_stack_1[:years_length, :, :], srad_stack_2[:years_length, :, :], srad_stack_3[:years_length, :, :]], axis=0)

    pos_stack = np.stack(pos_stack, axis=0)
    # pos_stack = pos_stack[:years_length, :, :]

    sos_stack = np.stack(sos_stack, axis=0)
    # sos_stack = sos_stack[:years_length, :, :]

    drought_year_stack = np.stack(drought_year_stack, axis=0)
    # drought_year_stack = drought_year_stack[:years_length, :, :]

    print('Stack end!')

    ########################### 4 逐像元根据最佳季前长度获取data #################################
    def get_preseason_data(i, j,
                           cor_len, cor_data, cor_pvalue_data,
                           sm_len, sm_data,
                           vpd_len, vpd_data,
                           ta_len, ta_data,
                           pre_len, pre_data,
                           srad_len, srad_data):

        def extract_by_len(length_val, data_stack):
            if np.isnan(length_val):
                return np.full(years_length, np.nan)
            idx = int(length_val) - 1  # 关键：将 1,2,3 转为 0,1,2
            return data_stack[idx, :]

        cor_preseason_value = extract_by_len(cor_len, cor_data)
        cor_pvalue_preseason_value = extract_by_len(cor_len, cor_pvalue_data)
        sm_preseason_value  = extract_by_len(sm_len, sm_data)
        vpd_preseason_value = extract_by_len(vpd_len, vpd_data)
        ta_preseason_value  = extract_by_len(ta_len, ta_data)
        pre_preseason_value = extract_by_len(pre_len, pre_data)
        srad_preseason_value = extract_by_len(srad_len, srad_data)

        return (i, j, cor_preseason_value, cor_pvalue_preseason_value, sm_preseason_value, vpd_preseason_value,
                      ta_preseason_value, pre_preseason_value, srad_preseason_value)



    ########################### 5 Remove outliers（/去不显著Cor的像元） ################################
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
            if len(x_flatten) < years_length/2:
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
                    return np.full_like(x, np.nan), i, j


        outlier_pos_stack = np.full((years_length, rows, cols), np.nan)
        outlier_sos_stack = np.full((years_length, rows, cols), np.nan)
        outlier_cor_stack = np.full((years_length, rows, cols), np.nan)
        outlier_sm_stack = np.full((years_length, rows, cols), np.nan)
        outlier_vpd_stack = np.full((years_length, rows, cols), np.nan)
        outlier_ta_stack = np.full((years_length, rows, cols), np.nan)
        outlier_pre_stack = np.full((years_length, rows, cols), np.nan)
        outlier_srad_stack = np.full((years_length, rows, cols), np.nan)

        ### IQR去 POS 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # pos_stack[:years_length, i, j], i, j, 25, 75
                pos_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_pos_stack[:, i, j] = data_mask

        ### IQR去 SOS 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # sos_stack[:years_length, i, j], i, j, 25, 75
                sos_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_sos_stack[:, i, j] = data_mask

        ### Remove Cor outliers using IQR
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # cor_stack[:years_length, i, j], i, j, 25, 75
                cor_pre_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_cor_stack[:, i, j] = data_mask

        ### Remove SM outliers using IQR
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # sm_stack[:years_length, i, j], i, j, 25, 75
                sm_pre_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_sm_stack[:, i, j] = data_mask

        ### Remove VPD outliers using IQR
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # vpd_stack[:years_length, i, j], i, j, 25, 75
                vpd_pre_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_vpd_stack[:, i, j] = data_mask

        ### IQR去 Ta 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # ta_stack[:years_length, i, j], i, j, 25, 75
                ta_pre_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_ta_stack[:, i, j] = data_mask

        ### IQR去 Pre 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # pre_stack[:years_length, i, j], i, j, 25, 75
                pre_pre_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_pre_stack[:, i, j] = data_mask

        ### IQR去 Srad 异常值
        results = Parallel(n_jobs=15, verbose=10)(
            delayed(Outlier_array_IQR)(
                # srad_stack[:years_length, i, j], i, j, 25, 75
                srad_pre_stack[analyzed_start: analyzed_end, i, j], i, j, 25, 75
            ) for i, j in zip(row_indices, col_indices)
        )

        for data_mask, i, j in results:
            outlier_srad_stack[:, i, j] = data_mask

        ### 筛出有效像元的mask ###
        mask = (np.isfinite(outlier_pos_stack) & np.isfinite(outlier_sos_stack) &
                np.isfinite(outlier_cor_stack) & np.isfinite(outlier_sm_stack) & np.isfinite(outlier_vpd_stack)
                & np.isfinite(outlier_ta_stack) & np.isfinite(outlier_pre_stack) & np.isfinite(outlier_srad_stack))

        # # ### ①用最佳窗口的cor取mean
        # # cor_stack_new = outlier_cor_stack[analyzed_start: analyzed_end, :, :]
        #
        # ### ②用POS前2个月的cor取mean，可以配合上干旱事件
        # cor_stack_2_lt0 = np.where(cor_stack_2<0, cor_stack_2, np.nan)
        # cor_stack_new = cor_stack_2_lt0[analyzed_start: analyzed_end, :, :]
        #
        # ### Cor mean
        # cor_mean = np.nanmean(cor_stack_new, axis=0)


    elif Outlier == 'No':

        mask = (np.isfinite(pos_stack) &
                np.isfinite(sos_stack) &
                np.isfinite(cor_pre_stack) &
                np.isfinite(sm_pre_stack) &
                np.isfinite(vpd_pre_stack) &
                np.isfinite(ta_pre_stack) &
                np.isfinite(pre_pre_stack) &
                np.isfinite(srad_pre_stack))


    if OutnosigCor == 'Yes':
        mask = (cor_pvalue_pre_stack[analyzed_start: analyzed_end,:,:] <= 0.1) & mask
    vaild_year_length = np.sum(mask, axis=0)

    space_mask = vaild_year_length >= (analyze_years_length/2)  # 形状：(rows, cols)

    print(f'原始有效像元数量：{np.count_nonzero(pos_stack[0, :, :])}\n'
          f'去除异常值后有效像元数量：{np.count_nonzero(space_mask)}')

    space_mask_3d = space_mask[np.newaxis, :, :]  # (1, rows, cols)
    space_mask_3d = np.repeat(space_mask_3d, analyze_years_length, axis=0)  # (years_length, rows, cols)


    ########################### 6 Preseason Pearson+Partial ################

    def standardize_data(data_stack, mask_3d):
        """Temporal standardization"""

        def standardize_func(data_pixel, i, j):
            original_nan_mask = np.isnan(data_pixel)

            spatio_mean = np.nanmean(data_pixel)
            spatio_std = np.nanstd(data_pixel)

            standardized_pixel_data = (data_pixel - spatio_mean) / spatio_std

            standardized_pixel_data[original_nan_mask] = np.nan

            return standardized_pixel_data, i, j

        standardized_data = np.full((analyze_years_length, rows, cols), np.nan)

        results = Parallel(n_jobs=15, verbose=10)(
            delayed(standardize_func)(
                data_stack[:, i, j],
                i, j
            ) for i, j in zip(row_indices, col_indices)
        )

        for standardized_pixel_data, i, j in results:
            standardized_data[:, i, j] = standardized_pixel_data

        standardized_data = np.where(mask_3d, standardized_data, np.nan)

        return standardized_data


    dtype = [('var', '<U10')]
    varname_first = np.empty((rows, cols), dtype=dtype)
    varname_second = np.empty((rows, cols), dtype=dtype)
    varname_third = np.empty((rows, cols), dtype=dtype)

    if the_most_important_var_method == 'Partial':

        ########################### 6 解释变量在Time上逐像元进行归一化 ###################
        if Outlier == 'Yes':
            pos_standardized = standardize_data(outlier_pos_stack, space_mask_3d)
            sos_standardized = standardize_data(outlier_sos_stack, space_mask_3d)
            cor_standardized = standardize_data(outlier_cor_stack, space_mask_3d)
            sm_standardized = standardize_data(outlier_sm_stack, space_mask_3d)
            vpd_standardized = standardize_data(outlier_vpd_stack, space_mask_3d)
            ta_standardized = standardize_data(outlier_ta_stack, space_mask_3d)
            pre_standardized = standardize_data(outlier_pre_stack, space_mask_3d)
            srad_standardized = standardize_data(outlier_srad_stack, space_mask_3d)

        elif Outlier == 'No':
            pos_standardized = standardize_data(pos_stack, space_mask_3d)
            sos_standardized = standardize_data(sos_stack, space_mask_3d)
            cor_standardized = standardize_data(cor_pre_stack, space_mask_3d)
            sm_standardized = standardize_data(sm_pre_stack, space_mask_3d)
            vpd_standardized = standardize_data(vpd_pre_stack, space_mask_3d)
            ta_standardized = standardize_data(ta_pre_stack, space_mask_3d)
            pre_standardized = standardize_data(pre_pre_stack, space_mask_3d)
            srad_standardized = standardize_data(srad_pre_stack, space_mask_3d)

        del pos_stack, sos_stack, cor_pre_stack, sm_pre_stack, vpd_pre_stack, ta_pre_stack, pre_pre_stack, srad_pre_stack

        print('Partial correlation calculation start!')

        if variable_type == 'sos and climate':
            var_list = ['POS', 'SOS', 'Ta', 'Pre', 'Srad', 'SM', 'VPD', 'Coupling']
        elif variable_type == 'only climate':
            var_list = ['POS', 'Ta', 'Pre', 'Srad', 'SM', 'VPD', 'Coupling']

        p_coefficient_r_first = np.full((rows, cols), np.nan)
        p_coefficient_r_second = np.full((rows, cols), np.nan)
        p_coefficient_r_third = np.full((rows, cols), np.nan)

        p_coefficient_p_first = np.full((rows, cols), np.nan)
        p_coefficient_p_second = np.full((rows, cols), np.nan)
        p_coefficient_p_third = np.full((rows, cols), np.nan)

        sos_pcor_matrix = np.full((rows, cols), np.nan)
        sm_pcor_matrix = np.full((rows, cols), np.nan)
        vpd_pcor_matrix = np.full((rows, cols), np.nan)
        cor_pcor_matrix = np.full((rows, cols), np.nan)
        ta_pcor_matrix = np.full((rows, cols), np.nan)
        pre_pcor_matrix = np.full((rows, cols), np.nan)
        srad_pcor_matrix = np.full((rows, cols), np.nan)

        # with parallel_backend('threading', n_jobs=15):
        results = Parallel(n_jobs=15, verbose=10)(
                delayed(cal_partial)(
                    i, j, var_list[1:], var_list[0], pos_standardized,  ##POS不Standardization：outlier_pos_stack  POSStandardization：pos_standardized
                                                    sos_standardized,
                                                    cor_standardized,
                                                    sm_standardized,
                                                    vpd_standardized,
                                                    ta_standardized,
                                                    pre_standardized,
                                                    srad_standardized
                )
                for i, j in zip(row_indices, col_indices)
            )

        for (i, j, first_varname, second_varname, third_varname,
             first_r, second_r, third_r,
             first_p, second_p, third_p,
             sos_pcor, sm_pcor, vpd_pcor, cor_pcor, ta_pcor, pre_pcor, srad_pcor) in results:
            varname_first[i, j] = first_varname
            varname_second[i, j] = second_varname
            varname_third[i, j] = third_varname

            p_coefficient_r_first[i, j] = first_r
            p_coefficient_r_second[i, j] = second_r
            p_coefficient_r_third[i, j] = third_r

            p_coefficient_p_first[i, j] = first_p
            p_coefficient_p_second[i, j] = second_p
            p_coefficient_p_third[i, j] = third_p

            sos_pcor_matrix[i, j] = sos_pcor
            sm_pcor_matrix[i, j] = sm_pcor
            vpd_pcor_matrix[i, j] = vpd_pcor
            cor_pcor_matrix[i, j] = cor_pcor
            ta_pcor_matrix[i, j] = ta_pcor
            pre_pcor_matrix[i, j] = pre_pcor
            srad_pcor_matrix[i, j] = srad_pcor


    elif the_most_important_var_method in ['RF', 'XGBoost']:

        print(f'{the_most_important_var_method} calculation start!')

        ML_imp_first = np.full((rows, cols), np.nan)
        ML_imp_second = np.full((rows, cols), np.nan)
        ML_imp_third = np.full((rows, cols), np.nan)

        sos_imp_matrix = np.full((rows, cols), np.nan)
        sm_imp_matrix = np.full((rows, cols), np.nan)
        vpd_imp_matrix = np.full((rows, cols), np.nan)
        cor_imp_matrix = np.full((rows, cols), np.nan)
        ta_imp_matrix = np.full((rows, cols), np.nan)
        pre_imp_matrix = np.full((rows, cols), np.nan)
        srad_imp_matrix = np.full((rows, cols), np.nan)

        if Outlier == 'Yes':
            results = Parallel(n_jobs=15, verbose=10)(
                    delayed(cal_ML)(
                        i, j, outlier_pos_stack,  ##POS不Standardization：outlier_pos_stack  POSStandardization：pos_standardized
                            outlier_sos_stack,
                            outlier_cor_stack,
                            outlier_sm_stack,
                            outlier_vpd_stack,
                            outlier_ta_stack,
                            outlier_pre_stack,
                            outlier_srad_stack
                    )
                    for i, j in zip(row_indices, col_indices)
                )
        elif Outlier == 'No':
            results = Parallel(n_jobs=15, verbose=10)(
                delayed(cal_ML)(
                    i, j, pos_stack,  ##POS不Standardization：outlier_pos_stack  POSStandardization：pos_standardized
                    sos_stack,
                    cor_pre_stack,
                    sm_pre_stack,
                    vpd_pre_stack,
                    ta_pre_stack,
                    pre_pre_stack,
                    srad_pre_stack
                )
                for i, j in zip(row_indices, col_indices)
            )

        for (i, j, first_varname, second_varname, third_varname,
             first_imp, second_imp, third_imp,
             sos_imp, cor_imp, sm_imp, vpd_imp, ta_imp, pre_imp, srad_imp) in results:

            varname_first[i, j] = first_varname
            varname_second[i, j] = second_varname
            varname_third[i, j] = third_varname

            ML_imp_first[i, j] = first_imp
            ML_imp_second[i, j] = second_imp
            ML_imp_third[i, j] = third_imp

            sos_imp_matrix[i, j] = sos_imp
            cor_imp_matrix[i, j] = cor_imp
            sm_imp_matrix[i, j] = sm_imp
            vpd_imp_matrix[i, j] = vpd_imp
            ta_imp_matrix[i, j] = ta_imp
            pre_imp_matrix[i, j] = pre_imp
            srad_imp_matrix[i, j] = srad_imp

    print('calculation end!')




    first_plot_data = varname_first.astype(str)
    second_plot_data = varname_second.astype(str)
    third_plot_data = varname_third.astype(str)


    ######### For Save ############
    # 变量名与整数编码的对应关系
    var_to_code = {
        "SOS": 1,
        "Coupling": 2,
        "SM": 3,
        "VPD": 4,
        "Ta": 5,
        "Pre": 6,
        "Srad": 7,
    }

    def encode_variable_raster(var_array, var_to_code, nodata=0):
        var_array = np.asarray(var_array, dtype=object)
        encoded = np.full(var_array.shape, nodata, dtype=np.int16)

        for name, code in var_to_code.items():
            encoded[var_array == name] = code

        return encoded

    for filename, var_array in [
        (fr"first_var{mean_or_yearPOS}.tif", first_plot_data),
        (rf"second_var{mean_or_yearPOS}.tif", second_plot_data),
        (rf"third_var{mean_or_yearPOS}.tif", third_plot_data),
    ]:
        out_put = os.path.join(
            output_partial_max_var_r_tif_path,
            rf"temporal\{the_most_important_var_method}",
            filename,
        )
        os.makedirs(os.path.dirname(out_put), exist_ok=True)

        encoded = encode_variable_raster(var_array, var_to_code)
        save_tif_gdal(out_put, encoded, rows, cols, crs, gt)

    print('Save tif done!')

    # sys.exit()

    # first_tif_data = p_coefficient_r_first
    # second_tif_data = p_coefficient_r_second
    # third_tif_data = p_coefficient_r_third


elif cal_done == 'yes':

    ################## 再次Input直接读取first、second、third的tif
    temporal_tif_dir = os.path.join(
        output_partial_max_var_r_tif_path,
        rf"temporal\{the_most_important_var_method}",
    )

    var_to_code = {
        "SOS": 1,
        "Coupling": 2,
        "SM": 3,
        "VPD": 4,
        "Ta": 5,
        "Pre": 6,
        "Srad": 7,
    }

    code_to_var = {code: name for name, code in var_to_code.items()}

    def read_tif_gdal(path):
        ds = gdal.Open(path)
        if ds is None:
            raise FileNotFoundError(f"无法打开TIF：{path}")

        data = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        return data

    def decode_variable_raster(encoded, code_to_var, nodata=0):
        # "nan"可保持与原先 var_array.astype(str) 的结果一致
        decoded = np.full(encoded.shape, "nan", dtype=object)

        for code, name in code_to_var.items():
            decoded[encoded == code] = name

        decoded[encoded == nodata] = "nan"
        return decoded

    first_encoded = read_tif_gdal(
        os.path.join(temporal_tif_dir, f"first_var{mean_or_yearPOS}.tif")
    )
    second_encoded = read_tif_gdal(
        os.path.join(temporal_tif_dir, f"second_var{mean_or_yearPOS}.tif")
    )
    third_encoded = read_tif_gdal(
        os.path.join(temporal_tif_dir, f"third_var{mean_or_yearPOS}.tif")
    )

    first_plot_data = decode_variable_raster(first_encoded, code_to_var)
    second_plot_data = decode_variable_raster(second_encoded, code_to_var)
    third_plot_data = decode_variable_raster(third_encoded, code_to_var)




############################# 耦合效应变化plot ##############################
######## ======= 不做区分plot =============== ###########
# def plot_NH_All(varname_data, r_data, grade_by, veg_type, plotdata_type, importance_grape):
def plot_NH_All(varname_data, importance_grape, grade_by, ax):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.gridspec as gridspec
    import numpy as np
    import os
    from brokenaxes import brokenaxes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # 创建子图布局
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[5, 0.3], hspace=0.15)
    ax1 = fig.add_subplot(gs_inner[0])  # map
    ax2 = fig.add_subplot(gs_inner[1])  # Colorbar

    ax.axis('off')

    plots = []  # Store the plot object for each subplot

    ########### 1 空间分布 #################
    ax1.set_box_aspect(1)  # Make the map axis square and fill the grid height

    ### 1.1 Create the map
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
                    latmax=90,  # Make meridians converge at the North Pole
                    labels=[0, 0, 0, 0],  # labels=[left, right, top, bottom] Control longitude label visibility
                    linewidth=0.5,
                    # fontsize=8,
                    color="gray")

    # # Fill continents
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Hide boundary lines

    ### 1.2 Plotdata
    # Color mapping
    # category_colors = {
    #     'SOS': '#ff7a00',
    #     'Ta': '#f2b701',
    #     'Srad': '#a5aa99',
    #     'Pre': '#e73f74',
    #     'SM': '#11a579',
    #     'VPD': '#3969ac',
    #     'Cor': '#7f3c8d'
    # }
    # category_colors = {
    #     'SOS': '#33a02c',  #绿
    #     'Ta': '#ff7f00',   #橙
    #     'Srad': '#a5aa99', #灰
    #     'Pre': '#1f78b4',  #蓝
    #     'SM': '#6a3d9a',   #紫
    #     'VPD': '#ffff99',  #黄
    #     'Cor': '#e31a1c'   #红
    # }
    category_colors = {
        'SOS': '#b2df8a',  #绿
        'Ta': '#fdbf6f',   #橙
        'Srad': '#a5aa99', #灰
        'Pre': '#a6cee3',  #蓝
        'SM': '#cab2d6',   #紫
        'VPD': '#ffff99',  #黄
        'Coupling': '#e31a1c'   #红
    }


    # 创建映射字典
    keys = list(category_colors.keys())
    str_to_int = {k: i for i, k in enumerate(keys)}

    # 向量化转换（处理NaN）
    def safe_map(x):
        if isinstance(x, np.ma.core.MaskedConstant) or str(x) == 'nan':
            return -1
        return str_to_int.get(x, -1)  # 不存在返回-1

    # 转换为整数矩阵
    int_data = np.vectorize(safe_map)(varname_data)
    int_data = np.ma.masked_where(int_data == -1, int_data)  # 屏蔽无效值

    # 创建Color mapping
    cmap = ListedColormap(list(category_colors.values()))
    norm = BoundaryNorm(np.arange(len(category_colors) + 1) - 0.5,
                        len(category_colors))

    # Plot空间分布图
    plot = m.pcolormesh(lons, lats, int_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Avoid tearing near the pole
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

    # # 北极点
    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    ### 4 Clip to the outer boundary
    from matplotlib.patches import Circle

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()

    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    # radius = (x1 - x0) / 2
    radius = min(x1 - x0, y1 - y0) / 2

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
        clip_on=False,  # Whether to clip this object to the axes rectangle
        zorder=4  # Place on top
    )

    ax1.add_patch(boundary_circle)

    ax1.axis('off')


    ##################### 2 Statistics图 #############################
    ### 2.1 Statistics
    counts = [np.sum(varname_data == k) for k in keys]
    total = np.sum([np.sum(varname_data == k) for k in keys if str(k) != 'nan'])
    print(f'counts:\n'
          f'{counts}')
    print(f'total={total}')

    fractions = [c / total * 100 for c in counts]

    ### 2.2 Plot
    bin_centers = ['SOS', 'Ta', 'Srad', 'Pre', 'SM', 'VPD', 'Coupling']
    if importance_grape == 'first':
        # if data_type == 'MCD12Q2':
        #     # 在 ax1 上创建 inset 容器
        #     inset_pos = [0.01, 0, 0.3, 0.3]
        #     axins = ax1.inset_axes(inset_pos)
        #
        #     axins.set_facecolor('white')
        #     axins.patch.set_alpha(1)
        #
        #     # 不显示刻度，但保留背景
        #     axins.set_xticks([])
        #     axins.set_yticks([])
        #
        #     # 隐藏边框
        #     for spine in axins.spines.values():
        #         spine.set_visible(False)
        #
        #     # 创建上下两个子轴（模拟断轴）
        #     axins_top = axins.inset_axes([0, 0.65, 1, 0.3])  # 上半部分
        #     axins_bottom = axins.inset_axes([0, 0, 1, 0.6])  # 下半部分
        #
        #     # 画柱状图（你的data直接用）
        #     x_indexes = np.arange(len(bin_centers))  # 生成 [0, 1, 2, 3, 4, 5, 6]
        #     plot_colors = [category_colors[k] for k in bin_centers]
        #
        #     axins_top.bar(x_indexes, fractions, color=plot_colors, width=0.7)
        #     axins_bottom.bar(x_indexes, fractions, color=plot_colors, width=0.7)
        #
        #     # 设置断轴范围
        #     total_max = max(arr.max() for arr in fractions)
        #     print(f"绝对最大值是: {total_max}")
        #     if OutnosigCor == 'No':
        #         if the_most_important_var_method == 'Partial':
        #             axins_top.set_ylim(40, 45)
        #         elif the_most_important_var_method == 'RF':
        #             axins_top.set_ylim(60, 65)
        #     if OutnosigCor == 'Yes':
        #         axins_top.set_ylim(45, 50)
        #
        #     if the_most_important_var_method == 'Partial':
        #         axins_bottom.set_ylim(0, 25.001)
        #         axins_bottom.set_yticks(np.arange(0, 25, 10))
        #     elif the_most_important_var_method == 'RF':
        #         axins_bottom.set_ylim(0, 30.001)
        #         axins_bottom.set_yticks(np.arange(0, 30, 10))
        #
        #
        #     # 美化（隐藏中间边框）
        #     axins_top.spines['bottom'].set_visible(False)
        #     axins_top.spines['top'].set_visible(False)
        #     axins_top.spines['right'].set_visible(False)
        #
        #     axins_bottom.spines['top'].set_visible(False)
        #     axins_bottom.spines['right'].set_visible(False)
        #
        #
        #     axins_top.tick_params(axis='x', bottom=False, labelbottom=False)  # 上图不显示x轴
        #     axins_top.tick_params(axis='y')
        #     axins_bottom.tick_params(axis='x', bottom=False, labelbottom=False)
        #     axins_bottom.tick_params(axis='y')
        #
        #     # 断裂符号
        #     d = 0.02
        #
        #     # kwargs = dict(transform=axins_top.transAxes, color='k', clip_on=False)
        #     # axins_top.plot((-d, +d), (-d, +d), **kwargs)
        #     # # axins_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        #     #
        #     # kwargs.update(transform=axins_bottom.transAxes)
        #     # axins_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        #     # # axins_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        #
        #     bbox_top = axins_top.get_position()
        #     bbox_bottom = axins_bottom.get_position()
        #
        #     ratio_top = bbox_top.height / bbox_top.width
        #     ratio_bottom = bbox_bottom.height / bbox_bottom.width
        #
        #     axins_top.plot((-d, +d), (-d * ratio_top, +d * ratio_top),
        #                    transform=axins_top.transAxes, color='k', clip_on=False)
        #
        #     axins_bottom.plot((-d, +d), (1 - d * ratio_bottom, 1 + d * ratio_bottom),
        #                       transform=axins_bottom.transAxes, color='k', clip_on=False)
        #
        #
        #     axins_top.set_ylabel("Frequency (%)")
        #     axins_top.yaxis.set_label_coords(-0.35, -0.35)
        if data_type == 'MCD12Q2':
            inset_pos = [0.01, 0, 0.3, 0.3]
            axins = ax1.inset_axes(inset_pos)
            # axins.set_axis_off()  # 确保容器轴本身不可见

            # 画柱状图（你的data直接用）
            x_indexes = np.arange(len(bin_centers))  # 生成 [0, 1, 2, 3, 4, 5, 6]
            plot_colors = [category_colors[k] for k in bin_centers]

            axins.bar(x_indexes, fractions, color=plot_colors, width=0.7)

            axins.set_ylim(0, 40)

            axins.set_yticks(np.arange(0, 40.01, 10))

            # 美化（隐藏中间边框）
            axins.spines['top'].set_visible(False)
            axins.spines['right'].set_visible(False)

            axins.tick_params(axis='x', bottom=False, labelbottom=False)
            axins.tick_params(axis='y')

            axins.set_ylabel("Frequency (%)")

        elif data_type == 'MODIS_GPP_8d':
            inset_pos = [0.01, 0, 0.3, 0.3]
            axins = ax1.inset_axes(inset_pos)
            # axins.set_axis_off()  # 确保容器轴本身不可见

            # 画柱状图（你的data直接用）
            x_indexes = np.arange(len(bin_centers))  # 生成 [0, 1, 2, 3, 4, 5, 6]
            plot_colors = [category_colors[k] for k in bin_centers]

            axins.bar(x_indexes, fractions, color=plot_colors, width=0.7)

            axins.set_ylim(0, 20)

            axins.set_yticks(np.arange(0, 20.01, 5))

            # 美化（隐藏中间边框）
            axins.spines['top'].set_visible(False)
            axins.spines['right'].set_visible(False)

            axins.tick_params(axis='x', bottom=False, labelbottom=False)
            axins.tick_params(axis='y')

            axins.set_ylabel("Frequency (%)")

    if importance_grape in ['second', 'third']:
        inset_pos = [0.01, 0, 0.3, 0.3]
        axins = ax1.inset_axes(inset_pos)
        # axins.set_axis_off()  # 确保容器轴本身不可见

        # 画柱状图（你的data直接用）
        x_indexes = np.arange(len(bin_centers))  # 生成 [0, 1, 2, 3, 4, 5, 6]
        plot_colors = [category_colors[k] for k in bin_centers]

        axins.bar(x_indexes, fractions, color=plot_colors, width=0.7)

        if importance_grape == 'second':
            if the_most_important_var_method == 'Partial':
                if data_type == 'MCD12Q2':
                    axins.set_ylim(0, 25)

                    axins.set_yticks(np.arange(0, 25.01, 5))
                elif data_type == 'MODIS_GPP_8d':
                    axins.set_ylim(0, 20)

                    axins.set_yticks(np.arange(0, 20.01, 5))
            elif the_most_important_var_method == 'RF':
                axins.set_ylim(0, 40)

                axins.set_yticks(np.arange(0, 40.01, 10))

        elif importance_grape == 'third':
            if data_type == 'MCD12Q2':
                axins.set_ylim(0, 25)

                axins.set_yticks(np.arange(0, 25.01, 5))
            elif data_type == 'MODIS_GPP_8d':
                axins.set_ylim(0, 20)

                axins.set_yticks(np.arange(0, 20.01, 5))

        # 美化（隐藏中间边框）
        axins.spines['top'].set_visible(False)
        axins.spines['right'].set_visible(False)

        axins.tick_params(axis='x', bottom=False, labelbottom=False)
        axins.tick_params(axis='y')

        axins.set_ylabel("Frequency (%)")


    # # 加背景透明
    # axins.set_facecolor('white')
    # axins.patch.set_alpha(1.0)  # 确保完全不透明
    # # axins.set_zorder(5)  # 确保在map图层（pcolormesh）之上


    ################### 3 Colorbar ######################
    if importance_grape == 'third':
        # 创建colorbar
        cbar_ax = ax2

        cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')

        # 设置colorbar标签和刻度
        cbar.set_ticks(np.arange(len(category_colors)))
        cbar.set_ticklabels(list(category_colors.keys()), fontsize=8)
        cbar.ax.tick_params(axis='x', which='major')
        cbar.ax.minorticks_off()
    else:
        ax2.set_visible(False)

        # # 手动调整colorbarposition使其居中
        # cbar.ax.set_position([0.21, 0.3, 0.6, 0.02])


    # # 显示图形
    # plt.tight_layout()

    # output_path1 = os.path.join(output_partial_max_var_png_path, rf"SM_VPD_Cor17\In temporal\Climate{climate_test_number}\{grade_by}\{Basedon}\Preseason\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}({importance_grape})_varname.png")  # SM_VPD_pearson_
    # plt.savefig(output_path1, dpi=600)
    # print(f'png已保存到：{output_path1}')
    # plt.show()

    return fractions, bin_centers, fig


def plot_NH_single(cor_first_data, cor_second_data, cor_third_data, varname, ax):
    """
    只显示 varname 对应的颜色映射
    varname: 变量名，如 'SM'、'VPD'、'Pre' 等
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy as np
    import matplotlib.colors as mcolors


    # 转换函数
    def safe_map_with_weight(x, weight):
        if isinstance(x, np.ma.core.MaskedConstant) or str(x) == 'nan' or x is None:
            return np.nan
        elif x == varname:
            return weight
        else:
            return 0

    ### 转换三个data集
    int_data1 = np.vectorize(lambda x: safe_map_with_weight(x, 1))(cor_first_data)  # first 权重1
    int_data2 = np.vectorize(lambda x: safe_map_with_weight(x, 2))(cor_second_data)  # second 权重2
    int_data3 = np.vectorize(lambda x: safe_map_with_weight(x, 3))(cor_third_data)  # third 权重3

    int_data_origin = int_data1 + int_data2 + int_data3  # 这样值范围 1-3

    int_data = np.ma.masked_where((int_data_origin == 0) | np.isnan(int_data_origin), int_data_origin)

    ### Color
    # 定义所有可能的Color mapping
    all_category_colors = {
        'SOS': '#b2df8a',  # 绿
        'Ta': '#fdbf6f',  # 橙
        'Srad': '#a5aa99',  # 灰
        'Pre': '#a6cee3',  # 蓝
        'SM': '#cab2d6',  # 紫
        'VPD': '#ffff99',  # 黄
        'Coupling': '#e31a1c'  # 红
    }

    # 获取基础color的 RGB 值
    base_rgb = np.array(mcolors.to_rgb(all_category_colors[varname]))


    def lighten(color, factor):
        return color + (1 - color) * factor

    def darken(color, factor):
        return color * (1 - factor)

    # colors = [
    #     base_rgb,# 深（原色）
    #     lighten(base_rgb, 0.3),  # 中
    #     lighten(base_rgb, 0.6)  # 浅
    # ]
    colors = [
        darken(base_rgb, 0.4),  # 深
        base_rgb,  # 原色
        lighten(base_rgb, 0.4)  # 浅
    ]

    # 限制在0-1
    colors = [np.clip(c, 0, 1) for c in colors]

    cmap = ListedColormap(colors)
    cmap.set_bad(alpha=0)

    vmin = int_data.min()
    vmax = int_data.max()
    norm = BoundaryNorm(
        np.arange(vmin - 0.5, vmax + 1, 1),
        int(vmax - vmin + 1)
    )

    ### plot
    # 创建子图布局
    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[5, 0.3], hspace=0.15)
    ax1 = fig.add_subplot(gs_inner[0])  # map
    ax2 = fig.add_subplot(gs_inner[1])  # Colorbar

    ax.axis('off')
    ax1.axis('off')

    plots = []

    # Create the map
    m = Basemap(ax=ax1,
                projection='npstere',
                boundinglat=30,
                lon_0=0,
                resolution='l')

    # Generate grid coordinates
    lons = np.linspace(-180, 180, cols, endpoint=False)
    lats = np.linspace(lat_max, lat_min, rows)
    lons, lats = np.meshgrid(lons, lats)

    # Plot经纬度线
    m.drawparallels(np.arange(30, 90, 30), labels=[0, 0, 0, 0],
                    linewidth=0.5, color="black")
    m.drawmeridians(np.arange(0, 360, 60), latmax=90,
                    labels=[0, 0, 0, 0], linewidth=0.5, color="gray")

    # Plot三层data（透明度叠加）
    plot = m.pcolormesh(lons, lats, int_data, cmap=cmap, norm=norm,
                         alpha=1.0, shading='nearest', latlon=True, zorder=1)

    plots.append(plot)

    # Plotboundaries
    terrence = r'D:/code/NH_terrence/NH_Terrence_30_90'
    m.readshapefile(terrence, 'NH_Terrence', drawbounds=False)
    for shape in m.NH_Terrence:
        points = np.array(shape)
        x, y = points[:, 0], points[:, 1]
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.1
        break_indices = np.where(dist > threshold)[0]

        start_idx = 0
        for break_idx in break_indices:
            ax1.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                     color='black', linewidth=0.3, zorder=3)
            start_idx = break_idx + 1
        ax1.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

    # 北极点
    # x_pole, y_pole = m(0, 90)
    # ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

    # 圆形裁剪
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    center = [(x0 + x1) / 2, (y0 + y1) / 2]
    radius = (x1 - x0) / 2

    clip_circle = Circle(center, radius, transform=ax1.transData)
    for artist in ax1.collections + ax1.lines + ax1.patches:
        artist.set_clip_path(clip_circle)

    boundary_circle = Circle(center, radius,
                             transform=ax1.transData,
                             facecolor='none',
                             edgecolor='black',
                             linewidth=0.8,
                             clip_on=False,  # Whether to clip this object to the axes rectangle
                             zorder=4)
    ax1.add_patch(boundary_circle)


    ##################### 2 Statistics图 #############################
    print(f'np.sum(np.isfinite(int_data_origin)):{np.sum(np.isfinite(int_data_origin))}')
    ### 2.1 Statistics
    counts = [np.sum(int_data == k) for k in range(1,4)]
    total = np.sum(np.isfinite(int_data_origin))
    fractions = [c / total * 100 for c in counts]
    print(f'fractions:{fractions}')

    ### 2.2 Plot
    bin_centers = ['First', 'Second', 'Third']

    # 在 ax1 上创建 inset 容器
    inset_pos = [0.01, 0, 0.3, 0.3]
    axins = ax1.inset_axes(inset_pos)

    axins.set_facecolor('white')
    axins.patch.set_alpha(1)

    # 不显示刻度，但保留背景
    axins.set_xticks([])
    axins.set_yticks([])

    # # 隐藏边框
    # for spine in axins.spines.values():
    #     spine.set_visible(False)

    # # 创建上下两个子轴（模拟断轴）
    # axins_top = axins.inset_axes([0, 0.65, 1, 0.2])  # 上半部分
    # axins_bottom = axins.inset_axes([0, 0, 1, 0.6])  # 下半部分

    # 画柱状图（你的data直接用）
    x_indexes = np.arange(len(bin_centers))  # 生成 [0, 1, 2, 3, 4, 5, 6]
    # bar_colors = [colors[0], colors[1], colors[2]]

    axins.bar(x_indexes, fractions, color=colors, width=0.7)

    # 设置断轴范围
    total_max = max(arr.max() for arr in fractions)
    print(f"绝对最大值是: {total_max}")

    # frac_max = max(fractions)
    # y_max = (math.ceil(frac_max / 5) + 1) * 5
    if the_most_important_var_method == 'Partial':
        y_max = 20
    elif the_most_important_var_method in ['RF', 'XGBoost']:
        y_max = 15


    axins.set_ylim(0, y_max + 0.001)
    axins.set_yticks(np.arange(0, y_max + 0.001, 5))
    tick_size = plt.rcParams['xtick.labelsize']
    axins.tick_params(axis='y', labelsize=tick_size)

    # 美化（隐藏中间边框）
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)

    axins.tick_params(axis='x', bottom=False, labelbottom=False)
    axins.tick_params(axis='y')

    axins.set_ylabel("Frequency (%)")


    ### Colorbar
    cbar = fig.colorbar(plots[0], cax=ax2, orientation='horizontal')

    ticks = np.arange(vmin, vmax + 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['First', 'Second', 'Third'][:len(ticks)], fontsize=8)

    cbar.ax.minorticks_off()

    # 可选：添加标题
    # ax1.set_title(f'{varname} Distribution', fontsize=10, fontweight='bold')

    return ax


def plot_Fig2_S5(first_plot_data, second_plot_data, third_plot_data):

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

    ################  ========= plot ============ ######################
    fig = plt.figure(figsize=(7.2, 7.2), dpi= 300)  # figsize=(6.2, 6.2)
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],  # 四列的宽度比
                           height_ratios=[5, 5],  # Reserve the last slot for the colorbar
                           hspace=0.03, wspace=0.15)
    # plt.tight_layout()
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])


    ######################## Distribution #####################
    fraction_first, bin_center, ax1 = plot_NH_All(first_plot_data, 'first', 'All', ax1)
    fraction_second, bin_center, ax2 = plot_NH_All(second_plot_data, 'second', 'All', ax2)
    fraction_third, bin_center, ax3 = plot_NH_All(third_plot_data, 'third', 'All', ax3)

    ax4 = plot_NH_single(first_plot_data, second_plot_data, third_plot_data, 'Coupling', ax4)

    print(f'Condition: Outlier({Outlier}) CornoSig({OutnosigCor})\n'
          f'    First:   Top2 Total:   Top3 Total:\n')
    for i, label in enumerate(bin_center):
        print(f"{label} - {fraction_first[i]:.1f}%     {(fraction_first[i]+fraction_second[i]):.1f}%    {(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%\n")
        # f"Second: {label} - {fraction_second[i]:.1f}%\n"
        # f"Third: {label} - {fraction_third[i]:.1f}%\n"
        # f"Top2 Total: {label} -{(fraction_first[i]+fraction_second[i]):.1f}%\n",
        # f"Top3 Total: {label} -{(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%")
    for i, label in enumerate(bin_center):
        print(f"Top3 Total: {label} -{(fraction_first[i]+fraction_second[i]+fraction_third[i]):.1f}%")

    # ######################## Colorbar ######################
    # cbar_ax = plt.subplot(gs[-1, 0])  # 第一个变量空间分布
    #
    # cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')
    #
    # category_colors = {
    #     'SOS': '#ff7a00',
    #     'Ta': '#f2b701',
    #     'Srad': '#a5aa99',
    #     'Pre': '#e73f74',
    #     'SM': '#11a579',
    #     'VPD': '#3969ac',
    #     'Cor': '#7f3c8d'
    # }
    #
    # # 设置colorbar标签和刻度
    # cbar.set_ticks(np.arange(len(category_colors)))
    # cbar.set_ticklabels(list(category_colors.keys()))
    # cbar.ax.tick_params(axis='x', which='major')
    #
    # # 手动调整colorbarposition使其居中
    # cbar.ax.set_position([0.21, 0.3, 0.6, 0.02])
    #
    #
    # output_path1 = os.path.join(output_partial_max_var_png_path, rf"SM_VPD_Cor17\In temporal\Climate{climate_test_number}\All\{Basedon}\Preseason\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    if variable_type == 'sos and climate':
        output_path1 = os.path.join(output_partial_max_var_png_path,
                                    rf"D:/code/All/Cor({coupling_method}_{partial_for}_lag={lag_day})_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname{mean_or_yearPOS}_vegS{veg_threshold}.png")  # SM_VPD_pearson_
    elif variable_type == 'only climate':
        output_path1 = os.path.join(output_partial_max_var_png_path,
                                    rf"D:/code/All/Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f'png已保存到：{output_path1}')
    # plt.show()




######## ======= 区分Vegetation type、AI梯度、Cor梯度plot =============== ###########
def plot_S6_8(varname_data_first, varname_data_second, varname_data_third, grade_by):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.gridspec as gridspec
    import numpy as np
    import os
    from brokenaxes import brokenaxes
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
        fig = plt.figure(figsize=(8.6, 13.5))
        gs = gridspec.GridSpec(5, 3,
                               width_ratios=[1, 1, 1],  # 四列的宽度比
                               height_ratios=[5, 5, 5, 5, 1],  # Reserve the last slot for the colorbar
                               hspace=0.001, wspace=0.2)
        # 定义四种Vegetation type
        types = ['Forest', 'Shrub', 'Savanna', 'Grass']#, 'Wet']
        titles = ['Forest', 'Shrub', 'Savanna', 'Grass']# 'Wet']

    elif grade_by == 'AI':
        fig = plt.figure(figsize=(8.6, 13.5))
        gs = gridspec.GridSpec(5, 3,
                               width_ratios=[1, 1, 1],  # 四列的宽度比
                               height_ratios=[5, 5, 5, 5, 1],  # Reserve the last slot for the colorbar
                               hspace=0.001, wspace=0.2)
        types = ['Arid', 'Semi-Arid', 'Dry sub-humid', 'Humid']
        titles = ['Arid', 'Semi-Arid', 'Dry sub-humid', 'Humid']
    elif grade_by == 'Cor mean':
        fig = plt.figure(figsize=(8.6, 15.5))
        gs = gridspec.GridSpec(6, 3,
                               width_ratios=[1, 1, 1],  # 四列的宽度比
                               height_ratios=[5, 5, 5, 5, 5, 1],  # Reserve the last slot for the colorbar
                               hspace=0.3, wspace=0.2)
        types = [#'Cor(<-0.5)',
                 'Cor(<-0.4)', 'Cor(-0.4~-0.3)', 'Cor(-0.3~-0.2)',
                 'Cor(-0.2~-0.1)', 'Cor(-0.1~0)']
        titles = [#'Cor(<-0.5)',
                  'Cor(<-0.4)', 'Cor(-0.4~-0.3)', 'Cor(-0.3~-0.2)',
                  'Cor(-0.2~-0.1)', 'Cor(-0.1~0)']




    for i, (type, title) in enumerate(zip(types, titles)):

        # 设置子图标题字母
        subplot_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        word = subplot_letters[i] if i < len(subplot_letters) else ''

        if grade_by == 'Veg':
            # 创建Vegetation type掩码
            if type == 'All':
                mask = np.isfinite(veg_type_data)
            elif type == 'Forest':
                ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[0, 2])  # 第一个变量空间分布
                mask = (veg_type_data == 1)
            elif type == 'Shrub':
                ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[1, 2])  # 第一个变量空间分布
                mask = (veg_type_data == 2)
            elif type == 'Savanna':
                ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[2, 2])  # 第一个变量空间分布
                mask = (veg_type_data == 3)
            elif type == 'Grass':
                ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[3, 2])  # 第一个变量空间分布
                mask = (veg_type_data == 4)
            elif type == 'Wet':
                mask = (veg_type_data == 5)
        elif grade_by == 'AI':
            # 创建Vegetation type掩码
            if type == 'All':
                mask = np.isfinite(ai_data)
            elif type == 'Hyper Arid':
                mask = (ai_data == 1)  # AI 0-0.03
            elif type == 'Arid':
                ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[0, 2])  # 第一个变量空间分布
                mask = (ai_data == 2)  # AI 0.03-0.2
            elif type == 'Semi-Arid':
                ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[1, 2])  # 第一个变量空间分布
                mask = (ai_data == 3) | (ai_data == 4)  # AI 0.2-0.35
            elif type == 'Dry sub-humid':
                ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[2, 2])  # 第一个变量空间分布
                mask = (ai_data == 5)  # AI 0.5-0.65
            elif type == 'Humid':
                ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[3, 2])  # 第一个变量空间分布
                mask = (ai_data == 6)  # AI > 0.65

        elif grade_by == 'Cor mean':

            if type == 'All':
                mask = np.isfinite(cor_mean_data)
            elif type == 'Cor(-0.1~0)':
                ax1 = plt.subplot(gs[0, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[0, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[0, 2])  # 第一个变量空间分布
                mask = (-0.1 <= cor_mean_data) & (cor_mean_data < 0)
            elif type == 'Cor(-0.2~-0.1)':
                ax1 = plt.subplot(gs[1, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[1, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[1, 2])  # 第一个变量空间分布
                mask = (-0.2 <= cor_mean_data) & (cor_mean_data < -0.1)
            elif type == 'Cor(-0.3~-0.2)':
                ax1 = plt.subplot(gs[2, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[2, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[2, 2])  # 第一个变量空间分布
                mask = (-0.3 <= cor_mean_data) & (cor_mean_data < -0.2)
            elif type == 'Cor(-0.4~-0.3)':
                ax1 = plt.subplot(gs[3, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[3, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[3, 2])  # 第一个变量空间分布
                mask = (-0.4 <= cor_mean_data) & (cor_mean_data < -0.3)
            elif type == 'Cor(<-0.4)':
                ax1 = plt.subplot(gs[4, 0])  # 第一个变量空间分布
                ax2 = plt.subplot(gs[4, 1])  # 第一个变量空间分布
                ax3 = plt.subplot(gs[4, 2])  # 第一个变量空间分布
                mask = (cor_mean_data < -0.4) #& (-0.5 <= cor_mean_data)
            # elif type == 'Cor(<-0.5)':
            #     ax1 = plt.subplot(gs[5, 0])  # 第一个变量空间分布
            #     ax2 = plt.subplot(gs[5, 1])  # 第一个变量空间分布
            #     ax3 = plt.subplot(gs[5, 2])  # 第一个变量空间分布
            #     mask = (cor_mean_data < -0.5)

        # 应用掩码
        plot_varname_data_first = np.where(mask, varname_data_first, np.nan)
        plot_varname_data_second = np.where(mask, varname_data_second, np.nan)
        plot_varname_data_third = np.where(mask, varname_data_third, np.nan)

        ############### =========== Statistics占比 =========== ##################
        # 各类别顺序需与图中的color及Statistics类别一致
        bin_center = ['SOS', 'Ta', 'Srad', 'Pre', 'SM', 'VPD', 'Coupling']

        def calculate_fraction(data):
            counts = np.array([
                np.count_nonzero(data == label)
                for label in bin_center
            ], dtype=float)

            total = counts.sum()

            if total == 0:
                return np.zeros(len(bin_center), dtype=float)

            return counts / total * 100

        fraction_first = calculate_fraction(plot_varname_data_first)
        fraction_second = calculate_fraction(plot_varname_data_second)
        fraction_third = calculate_fraction(plot_varname_data_third)

        print(
            f'==== {type} ===='
            f'Condition: Outlier({Outlier}) CornoSig({OutnosigCor}) '
            f'Grade({grade_by}: {type})\n'
            f'    First:   Top2 Total:   Top3 Total:'
        )

        for i, label in enumerate(bin_center):
            top2_total = fraction_first[i] + fraction_second[i]
            top3_total = top2_total + fraction_third[i]

            print(
                f'{label} - {fraction_first[i]:.1f}%'
                f'     {top2_total:.1f}%'
                f'     {top3_total:.1f}%'
            )

        ################  ========= plot ============ ######################
        plots = []  # Store the plot object for each subplot

        ########### 1 空间分布及频率 #################
        def plot_distribution_frequency(plot_data, input_ax, importance_grape):
            ########### 1 空间分布 #################
            input_ax.set_box_aspect(1)  # Make the map axis square and fill the grid height
            input_ax.axis('off')
            ### 1.1 Create the map
            m = Basemap(ax=input_ax,
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
                            labels=[0, 0, 0, 1],  # labels=[left, right, top, bottom] Control longitude label visibility
                            linewidth=0.6,
                            # fontsize=8,
                            color="gray")

            # # Fill continents
            # m.fillcontinents(color='white', lake_color='white', zorder=1)

            # map_boundary = m.drawmapboundary(linewidth=0)  # Hide boundary lines

            ### 1.2 Plotdata
            # Color mapping
            category_colors = {
                'SOS': '#b2df8a',  # 绿
                'Ta': '#fdbf6f',  # 橙
                'Srad': '#a5aa99',  # 灰
                'Pre': '#a6cee3',  # 蓝
                'SM': '#cab2d6',  # 紫
                'VPD': '#ffff99',  # 黄
                'Coupling': '#e31a1c'  # 红
            }

            # 创建映射字典
            keys = list(category_colors.keys())
            str_to_int = {k: i for i, k in enumerate(keys)}

            # 向量化转换（处理NaN）
            def safe_map(x):
                if isinstance(x, np.ma.core.MaskedConstant) or str(x) == 'nan':
                    return -1
                return str_to_int.get(x, -1)  # 不存在返回-1

            # 转换为整数矩阵
            int_data = np.vectorize(safe_map)(plot_data)
            int_data = np.ma.masked_where(int_data == -1, int_data)  # 屏蔽无效值

            # 创建Color mapping
            cmap = ListedColormap(list(category_colors.values()))
            norm = BoundaryNorm(np.arange(len(category_colors) + 1) - 0.5,
                                len(category_colors))

            # Plot空间分布图
            plot = m.pcolormesh(lons, lats, int_data, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                                zorder=1)  # Avoid tearing near the pole
            plots.append(plot)  # Save the plot object


            ### 1.3 Plotboundaries
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
                threshold = (input_ax.get_xlim()[1] - input_ax.get_xlim()[0]) * 0.1

                # Find the indices of jumps
                break_indices = np.where(dist > threshold)[0]

                if len(break_indices) == 0:
                    # Draw the full line if there are no jumps
                    input_ax.plot(x, y, color='black', linewidth=0.3, zorder=3)
                else:
                    # Split the line at jumps and draw each segment
                    # This removes lines crossing the center while retaining valid boundaries
                    start_idx = 0
                    for break_idx in break_indices:
                        input_ax.plot(x[start_idx:break_idx + 1], y[start_idx:break_idx + 1],
                                 color='black', linewidth=0.3, zorder=3)
                        start_idx = break_idx + 1
                    # Draw the final segment
                    input_ax.plot(x[start_idx:], y[start_idx:], color='black', linewidth=0.3, zorder=3)

            # # 北极点
            # x_pole, y_pole = m(0, 90)
            # input_ax.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

            ### 4 Clip to the outer boundary
            from matplotlib.patches import Circle

            x0, x1 = input_ax.get_xlim()
            y0, y1 = input_ax.get_ylim()

            center = [(x0 + x1) / 2, (y0 + y1) / 2]
            radius = (x1 - x0) / 2

            clip_circle = Circle(center, radius, transform=input_ax.transData)

            for artist in input_ax.collections + input_ax.lines + input_ax.patches:
                artist.set_clip_path(clip_circle)

            boundary_circle = Circle(
                center,
                radius,
                transform=input_ax.transData,
                facecolor='none',
                edgecolor='black',  # color
                linewidth=1,
                clip_on = False,
                zorder=4  # Place on top
            )

            input_ax.add_patch(boundary_circle)


            # input_ax.set_title(f'({word}) {type}')

            ##################### 2 Statistics图 #############################
            ### 2.1 Statistics
            counts = [np.sum(plot_data == k) for k in keys]
            total = np.sum([np.sum(plot_data == k) for k in keys if str(k) != 'nan'])
            fractions = [c / total * 100 for c in counts]

            ### 2.2 Plot
            bin_centers = ['SOS', 'Ta', 'Srad', 'Pre', 'SM', 'VPD', 'Coupling']

            inset_pos = [0.01, 0, 0.3, 0.3]
            axins = input_ax.inset_axes(inset_pos)
            # axins.set_axis_off()  # 确保容器轴本身不可见

            # 画柱状图
            x_indexes = np.arange(len(bin_centers))  # 生成 [0, 1, 2, 3, 4, 5, 6]
            plot_colors = [category_colors[k] for k in bin_centers]

            axins.bar(x_indexes, fractions, color=plot_colors, width=0.7)

            if importance_grape == 'first':
                if data_type == 'MCD12Q2':
                    axins.set_ylim(0, 55)
                    axins.set_yticks(np.arange(0, 55.01, 10))
                elif data_type == 'MODIS_GPP_8d':
                    axins.set_ylim(0, 25)
                    axins.set_yticks(np.arange(0, 25.01, 5))

            if importance_grape in ['second', 'third']:
                axins.set_ylim(0, 25)
                axins.set_yticks(np.arange(0, 25.01, 5))

            # 美化（隐藏中间边框）
            axins.spines['top'].set_visible(False)
            axins.spines['right'].set_visible(False)

            axins.tick_params(axis='x', bottom=False, labelbottom=False)
            axins.tick_params(axis='y', labelsize=7)

            axins.set_ylabel("Frequency (%)")
            # # 加背景透明
            # axins.set_facecolor('white')
            # axins.patch.set_alpha(1.0)  # 确保完全不透明
            # # axins.set_zorder(5)  # 确保在map图层（pcolormesh）之上

        plot_distribution_frequency(plot_varname_data_first, ax1, 'first')
        plot_distribution_frequency(plot_varname_data_second, ax2, 'second')
        plot_distribution_frequency(plot_varname_data_third, ax3, 'third')

    ################### 2 Colorbar ######################
    # 创建colorbar
    cbar_ax = plt.subplot(gs[-1, 0])  # 第一个变量空间分布

    cbar = fig.colorbar(plots[0], cax=cbar_ax, orientation='horizontal')

    # 设置colorbar标签和刻度
    category_colors = {
        'SOS': '#b2df8a',  # 绿
        'Ta': '#fdbf6f',  # 橙
        'Srad': '#a5aa99',  # 灰
        'Pre': '#a6cee3',  # 蓝
        'SM': '#cab2d6',  # 紫
        'VPD': '#ffff99',  # 黄
        'Coupling': '#e31a1c'  # 红
    }

    cbar.set_ticks(np.arange(len(category_colors)))
    cbar.set_ticklabels(list(category_colors.keys()))
    cbar.ax.tick_params(axis='x', which='major')

    # 手动调整colorbarposition使其居中
    if grade_by in ['Veg', 'AI']:
        cbar.ax.set_position([0.21, 0.14, 0.6, 0.01])
    elif grade_by == 'Cor mean':
        cbar.ax.set_position([0.21, 0.14, 0.6, 0.01])

    # 显示图形
    plt.tight_layout()

    # output_path1 = os.path.join(output_partial_max_var_png_path, rf"SM_VPD_Cor17\In temporal\Climate{climate_test_number}\{grade_by}\{Basedon}\Preseason\SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    if variable_type == 'sos and climate':
        output_path1 = os.path.join(output_partial_max_var_png_path,
                                    rf"D:/code/{grade_by}/Cor({coupling_method}_{partial_for}_lag={lag_day})_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname{mean_or_yearPOS}_vegS{veg_threshold}.png")  # SM_VPD_pearson_
    elif variable_type == 'only climate':
        output_path1 = os.path.join(output_partial_max_var_png_path,
                                    rf"D:/code/{grade_by}/SM_VPD_Cor{cor_test_number}_{analyzed_start_year}-{analyzed_end_year}_Outlier({Outlier})_Outnosig({OutnosigCor})_{the_most_important_var_method}_varname.png")  # SM_VPD_pearson_
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f'png已保存到：{output_path1}')

    # if grade_by == 'Cor mean':
    #     cor_mean_tif_path = 'D:/code/Cor_mean'
    #     out_put = os.path.join(cor_mean_tif_path, 'NH_cor_mean_55km(Python).tif')
    #     save_tif_gdal(out_put, cor_mean_data, rows, cols, crs, gt)
    # plt.show()



######## ======= 看变量的r ============= ##########
def plot_cor_pos_slope_forAllvegType(plot_data_slope, plot_data_r2, plot_data_pvalue, colorbarmin, colorbarmax, data_sig, name, ax):

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

    fig = ax.get_figure()
    gs_inner = ax.get_subplotspec().subgridspec(2, 2,
                                                width_ratios=[5, 0.8],
                                                height_ratios=[5, 0.3],
                                                hspace=0.17, wspace=0.01)

    # Hide the parent ax, which is a placeholder
    ax.axis('off')

    plots = []  # Store the plot object for each subplot

    # ax1 = plt.subplot(gs[0, 0])
    # ax2 = plt.subplot(gs[0, 1])
    # ax3 = plt.subplot(gs[0, 2])
    # Create the three inner axes
    ax1 = fig.add_subplot(gs_inner[0, 0])  # map
    ax2 = fig.add_subplot(gs_inner[0, 1])  # Latitude profile
    ax3 = fig.add_subplot(gs_inner[1, :])  # Colorbar spans both columns



    ########### Subplot 1: spatial distribution #################
    ax1.set_box_aspect(1)  #Make the map axis square and fill the grid height
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
                    labels=[0, 0, 0, 1],  #labels=[left, right, top, bottom] Control longitude label visibility
                    linewidth=0.6,
                    # fontsize=8,
                    color="gray")

    # # Fill continents
    # m.fillcontinents(color='white', lake_color='white', zorder=1)

    # map_boundary = m.drawmapboundary(linewidth=0)  # Hide boundary lines


    ### Plotdata
    # Color mapping
    piyg_cmap = mpl.colormaps['PiYG']
    colors = piyg_cmap(np.linspace(0, 1, 8))
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
    plot = m.pcolormesh(lons, lats, plot_data_slope, cmap=cmap, norm=norm, shading='nearest', latlon=True,
                        zorder=1)  # Avoid tearing near the pole

    plots.append(plot)  # Save the plot object

    if data_sig == 'Yes':
        # Add significance markers
        significant_mask = (plot_data_pvalue < 0.05) & np.isfinite(plot_data_pvalue) & np.isfinite(plot_data_slope)

        if np.any(significant_mask):
            sig_rows, sig_cols = np.where(significant_mask)
            sig_lons = lons[sig_rows, sig_cols]
            sig_lats = lats[sig_rows, sig_cols]

            # Convert geographic coordinates to projected coordinates
            sig_x, sig_y = m(sig_lons, sig_lats)

            ax1.scatter(sig_x, sig_y, marker='.', color='black', s=0.5,
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

    x_pole, y_pole = m(0, 90)
    ax1.scatter(x_pole, y_pole, color='black', s=5, zorder=3)

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
        zorder=4  # Place on top
    )

    ax1.add_patch(boundary_circle)

    data_gte0 = plot_data_slope[(plot_data_slope >= 0) & np.isfinite(plot_data_slope)]
    data_lt0 = plot_data_slope[(plot_data_slope < 0) & np.isfinite(plot_data_slope)]
    sum_count = np.sum(np.isfinite(plot_data_r2))

    data_gte0_count = np.sum(np.isfinite(data_gte0))
    data_lt0_count = np.sum(np.isfinite(data_lt0))

    data_gte0_ratio = data_gte0_count / sum_count * 100
    data_lt0_ratio = data_lt0_count / sum_count * 100

    h = 0.25
    v = 0.82
    ax1.text(h, v,
             f'Mean = {np.nanmean(plot_data_slope):.2f}\n'
             f'P = {np.nanmean(data_gte0):.2f} ({data_gte0_ratio:.1f}%)\n'
             f'N = {np.nanmean(data_lt0):.2f} ({data_lt0_ratio:.1f}%)',
             transform=ax1.transAxes,  # Use relative coordinates for positioning
             multialignment='center',  # Center vertically
             fontsize = 10)

    ### 显著性Statistics ###
    data_gt0_sig_005 = plot_data_slope[(plot_data_slope > 0) & (plot_data_pvalue <= 0.05) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    data_gt0_sig_010 = plot_data_slope[(plot_data_slope > 0) & (plot_data_pvalue <= 0.1) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    data_lte0_sig_005 = plot_data_slope[(plot_data_slope <= 0) & (plot_data_pvalue <= 0.05) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    data_lte0_sig_010 = plot_data_slope[(plot_data_slope <= 0) & (plot_data_pvalue <= 0.1) & np.isfinite(plot_data_slope) & np.isfinite(plot_data_pvalue)]
    sum_count = np.sum(np.isfinite(plot_data_r2))

    data_gt0_sig_005_count = np.sum(np.isfinite(data_gt0_sig_005))
    data_gt0_sig_010_count = np.sum(np.isfinite(data_gt0_sig_010))
    data_lte0_sig_005_count = np.sum(np.isfinite(data_lte0_sig_005))
    data_lte0_sig_010_count = np.sum(np.isfinite(data_lte0_sig_010))

    data_gt0_sig_005ratio = data_gt0_sig_005_count / sum_count * 100
    data_gt0_sig_010ratio = data_gt0_sig_010_count / sum_count * 100
    data_lte0_sig_005ratio = data_lte0_sig_005_count / sum_count * 100
    data_lte0_sig_010ratio = data_lte0_sig_010_count / sum_count * 100
    print(f'data_gt0_sig_005ratio = {data_gt0_sig_005ratio}\n'
          f'data_gt0_sig_010ratio = {data_gt0_sig_010ratio}\n'
          f'data_lte0_sig_005ratio = {data_lte0_sig_005ratio}\n'
          f'data_lte0_sig_010ratio = {data_lte0_sig_010ratio}')


    ########### Subplot 2: variation by latitude

    # Use actual latitude values on the y-axis
    lat_centers = lats[:, 0]

    plot_data_lat = np.nanmean(plot_data_slope, axis=1)

    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    ax2.plot(plot_data_lat, lat_centers, color='red', linewidth=1, alpha=0.8)

    ax2.set_xlim(-1, 1)
    ax2.set_xticks(np.arange(-1, 1.01, 1))
    ax2.set_xticklabels(['-1', '0', '1'])  # Set labels manually

    # tick_size = plt.rcParams['xtick.labelsize']

    # ax2.text(
    #     0.45,  # x = Tick position in data coordinates
    #     -0.005,  # y = Slightly lower in axes coordinates
    #     r'$×10^{-1}$',  # Text to display
    #     transform=ax2.get_xaxis_transform(),
    #     ha='left',  # Align rightward to avoid compression
    #     va='top',
    #     fontsize = tick_size,
    #     clip_on=False
    # )

    ax2.set_ylim(30, 90)
    ax2.set_yticks(np.arange(30, 91, 10))
    ax2.set_yticklabels(f'{x}°' for x in np.arange(30, 91, 10))

    ax2.tick_params(axis='both', which='major', length = 2, pad=3)

    ########### Subplot 3：Colorbar
    ### Create the colorbar in the final position
    cbar = fig.colorbar(plots[0], cax=ax3, orientation='horizontal')

    cbar.set_ticks(bins)

    cbar.set_ticklabels(['0' if x == 0 else
         # f'{int(x * 10)}' if x * 10 == int(x * 10) else
         # f'{x * 10}'
         f'{x:.1f}'
         for x in bins
         ])

    #
    # ax3.text(
    #     1.05,  # x = Tick position in data coordinates
    #     -0.25,  # y = Slightly lower in axes coordinates
    #     r'$×10^{-1}$',  # Text to display
    #     transform=ax3.transAxes,
    #     ha='left',  # Align rightward to avoid compression
    #     va='top',
    #     fontsize=tick_size,
    #     clip_on=False
    # )

    cbar.set_label('Slope between coupling and PPT')


    plt.tight_layout()

    # Current lower-left corner of ax1
    pos1 = ax1.get_position()

    pos2 = ax2.get_position()

    pos3 = ax3.get_position()

    if name == 'All':
        # Reposition ax1
        ax1.set_position([
            pos1.x0 -0.06,  # Keep the left edge unchanged
            pos2.y0,  # Align the bottom with ax2
            pos2.height,
            pos2.height
        ])  # [left, bottom, width, height]
    else:
        ax1.set_position([
            pos1.x0 - 0.11,  # Keep the left edge unchanged
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

    plt.tight_layout()

    plt.show()
    plt.close()


##### Fig2 ######
mcd12q2_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
mcd12q2_mask_tif = gdal.Open(mcd12q2_mask_tif)
mcd12q2_mask = mcd12q2_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
mcd12q2_mask = np.isfinite(mcd12q2_mask)

gpp_mask_tif = rf'D:/code/mean/POS_mean_{scale}km_All.tif'
gpp_mask_tif = gdal.Open(gpp_mask_tif)
gpp_mask = gpp_mask_tif.GetRasterBand(1).ReadAsArray().astype(np.float32)
gpp_mask = np.isfinite(gpp_mask)

cor_lt0_maks = cor_mean_data <= 0
# cor_lt0_maks = np.isfinite(cor_mean_data)
first_plot_data = np.where(cor_lt0_maks & mcd12q2_mask & gpp_mask & np.isfinite(veg_type_data), first_plot_data, np.nan)
second_plot_data = np.where(cor_lt0_maks & mcd12q2_mask & gpp_mask & np.isfinite(veg_type_data), second_plot_data, np.nan)
third_plot_data = np.where(cor_lt0_maks & mcd12q2_mask & gpp_mask & np.isfinite(veg_type_data), third_plot_data, np.nan)


plot_Fig2_S5(first_plot_data, second_plot_data, third_plot_data)
print('Fig 2 plot done!')
sys.exit()
# ###### SI ######
plot_S6_8(first_plot_data, second_plot_data, third_plot_data, 'Veg')   # VegType / AI / Cor mean
print('S6 plot done!')
plot_S6_8(first_plot_data, second_plot_data, third_plot_data, 'AI')   # VegType / AI / Cor mean
print('S7 plot done!')
plot_S6_8(first_plot_data, second_plot_data, third_plot_data, 'Cor mean')   # VegType / AI / Cor mean
print('S8 plot done!')

sys.exit()