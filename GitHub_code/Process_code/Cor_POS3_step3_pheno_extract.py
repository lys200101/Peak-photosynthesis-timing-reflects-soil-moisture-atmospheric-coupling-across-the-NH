
############ Vision2.0 ###############
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy import optimize


# def process_single_pixel(args):
#     # 1. Extract the 24-year pixel time series (24 * 46 = 1104)
#     all_years_data = args['all_years_data']
#     years = args['years']  # e.g. [2001, 2002, ..., 2024]
#     pts_per_year = args['pts_per_year']  # fixed at 46
#
#     pixel_result = {}
#
#     # 2. Revised loop over years
#     for i, year in enumerate(years):
#         # Calculate exact start and end indices for this year
#         # Year 2001 (i=0): 0 to 46
#         # Year 2002 (i=1): 46 to 92 ...
#         start_idx = i * pts_per_year
#         end_idx = start_idx + pts_per_year
#
#         # Slice the 46 GPP observations for this year
#         year_ts_data = all_years_data[start_idx:end_idx]
#
#         # 3. Replace invalid values such as -9999.0 from GEE with NaN
#         year_ts_data[year_ts_data == -9999.0] = np.nan
#
#         # Skip the year if all observations are NaN
#         if np.isnan(year_ts_data).all():
#             continue
#
#         # 4. Alternatively calculate the annual mean as a test
#
#         def read_doy_by_year(inputfile, year):
#
#             gpp_list = []
#             for doy in range(1, 369, 8):
#                 tile_path = os.path.join(inputfile, "GPP_8_day_%s%s_China_grass.tif" % (str(year), str(doy + 1000)[1:]))
#                 ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
#                 if ds is not None:  # !=
#                     data_ds = ds.ReadAsArray().astype(np.float32)  # data.dtype--dtype('uint16')
#                 else:
#                     data_ds = np.ones([rows * cols], dtype=np.float32).reshape(rows, cols) * (-9999.0)
#                     print
#                     'Could not open' + tile_path
#                 gpp_list.append(data_ds)
#                 data_ds = None
#
#             data_array = np.array(gpp_list)
#             return data_array
#
#         def reject_outliers(data, m=3):
#             # data = np.where(data== -9999.0, np.nan, data)
#             data[abs(data - np.nanmean(data, axis=0)) > m * np.nanstd(data, axis=0)] = np.nan
#             return data
#
#         def nan_helper(y):
#             return np.isnan(y), lambda z: z.nonzero()[0]
#
#         def interpolate_daily_with_nan(x_orig, y_orig, x_interp, plot=False):
#             """A function that interpolates 8-day GPP values to daily
#             # numpy.interp vs scipy.interpolate.interp1d (with kind='linear')
#             # numpy.interp(x, xp, fp, left=None, right=None, period=None)
#             """
#             if np.count_nonzero(np.isnan(y_orig), axis=0) <= len(y_orig) - 3:
#                 # ValueError: array of sample points is empty
#                 nans, x = nan_helper(y_orig)
#                 y_orig[nans] = np.interp(x(nans), x(~nans), y_orig[~nans])
#                 # method1
#                 # y_interp = np.interp (x_interp, x_orig, y_orig)
#
#                 # method2
#                 # https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.interpolate.interp1d.html
#                 # scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)
#                 # f_linear = interpolate.interp1d(x_orig, y_orig, fill_value="extrapolate")
#                 # y_interp = f_linear(x_interp)
#
#                 # method3
#                 # https://www.cnblogs.com/onemorepoint/p/8135368.html
#                 # scipy.interpolate.splev(x, tck, der=0, ext=0)
#                 # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html
#                 tck = interpolate.splrep(x_orig, y_orig)
#                 y_interp = interpolate.splev(x_interp, tck)
#
#             else:
#                 y_interp = np.array([np.nan] * len(x_interp))  # -9999.0
#
#             if plot:
#                 plt.plot(x_orig, y_orig, "o", label="original data")
#                 # plt.scatter(x_orig, y_orig, 25, "red")
#                 # plt.plot(xnew, ynew, label="linear interpolation")
#                 plt.plot(x_interp, y_interp, label="B-spline interpolation")
#                 plt.xticks(fontsize=15)
#                 plt.yticks(fontsize=15)
#                 plt.legend()
#                 plt.show()
#             return y_interp
#
#         def fit_GAM_model(x, y, year=2000, plot=False):
#             # simulated independent variables
#             x = np.array(x, copy=True)
#             y = np.array(y, copy=True)
#
#             if year % 4 == 0:
#                 xvals_array = np.arange(1, 367)  # Leap year
#             else:
#                 xvals_array = np.arange(1, 366)
#             # gam = LinearGAM(n_splines = 25, spline_order = 3).gridsearch(DOY_midday, y, n_splines=np.arange(20))
#
#             gam = LinearGAM().gridsearch(x, y)
#             # gam = LinearGAM(n_splines=25, spline_order=3).gridsearch(DOY_midday, y, n_splines=np.arange(50))
#             # lam=np.logspace(-3, 3, 11), n_splines=np.arange(50)
#             # simulated dependent variables
#             yvals_array = gam.predict(xvals_array)
#             # ==============================================================================
#             # C:\Anaconda2\lib\site-packages\pygam\utils.py:209: UserWarning: Expected 2D input data array, but found 1D. Expanding to 2D.
#             #   warnings.warn(msg)
#             # C:\Anaconda2\lib\site-packages\pygam\utils.py:69: UserWarning: Could not import Scikit-Sparse or Suite-Sparse.
#             # This will slow down optimization for models with monotonicity/convexity penalties and many splines.
#             # See installation instructions for installing Scikit-Sparse and Suite-Sparse via Conda.
#             #   warnings.warn(msg)
#             # ==============================================================================
#             if plot:
#                 plt.plot(x, y, 'k.')
#                 plt.plot(xvals_array, yvals_array, 'b-')
#             return yvals_array
#
#         def Double_logistic_func(x, base, frange, srange, mS, mA, S, A):
#
#             """A double logistic model, as in Sobrino and Juliean, or Zhang et al"""
#             """
#             # where base denotes the annual minimum NDVI value; frange denotes the range from spring
#             # minimum NDVI value to annual maximum NDVI value; srange denotes the range from
#             # autumn minimum NDVI value to annual maximum NDVI value; mS and mA are the
#             # maximum slope values of the fitted curve in spring and autumn; S and A are corresponding
#             # dates of mS and mA, namely SOS and EOS. Before extracting SOS and EOS, we also
#             # eliminated pixels with large root mean square errors between fitted and observed NDVI time
#             # series (≥0.15).
#             """
#             return base + frange / (np.exp(-mS * (x - S)) + 1) - srange / (np.exp(-mA * (x - A)) + 1)
#
#         def fit_Double_Logistic_model(x, y, year=2000, plot=False):
#
#             x = np.array(x, copy=True)
#             y = np.array(y, copy=True)
#
#             if year % 4 == 0:
#                 xvals_array = np.arange(1, 367)  # Leap year
#             else:
#                 xvals_array = np.arange(1, 366)
#
#             try:
#                 # The starting estimate for the minimization
#                 # Dbl_logistic might require sensible starting point
#                 # p0 = [y_daily.min(), y_daily.max() - y_daily.min(), y_daily.max() - y_daily.min(), 0.7, 0.6, 16, 34]
#                 # plsq  = optimize.leastsq(residuals, p0, args=(y_daily, x_daily), maxfev= 1000000)
#                 # logistic_y = logistic7_func(x_daily, plsq[0])
#                 # scipy.optimize.leastsq; https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
#
#                 #        popt, pcov = optimize.curve_fit(logistic7_func, x_daily, y_daily, p0, bounds=([0, 0, 0, 0,-np.inf, 1, 1], [np.inf, np.inf, np.inf, np.inf, 0, 366, 366]), \
#                 #                                        method="trf", maxfev= 1000000)
#                 popt, pcov = optimize.curve_fit(Double_logistic_func, x, y, bounds=(
#                 [-1, 0, 0, 0, -1, 1, 1], [np.inf, np.inf, np.inf, np.inf, np.inf, 366, 366]),
#                                                 maxfev=1000000)  # , maxfev= 1000000
#                 # popt, pcov = optimize.curve_fit(Double_logistic_func, x_daily, y_daily, bounds=([0,0,0,1,0,0,1], [1,1,1,366,1,1,366])); [0.017, 6.787, 7.015, 0.068, 0.034, 174.676, 231.35]
#
#                 yvals_array = Double_logistic_func(xvals_array, *popt)
#
#                 if plot:
#                     plt.plot(x, y, 'k.', label="original data")
#                     plt.plot(xvals_array, yvals_array, 'c-',
#                              label='logistic_fitted data')  # label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)
#                     #            plt.ylim(0, 0.1)
#                     #            plt.xticks([1., 2., 3., 4., 5.], ['C1','CN2','N3','CN4','S5'])
#                     #            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%','20%','40%','60%','80%','100%'])
#                     #            plt..legend(loc='best')
#                     #            plt.xlabel('Conditions')
#                     #            plt.ylabel('% perceived more sin like')
#                     plt.legend()
#                     plt.show()
#                     # print (u"Initial parameters:", p0)
#                 # print (u"Fitted parameters", popt) # Parameters fitted to experimental data
#             except RuntimeError:
#                 yvals_array = np.array([-9999.0] * len(xvals_array))
#                 # popt = np.array([np.nan]* 7)
#             # raise ValueError("`x0` is infeasible.")
#             return yvals_array  # popt,
#
#         def Threshold_point(yvals_array, fillvalue):  # dynamic local threshold
#
#             # maximum GPP and its index
#             index_y_max = np.argmax(yvals_array, axis=0)
#             y_max = yvals_array[index_y_max]
#             # Initialize the list of result variables.
#
#             # get SOS, EOS, LGS
#             try:
#                 # The lowest value of the two periods
#                 front_bottom = yvals_array[:index_y_max].min()
#                 # Note! The 6th order polynomial occurs exceptions in this place, the maximum appears on the first day
#                 if front_bottom < 0: front_bottom = 0
#                 latter_bottom = yvals_array[index_y_max:].min()
#                 if latter_bottom < 0: latter_bottom = 0
#
#                 # The dynamic threshold of the two time periods
#                 front_amplitude = y_max - front_bottom
#                 SOS_threshold = 0.0918 * front_amplitude + front_bottom
#                 latter_amplitude = y_max - latter_bottom
#                 EOS_threshold = 0.0918 * latter_amplitude + latter_bottom  # x_interpolation = range(0, 365)
#
#                 SOS = np.abs(yvals_array[:index_y_max] - SOS_threshold).argmin()
#                 GPP_SOS = yvals_array[SOS]
#                 EOS = index_y_max + np.abs(yvals_array[index_y_max:] - EOS_threshold).argmin()
#                 GPP_EOS = yvals_array[EOS]
#
#                 if EOS <= SOS:
#                     y_max, index_y_max, SOS, EOS, LGS, GPP_SOS, GPP_EOS = ([fillvalue] * 7)
#                 else:
#                     LGS = EOS - SOS
#
#             except Exception, e:
#                 print
#                 'Error: ', e
#                 y_max, index_y_max, SOS, EOS, LGS, GPP_SOS, GPP_EOS = ([fillvalue] * 7)
#
#             return (y_max, index_y_max, SOS, EOS, LGS, GPP_SOS, GPP_EOS)
#
#         def get_pheno_indexs(data, nodata, pheno_model="fit_GAM_model", extraction="threshold", fillvalue=-9999.0,
#                              year=2000, plot=False):  # fit_DL_model
#             """
#             Returns the phenological indicators for a given longitude and latitude from 2000 to 2016.
#             Optionally, it will also plot the data.
#             """
#             DOY_midday = []
#             for doy in range(5, 369, 8):
#                 DOY_midday.append(doy)
#                 # The independent variable used to fit(8-day interval, in total, 46 points)
#             DOY_midday = np.array(DOY_midday, copy=True)
#
#             # The independent variable used to fit(8-day interval, in total, 46 points)
#             data_midday = np.array(data, copy=True)
#             data_midday = np.where(data_midday == nodata, np.nan, data_midday)
#             # reject_outliers
#             data_midday_outliers = reject_outliers(data_midday, m=3)
#
#             # Initialize the list of result variables.
#             (SOS, EOS, LGS, PPT, GPP_SOS, GPP_EOS, GPP_MAX) = tuple([[]] * 7)
#
#             # Rule out 1: null values or irrational Numbers
#             finitemask = np.isfinite(data_midday_outliers)
#             specvalmask = data_midday_outliers != nodata
#             validmask = np.logical_and(finitemask, specvalmask)
#
#             # if all is nan, np.unique(finitemask).any()= False
#             # Rule out 2: most or all of the the sequence values are the same
#             if (np.count_nonzero(validmask == False) <= len(data) / 2) and (
#                     np.unique(data_midday_outliers).size > 5):  # Fewer than half of values are NaN, with more than 5 unique values
#                 # index() returns the first index of value!
#                 # https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python
#                 # Replace invalid value
#
#                 # ==============================================================================
#                 #         index_invalid = [i for i, j in enumerate(validmask) if j == False]
#                 #         # python -m timeit "[i for i, j in enumerate(['foo', 'bar', 'baz']*500) if j == 'bar']"
#                 #
#                 #         for i in index_invalid:
#                 #             if i == 0:
#                 #                 data_midday[i]= data_midday[i+1]
#                 #             elif i == len(data_midday)-1:
#                 #                 data_midday[i]= data_midday[i-1]
#                 #             else:
#                 #                 data_midday[i] = np.nanmean((data_midday[i-1], data_midday[i+1]))
#                 # ==============================================================================
#
#                 # 1.1. Iteration five times
#                 # ==============================================================================
#                 #         count = 0
#                 #         while count <= 5:
#                 #             for i in range(len(data_midday) - 2):
#                 #                 if data_midday[i+1] == min(data_midday[i], data_midday[i+1], data_midday[i+2]):
#                 #                     data_midday[i+1] = (data_midday[i] + data_midday[i+2]) / 2
#                 #             count += 1
#                 # ==============================================================================
#
#                 """
#                 # 1.2. Iteration until meet the conditions of y[i+1] is not the min
#                 identify = 1
#                 while identify:
#                     identify = 0
#                     for i in range(len(y) - 2):
#                         if y[i+1] == min(y[i], y[i+1], y[i+2]):
#                             y[i+1] = (y[i] + y[i+2]) / 2
#                             identify += 1
#                         else: identify += 0
#                 """
#                 data_midday_interp = interpolate_daily_with_nan(DOY_midday, data_midday_outliers, DOY_midday,
#                                                                 plot=False)
#                 data_midday_sgfilter = savgol_filter(data_midday_interp, axis=0, window_length=5, polyorder=3)
#
#                 # Find the number of parameters and a pointer to the phenology model func.
#                 # simulated dependent variables
#
#                 if pheno_model == "fit_GAM_model":
#                     yvals_array = fit_GAM_model(DOY_midday, data_midday_sgfilter, year=year, plot=False)
#
#                 elif pheno_model == "fit_DL_model":
#                     yvals_array = fit_Double_Logistic_model(DOY_midday, data_midday_sgfilter, year=year, plot=False)
#
#                     #        plt.plot(DOY_midday, data_midday_sgfilter, 'k.',label='raw points')
#                 #        plt.plot(xvals_array, yvals_array1, 'r-',label='GAM')
#                 #        plt.plot(xvals_array, yvals_array2, 'b--', label='DL')
#                 #        plt.legend()
#                 #        plt.show()
#                 ##############################################################################################################
#                 # get SOS, EOS, LGS
#
#                 if extraction == "threshold":
#                     GPP_MAX, PPT, SOS, EOS, LGS, GPP_SOS, GPP_EOS = Threshold_point(yvals_array, fillvalue)
#
#                 if plot == True:
#                     if year % 4 == 0:
#                         xvals_array = np.arange(1, 367)  # Leap year
#                     else:
#                         xvals_array = np.arange(1, 366)
#
#                     plt.figure(figsize=(8, 4))
#                     # original data point
#                     plt.plot(DOY_midday, data, 'm.', ms=10, label="original points")
#                     plt.plot(DOY_midday, data_midday_outliers, 'k.', ms=10,
#                              label="after removed outliers")  # 'ro'  alpha=0.3,
#                     # linear interpolation
#                     plt.plot(DOY_midday, data_midday_interp, 'b--', label="interpolation", lw=1.5)
#                     plt.plot(xvals_array, yvals_array, 'r-', label=pheno_model, lw=2)  # b--, :,-.
#
#                     # plt.scatter(x[idx_1,1], x[idx_1,0], marker = 'x', color = 'm', label='1', s = 30)
#
#                     # Mark phenological parameters
#                     plt.plot(PPT, GPP_MAX, marker='x', color='magenta', label='PPT', ms=12)  # o, x >,
#                     plt.plot(SOS, GPP_SOS, marker='+', color='aqua', label='SOS', ms=12)
#                     plt.plot(EOS, GPP_EOS, marker='*', color='b', label='EOS', ms=12)
#
#                     # Horizontal and vertical dividing lines: vlines(x, ymin, ymax), hlines(y, xmin, xmax)
#                     plt.vlines(SOS, 0, 0.5, colors="c", linestyles="dashed")
#                     plt.hlines(GPP_SOS, 50, 200, colors="c", linestyles="dashed")
#                     plt.vlines(EOS, 0, 0.5, colors="c", linestyles="dashed")
#                     plt.hlines(GPP_EOS, 200, 350, colors="c", linestyles="dashed")
#                     plt.grid(True)
#                     plt.xlabel("DOY [-]")
#                     plt.xticks(range(0, 373, 10), range(0, 373, 10), size='small', rotation=270)
#                     plt.xlim((-1, 366))
#                     plt.ylabel('GPP [g C$m^{-2}$ $day^{-1}$]')
#                     plt.ylim((-1, 15.0))
#                     #            plt.title('Phenological indicators at location ( %3.4f, %3.4f ) in year(%04d)' % (longitude, latitude, year))
#                     #            plt.title('8-day GPP values at location ( %3.4f, %3.4f ) at (% 04d)' % (longitude, latitude, year) )
#                     # plt.xticks([w*7*24 for w in range(10)],  ['week %i'%w for w in range(10)])
#                     # plt.autoscale(tight=True)
#                     plt.savefig('8-day GPP pheno indexes.png')
#                     plt.legend(loc='best', ncol=1)  # loc='lower left'
#                     plt.show()
#             else:
#                 (SOS, EOS, LGS, PPT, GPP_SOS, GPP_EOS, GPP_MAX) = ([fillvalue] * 7)
#
#             #        return (GPP_MAX, PPT, SOS, EOS, LGS)
#             return (SOS, EOS, LGS, PPT, GPP_SOS, GPP_EOS, GPP_MAX)
#
#         def export_pheno_indexs(InFol, outFol, GeoT, proJ, ncols, nrows, fillvalue, driverName='GTiff',
#                                 pheno_model='fit_GAM_model', method='GAM'):  # CubicSpline, SavGol, Lowess
#             for year in range(2006, 2011):
#                 GPP_array = read_doy_by_year(InFol, year)
#                 GPP_reshape = GPP_array.reshape(GPP_array.size / cols / rows, cols * rows)
#                 SOS, EOS, LGS, PPT, GPP_SOS, GPP_EOS, GPP_MAX = [np.zeros((1, cols * rows)) for _ in range(7)]  # array
#                 args = [nodatav]
#                 kwargs = {'pheno_model': pheno_model, 'extraction': 'threshold', 'fillvalue': -9999.0, 'year': year,
#                           'plot': False}
#                 outArray = np.apply_along_axis(get_pheno_indexs, 0, GPP_reshape, *args, **kwargs)
#                 #  get_pheno_indexs(data, nodatav, pheno_model="fit_GAM_model", extraction="threshold", fillvalue= -9999.0, year= 2000, plot= False)
#                 # fit_GAM_model, fit_CubicSpline_model, fit_SavGol_model, fit_Lowess_model
#                 # outList = list(map ( (lambda x, y: polyfit_func(x, y)), DOY_midday, GPP_reshape))
#
#                 SOS = outArray[0]  # [1D]
#                 EOS = outArray[1]  # [1D]
#                 LGS = outArray[2]
#                 PPT = outArray[3]
#                 GPP_SOS = outArray[4]
#                 GPP_EOS = outArray[5]
#                 GPP_MAX = outArray[6]
#
#                 outShape = (rows, cols)  # outShape = (rows, cols)
#                 outTuple = (
#                     SOS.reshape(outShape),  # 2D array
#                     EOS.reshape(outShape),
#                     LGS.reshape(outShape),
#                     PPT.reshape(outShape),  #
#                     GPP_SOS.reshape(outShape),
#                     GPP_EOS.reshape(outShape),  #
#                     GPP_MAX.reshape(outShape),
#                 )
#                 # # return outTuple
#                 # # convert the arrays back to rasters and save them to disk
#                 # outNames = (
#                 #     'SOS_' + str(year) + '_' + method + '.tif',
#                 #     'EOS_' + str(year) + '_' + method + '.tif',
#                 #     'LGS_' + str(year) + '_' + method + '.tif',
#                 #     'PPT_' + str(year) + '_' + method + '.tif',
#                 #     'GPP_SOS_' + str(year) + '_' + method + '.tif',
#                 #     'GPP_EOS_' + str(year) + '_' + method + '.tif',
#                 #     'GPP_MAX_' + str(year) + '_' + method + '.tif'
#                 # )
#                 #
#                 # for fname, i in zip(outNames, range(len(outNames))):
#                 #     write_file(outFol + fname, outTuple[i], GeoT, proJ, ncols, nrows, fillvalue, "GTiff")
#                 # print
#                 # str(year) + '.tif' + ' has  done'
#
#                 return SOS, EOS, LGS, PPT, GPP_SOS, GPP_EOS, GPP_MAX
#
#         SOS_pixel, EOS_pixel, LGS_pixel, POS_pixel, GPP_SOS_pixel, GPP_EOS_pixel, GPP_MAX_pixel = export_pheno_indexs(inputfile, OutFol_Phenology, geoTran, geoProj, cols, rows, fillvalue=-9999.0,
#                             driverName='GTiff', pheno_model='fit_GAM_model', method='GAM')
#
#
#         # Store the year's phenology results in the dictionary
#         pixel_result[year] = {
#             'SOS': SOS_pixel,
#             'EOS': EOS_pixel,
#             'LGS': LGS_pixel,
#             'POS': POS_pixel,
#             'GPP_SOS': GPP_SOS_pixel,
#             'GPP_EOS': GPP_EOS_pixel,
#             'GPP_POS': GPP_MAX_pixel
#
#         }
#
#     return {'pixel_id': args['pixel_idx'], 'status': 'processed', 'data': pixel_result}


############## Vision3.0 ##################
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy import optimize
from pygam import LinearGAM
import sys


def reject_outliers(data, m=3):
    """Filter outliers."""
    data = np.array(data, copy=True)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0:
        return data
    data[abs(data - mean) > m * std] = np.nan
    return data


def nan_helper(y):
    """Find NaN positions for interpolation."""
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_daily_with_nan(x_orig, y_orig, x_interp):
    """A function that interpolates 8-day GPP values to daily
    # numpy.interp vs scipy.interpolate.interp1d (with kind='linear')
    # numpy.interp(x, xp, fp, left=None, right=None, period=None)
    """
    if np.count_nonzero(np.isnan(y_orig), axis=0) <= len(y_orig) - 3:
        # ValueError: array of sample points is empty
        nans, x = nan_helper(y_orig)
        y_orig[nans] = np.interp(x(nans), x(~nans), y_orig[~nans])
        # method1
        # y_interp = np.interp (x_interp, x_orig, y_orig)

        # method2
        # https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.interpolate.interp1d.html
        # scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)
        # f_linear = interpolate.interp1d(x_orig, y_orig, fill_value="extrapolate")
        # y_interp = f_linear(x_interp)

        # method3
        # https://www.cnblogs.com/onemorepoint/p/8135368.html
        # scipy.interpolate.splev(x, tck, der=0, ext=0)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html
        tck = interpolate.splrep(x_orig, y_orig)
        y_interp = interpolate.splev(x_interp, tck)
        y_interp = np.asarray(y_interp).flatten()

    else:
        y_interp = np.array([np.nan] * len(x_interp))  # -9999.0

    return y_interp



def fit_GAM_model(x, y, year):
    """Fit a GAM model and predict daily values."""

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    # Determine annual length based on leap year
    days_in_year = 366 if year % 4 == 0 else 365

    xvals_array = np.arange(1, days_in_year + 1).reshape(-1, 1)

    gam = LinearGAM().gridsearch(x, y, progress=False)

    yvals_array = gam.predict(xvals_array)

    return yvals_array


def fit_Double_Logistic_model(x, y, year):
    """Fit a double logistic model."""
    x = np.array(x, copy=True)
    y = np.array(y, copy=True)

    # x = np.asarray(x, dtype=float).reshape(-1, 1)
    # y = np.asarray(y, dtype=float)

    days_in_year = 366 if (year % 4 == 0) else 365
    xvals_array = np.arange(1, days_in_year + 1)

    def Double_logistic_func(x, base, frange, srange, mS, mA, S, A):
        return base + frange / (np.exp(-mS * (x - S)) + 1) - srange / (np.exp(-mA * (x - A)) + 1)

    try:
        popt, _ = optimize.curve_fit(
            Double_logistic_func, x, y,
            bounds=([-1, 0, 0, 0, -1, 1, 1], [np.inf, np.inf, np.inf, np.inf, np.inf, 366, 366]),
            maxfev=1000000
        )
        yvals_array = Double_logistic_func(xvals_array, *popt)
    except RuntimeError:
        yvals_array = np.array([np.nan] * len(xvals_array))
    return yvals_array




def Threshold_point(yvals_array, original_yvals_array, original_yvals_doy, ta_vals_array, pheno_start_and_end_threshold, data_type, fillvalue=-9999.0):
    """Extract phenological points using a dynamic local threshold."""
    if np.isnan(yvals_array).all():
        return [fillvalue] * 6

    # index_y_max = int(np.argmax(yvals_array))
    # y_max = yvals_array[index_y_max]

    ## POS selection criteria
    # (1) POS is between DOY 90 and 300
    start_idx = 90
    end_idx = 300

    # Local arrays
    y_subset = yvals_array[start_idx:end_idx + 1]
    ta_subset = ta_vals_array[start_idx:end_idx + 1]

    # Temperature criterion
    ta_valid_mask = np.isfinite(ta_subset) & (ta_subset >= 0)

    # No dates satisfy the temperature criterion between DOY 90 and 300
    if not np.any(ta_valid_mask):
        return [fillvalue] * 6

    # # Keep GAM values for dates satisfying the temperature criterion
    y_temp_valid = y_subset[ta_valid_mask]

    # Find the maximum among dates satisfying the temperature criterion
    local_max_idx = np.argmax(y_temp_valid)

    # Map the local valid index back to the original array
    valid_indices = np.where(ta_valid_mask)[0]
    index_y_max = start_idx + valid_indices[local_max_idx]

    # Maximum GAM value corresponding to POS
    y_max = yvals_array[index_y_max]
    try:
        front_bottom = yvals_array[:index_y_max].min() #if len(yvals_array[:index_y_max]) > 0 else 0
        if front_bottom < 0: front_bottom = 0

        latter_bottom = yvals_array[index_y_max:].min() #if len(yvals_array[index_y_max:]) > 0 else 0
        if latter_bottom < 0: latter_bottom = 0

        front_amplitude = y_max - front_bottom
        SOS_threshold = pheno_start_and_end_threshold * front_amplitude + front_bottom
        latter_amplitude = y_max - latter_bottom
        EOS_threshold = pheno_start_and_end_threshold * latter_amplitude + latter_bottom


        ## SOS selection
        # (0) Require SOS threshold > 0.1 for NDVI and > 0.08 for EVI2
        if data_type == "GIMMS_NDVI" and SOS_threshold < 0.1:
            SOS_threshold = 0.1
        if data_type == "MODIS_EVI2" and SOS_threshold < 0.08:
            SOS_threshold = 0.08
        # # (1) Method 1: select the first DOY reaching the SOS threshold
        # SOS_initial = int(np.abs(yvals_array[:index_y_max] - SOS_threshold).argmin())
        # (2) Method 2
        SOS_initial = None

        for i in range(index_y_max - 1, -1, -1):  # Iterate backwards to index 0

            if yvals_array[i] >= SOS_threshold:
                SOS_initial = i
            else:
                # The series has crossed from above to below the threshold
                # No need to continue searching left
                break

        if SOS_initial is None:
            return [fillvalue] * 6

        # Condition: temperature > 5°C
        SOS_candidates = np.where(ta_vals_array[SOS_initial:index_y_max + 1] >= 0)[0] # Search forward from initial SOS for Ta >= 0°C

        if len(SOS_candidates) == 0:
            # No points have Ta >= 0°C between SOS and POS
            return [fillvalue] * 6

        # First point satisfying the temperature criterion
        SOS = SOS_initial + SOS_candidates[0]
        GPP_SOS = yvals_array[SOS]
        # SOS_ta = ta_vals_array[SOS]
        ## Check whether the SOS value is the minimum within SOS–POS
        # GAM index → DOY
        SOS_doy = SOS + 1
        POS_doy = index_y_max + 1

        original_doy = np.asarray(original_yvals_doy)
        original_values = np.asarray(original_yvals_array)

        # Keep original observations within SOS–POS
        raw_mask = ((original_doy >= SOS_doy) & (original_doy <= POS_doy) & np.isfinite(original_values))

        if not np.any(raw_mask):
            return [fillvalue] * 6

        original_doy_sos_pos = original_doy[raw_mask]
        original_values_sos_pos = original_values[raw_mask]

        # Minimum original observation within SOS–POS
        min_original_idx = np.argmin(original_values_sos_pos)
        min_original_value = original_values_sos_pos[min_original_idx]
        min_original_doy = int(original_doy_sos_pos[min_original_idx])

        # =====================================================
        # If the observed value corresponding to SOS is not the minimum,
        # consider the SOS extraction incorrect
        # and change SOS to the DOY of the minimum observed value
        # =====================================================
        if ((data_type == "GIMMS_NDVI" and min_original_value > 0.1) or
                (data_type == "MODIS_EVI2" and min_original_value > 0.08) or
                (data_type =="MODIS_GPP" and min_original_value > 0)):
            if GPP_SOS > min_original_value:

                SOS = min_original_doy - 1

                # Check the temperature criterion again for corrected SOS
                while SOS < index_y_max and ta_vals_array[SOS] < 5:
                    SOS += 1

                if SOS >= index_y_max:
                    return [fillvalue] * 6

                GPP_SOS = yvals_array[SOS]

        EOS = index_y_max + int(np.abs(yvals_array[index_y_max:] - EOS_threshold).argmin())
        GPP_EOS = yvals_array[EOS]

        # EOS_ta = ta_vals_array[EOS]

        if EOS <= SOS:
            return [fillvalue] * 6
        else:
            LGS = EOS - SOS
            PPT = index_y_max  # DOY corresponding to the peak
            return (y_max, PPT, GPP_SOS, SOS, GPP_EOS, EOS)#, LGS)
    except Exception:
        # return [fillvalue] * 7
        return [fillvalue] * 6


def process_single_pixel(args):
    """
    Core Colab / multiprocessing function: mathematical calculations only
    """
    all_years_data = np.array(args['all_years_data'], dtype=np.float32)
    ta_data = np.array(args['ta_data'], dtype=np.float32)
    years = args['years']  # e.g. [2001, 2002, ..., 2024]
    dates = args['dates']
    ta_dates = args['ta_dates']
    pheno_model = args['pheno_model']  # Options: 'fit_GAM_model' or 'fit_DL_model'
    # output = args['output']
    data_type = args['data_type']
    fillvalue = -9999.0
    timescale = args['timescale']  # '8d', '16d', '1d'

    pheno_start_and_end_threshold = args['pheno_start_and_end_threshold'] # 0.1 / 0.15 / 0.2 / 0.5
    sg_window_size = args['sg_window_size']
    sg_polyorder = args['sg_polyorder']

    # ### Only used to plot site curves
    # site_name = args['site_name']
    # csv_files_path = args['csv_files_path']

    pixel_result = {}
    df_export_all = {}

    if data_type in ["Site", "MODIS_EVI2", "MODIS_GPP", "CSIF"]:
        if timescale == '1d':
            threshold = 76
        elif timescale == '8d':
            threshold = 7
        elif timescale == '16d':
            threshold = 5
    elif data_type == "GIMMS_NDVI":
        threshold = 5

    # Process timestamps outside the loop for performance
    dates = pd.to_datetime(dates)
    ta_dates = pd.to_datetime(ta_dates)
    # dates_np = np.array(dates)
    for i, year in enumerate(years):
        year_mask = (dates.year == year)
        ta_year_mask = (ta_dates.year == year)

        # Get data and dates for the current year
        year_data = all_years_data[year_mask]
        ta_year_data = ta_data[ta_year_mask]

        if data_type != 'GIMMS_NDVI':
            # year_DOY = dates[year_mask].dayofyear.values  # Standard daily time axis (365 points)
            year_DOY = np.array(dates[year_mask].dayofyear)  # Standard daily time axis (365 points)
        elif data_type == 'GIMMS_NDVI':   ### GIMMS dates in GEE are 0101 and 0102; set the time points manually
            new_dates = []

            for month in range(1, 13):
                new_dates.append(pd.Timestamp(year, month, 7))
                new_dates.append(pd.Timestamp(year, month, 23))

            # print(f'dates:{dates}')
            year_DOY = np.array([d.dayofyear for d in new_dates])
            # print(f'year_DOY:{year_DOY}\n'
            #       f'year_data:{year_data}')

        # 2. Clean invalid and NoData values
        data_midday = np.array(year_data, copy=True)
        data_midday = np.where((data_midday == fillvalue) | (data_midday == 65535) , np.nan, data_midday)
        if data_type in ['Site', 'MODIS_GPP']:
            data_midday = np.where(np.isfinite(data_midday) &  (data_midday < 0), 0, data_midday)
        elif data_type == 'GIMMS_NDVI':
            data_midday = np.where(np.isfinite(data_midday) & (data_midday < 0.1), 0.1, data_midday)
        elif data_type == 'MODIS_EVI2':
            data_midday = np.where(np.isfinite(data_midday) & (data_midday < 0.08), 0.08, data_midday)

        # Skip the year if all values are NaN
        if np.isnan(data_midday).all():
            pixel_result[year] = {'POS': fillvalue, 'SOS': fillvalue, 'EOS': fillvalue, f'{data_type}_POS': fillvalue, f'{data_type}_SOS': fillvalue, f'{data_type}_EOS': fillvalue}
            continue

        # 3. Filter outliers
        data_midday_outliers = reject_outliers(data_midday, m=3)

        # 4. Get values at the chosen time interval
        if timescale != '1d':

            ### Used to evaluate the May–September growing season ###
            start_doy_5 = pd.Timestamp(f'{year}-05-01').dayofyear
            end_doy_9 = pd.Timestamp(f'{year}-09-30').dayofyear

            mask_5_9 = (year_DOY >= start_doy_5) & (year_DOY <= end_doy_9)

            if data_type != 'GIMMS_NDVI':
                interval = 8 if timescale == '8d' else 16

                if data_type in ['MODIS_GPP', 'Site']:
                    ######### ========== Extract one fixed-date value every 8/16 days =========== ###########
                    # Find indices of the first and last valid values
                    valid_mask = np.isfinite(data_midday)

                    if valid_mask.sum() == 0:
                        # Skip if the entire year has no valid values
                        doy_coarse = np.array([])
                        data_coarse = np.array([])
                        data_5_9 = np.array([])
                        valid_days_5_9 = 0
                    else:
                        first_valid_idx = np.argmax(valid_mask)  # Position of first True
                        last_valid_idx = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])  # Position of last True

                        start_doy = year_DOY[first_valid_idx]
                        end_doy = year_DOY[last_valid_idx]

                        # Create an 8-day or 16-day interval mask within [start_doy, end_doy]
                        mask_interval = (
                                (year_DOY >= start_doy) &
                                (year_DOY <= end_doy) &
                                ((year_DOY - start_doy) % interval == 0)
                        )

                        # First selection: extract times and data at 8d/16d intervals
                        doy_coarse = year_DOY[mask_interval]
                        data_coarse = data_midday_outliers[mask_interval]

                        data_5_9 = data_midday_outliers[mask_5_9 & mask_interval]

                        valid_days_5_9 = np.sum(np.isfinite(data_5_9))
                elif data_type == 'MODIS_EVI2':
                    ######### ============ Extract the maximum and its date in each 8/16-day window ========= ###########
                    doy_coarse_list = []
                    data_coarse_list = []

                    start_doy = year_DOY[0]
                    end_doy = year_DOY[-1]

                    for window_start in range(start_doy, end_doy + 1, interval):

                        window_end = window_start + interval - 1

                        # Current interval-day window
                        mask_window = (
                                (year_DOY >= window_start) &
                                (year_DOY <= window_end)
                        )

                        doy_window = year_DOY[mask_window]
                        data_window = data_midday_outliers[mask_window]

                        # Remove NaNs
                        valid_window = np.isfinite(data_window)

                        if not np.any(valid_window):
                            continue

                        doy_valid = doy_window[valid_window]
                        data_valid = data_window[valid_window]

                        # Find the index of the maximum within the window
                        max_idx = np.argmax(data_valid)

                        max_value = data_valid[max_idx]
                        max_doy = doy_valid[max_idx]

                        # Save the maximum and its actual date
                        data_coarse_list.append(max_value)
                        doy_coarse_list.append(max_doy)

                    # Convert to NumPy arrays
                    doy_coarse = np.array(doy_coarse_list)
                    data_coarse = np.array(data_coarse_list)

                    # Number of valid maxima in the May–September window
                    valid_5_9 = (
                            (doy_coarse >= start_doy_5) &
                            (doy_coarse <= end_doy_9)
                    )

                    valid_days_5_9 = np.sum(valid_5_9)


            elif data_type == 'GIMMS_NDVI':
                doy_coarse = year_DOY
                data_coarse = data_midday_outliers

                data_5_9 = data_midday_outliers[mask_5_9]

                valid_days_5_9 = np.sum(np.isfinite(data_5_9))

            # Skip if too few valid points remain after removing NaNs to fit a GAM curve
            if valid_days_5_9 < threshold:
                pixel_result[year] = {'POS': fillvalue, 'SOS': fillvalue, 'EOS': fillvalue, f'{data_type}_POS': fillvalue, f'{data_type}_SOS': fillvalue, f'{data_type}_EOS': fillvalue}
                continue

            # Second filter: remove NaN points from these nodes
            valid_mask = np.isfinite(data_coarse)
            doy_for_interp = doy_coarse[valid_mask]
            data_for_interp = data_coarse[valid_mask]

        else:
            # For '1d', use the original daily data after removing NaNs
            valid_mask = ~np.isnan(data_midday_outliers)
            doy_for_interp = year_DOY[valid_mask]
            data_for_interp = data_midday_outliers[valid_mask]

            # Skip if too few valid points remain after removing NaNs to fit a GAM curve
            if len(data_for_interp) < len(year_data) / 2:
                pixel_result[year] = {'POS': fillvalue, 'SOS': fillvalue, 'EOS': fillvalue, f'{data_type}_POS': fillvalue, f'{data_type}_SOS': fillvalue, f'{data_type}_EOS': fillvalue}

                continue

        # 5. Linear interpolation
        if timescale != '1d':
            # if data_type == 'MODIS_GPP':
            #     data_interp = interpolate_daily_with_nan(doy_for_interp, data_for_interp, year_DOY)
            #     doy_for_fit = year_DOY
            # elif data_type == 'MODIS_EVI2':
            data_interp = interpolate_daily_with_nan(doy_for_interp, data_for_interp, doy_coarse)
            doy_for_fit = doy_coarse

        elif timescale == '1d':

            data_interp = interpolate_daily_with_nan(doy_for_interp, data_for_interp, year_DOY)
            doy_for_fit = year_DOY


        # 6. Savitzky-Golay smoothing
        data_interp = np.where(data_interp < 0, 0, data_interp)
        data_filtered = savgol_filter(data_interp, axis=0, window_length=sg_window_size, polyorder=sg_polyorder)
        # data_filtered = data_interp

        # print(f'doy_for_fit:{doy_for_fit}\n'
        #       f'data_interp:{data_interp}\n'
        #       f'data_filtered:{data_filtered}')

        # 7. Model fitting
        try:
            data_filtered = np.where(data_filtered < 0, 0, data_filtered)
            if pheno_model == "fit_GAM_model":
                yvals_array = fit_GAM_model(doy_for_fit, data_filtered, year=year)
            elif pheno_model == "fit_DL_model":
                yvals_array = fit_Double_Logistic_model(doy_for_fit, data_filtered, year=year)
        except Exception as e:
            # Handle numerical fitting failures for individual pixels
            pixel_result[year] = {'POS': fillvalue, 'SOS': fillvalue, 'EOS': fillvalue, f'{data_type}_POS': fillvalue, f'{data_type}_SOS': fillvalue, f'{data_type}_EOS': fillvalue}
            continue


        # ###### ========== Export interpolated, smoothed and GAM data locally ========== #######
        #
        # # ============================================================
        # # 1. Create the daily GAM time axis
        # # ============================================================
        # days_in_year = 366 if year % 4 == 0 else 365
        #
        # gam_doy = np.arange(1, days_in_year + 1)
        #
        # gam_dates = pd.to_datetime(
        #     [f'{year}-{int(doy):03d}' for doy in gam_doy],
        #     format='%Y-%j'
        # )
        #
        # # Build the final DataFrame using daily GAM as the basis
        # df_export = pd.DataFrame({
        #     'Date': gam_dates,
        #     'DOY': gam_doy,
        #     f'{data_type}_GAM': yvals_array
        # })
        #
        # # ============================================================
        # # 2. Original data
        # # ============================================================
        # year_dates = pd.to_datetime(dates[year_mask])
        # year_doy = np.array(year_dates.dayofyear)
        #
        # df_original = pd.DataFrame({
        #     'Date': year_dates,
        #     f'{data_type}_original': data_midday
        # })
        #
        # # ============================================================
        # # 3. Generate actual dates for 8d / 16d data
        # # ============================================================
        # dates_for_fit = pd.to_datetime(
        #     [f'{year}-{int(doy):03d}' for doy in doy_for_fit],
        #     format='%Y-%j'
        # )
        #
        # df_coarse = pd.DataFrame({
        #     'Date': dates_for_fit,
        #     f'{data_type}_interp': data_interp,
        #     # f'{data_type}_SG': data_filtered
        # })
        #
        # # ==================================
        # # 4. Temperature
        # # =================================
        # ta_year_dates = ta_dates[ta_year_mask]  # Key: select dates for the current year only
        # ta_original = pd.DataFrame({
        #     'Date': ta_year_dates,
        #     'Ta_original': ta_year_data
        # })
        #
        #
        # # ============================================================
        # # 4. Merge original, interpolated and SG data onto daily GAM time axis by Date
        # # ============================================================
        # df_export = df_export.merge(
        #     df_original,
        #     on='Date',
        #     how='left'
        # )
        #
        # df_export = df_export.merge(
        #     df_coarse,
        #     on='Date',
        #     how='left'
        # )
        #
        # df_export = df_export.merge(
        #     ta_original,
        #     on='Date',
        #     how='left'
        # )
        #
        # # ============================================================
        # # 5. Adjust column order
        # # ============================================================
        # df_export = df_export[
        #     [
        #         'Date',
        #         'DOY',
        #         f'{data_type}_original',
        #         f'{data_type}_interp',
        #         # f'{data_type}_SG',
        #         f'{data_type}_GAM',
        #         f'Ta_original'
        #     ]
        # ]
        # df_export_all[year] = df_export.copy()
        #
        # # # ============================================================
        # # # 6. Export
        # # # ============================================================
        # # df_export.to_csv(
        # #     rf'{csv_files_path}\Fitted_curve\{data_type}_500m_{site_name}_{year}_Original_Interp_SG_GAM_SOS{pheno_start_and_end_threshold}_SGwin{sg_window_size}_SGpol{sg_polyorder}.csv',
        # #     index=False
        # # )

        ######## ==================================== ###########
        ta_vals_array = np.array(ta_year_data)

        # 7. Extract phenology indicators and corresponding GPP values
        # print(f'yvals_array:{yvals_array}')
        y_max, PPT, SOS_GPP, SOS, EOS_GPP, EOS = Threshold_point(yvals_array, data_midday_outliers, year_DOY, ta_vals_array, pheno_start_and_end_threshold, data_type, fillvalue)

        # 8. Store results in the dictionary
        pixel_result[year] = {
            'POS': int(round(PPT)) if PPT != fillvalue else fillvalue,
            'SOS': int(round(SOS)) if SOS != fillvalue else fillvalue,
            'EOS': int(round(EOS)) if EOS != fillvalue else fillvalue,
            f'{data_type}_POS': round(float(y_max), 6) if y_max != fillvalue else fillvalue,
            f'{data_type}_SOS': round(float(SOS_GPP), 6) if SOS_GPP != fillvalue else fillvalue,
            f'{data_type}_EOS': round(float(EOS_GPP), 6) if EOS_GPP != fillvalue else fillvalue
        }

    return {'pixel_idx': args['pixel_idx'], 'status': 'processed', 'data': pixel_result} #, 'gpp_export': df_export_all}



