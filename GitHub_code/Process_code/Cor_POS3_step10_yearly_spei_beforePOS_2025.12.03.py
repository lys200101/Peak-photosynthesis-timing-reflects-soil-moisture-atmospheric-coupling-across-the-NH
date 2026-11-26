import glob
import os
import re

import numpy as np
from osgeo import gdal, osr


# ============================================================
# 1. Parameters
# ============================================================

startyear = 2001
endyear = 2024

data_type = "MODIS_GPP_8d" # GIMMS_NDVI_16d / MODIS_GPP_16d / MODIS_GPP_8d / MODIS_EVI2_8d / MCD12Q2
scale = 11

min_lat = 30
max_lat = 84

spei_length = 2       # SPEI-2; read band 2 of each SPEI file
drought_percent = 10  # 10th percentile of each 55 km pixel time series
wet_percent = 90      # 90th percentile of each 55 km pixel time series

data_path = (
    rf"E:\CAU\phenology_extract\Result_3_for_region_R2_RMSE"
    rf"\{data_type}\Global_test2_{scale}000m\Data"
)

pos_input_path = rf"{data_path}\Pheno\POS_{scale}km"
spei_root_path = rf"{data_path}\SPEI\NH_SPEI01_03"

drought_event_outputpath = (
    rf"{data_path}\SPEI\Drought_Event"
    rf"\drought_event(POS_SPEI{spei_length}_threshold10%_way3)"
)

spei_strength_outputpath = (
    rf"{data_path}\SPEI\NH_SPEI{spei_length}_{spei_length}monthBeforePOS"
)

os.makedirs(drought_event_outputpath, exist_ok=True)
os.makedirs(spei_strength_outputpath, exist_ok=True)


# ============================================================
# 2. Utility functions
# ============================================================

def read_band(path, band_number=1):
    """Read a band and convert NoData and infinite values to NaN."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开文件：{path}")

    if band_number < 1 or band_number > ds.RasterCount:
        raise ValueError(
            f"{path} 有 {ds.RasterCount} 个波段，"
            f"无法读取第 {band_number} 波段。"
        )

    band = ds.GetRasterBand(band_number)
    array = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()

    if nodata is not None:
        if np.isnan(nodata):
            array[~np.isfinite(array)] = np.nan
        else:
            array[array == nodata] = np.nan

    array[~np.isfinite(array)] = np.nan
    return array, ds


def spatial_refs_same(wkt1, wkt2):
    """Check whether two rasters have the same CRS."""
    srs1 = osr.SpatialReference()
    srs1.ImportFromWkt(wkt1)

    srs2 = osr.SpatialReference()
    srs2.ImportFromWkt(wkt2)

    if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        srs1.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs2.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return bool(srs1.IsSame(srs2))


def crop_pos_rows_by_latitude(ds, lat_min, lat_max):
    """
    Crop rows by the latitude of POS pixel centers.
    Applicable to unrotated latitude-longitude rasters.
    """
    gt = ds.GetGeoTransform()

    if not np.isclose(gt[2], 0) or not np.isclose(gt[4], 0):
        raise ValueError("POS 栅格带有旋转参数，当前裁剪方法不适用。")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjectionRef())

    if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if not srs.IsGeographic():
        raise ValueError(
            "POS 不是经纬度坐标系，不能直接将 Y 坐标当作纬度裁剪。"
        )

    row_centers = gt[3] + (np.arange(ds.RasterYSize) + 0.5) * gt[5]
    keep = (row_centers >= lat_min) & (row_centers <= lat_max)

    if not np.any(keep):
        raise ValueError(f"POS 栅格没有纬度 {lat_min}–{lat_max} 内的像元。")

    kept_rows = np.flatnonzero(keep)
    row_start = int(kept_rows[0])
    row_end = int(kept_rows[-1] + 1)

    cropped_gt = (
        gt[0] + row_start * gt[2],
        gt[1],
        gt[2],
        gt[3] + row_start * gt[5],
        gt[4],
        gt[5],
    )

    return row_start, row_end, cropped_gt


def doy_to_month(doy_array, year):
    """Convert DOY to month; return (month array, valid DOY mask)."""
    leap_year = (
        year % 4 == 0
        and (year % 100 != 0 or year % 400 == 0)
    )

    month_lengths = np.array(
        [31, 29 if leap_year else 28, 31, 30, 31, 30,
         31, 31, 30, 31, 30, 31],
        dtype=np.int32
    )
    cumulative_days = np.cumsum(month_lengths)
    days_in_year = 366 if leap_year else 365

    valid = (
        np.isfinite(doy_array)
        & (doy_array >= 1)
        & (doy_array <= days_in_year)
    )

    month = np.zeros(doy_array.shape, dtype=np.uint8)
    doy_int = np.zeros(doy_array.shape, dtype=np.int32)
    doy_int[valid] = np.rint(doy_array[valid]).astype(np.int32)

    month[valid] = (
        np.searchsorted(
            cumulative_days,
            doy_int[valid] - 1,
            side="right"
        ) + 1
    ).astype(np.uint8)

    return month, valid


def get_year_month_from_filename(path):
    """
    Parse the following example filename:
    NH_SPEI01_03_2001_1_55000m.tif
    Returns year=2001, month=1.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"_(\d{4})_(\d{1,2})_55000m$", stem, re.IGNORECASE)

    if match is None:
        raise ValueError(
            "无法从 SPEI 文件名解析年月："
            f"{os.path.basename(path)}\n"
            "预期文件名类似 NH_SPEI01_03_2001_1_55000m.tif"
        )

    return int(match.group(1)), int(match.group(2))


def write_geotiff(path, array, geotransform, projection, nodata=-9999.0):
    """Write a Float32 GeoTIFF and set NoData correctly."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    out_array = np.where(
        np.isfinite(array),
        array,
        nodata
    ).astype(np.float32)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path,
        array.shape[1],  # Number of columns
        array.shape[0],  # Number of rows
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"]
    )

    if ds is None:
        raise RuntimeError(f"无法创建输出文件：{path}")

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    band = ds.GetRasterBand(1)
    band.WriteArray(out_array)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    ds.FlushCache()
    ds = None


# ============================================================
# 3. POS input and output grid
# ============================================================

years = list(range(startyear, endyear + 1))

pos_files = sorted(glob.glob(os.path.join(pos_input_path, "*.tif")))

if len(pos_files) != len(years):
    raise ValueError(
        f"找到 {len(pos_files)} 个 POS 文件，但研究期有 {len(years)} 年。"
        "请确认每年一个POS文件，并且没有其他tif混在目录中。"
    )

pos_sample = gdal.Open(pos_files[0], gdal.GA_ReadOnly)
if pos_sample is None:
    raise FileNotFoundError(pos_files[0])

pos_row_start, pos_row_end, output_gt = crop_pos_rows_by_latitude(
    pos_sample,
    min_lat,
    max_lat
)

output_rows = pos_row_end - pos_row_start
output_cols = pos_sample.RasterXSize
output_crs = pos_sample.GetProjectionRef()

print("POS output rows and columns: ", output_rows, output_cols)
print("POS output GeoTransform: ", output_gt)


# ============================================================
# 4. Read SPEI-2 by year and month
# ============================================================

# Parse numeric year and month from filenames instead of relying on lexicographic order.
spei_files_by_year_month = {}

for year in years:
    year_dir = os.path.join(spei_root_path, str(year))
    year_files = glob.glob(os.path.join(year_dir, "*.tif"))

    if not year_files:
        raise FileNotFoundError(f"{year} 年没有找到 SPEI tif：{year_dir}")

    for path in year_files:
        file_year, file_month = get_year_month_from_filename(path)

        if file_year != year:
            continue

        if not 1 <= file_month <= 12:
            raise ValueError(f"文件月份不合法：{path}")

        key = (file_year, file_month)
        if key in spei_files_by_year_month:
            raise ValueError(f"发现重复的 SPEI 年月文件：{key}")

        spei_files_by_year_month[key] = path

    missing_months = [
        month for month in range(1, 13)
        if (year, month) not in spei_files_by_year_month
    ]

    if missing_months:
        raise ValueError(
            f"{year} 年缺少 SPEI 月份文件：{missing_months}"
        )

# Read one SPEI file as the source grid reference
first_spei_path = spei_files_by_year_month[(startyear, 1)]
spei_sample = gdal.Open(first_spei_path, gdal.GA_ReadOnly)

if spei_sample is None:
    raise FileNotFoundError(first_spei_path)

spei_rows = spei_sample.RasterYSize
spei_cols = spei_sample.RasterXSize
spei_gt = spei_sample.GetGeoTransform()
spei_crs = spei_sample.GetProjectionRef()

if spei_length > spei_sample.RasterCount:
    raise ValueError(
        f"要求读取 SPEI 第 {spei_length} 波段，"
        f"但样本文件只有 {spei_sample.RasterCount} 个波段。"
    )

if not spatial_refs_same(output_crs, spei_crs):
    raise ValueError(
        "POS 和 SPEI 的 CRS 不同；本代码按你提供的 EPSG:4326 "
        "网格直接映射，不会自动进行 CRS 转换。"
    )

print("SPEI source grid rows and columns: ", spei_rows, spei_cols)
print("SPEI source GeoTransform: ", spei_gt)
print("SPEI band number: ", spei_length)


# Read SPEI-2 for all years and months in January-to-December order.
number_of_months = len(years) * 12
spei_stack = np.full(
    (number_of_months, spei_rows, spei_cols),
    np.nan,
    dtype=np.float32
)

for year_index, year in enumerate(years):
    for month in range(1, 13):
        path = spei_files_by_year_month[(year, month)]
        array, ds = read_band(path, spei_length)

        if array.shape != (spei_rows, spei_cols):
            raise ValueError(f"SPEI 行列数与参考网格不同：{path}")

        if not np.allclose(
            ds.GetGeoTransform(),
            spei_gt,
            atol=1e-10,
            rtol=0
        ):
            raise ValueError(f"SPEI GeoTransform 与参考网格不同：{path}")

        if not spatial_refs_same(ds.GetProjectionRef(), spei_crs):
            raise ValueError(f"SPEI CRS 与参考网格不同：{path}")

        time_index = year_index * 12 + (month - 1)
        spei_stack[time_index] = array
        ds = None

print("SPEI time series shape: ", spei_stack.shape)


# ============================================================
# 5. Calculate per-pixel percentile thresholds on the 55 km source grid
# ============================================================

# Use way3: calculate each 55 km pixel's thresholds from all monthly values in the study period.
# These thresholds are percentiles of each pixel's own time series, not SPEI < -1.
spei_drought_threshold = np.nanpercentile(
    spei_stack,
    drought_percent,
    axis=0
).astype(np.float32)

spei_wet_threshold = np.nanpercentile(
    spei_stack,
    wet_percent,
    axis=0
).astype(np.float32)


# ============================================================
# 6. Find the 55 km SPEI pixel corresponding to each 11 km POS pixel center
# ============================================================

row_grid, col_grid = np.indices((output_rows, output_cols))

# POS pixel center coordinates. Both datasets use EPSG:4326.
x_pos = (
    output_gt[0]
    + (col_grid + 0.5) * output_gt[1]
    + (row_grid + 0.5) * output_gt[2]
)
y_pos = (
    output_gt[3]
    + (col_grid + 0.5) * output_gt[4]
    + (row_grid + 0.5) * output_gt[5]
)

# Convert to floating-point SPEI source grid row and column coordinates, then floor to find the source pixel.
# This applies directly to the supplied north-up GeoTransform.
spei_col_index = np.floor(
    (x_pos - spei_gt[0]) / spei_gt[1]
).astype(np.int32)

spei_row_index = np.floor(
    (y_pos - spei_gt[3]) / spei_gt[5]
).astype(np.int32)

valid_spei_location = (
    (spei_row_index >= 0)
    & (spei_row_index < spei_rows)
    & (spei_col_index >= 0)
    & (spei_col_index < spei_cols)
)

print(
    "POS pixels within the SPEI extent: ",
    int(np.count_nonzero(valid_spei_location))
)

# Assign 55 km thresholds to 11 km pixels using the same spatial mapping
drought_threshold_11km = np.full(
    (output_rows, output_cols), np.nan, dtype=np.float32
)
wet_threshold_11km = np.full(
    (output_rows, output_cols), np.nan, dtype=np.float32
)

drought_threshold_11km[valid_spei_location] = (
    spei_drought_threshold[
        spei_row_index[valid_spei_location],
        spei_col_index[valid_spei_location]
    ]
)

wet_threshold_11km[valid_spei_location] = (
    spei_wet_threshold[
        spei_row_index[valid_spei_location],
        spei_col_index[valid_spei_location]
    ]
)


# Print a spatial mapping example for checking in ArcGIS
example_row = 0
example_col = 0

if valid_spei_location[example_row, example_col]:
    example_spei_row = spei_row_index[example_row, example_col]
    example_spei_col = spei_col_index[example_row, example_col]

    print(
        "Mapping example: POS(row, col)=",
        (example_row, example_col),
        "-> SPEI(row, col)=",
        (example_spei_row, example_spei_col)
    )


# ============================================================
# 7. Extract and classify SPEI-2 for each year's POS month
# ============================================================

for year_index, year in enumerate(years):
    pos_path = pos_files[year_index]
    pos_full, pos_ds = read_band(pos_path, band_number=1)

    if pos_full.shape != (
        pos_sample.RasterYSize,
        pos_sample.RasterXSize
    ):
        raise ValueError(f"POS 行列数与参考网格不同：{pos_path}")

    if not np.allclose(
        pos_ds.GetGeoTransform(),
        pos_sample.GetGeoTransform(),
        atol=1e-10,
        rtol=0
    ):
        raise ValueError(f"POS GeoTransform 与参考网格不同：{pos_path}")

    if not spatial_refs_same(pos_ds.GetProjectionRef(), output_crs):
        raise ValueError(f"POS CRS 与参考网格不同：{pos_path}")

    # Crop POS to match the output extent
    pos = pos_full[pos_row_start:pos_row_end, :]

    # Convert POS DOY to calendar month, accounting for leap years
    pos_month, valid_pos = doy_to_month(pos, year)

    # For each 11 km pixel, look up SPEI-2 by year and POS month.
    spei_at_pos_month = np.full(
        (output_rows, output_cols),
        np.nan,
        dtype=np.float32
    )

    for month in range(1, 13):
        mask = (
            valid_pos
            & valid_spei_location
            & (pos_month == month)
        )

        if not np.any(mask):
            continue

        time_index = year_index * 12 + (month - 1)
        source_values = spei_stack[time_index]

        spei_at_pos_month[mask] = source_values[
            spei_row_index[mask],
            spei_col_index[mask]
        ]

    valid_values = (
        valid_pos
        & valid_spei_location
        & np.isfinite(spei_at_pos_month)
        & np.isfinite(drought_threshold_11km)
        & np.isfinite(wet_threshold_11km)
    )

    # Classes: 1 = drought, 2 = wet, 0 = normal; missing pixels are NaN
    drought_event = np.full(
        (output_rows, output_cols),
        np.nan,
        dtype=np.float32
    )

    drought_event[valid_values] = 0

    drought_mask = (
        valid_values
        & (spei_at_pos_month <= drought_threshold_11km)
    )
    wet_mask = (
        valid_values
        & (spei_at_pos_month >= wet_threshold_11km)
    )

    drought_event[drought_mask] = 1
    drought_event[wet_mask] = 2

    # Save SPEI-2 values for the POS month
    spei_strength = np.full(
        (output_rows, output_cols),
        np.nan,
        dtype=np.float32
    )
    spei_strength[valid_values] = spei_at_pos_month[valid_values]

    event_output_path = os.path.join(
        drought_event_outputpath,
        f"SPEI{spei_length}_POSmonth_event_{year}.tif"
    )

    strength_output_path = os.path.join(
        spei_strength_outputpath,
        f"SPEI{spei_length}_POSmonth_strength_{year}.tif"
    )

    write_geotiff(
        event_output_path,
        drought_event,
        output_gt,
        output_crs
    )

    write_geotiff(
        strength_output_path,
        spei_strength,
        output_gt,
        output_crs
    )

    print(f"{year} event map saved: {event_output_path}")
    print(f"{year} POS-month SPEI-2 saved: {strength_output_path}")

print("Processing complete for all years.")