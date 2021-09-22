import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, plot_precision_recall_curve
from matplotlib import pyplot as plt

features_ = [
    'Fpar_500m',
    'Lai_500m',
    'LST_Day_1km',
    'LST_Night_1km',
    '1 km 16 days NDVI',
    '1 km 16 days EVI',
    # 'ET_500m',
    # 'LE_500m',
    # 'PET_500m',
    # 'PLE_500m',
    'era5_max_u10',
    'era5_max_v10',
    'era5_max_t2m',
    'era5_max_tp',
    'era5_min_u10',
    'era5_min_v10',
    'era5_min_t2m',
    # 'era5_min_tp',
    # 'fwi',
]

dynamic_features = [
    'LST_Day_1km',
    'LST_Night_1km',
    'era5_max_u10',
    'era5_max_v10',
    'era5_max_t2m',
    'era5_max_tp',
    'era5_min_u10',
    'era5_min_v10',
    'era5_min_t2m',
]

features = features_ + [f'{x}_10day_mean' for x in features_]

coordinates = ['x', 'y']

static_features = [
    'dem_mean',
    # 'dem_std',
    'aspect_mean',
    # 'aspect_std',
    'slope_mean',
    # 'slope_std',
    'roads_density_2020',
    'population_density',
    # 'clc'
]

target = 'burned'

features_to_exclude = [
    #  'Fpar_500m',
    #  'Lai_500m',
    #  'FparStdDev_500m',
    #  'LaiStdDev_500m',
    #  'LST_Day_1km',
    #  'Clear_day_cov',
    #  'Clear_night_cov',
    #  'LST_Night_1km',
    #  '1 km 16 days NDVI',
    #  '1 km 16 days EVI',
    #  'ET_500m',
    #  'LE_500m',
    #  'PET_500m',
    #  'PLE_500m',
    #  'era5_min_u10',
    #  'era5_min_v10',
    #  'era5_min_t2m',
    #  'era5_min_tp'
]

dataset_path = Path.home() / 'jh-shared/iprapas/uc3/datasets/pixel_dataset_clc_wmean.nc'

if __name__ == '__main__':
    ds = xr.open_dataset(dataset_path)
    dates = pd.DatetimeIndex(ds['time_'].values)

    val_year = 2020
    test_year = 2020
    train_ds = ds.isel(patch=list(np.where(~dates.year.isin([val_year, test_year]))[0]))
    val_ds = ds.isel(patch=list(np.where(dates.year == val_year)[0]))
    test_ds = ds.isel(patch=list(np.where(dates.year == test_year)[0]))
    # train_ds = ds.isel(patch=slice(0, 17068))
    # test_ds = ds.isel(patch=slice(17068, len(ds.patch)))

    train_dict = {}
    test_dict = {}

    for feat in features + static_features + ['burned']:
        train_dict[feat] = train_ds[feat].values
        test_dict[feat] = test_ds[feat].values

    nan_value = -9999
    train_df = pd.DataFrame.from_dict(train_dict).fillna(nan_value)
    test_df = pd.DataFrame.from_dict(test_dict).fillna(nan_value)

    features_filtered = [x for x in features if x not in features_to_exclude]
    X_train, X_test = train_df[features_filtered + static_features], test_df[features_filtered + static_features]
    y_train, y_test = train_df[target], test_df[target]

    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()
    plot_precision_recall_curve(clf, X_test, y_test)
    plt.show()
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    print("AUPRC:", auc(recall, precision))
    print("AUROC:", roc_auc_score(y_test, y_pred_proba))

    from sklearn.inspection import permutation_importance
    import time
    start_time = time.time()
    result = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: "
          f"{elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=features+static_features)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
