import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=123)

# Average CV score on the training set was:-8127.713225014512
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.6000000000000001, min_samples_leaf=19, min_samples_split=20, n_estimators=100)),
    ZeroCount(),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.9500000000000001, tol=0.001)),
    VarianceThreshold(threshold=0.01),
    MinMaxScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.6000000000000001, min_samples_leaf=19, min_samples_split=19, n_estimators=100)),
    XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=14, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.7000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
