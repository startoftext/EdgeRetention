import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-9094.90942490662
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=11, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.15000000000000002)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=11, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.15000000000000002)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=5, min_child_weight=11, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.5)),
    RobustScaler(),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.75, tol=0.0001)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=11, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.5)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    RobustScaler(),
    StackingEstimator(estimator=RidgeCV()),
    XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=1, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
