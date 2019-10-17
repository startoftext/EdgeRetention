import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=1234)

# Average CV score on the training set was:-1793.0023079905532
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.005),
    Nystroem(gamma=0.9, kernel="linear", n_components=10),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=13, min_samples_split=3, n_estimators=100)),
    ZeroCount(),
    ZeroCount(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=13, min_samples_split=17, n_estimators=100)),
    StackingEstimator(estimator=RidgeCV()),
    XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=4, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.9500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
