import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=1234)

# Average CV score on the training set was:0.8525458142633526
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.25, min_samples_leaf=13, min_samples_split=11, n_estimators=100)),
    SelectPercentile(score_func=f_regression, percentile=20),
    StackingEstimator(estimator=LinearSVR(C=0.1, dual=True, epsilon=0.001, loss="squared_epsilon_insensitive", tol=0.1)),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=2, min_samples_split=10, n_estimators=100)),
    SelectPercentile(score_func=f_regression, percentile=20),
    ZeroCount(),
    StandardScaler(),
    StackingEstimator(estimator=XGBRegressor(learning_rate=1.0, max_depth=5, min_child_weight=15, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.4)),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=2, min_samples_split=5, n_estimators=100)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=18, min_samples_split=4)),
    LassoLarsCV(normalize=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
