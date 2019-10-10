import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFwe, VarianceThreshold, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from tpot.builtins import StackingEstimator
try:
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    from sklearn.preprocessing import Imputer

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was:-12322.859967038718
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            SelectFwe(score_func=f_regression, alpha=0.018000000000000002),
            VarianceThreshold(threshold=0.001)
        ),
        StandardScaler()
    ),
    VarianceThreshold(threshold=0.0005),
    MaxAbsScaler(),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.6500000000000001, min_samples_leaf=18, min_samples_split=18, n_estimators=100)),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=2, max_features=0.7000000000000001, min_samples_leaf=3, min_samples_split=19, n_estimators=100, subsample=0.8500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
