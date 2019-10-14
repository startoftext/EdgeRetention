from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn import preprocessing
import numpy as np

pathToData = "cut_data_cleaned.csv"
data = pd.read_csv(pathToData)
print("Data head")
print(data.head())

y = data.Cuts
# X = data.drop('Cuts', axis=1)
# Select out the columns we want as features. I could have also just removed them from the CSV but I did it this way
# so I could keep the extra data in the CSV.
X = data[['Fe (Iron)',
          'C (Carbon)', 'Cr (Chromium)', 'Co (Cobalt)', 'Cu (Copper)', 'Mn (Manganese)', 'Mo (Molybdenum)',
          'N (Nickel)', 'Nb (Niobium)', 'N (Nitrogen)', 'Phosphorous (P)', 'Si (Silicon)', 'Sulfur (S)',
          'Ti (titanium)', 'W (Tungsten)', 'V (Vanadium)', 'Edge angle']]

# Before scaling or normalizing we want to impute missing values
# In this case the only missing values should be the edge angle that is missing in some cases
# TODO I could try mean, median and most frequent for imputing missing values
imputer = preprocessing.Imputer(strategy='median')
X = imputer.fit_transform(X)

# Scale our features
print("\nMean of X before scaling")
print(X.mean(axis=0))
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# We want to look for a mean that is zero (other then floating point errors)
print("\nMean of X after scaling")
print(X.mean(axis=0))

# TODO I could also try scaling by using min/max scalers

# Normalize our features
X = preprocessing.normalize(X)

print("\nSum of X*X along axis 1 should be 1 after normalizing")
print((X*X).sum(axis=1))

# TODO use PCA or something else to remove outliers

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# n_jobs = -1 means use all cores -2 means use n cores -1
# Maybe try setting memory to 'auto' to allow caching fitness calculations
# cv=7 means use 7-fold validation
tpot = TPOTRegressor(generations=100, population_size=100, verbosity=2, n_jobs=-1,
                     periodic_checkpoint_folder='tpot_checkpoints', memory='auto',
                     cv=5, mutation_rate=0.5, crossover_rate=0.5)
tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
tpot.export('generated_model_pipeline.py')
