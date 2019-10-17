import matplotlib
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn import decomposition

rand_state = 1234

pathToData = "cut_data_cleaned - cut tests 20 degree.csv"
data = pd.read_csv(pathToData)
print("Data head")
print(data.head())

y = data.Cuts
# X = data.drop('Cuts', axis=1)
# Select out the columns we want as features. I could have also just removed them from the CSV but I did it this way
# so I could keep the extra data in the CSV.
# I have removed 'Fe (Iron)', 'Edge angle'
X = data[[
          'C (Carbon)', 'Cr (Chromium)', 'Co (Cobalt)', 'Cu (Copper)', 'Mn (Manganese)', 'Mo (Molybdenum)',
          'N (Nickel)', 'Nb (Niobium)', 'N (Nitrogen)', 'Phosphorous (P)', 'Si (Silicon)', 'Sulfur (S)',
          'Ti (titanium)', 'W (Tungsten)', 'V (Vanadium)']]

# Before scaling or normalizing we want to impute missing values
# In this case the only missing values should be the edge angle that is missing in some cases
# TODO I could try mean, median and most frequent for imputing missing values
# For now i do not need the imputer because I have limited the data to 20 degree angles
# imputer = preprocessing.Imputer(strategy='median')
# X = imputer.fit_transform(X)

# Scale our features (tpot only does StandardScaler and I need to scale to do PCA later)
print("\nMean of X before scaling")
print(X.mean(axis=0))
# scalar = preprocessing.StandardScaler()
# TODO I am still not sure if minMax is that much better then standardScalar
scalar = preprocessing.MinMaxScaler()

scalar.fit(X)
X = scalar.transform(X)

# We want to look for a mean that is zero (other then floating point errors)
print("\nMean of X after scaling")
print(X.mean(axis=0))



# Normalize our features
# X = preprocessing.normalize(X)
#
# print("\nSum of X*X along axis 1 should be 1 after normalizing")
# print((X*X).sum(axis=1))

# TODO use PCA or something else to remove outliers like kmeans
# pca = decomposition.PCA(random_state=rand_state)
# X_pca = pca.fit_transform(X)
# print("\nPCA explained_variance_ratio: " + str(pca.explained_variance_ratio_))


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=rand_state)

# n_jobs = -1 means use all cores -2 means use n cores -1
# Maybe try setting memory to 'auto' to allow caching fitness calculations
# cv=7 means use 7-fold validation
tpot = TPOTRegressor(generations=150,
                     population_size=250,
                     verbosity=2,
                     n_jobs=-1,
                     periodic_checkpoint_folder='tpot_checkpoints',
                     memory='auto',
                     cv=5,
                     mutation_rate=0.8,
                     crossover_rate=0.2,
                     random_state=rand_state)
tpot.fit(X_train, y_train)

print("\nTest data score:"+str(tpot.score(X_test, y_test))+"\n")

# TODO print out predictions for many samples
predictions = tpot.predict(X)
df = pd.DataFrame()
df['Brand'] = data.Brand.astype(str)
df['Knife'] = data.Knife.astype(str)
df['Steel'] = data.Steel.astype(str)
df['Pred Cuts'] = predictions
df['Actual Cuts'] = data.Cuts

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 200, 'display.width', None):  # more options can be specified also
    print(df)

tpot.export('generated_model_pipeline.py')
