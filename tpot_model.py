from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

pathToData = "cut_data_cleaned.csv"
data = pd.read_csv(pathToData)
# print(data.head())

y = data.Cuts
# X = data.drop('Cuts', axis=1)
# Select out the columns we want as features. I could have also just removed them from the CSV but I did it this way
# so I could keep the extra data in the CSV.
X = data[['Fe (Iron)', 'C (Carbon)', 'Cr (Chromium)', 'Co (Cobalt)', 'Cu (Copper)', 'Mn (Manganese)', 'Mo (Molybdenum)',
          'N (Nickel)', 'Nb (Niobium)', 'N (Nitrogen)', 'Phosphorous (P)', 'Si (Silicon)', 'Sulfur (S)',
          'Ti (titanium)', 'W (Tungsten)', 'V (Vanadium)', 'Edge angle']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# n_jobs = -1 means use all cores -2 means use n cores -1
# Maybe try setting memory to 'auto' to allow caching fitness calculations
# cv=7 means use 7-fold validation
tpot = TPOTRegressor(generations=20, population_size=100, verbosity=2, n_jobs=-1,
                     periodic_checkpoint_folder='tpot_checkpoints', memory='auto',
                     cv=17, mutation_rate=0.8, crossover_rate=0.2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_cut_model_pipeline.py')
