import inline as inline
import matplotlib
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import numpy as np
from sklearn import decomposition
import seaborn as sns

pathToData = "cut_data_cleaned.csv"
data = pd.read_csv(pathToData)

X = data[[
          'Cuts', 'C (Carbon)', 'Cr (Chromium)', 'Co (Cobalt)', 'Cu (Copper)', 'Mn (Manganese)', 'Mo (Molybdenum)',
          'N (Nickel)', 'Nb (Niobium)', 'N (Nitrogen)', 'Phosphorous (P)', 'Si (Silicon)', 'Sulfur (S)',
          'Ti (titanium)', 'W (Tungsten)', 'V (Vanadium)', 'Edge angle']]

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 200, 'display.width', None):
    print(X.describe(include='all'))

plt.figure()
X.Cuts.plot.hist(data.Cuts, bins=30, grid=True)
plt.xlabel("Cuts")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.savefig("visuals/cuts_histogram.png")

# fig = plt.figure()
# plt.matshow(X.corr(), names=list(X.columns))
# ax1 = fig.add_subplot(111)
#
# plt.savefig("visuals/correlation_matrix.png")

plt.figure()
# play with the figsize until the plot is big enough to plot all the columns
# of your dataset, or the way you desire it to look like otherwise
sns.heatmap(X.corr(), cmap='Blues')
plt.savefig("visuals/correlation_matrix.png", bbox_inches="tight")

