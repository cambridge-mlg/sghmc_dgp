import numpy as np
from scipy.stats import norm
import pandas

from models import RegressionModel

path = './data/kin8nm.csv'
data = pandas.read_csv(path, header=None).values

X_full = data[:, :-1]
Y_full = data[:, -1:]


N = X_full.shape[0]
n = int(N * 0.8)
ind = np.arange(N)

np.random.shuffle(ind)
train_ind = ind[:n]
test_ind = ind[n:]

X = X_full[train_ind]
Xs = X_full[test_ind]
Y = Y_full[train_ind]
Ys = Y_full[test_ind]

X_mean = np.mean(X, 0)
X_std = np.std(X, 0)
X = (X - X_mean) / X_std
Xs = (Xs - X_mean) / X_std
Y_mean = np.mean(Y, 0)
Y = (Y - Y_mean)
Ys = (Ys - Y_mean)

model = RegressionModel()
model.fit(X, Y)

m, v = model.predict(Xs)
print('MSE', np.mean(np.square(Ys - m)))
print('MLL', np.mean(model.calculate_density(Xs, Ys)))
