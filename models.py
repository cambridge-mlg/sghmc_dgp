from sghmc_dgp import DGP

import numpy as np
from scipy.stats import norm
from scipy.misc import logsumexp
from kernels import SquaredExponential
from likelihoods import Gaussian


class RegressionModel(object):
    def __init__(self):
        class ARGS:
            num_inducing = 100
            iterations = 10000
            minibatch_size = 10000
            window_size = 100
            num_posterior_samples = 100
            posterior_sample_spacing = 50
        self.ARGS = ARGS
        self.model = None

    def fit(self, X, Y):
        lik = Gaussian(np.var(Y, 0))
        return self._fit(X, Y, lik)

    def _fit(self, X, Y, lik, **kwargs):
        if len(Y.shape) == 1:
            Y = Y[:, None]

        kerns = []
        if not self.model:
            for _ in range(2):
                kerns.append(SquaredExponential(X.shape[1], ARD=True, lengthscales=float(X.shape[1])**0.5))

            mb_size = self.ARGS.minibatch_size if X.shape[0] > self.ARGS.minibatch_size else X.shape[0]

            self.model = DGP(X, Y, 100, kerns, lik,
                             minibatch_size=mb_size,
                             window_size=self.ARGS.window_size,
                             **kwargs)

        self.model.reset(X, Y)

        try:
            for _ in range(self.ARGS.iterations):
                self.model.sghmc_step()
                self.model.train_hypers()
                if _ % 100 == 1:
                    print('Iteration {}'.format(_))
                    self.model.print_sample_performance()
            self.model.collect_samples(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing)

        except KeyboardInterrupt:  # pragma: no cover
            pass

    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 100, 1)  # predict in small batches
        for xs in np.array_split(Xs, n):
            m, v = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1)  # num_posterior_samples, N_test, D_y

    def predict(self, Xs):
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v

    def calculate_density(self, Xs, Ys):
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        logps = norm.logpdf(np.repeat(Ys[None, :, :], self.ARGS.num_posterior_samples, axis=0), ms, np.sqrt(vs))
        return logsumexp(logps, axis=0) - np.log(self.ARGS.num_posterior_samples)

    def sample(self, Xs, S):
        ms, vs = self._predict(Xs, S)
        return ms + vs**0.5 * np.random.randn(*ms.shape)

