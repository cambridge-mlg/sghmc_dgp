import tensorflow as tf
import numpy as np

from sghmc_base import BaseModel
import conditionals
from scipy.cluster.vq import kmeans2


class Layer(object):
    def __init__(self, kern, outputs, n_inducing, fixed_mean, X):
        self.inputs, self.outputs, self.kernel = kern.input_dim, outputs, kern
        self.M, self.fixed_mean = n_inducing, fixed_mean

        self.Z = tf.Variable(kmeans2(X, self.M, minit='points')[0], dtype=tf.float64, name='Z')
        if self.inputs == outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.outputs:
            self.mean = np.concatenate([np.eye(self.inputs), np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.outputs, :].T

        self.U = tf.Variable(np.zeros((self.M, self.outputs)), dtype=tf.float64, trainable=False, name='U')

    def conditional(self, X):
        # Caching the covariance matrix from the sghmc steps gives a significant speedup. This is not being done here.
        mean, var = conditionals.conditional(X, self.Z, self.kernel, self.U, white=True)

        if self.fixed_mean:
            mean += tf.matmul(X, tf.cast(self.mean, tf.float64))
        return mean, var

    def prior(self):
        return -tf.reduce_sum(tf.square(self.U)) / 2.0


class DGP(BaseModel):
    def propagate(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            mean, var = layer.conditional(Fs[-1])
            eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            F = mean + eps * tf.sqrt(var)
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

    def __init__(self, X, Y, n_inducing, kernels, likelihood, minibatch_size, window_size,
                 adam_lr=0.01, epsilon=0.01, mdecay=0.05):
        self.n_inducing = n_inducing
        self.kernels = kernels
        self.likelihood = likelihood
        self.minibatch_size = minibatch_size
        self.window_size = window_size

        n_layers = len(kernels)
        N = X.shape[0]

        self.layers = []
        X_running = X.copy()
        for l in range(n_layers):
            outputs = self.kernels[l+1].input_dim if l+1 < n_layers else Y.shape[1]
            self.layers.append(Layer(self.kernels[l], outputs, n_inducing, fixed_mean=(l+1 < n_layers), X=X_running))
            X_running = np.matmul(X_running, self.layers[-1].mean)

        super().__init__(X, Y, [l.U for l in self.layers], minibatch_size, window_size)
        self.f, self.fmeans, self.fvars = self.propagate(self.X_placeholder)
        self.y_mean, self.y_var = self.likelihood.predict_mean_and_var(self.fmeans[-1], self.fvars[-1])

        self.prior = tf.add_n([l.prior() for l in self.layers])
        self.log_likelihood = self.likelihood.predict_density(self.fmeans[-1], self.fvars[-1], self.Y_placeholder)

        self.nll = - tf.reduce_sum(self.log_likelihood) / tf.cast(tf.shape(self.X_placeholder)[0], tf.float64) \
                   - (self.prior / N)

        self.generate_update_step(self.nll, epsilon, mdecay)
        self.adam = tf.train.AdamOptimizer(adam_lr)
        self.hyper_train_op = self.adam.minimize(self.nll)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def predict_y(self, X, S):
        assert S <= len(self.posterior_samples)
        ms, vs = [], []
        for i in range(S):
            feed_dict = {self.X_placeholder: X}
            feed_dict.update(self.posterior_samples[i])
            m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
            ms.append(m)
            vs.append(v)
        return np.stack(ms, 0), np.stack(vs, 0)
