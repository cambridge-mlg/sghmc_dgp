# Credit to GPFlow.

import tensorflow as tf
import numpy as np


class Gaussian(object):
    def logdensity(self, x, mu, var):
        return -0.5 * (np.log(2 * np.pi) + tf.log(var) + tf.square(mu-x) / var)

    def __init__(self, variance=1.0, **kwargs):
        self.variance = tf.exp(tf.Variable(np.log(variance), dtype=tf.float64, name='lik_log_variance'))

    def logp(self, F, Y):
        return self.logdensity(Y, F, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance
