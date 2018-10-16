# Credit to GPflow

import tensorflow as tf
import numpy as np


class Kernel(object):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.
        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.
        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        self.input_dim = int(input_dim)
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif isinstance(active_dims, slice):
            self.active_dims = active_dims
            if active_dims.start is not None and active_dims.stop is not None and active_dims.step is not None:
                assert len(range(active_dims.start, active_dims.stop, active_dims.step)) == input_dim
        else:
            self.active_dims = np.array(active_dims, dtype=np.int32)
            assert len(active_dims) == input_dim

    def _validate_ard_shape(self, name, value, ARD=None):
        """
        Validates the shape of a potentially ARD hyperparameter
        :param name: The name of the parameter (used for error messages)
        :param value: A scalar or an array.
        :param ARD: None, False, or True. If None, infers ARD from shape of value.
        :return: Tuple (value, ARD), where _value_ is a scalar if input_dim==1 or not ARD, array otherwise.
            The _ARD_ is False if input_dim==1 or not ARD, True otherwise.
        """
        if ARD is None:
            ARD = np.asarray(value).squeeze().shape != ()

        if ARD:
            # accept float or array:
            value = value * np.ones(self.input_dim, dtype=float)

        if self.input_dim == 1 or not ARD:
            correct_shape = ()
        else:
            correct_shape = (self.input_dim,)

        if np.asarray(value).squeeze().shape != correct_shape:
            raise ValueError("shape of {} does not match input_dim".format(name))

        return value, ARD

    def compute_K(self, X, Z):
        return self.K(X, Z)

    def compute_K_symm(self, X):
        return self.K(X)

    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    def on_separate_dims(self, other_kernel):
        """
        Checks if the dimensions, over which the kernels are specified, overlap.
        Returns True if they are defined on different/separate dimensions and False otherwise.
        """
        if isinstance(self.active_dims, slice) or isinstance(other_kernel.active_dims, slice):
            # Be very conservative for kernels defined over slices of dimensions
            return False

        if np.any(self.active_dims.reshape(-1, 1) == other_kernel.active_dims.reshape(1, -1)):
            return False

        return True

    def _slice(self, X, X2):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims`.
        :param X: Input 1 (NxD).
        :param X2: Input 2 (MxD), may be None.
        :return: Sliced X, X2, (Nxself.input_dim).
        """
        if isinstance(self.active_dims, slice):
            X = X[..., self.active_dims]
            if X2 is not None:
                X2 = X2[..., self.active_dims]
        else:
            X = tf.gather(X, self.active_dims, axis=-1)
            if X2 is not None:
                X2 = tf.gather(X2, self.active_dims, axis=-1)

        input_dim_shape = tf.shape(X)[-1]
        input_dim = tf.convert_to_tensor(self.input_dim, dtype=tf.int32)
        with tf.control_dependencies([tf.assert_equal(input_dim_shape, input_dim)]):
            X = tf.identity(X)

        return X, X2

    def _slice_cov(self, cov):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims` for covariance matrices. This requires slicing the
        rows *and* columns. This will also turn flattened diagonal
        matrices into a tensor of full diagonal matrices.
        :param cov: Tensor of covariance matrices (NxDxD or NxD).
        :return: N x self.input_dim x self.input_dim.
        """
        cov = tf.cond(tf.equal(tf.rank(cov), 2), lambda: tf.matrix_diag(cov), lambda: cov)

        if isinstance(self.active_dims, slice):
            cov = cov[..., self.active_dims, self.active_dims]
        else:
            cov_shape = tf.shape(cov)
            covr = tf.reshape(cov, [-1, cov_shape[-1], cov_shape[-1]])
            gather1 = tf.gather(tf.transpose(covr, [2, 1, 0]), self.active_dims)
            gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), self.active_dims)
            cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
                             tf.concat([cov_shape[:-2], [len(self.active_dims), len(self.active_dims)]], 0))
        return cov


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on
        r = || x - x' ||
    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=0.01, lengthscales=1.0,
                 active_dims=None, ARD=None, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - if ARD is not None, it specifies whether the kernel has one
          lengthscale per dimension (ARD=True) or a single lengthscale
          (ARD=False). Otherwise, inferred from shape of lengthscales.
        """
        super().__init__(input_dim, active_dims, name=name)
        self.variance = tf.exp(tf.Variable(np.log(variance), dtype=tf.float64, name='log_variance'))

        lengthscales, self.ARD = self._validate_ard_shape("lengthscales", lengthscales, ARD)
        self.lengthscales = tf.exp(tf.Variable(np.log(lengthscales), dtype=tf.float64, name='log_lengthscales'))

    def _scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.matrix_transpose(Xs)
            return dist

        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += Xs + tf.matrix_transpose(X2s)
        return dist

    def _clipped_sqrt(r2):
        # Clipping around the (single) float precision which is ~1e-45.
        return tf.sqrt(tf.maximum(r2, 1e-40))

    def scaled_square_dist(self, X, X2):  # pragma: no cover
        return self._scaled_square_dist(X, X2)

    def scaled_euclid_dist(self, X, X2):  # pragma: no cover
        """
        Returns |(X - X2ᵀ)/lengthscales| (L2-norm).
        """

        r2 = self.scaled_square_dist(X, X2)
        return self._clipped_sqrt(r2)

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

    def K(self, X, X2=None, presliced=False):
        """
        Calculates the kernel matrix K(X, X2) (or K(X, X) if X2 is None).
        Handles the slicing as well as scaling and computes k(x, x') = k(r),
        where r² = ((x - x')/lengthscales)².
        Internally, this calls self.K_r2(r²), which in turn computes the
        square-root and calls self.K_r(r). Classes implementing stationary
        kernels can either overwrite `K_r2(r2)` if they only depend on the
        squared distance, or `K_r(r)` if they need the actual radial distance.
        """
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.K_r2(self.scaled_square_dist(X, X2))

    def K_r(self, r):
        """
        Returns the kernel evaluated on `r`, which is the scaled Euclidean distance
        Should operate element-wise on r
        """
        raise NotImplementedError

    def K_r2(self, r2):
        """
        Returns the kernel evaluated on `r2`, which is the scaled squared distance.
        Will call self.K_r(r=sqrt(r2)), or can be overwritten directly (and should operate element-wise on r2).
        """
        r = self._clipped_sqrt(r2)
        return self.K_r(r)


class SquaredExponential(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """

    def K_r2(self, r2):
        return self.variance * tf.exp(-r2 / 2.)
