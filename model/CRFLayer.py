"""
MIT License

Copyright (c) 2021 Libin Jiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import numpy as np
import tensorflow as tf


def _diagonal_compatibility(shape):
    return tf.eye(shape[0], shape[1], dtype=np.float32)


def _potts_compatibility(shape):
    return -1 * _diagonal_compatibility(shape)


class GaussianLayer(tf.keras.layers.Layer):
    """ Gaussian layer """

    def __init__(self, theta_gamma, truncate=4.):
        super(GaussianLayer, self).__init__()
        self.theta_gamma = theta_gamma
        
        r = int(theta_gamma * truncate + .5)
        x, y = np.mgrid[-r:r + 1, -r:r + 1]
        k = np.exp(-(x ** 2 + y ** 2) / (2 * theta_gamma ** 2), dtype=np.float32)
        k[r, r] = 0
        
        self.r, self.k = r, k


    def call(self, src):
        """ Gaussian filter with FFT """
        
        # Reshape, to [bs, c, h, w]
        src_shape = tf.shape(src)
        bs, h, w = src_shape[0], src_shape[1], src_shape[2]
        src = tf.reshape(src, [bs, h, w, -1])
        src = tf.transpose(src, [0, 3, 1, 2])

        # FFT(src) .x FFT(k)
        src = tf.pad(src, [[0, 0], [0, 0], [self.r, self.r], [self.r, self.r]], mode='constant', constant_values=0)
        ext_k = tf.pad(self.k, [[0, h - 1], [0, w - 1]], mode='constant', constant_values=0)
        src_fft, k_fft = tf.signal.rfft2d(src), tf.signal.rfft2d(ext_k)
        dst_fft = src_fft * k_fft[tf.newaxis, tf.newaxis, ...]
        dst = tf.signal.irfft2d(dst_fft)
        dst = dst[..., 2 * self.r:, 2 * self.r:]
        
        # Reshape, to [bs, h, w, c_1, ..., c_n]
        dst = tf.transpose(dst, [0, 2, 3, 1])
        dst = tf.reshape(dst, src_shape)
        
        return dst

class BilateralLayer(tf.keras.layers.Layer):
    """ Bilateral layer: Gaussian, Two-order Taylor expansion """

    def __init__(self, r, eps, truncate=4.0):
        super(BilateralLayer, self).__init__()
        self.r, self.eps = r, eps
        self.theta_alpha = r / 4.
        
        x, y = np.mgrid[-r:r + 1, -r:r + 1]
        k = np.exp(-.5 * (x ** 2 + y ** 2) / (self.theta_alpha ** 2), dtype=np.float32)
        k[r, r] = 0
        
        self.k = k

    def call(self, I, p):
        def gaussian_filter(src):
            """ Gaussian filter with FFT """

            # Reshape, to [bs, c, h, w]
            src_shape = tf.shape(src)
            bs, h, w = src_shape[0], src_shape[1], src_shape[2]
            src = tf.reshape(src, [bs, h, w, -1])
            src = tf.transpose(src, [0, 3, 1, 2])

            # FFT(src) .x FFT(k)
            src = tf.pad(src, [[0, 0], [0, 0], [self.r, self.r], [self.r, self.r]], mode='constant', constant_values=0)
            ext_k = tf.pad(self.k, [[0, h - 1], [0, w - 1]], mode='constant', constant_values=0)
            src_fft, k_fft = tf.signal.rfft2d(src), tf.signal.rfft2d(ext_k)
            dst_fft = src_fft * k_fft[tf.newaxis, tf.newaxis, ...]
            dst = tf.signal.irfft2d(dst_fft)
            dst = dst[..., 2 * self.r:, 2 * self.r:]
            
            # Reshape, to [bs, h, w, c_1, ..., c_n]
            dst = tf.transpose(dst, [0, 2, 3, 1])
            dst = tf.reshape(dst, src_shape)
            
            return dst

        def guided_filter(I, p):
            I_shape, p_shape = tf.shape(I), tf.shape(p)

            # N, [bs, h, w, 1]
            N = gaussian_filter(tf.ones([1, I_shape[1], I_shape[2], 1], dtype=I.dtype))

            # mean_I, [bs, h, w, c]
            mean_I = gaussian_filter(I) / N
            # mean_p, [bs, h, w, n]
            mean_p = gaussian_filter(p) / N

            # cov_Ip, [bs, h, w, c, n]
            cov_Ip = gaussian_filter(I[..., tf.newaxis] * p[..., tf.newaxis, :]) / N[..., tf.newaxis] - mean_I[..., tf.newaxis] * mean_p[..., tf.newaxis, :]

            # var_I, [bs, h, w, c, c]
            var_I = gaussian_filter(tf.matmul(I[..., tf.newaxis], I[..., tf.newaxis, :])) / N[..., tf.newaxis] - tf.matmul(mean_I[..., tf.newaxis], mean_I[..., tf.newaxis, :])

            # Compute a, [bs, h, w, c, n]
            inv = tf.linalg.lu_matrix_inverse(*tf.linalg.lu(var_I + self.eps * tf.eye(I_shape[-1], dtype=I.dtype)[tf.newaxis, tf.newaxis, tf.newaxis, ...]))
            a = tf.matmul(inv, cov_Ip)
            # b, [bs, h, w, n, 1]
            b = mean_p[..., tf.newaxis] - tf.matmul(a, mean_I[..., tf.newaxis], transpose_a=True)

            # mean_a, [bs, h, w, c, n]
            mean_a = gaussian_filter(a) / N[..., tf.newaxis]
            # mean_b, [bs, h, w, n, 1]
            mean_b = gaussian_filter(b) / N[..., tf.newaxis]

            # q, [bs, h, w, n, 1]
            q = tf.matmul(mean_a, I[..., tf.newaxis], transpose_a=True) + mean_b
            # Squeeze
            q = tf.reshape(q, p_shape)

            return q

        return guided_filter(I, p)



class CRFLayer(tf.keras.layers.Layer):
    """ A layer implementing CRF """

    def __init__(self, num_classes, r, eps, theta_gamma, spatial_compat, bilateral_compat, num_iterations):
        super(CRFLayer, self).__init__()
        self.num_classes = num_classes
        self.r, self.eps = r, eps
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations

        self.spatial_weights = spatial_compat * _diagonal_compatibility((num_classes, num_classes))
        self.bilateral_weights = bilateral_compat * _diagonal_compatibility((num_classes, num_classes))
        self.compatibility_matrix = _potts_compatibility((num_classes, num_classes))

        self.bilateral = BilateralLayer(r, eps)
        self.gaussian = GaussianLayer(theta_gamma)

    @tf.function
    def call(self, image, unary):
        """
        The order of parameters: I, p
        """
        assert len(image.shape) == 4 and len(unary.shape) == 4

        unary_shape = tf.shape(unary)
        all_ones = tf.ones([unary_shape[0], unary_shape[1], unary_shape[2], self.num_classes], dtype=tf.float32)
        spatial_norm_vals = self.gaussian(all_ones)
        bilateral_norm_vals = self.bilateral(image, all_ones)

        # Initialize Q
        Q = tf.nn.softmax(-unary)

        for i in range(self.num_iterations):
            tmp1 = -unary

            # Message passing - spatial
            spatial_out = self.gaussian(Q)
            spatial_out /= spatial_norm_vals

            # Message passing - bilateral
            bilateral_out = self.bilateral(image, Q)
            bilateral_out /= bilateral_norm_vals

            # Message passing
            spatial_out = tf.reshape(spatial_out, [-1, self.num_classes])
            spatial_out = tf.matmul(spatial_out, self.spatial_weights)
            bilateral_out = tf.reshape(bilateral_out, [-1, self.num_classes])
            bilateral_out = tf.matmul(bilateral_out, self.bilateral_weights)
            message_passing = spatial_out + bilateral_out

            # Compatibility transform
            pairwise = tf.matmul(message_passing, self.compatibility_matrix)
            pairwise = tf.reshape(
                pairwise, [unary_shape[0], unary_shape[1], unary_shape[2], self.num_classes])

            # Local update
            tmp1 -= pairwise

            # Normalize
            Q = tf.nn.softmax(tmp1)

        return Q
