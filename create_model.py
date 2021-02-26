# -*- coding: utf-8 -*-

import os

import numpy as np
import tensorflow as tf

from model.CRFLayer import CRFLayer
from model.UNet import UNet


def create_refined_unet_v3(input_channels, num_classes, r=100, eps=1e-4, theta_gamma=3.0, spatial_compat=3.0, bilateral_compat=10.0, num_iterations=10, gt_prob=0.7, unet_pretrained=None):
    """ Create Refined UNet v2 """

    # Input
    inputs = tf.keras.Input(
        shape=[None, None, input_channels], name='inputs')

    # Create UNet
    unet = UNet()

    # Restore pretrained UNet
    if unet_pretrained:
        checkpoint = tf.train.Checkpoint(model=unet)
        checkpoint.restore(tf.train.latest_checkpoint(unet_pretrained))
        print('Checkpoint restored, at {}'.format(
            tf.train.latest_checkpoint(unet_pretrained)))

    # Create CRF layer
    crf = CRFLayer(num_classes, r, eps, theta_gamma,
                   spatial_compat, bilateral_compat, num_iterations)

    # RGB channels, scale [0, 1]
    image = inputs[..., 4:1:-1]

    # Only forward
    logits = unet(inputs)
    probs = tf.nn.softmax(logits, name='logits2probs')
    unary = -tf.math.log(probs * gt_prob, name='probs2unary')

    refined_logits = crf(image=image, unary=unary)

    return tf.keras.Model(inputs=inputs, outputs=[logits, refined_logits])
