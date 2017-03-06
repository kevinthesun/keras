import tensorflow as tf
import os
from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda

global GPU_NUM
GPU_NUM = 8

def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]


def to_multi_gpu(model, n_gpus=2):
    multi_input = True if type(model.input_shape) is list else False
    x = [Input(shape[1:]) for shape in model.input_shape] if multi_input else Input(model.input_shape[1:])
    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = [Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(y) for y in x] \
                      if type(x) is list \
                      else Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    return Model(input= x if type(x) is list else [x], output=merged)


def make_model(model, **kwargs):
    backend = os.environ['KERAS_BACKEND']
    gpu_list = []
    for i in range(GPU_NUM):
        gpu_list.append('gpu(%d)' % i)
    if backend == 'tensorflow':
        model = to_multi_gpu(model, GPU_NUM)
    if backend == 'mxnet':
        kwargs['context'] = gpu_list
        model.compile(**kwargs)
    else:
        model.compile(**kwargs)
    return model
