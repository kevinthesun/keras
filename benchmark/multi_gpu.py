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
    multi_output = True if len(model.outputs) > 1 else False
    x = [Input(shape[1:]) for shape in model.input_shape] if multi_input else Input(model.input_shape[1:])
    towers = []
    outputs = []
    for i in range(len(model.outputs)):
        outputs.append([])
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = [Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(y) for y in x] \
                      if multi_input \
                      else Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':n_gpus, 'part':g})(x)
            output_model = model(slice_g)
            if multi_output:
                for num in range(len(output_model):
                    outputs[num].append(output_model[num])
            else:
                 towers.append(output_model)

    with tf.device('/cpu:0'):
        merged = []
        if multi_output:
            merged = []
            for output in outputs:
                merged.append(merge(output, mode='concat', concat_axis=0))
        else:
            merged = merge(towers, mode='concat', concat_axis=0)

    return Model(input= x if type(x) is list else [x], output=merged)


def make_model(model, **kwargs):
    backend = os.environ['KERAS_BACKEND']
    gpu_list = []
    for i in range(GPU_NUM):
        gpu_list.append('gpu(%d)' % i)
    if backend == 'tensorflow' and GPU_NUM > 1:
        model = to_multi_gpu(model, GPU_NUM)
    if backend == 'mxnet':
        kwargs['context'] = gpu_list
        model.compile(**kwargs)
    else:
        model.compile(**kwargs)
    return model
