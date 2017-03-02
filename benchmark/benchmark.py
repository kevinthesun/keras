import os
import sys
import copy
import importlib

backend = ["tensorflow", "theano", "mxnet"]
metrics = ["training_time", "max_memory", "memory_variance", "training_accuracy", "test_accuracy"]
module_name = ["addition_rnn_cpu", "babi_rnn_cpu", "mnist_hierarchical_rnn_cpu"]
result = dict()
test_summary = open('test_summary.txt', 'w')


for back in backend:
    os.environ['KERAS_BACKEND'] = back
    import keras
    reload(keras.backend)
    if back == "theano" or back == "mxnet":
        keras.backend.set_image_dim_ordering('th')
    else:
        keras.backend.set_image_dim_ordering('tf')
    result[back] = dict()
    for module in module_name:
        example = importlib.import_module(module)
        result[back][module] = copy.deepcopy(example.ret_dict)
        del sys.modules[module]

    output = ''
    output += "{backend:<20}\n".format(backend=back)
    output += "{describe:<40}".format(describe='exampe/metric')
    for metric in metrics:
        output += "{metric:<25}".format(metric=metric)
    output += '\n'
    for module in module_name:
        output += "{module:<40}".format(module=module)
        for metric in metrics:
            output += "{metric:<25}".format(metric=result[back][module][metric])
        output += '\n'
    output += '\n'
    test_summary.write(output)
    print output
