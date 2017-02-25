import os
import sys
import copy
import importlib

backend = ["mxnet", "tensorflow", "theano"]
metrics = ["training_time", "training_memory", "training_accuracy", "test_accuracy"]
module_name = ["mnist_hierarchical_rnn_cpu", "addition_rnn_cpu", "babi_rnn_cpu"]
result = dict()
test_summary = open('test_summary.txt', 'w')
keras_set = os.path.expanduser("~") + '/.keras/keras.json'


for back in backend:
    os.environ['KERAS_BACKEND'] = back
    result[back] = dict()
    for module in module_name:
        example = importlib.import_module(module)
        result[back][module] = copy.deepcopy(example.ret_dict)
        del sys.modules[module]
    del sys.modules['keras']

output = ''
for back in backend:
    output += "{backend:<20}\n".format(backend=back)
    output += "{describe:<20}".format(describe='exampe/metric')
    for metric in metrics:
        output += "{metric:<25}".format(metric=metric)
    output += '\n'
    for module in module_name:
        output += "{module:<20}".format(module=module)
        for metric in metrics:
            output += "{metric:<25}".format(metric=result[back][module][metric])
        output += '\n'
    output += '\n'

test_summary.write(output)
print output
