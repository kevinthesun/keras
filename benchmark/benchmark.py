import os
import sys
import json
import copy
import importlib

backend = ["mxnet", "tensorflow", "theano"]
metrics = ["training_time", "training_memory", "training_accuracy", "test_accuracy"]
module_name = ["babi_rnn_cpu"]
result = dict()
test_summary = open('test_summary.txt', 'w')
keras_set = os.path.expanduser("~") + '/.keras/keras.json'


for back in backend:
    keras_set_js = json.load(open(keras_set))
    with open(keras_set, 'r') as f:
        keras_set_js = json.load(f)
        keras_set_js['backend'] = back

    with open(keras_set, 'w') as f:
        f.write(json.dumps(keras_set_js))

    result[back] = dict()
    for module in module_name:
        example = importlib.import_module(module)
        result[back][module] = copy.deepcopy(example.ret_dict)
        del sys.modules[module]

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
