import json
import copy

backend = ["mxnet", "tensorflow", "theano"]
metrics = ["training_time", "training_memory", "training_accuracy", "test_accuracy"]
module_name = ["babi_rnn_cpu"]
result = dict()
test_summary = open('test_summary.txt', 'w')


for back in backend:
    keras_set_js = json.load(open(~/.keras/keras.json))
    with open('~/.keras/keras.json', 'r') as f:
        keras_set_js = json.load(f)
        keras_set_js['backend'] = back

    with open('~/.keras/keras.json', 'w') as f
        f.write(json.dumps(keras_set_js))

    result[back] = dict()
    for module in module_name:
        import module
        result[back][module] = copy.deepcopy(module.ret_dict)

output = ''
for back in backend:
    output += "{backend:<15}\n".format(backend=back)
    output += "{describe:<15}".format(describe='exampe/metric')
    for metric in metrics:
        output += "{metric:<15}".format(metric=metric)
    output += '\n'
    for module in module_name:
        output += "{module:<15}".format(module=module)
        for metric in metrics:
            output += "{metric:<15}".format(metric=result[back][module][metric])
        output += '\n'
    output += '\n'

test_summary.write(output)
print output
