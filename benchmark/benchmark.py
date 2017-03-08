import os
import sys
import copy
import importlib

back = os.environ['KERAS_BACKEND']
GPU_NUM = int(os.environ['GPU_NUM'])
metrics = ["training_time", "max_memory", "memory_variance", "training_accuracy", "test_accuracy"]
module_name = ["mnist_acgan_gpu"]#["mnist_hierarchical_rnn_gpu", "addition_rnn_gpu", "babi_rnn_gpu"]
result = dict()


def run_benchmark():
    result[back] = dict()
    test_summary = open('test_summary_' + str(back) +
                        str(GPU_NUM) + '.txt', 'w')
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
    test_summary.close()
    print output

if __name__ == '__main__':
    run_benchmark()
