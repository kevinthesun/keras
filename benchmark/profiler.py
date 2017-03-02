import os
import time
import csv

GPU_NUM = 8

def mem_extract(file_name, ret_dict):
    row_count = 0
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        last_line_broken = False
        for row in csv_reader:
            if row_count == 0:
                row_count += 1
                continue
            if not 'MiB' in row[1]:
                last_line_broken = True
            row_count += 1
        row_count -= 1
        if row_count % GPU_NUM == 0 and last_line_broken:
            row_count -= GPU_NUM
        else:
            row_count -= row_count % GPU_NUM

    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        max_usage = 0
        current_usage = 0
        mem_recoder = [0] * GPU_NUM
        row_num = 0
        for row in csv_reader:
            if row_num == 0:
                row_num += 1
                continue
            mem_str = row[1].lstrip().rstrip()[:-4]
            mem_num = float(mem_str)
            current_usage += mem_num
            mem_recoder[(row_num - 1) % GPU_NUM] += mem_num
            if row_num % GPU_NUM == 0:
                max_usage = max(max_usage, current_usage)
                current_usage = 0
            row_num += 1
            if row_num > row_count:
                break
        row_num -= 1
        ret_dict['max_memory'] = max_usage
        avg_mem = 0
        var_mem = 0
        for num in mem_recoder:
            avg_mem += num/row_num
        avg_mem /= GPU_NUM
        for num in mem_recoder:
            var_mem += (avg_mem - num/row_num) * (avg_mem - num/row_num)
        var_mem /= GPU_NUM
        ret_dict["memory_variance"] = var_mem
    os.remove(file_name)
            
class Timer(object):
    def __init__(self, ret_dict):
        self.__start = time.time()
        self.__ret_dict = ret_dict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.__start
        self.__ret_dict["training_time"] = runtime
