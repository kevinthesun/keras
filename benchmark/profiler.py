import time

class Timer(object):
    def __init__(self, ret_dict):
        self.__start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.__start
        ret_dict["training_time"] = runtime
