import time
class SimpleProfile:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t1 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        print(self.name, ':', time.perf_counter()-self.t1)
