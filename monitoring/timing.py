import time


class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer was not started")

        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
