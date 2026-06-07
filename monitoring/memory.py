import os
import psutil


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class MemoryTracker:
    def __init__(self, name=None):
        self.name = name
        self.start_memory_mb = None
        self.end_memory_mb = None
        self.delta_memory_mb = None
        self.peak_memory_mb = None

    def start(self):
        memory = get_memory_usage_mb()
        self.start_memory_mb = memory
        self.peak_memory_mb = memory

    def update_peak(self):
        current = get_memory_usage_mb()
        if self.peak_memory_mb is None or current > self.peak_memory_mb:
            self.peak_memory_mb = current
        return current

    def stop(self):
        self.end_memory_mb = get_memory_usage_mb()
        self.delta_memory_mb = self.end_memory_mb - self.start_memory_mb

        if self.peak_memory_mb is None or self.end_memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = self.end_memory_mb

        return {
            "name": self.name,
            "start_memory_mb": float(self.start_memory_mb),
            "end_memory_mb": float(self.end_memory_mb),
            "delta_memory_mb": float(self.delta_memory_mb),
            "peak_memory_mb": float(self.peak_memory_mb),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
