import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from monitoring.memory import get_memory_usage_mb, MemoryTracker


def main():
    print("Current memory MB:", get_memory_usage_mb())

    with MemoryTracker("allocation_test") as tracker:
        data = [0] * 1_000_000
        tracker.update_peak()

    result = tracker.stop()
    print("Memory tracking result:")
    print(result)


if __name__ == "__main__":
    main()
