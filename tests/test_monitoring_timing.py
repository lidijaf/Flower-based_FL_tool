import os
import sys
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from monitoring.timing import Timer


def main():
    timer = Timer("sleep_test")

    timer.start()
    time.sleep(1)
    elapsed = timer.stop()

    print("Elapsed:", elapsed)

    with Timer("context_manager") as t:
        time.sleep(0.5)

    print("Context elapsed:", t.elapsed)


if __name__ == "__main__":
    main()
