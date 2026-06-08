import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("run_inference.py exists:", os.path.exists("run_inference.py"))

    result = subprocess.run(
        ["python3", "-m", "py_compile", "run_inference.py"],
        capture_output=True,
        text=True,
    )

    print("return code:", result.returncode)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    assert result.returncode == 0


if __name__ == "__main__":
    main()
