import subprocess
import sys


def main():
    command = [sys.executable, "-m", "streamlit", "run", "demo_streamlit.py", *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
