import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        'src/k9kmeans/app/dashboard.py',
        '--',
        f'--csv={args.csv}',
    ]

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
