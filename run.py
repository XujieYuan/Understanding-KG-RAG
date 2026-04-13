import subprocess
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Pilot KG-RAG pipeline runner")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    pilot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'KG-RAG', 'Pilot.py')

    subprocess.run([sys.executable, pilot_path, '--config', config_path], check=True)


if __name__ == '__main__':
    main()
