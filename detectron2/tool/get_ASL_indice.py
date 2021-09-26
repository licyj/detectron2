from argparse import ArgumentParser
import os
from pathlib import Path
import argparse




def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument('label_json', type=str, help='path/to/label_json_path')
    return args


def main():
    args = get_args()


if __name__ == '__main__':
    main()