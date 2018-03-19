#! /usr/bin/env python3
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path='./.env', verbose=True)

WSI_LIST_PATH = Path(os.getenv('WSI_LIST'))
OUTPUT_DIR_PATH = Path(os.getenv('PATCH_OUTPUT_DIR')) / 'orig'

NEGATIVE_OUTPUT_DIR_PATH = OUTPUT_DIR_PATH / 'negative'
POSITIVE_OUTPUT_DIR_PATH = OUTPUT_DIR_PATH / 'positive'


def main():
    print('Hello')


if __name__ == '__main__':
    main()
