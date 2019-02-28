#!/usr/bin/env python

import argparse
from molehill.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate ML workflow from yaml file")
    parser.add_argument('yaml', metavar='file', type=str,
                        help='yaml file path to convert')
    parser.add_argument('--dest', type=str,
                        help='output file name')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing file/dir')

    args = parser.parse_args()

    print(f"Start converting: {args.yaml}")
    pipe = Pipeline()
    pipe.dump_pipeline(args.yaml, args.dest, overwrite=args.overwrite)
    print(f"Finish dump file: {pipe.workflow_path}")


if __name__ == '__main__':
    main()
