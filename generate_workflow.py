#!/usr/bin/env python

import argparse
from molehill.pipeline import Pipeline

parser = argparse.ArgumentParser(description="Generate ML workflow from yaml file")
parser.add_argument('yaml', metavar='file', type=str,
                    help='yaml file path to convert')

args = parser.parse_args()
print("Start converting: {}".format(args.yaml))
pipe = Pipeline()
pipe.dump_pipeline(args.yaml)
print("Finish dump file: {}".format(pipe.workflow_path))