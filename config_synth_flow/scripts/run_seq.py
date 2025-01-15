import yaml
from argparse import ArgumentParser

from ..executor import SequentialExecutor

def main():
    parser = ArgumentParser()
    parser.add_argument("-cfg_path", type=str, required=True)
    args = parser.parse_args()
    
    executor: SequentialExecutor = SequentialExecutor.from_yaml(args.cfg_path)
    executor.chunked_run()
    
