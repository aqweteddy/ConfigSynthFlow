from argparse import ArgumentParser

from ..executor import BaseExecutor


def main():
    parser = ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()

    executor: BaseExecutor = BaseExecutor.from_yaml(args.cfg_path)
    executor.execute()
