import importlib
from typing import Union, Generator

from .base_pipeline import BasePipeline


class LambdaFunc(BasePipeline):
    def post_init(
        self,
        lambda_func: str,
    ):
        """
        Initialize the lambda function.
        There are two ways to use the lambda function:
            - If the lambda function returns a dict, update the input dict with the returned dict.
            - If the lambda function returns a bool, yield the input dict if the returned value is True.
        if the lambda function doesn't start with "lambda", add "lambda x:" to the beginning of the function.

        """
        if not lambda_func.startswith("lambda"):
            lambda_func = "lambda x: " + lambda_func
        self.lambda_func = eval(lambda_func)

    def run_each(self, dct: dict) -> Union[dict, Generator[dict, None, None]]:
        res = self.lambda_func(dct)
        if isinstance(res, dict):
            for k, v in res.items():
                dct[k] = v
            return dct
        elif isinstance(res, list):
            return res
        elif isinstance(res, bool):
            if res:
                return dct
        else:
            raise ValueError("lambda function must return either a dict or a bool")


class CfgFunc(BasePipeline):
    def post_init(
        self,
        func: str,
    ):
        namespace = {}
        exec(func, namespace)
        self.func = namespace["func"]

    def run_each(self, dct: dict) -> Union[dict, Generator[dict, None, None]]:
        res = self.func(dct)
        if isinstance(res, dict):
            for k, v in res.items():
                dct[k] = v
            return dct
        elif isinstance(res, bool):
            if res:
                return dct
        elif isinstance(res, list):
            return res
        else:
            raise ValueError("CfgFunc function must return either a dict, bool, or list")
