from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from config_synth_flow.base import JudgePipeline


class Validator(BaseModel):
    judge: JudgePipeline | str | Callable[[dict], float] | None = None
    judge_kwargs: dict | None = None
    criteria_lambda: str | Callable[[float], bool] = "lambda x: x"
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def name(self) -> str:
        if isinstance(self.judge, str):
            return self.judge
        elif isinstance(self.judge, JudgePipeline):
            return self.judge.class_name
        elif isinstance(self.judge, Callable):
            return self.judge.__name__
        else:
            return "judge"

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.judge, str):
            self.judge = self.import_judge_func(self.judge)

        if not self.criteria_lambda.startswith("lambda"):
            self.criteria_lambda = "lambda x: " + self.criteria_lambda
        self.criteria_lambda = eval(self.criteria_lambda)
        self.judge_kwargs = self.judge_kwargs or {}

    def import_judge_func(self, judge_func_name: str) -> Callable[[dict], float]:
        try:
            import importlib

            judge_module = importlib.import_module(
                "config_synth_flow.pipelines.papers.magpie.judge_func"
            )
            return getattr(judge_module, judge_func_name)
        except Exception as e:
            try:
                return importlib.import_module(judge_func_name)
            except Exception as e:
                raise ValueError(f"Error importing judge function {judge_func_name}: {e}")

    async def validate(self, dct: dict, save_judge_result: bool = False) -> bool:
        if isinstance(self.judge, JudgePipeline):
            try:
                res = await self.judge.judge(dct, **self.judge_kwargs)
                score = await self.judge.get_score(res)
                if not isinstance(res, str):
                    res = str(res)
            except Exception as e:
                return False, {}
        elif self.judge is not None:
            score = self.judge(dct, **self.judge_kwargs)
            res = None

        dct = {"reason": res, "score": score}

        return (
            self.criteria_lambda(score)
            if not save_judge_result
            else (self.criteria_lambda(score), dct)
        )
