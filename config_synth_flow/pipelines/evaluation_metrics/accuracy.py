from collections import defaultdict
from ...base import BasePipeline, DictsGenerator


class AccuracyMetric(BasePipeline):
    def __post_init__(self, 
                      pred_lambda_col: str = None,
                      ground_lambda_col: str = None,
                      group_by_col: str = None,
    ):
        self.pred_lambda_col = eval(pred_lambda_col)
        self.ground_lambda_col = eval(ground_lambda_col)
        self.group_by_col = group_by_col
    
        self.grp_dct = defaultdict(list)
        self.total_cnt = 0
        self.correct_cnt = 0
        
    def __call__(self, dcts: DictsGenerator) -> DictsGenerator:
        
        for dct in dcts:
            is_correct = self.pred_lambda_col(dct) == self.ground_lambda_col(dct)
            self.grp_dct[dct[self.group_by_col]].append(is_correct)
            self.total_cnt += 1
            self.correct_cnt += is_correct
            yield dct
    
    def __del__(self):
        
        self.logger.info(f"Total count: {self.total_cnt}")
        self.logger.info(f"Correct count: {self.correct_cnt}")
        self.logger.info(f"Accuracy: {self.correct_cnt/self.total_cnt}")
        res_str = ''
        for k, v in self.grp_dct.items():
            total_cnt = len(v)
            correct_cnt = sum(v)
            res_str += f"Group {k} accuracy: {correct_cnt/total_cnt:.4f}\n"
        self.logger.info(res_str)