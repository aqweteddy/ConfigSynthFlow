from ...base import BasePipeline, DictsGenerator


class SetExactMatch(BasePipeline):
    def __post_init__(self, text_col: str = 'text'):
        self.text_col = text_col
        
    def __call__(self, dcts: DictsGenerator):
        used = set()
        
        for dct in dcts:
            text = dct[self.text_col]
            if text not in used:
                yield dct
                used.add(text)
lambda x: {
        "text": f"""- 主題: {x['_topic']}\n- 主要實體: {x['_ent_str']}\n- 關鍵字: {x['_kwd_str']}\n\n{x['text'][:10000]}"""
    }