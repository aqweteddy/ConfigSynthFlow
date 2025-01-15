import yaml
from ..executor import SeqentialExecutor

def main(
    cfg_path: str,
):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    executor: SeqentialExecutor = SeqentialExecutor.from_config(cfg)
    executor.chunked_run()
    
    
if __name__ == "__main__":
    from fire import Fire
    Fire(main)