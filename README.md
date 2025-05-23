# Config Synth Flow

Configurable Workflows for Synthetic Data Generation.

## Simple Usage

### Project Spotlight
Config Synth Flow provides a configurable, flexible pipeline system that allows you to:
- Create modular from data prcossing to data generation pipelines through YAML configuration
- Sytnth data asynchronously with high concurrency.
- Designed for easy extensibility with custom pipeline components.


### Environment Preparation
1. This project uses Poetry for dependency management. Install Poetry first:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

### Running a Simple Example

To run a simple pipeline, use the `run-seq` command:

```bash
run-seq configs/examples/magpie.yml
```

### Writing a Configuration File

ConfigSynthFlow uses YAML configuration files to define the pipeline structure. Here's a simple example:

```yaml
import_path: SequentialExecutor
init_kwargs:
  reader:
    import_path: HfDatasetReader
    init_kwargs:
      debug: true # debug mode
      resume: false # resume from the last processed data
      dataset_kwargs:
        path: json
        num_proc: 10
        data_files:
        - <your_data_path>
  writer:
    import_path: HfWriter # will output in jsonl format
    init_kwargs:
      chunk_size: 5000 # number of data to save in one file
      output_path: result/test # output path
  pipes:
  - import_path: ChatGenerator # chat generator
    init_kwargs:
      prompt_type_col: role # prompt type column
      litellm_kwargs:
        model: gpt-4.1-nano # llm model
  - import_path: RemoveColumns # remove columns
    async_cfg:
      concurrency: 100
      batch_size: 10000
    init_kwargs:
      prompt_type_col: role
      litellm_kwargs:
        model: gpt-4.1-nano
      system_template:
      - name: zhtw # template name
        weight: 1 # sample weight
        template_str: | # template string
          你是一個善於改寫長文本的AI助理，請幫我改寫以下文章，不要有任何多餘的前後綴。你的改寫必須要盡可能涵蓋所有原始文章的重點，且不要省略任何重要的細節。
      user_template:
      - name: zhtw
        template_str: | # jinja template to format the prompt
          # 任務

          - 把文章改寫成 wiki 的文章段落，並去除不重要的部分。

          # 文章

          {{ text }}
      output_col: rephrase # output key to save
```

Configuration files support:
- Pipelines instantiation via `import_path` and `init_kwargs`
- Recursive pipeline definition (pipelines can contain other pipelines)
- Fuzzy matching of pipeline names (`import_path`).
- Environment variable interpolation for sensitive values

you can see `configs/examples/` for more examples.


### Base Classes Overview

#### BasePipeline

The foundational class for all pipelines, providing:
- Configuration loading and parsing
- Logging capabilities

```python
from config_synth_flow.base import BasePipeline, PipelineConfig

# Create from a YAML file
pipeline = BasePipeline.from_yaml("config.yml")

# Create from a config object
config = PipelineConfig(import_path="path.to.Class", init_kwargs={...})
pipeline = BasePipeline(config)
```

#### AsyncBasePipeline

Extends BasePipeline with asynchronous processing capabilities, allowing for:
- Concurrent data processing
- Batch processing with configurable batch sizes
- Asynchronous API calls

#### MultiprocessBasePipeline

Provides parallel processing capabilities using Python's multiprocessing:
- Process data across multiple CPU cores
- Handle CPU-bound tasks efficiently
- Scale processing across available hardware

#### Reader

The base input handler for pipelines:
- Reads data from various sources (files, databases, etc.)
- Provides resumable processing support
- Handles data ID generation and tracking

#### Writer

The base output handler for pipelines:
- Writes processed data to various destinations
- Supports different output formats
- Handles output file management

#### Executor
Coordinates the execution flow between readers and writers:
- Connects readers to writers
- Manages the execution lifecycle
- Handles configuration serialization

### Core Components

#### Reader Components
Readers are responsible for loading data into the pipeline system. Config Synth Flow provides several built-in reader implementations:

- **BaseReader**: Abstract base class for all readers with support for:
  - Resumable processing through unique ID tracking
  - Automatic generation of hash IDs for data objects
  - Integration with writers for saving progress

- **HfDatasetReader**: Loads data from Hugging Face datasets with features like:
  - Support for both Dataset and IterableDataset types
  - Debug mode for quick testing with limited samples
  - Shuffling functionality for randomized data processing
  - Customizable loading through dataset_kwargs

Example reader configuration:
```yaml
reader:
  import_path: config_synth_flow.reader.HfDatasetReader
  init_kwargs:
    dataset_kwargs:
      path: "your_dataset_name"
      split: "train"
    resume: true
    shuffle: true
```

#### Writer Components
Writers handle the output of processed data in various formats. Available writers include:

- **BaseWriter**: Foundational writer class with common output management functions
  - Configurable output paths
  - Basic output handling

- **HfWriter**: Specialized writer for saving data in Hugging Face dataset formats
  - Supports multiple output formats (jsonl, json, csv, parquet)
  - Chunk-based saving for large datasets
  - Automatic chunk naming and management

Example writer configuration:
```yaml
writer:
  import_path: config_synth_flow.writer.HfWriter
  init_kwargs:
    output_path: "path/to/output"
    chunk_size: 1000
    output_format: "jsonl"
```

#### Judge Components
Judge pipelines evaluate and score content generated within the system:

- **SglangRmJudgePipeline**: Evaluates conversations using reward models
  - Integration with SgLang served reward models
  - Support for per-round or full conversation judging
- **OpenaiLmPplPipeline**: Calculates perplexity scores using OpenAI models
- **InfinitySlidingWindowEduClassifier**: Specialized classifier for educational content

#### Paper Implementation Components

The Papers pipelines implement algorithms and techniques from academic papers:

- **[Magpie](https://arxiv.org/abs/2406.08464)**: Implementation of the Magpie approach for instruction tuning data generation
  - Built on `AsyncChatBasePipeline` for efficient processing
- **[ContextualMagpie](https://arxiv.org/abs/2406.08464)**: Implementation of the ContextualMagpie approach for instruction tuning data generation

## Advanced Usage

### Architecture of Base Classes

The pipeline system follows a hierarchical structure:

1. **Core Mixins**: Provide shared functionality like logging, async support, and serialization
2. **Base Classes**: Build on mixins to define core behaviors
3. **Specialized Pipelines**: Implement specific functionality for different use cases

The main execution flow follows the pattern controlled by `Executor`:
```
Reader → [\*Pipes] → Writer
```

Key components communicate through well-defined interfaces, making the system modular and extensible.

### Implementing Custom Components

#### Custom Reader
```python
from config_synth_flow.base.io import BaseReader
from config_synth_flow.base.pipeline import DictsGenerator

class MyCustomReader(BaseReader):
    def post_init(self, input_path: str, resume: bool = False):
        super().post_init(resume=resume)
        self.input_path = input_path
        
    def read(self) -> DictsGenerator:
        # Implement your reading logic here
        for item in my_data_source:
            yield {"data": item}
```

#### Custom Writer
```python
from config_synth_flow.base.io import BaseWriter

class MyCustomWriter(BaseWriter):
    def post_init(self, output_path: str, save_format: str = "jsonl"):
        self.output_path = output_path
        self.save_format = save_format
        
    def write(self, data):
        # Implement your writing logic here
        with open(self.output_path, "a") as f:
            f.write(f"{data}\n")
```

#### Custom Pipeline
```python
from config_synth_flow.base.pipeline import BasePipeline

class MyCustomPipeline(BasePipeline):
    def post_init(self, param1: str, param2: int = 0):
        self.param1 = param1
        self.param2 = param2
      
    def __call__(self, data: DictsGenerator) -> DictsGenerator:
        # Implement your processing logic here
        for item in data:
            yield self.run_each(item)
      
    def run_each(self, data: dict) -> dict:
        # Implement your processing logic here
        return {"processed": data, "param1": self.param1}
```

#### Async Chat Pipeline
```python
from config_synth_flow.base import AsyncChatBasePipeline
from config_synth_flow.base import PromptTemplate

class MyAsyncChatPipeline(AsyncChatBasePipeline):
    def post_init(self, 
                 litellm_kwargs: dict,
                 prompt_template: PromptTemplate):
        super().post_init(litellm_kwargs)
        self.prompt_template = prompt_template
        
    async def run_each(self, data: dict) -> dict:
        messages = [{"role": "user", "content": self.prompt_template.render(**data)}]
        response = await self.chat(messages=messages)
        data["response"] = response.choices[0].message.content
        return data
```

## Citation
```
@software{config_synth_flow,
  author = {aqweteddy},
  title = {ConfigSynthFlow: Configurable Workflows for Synthetic Data Generation.},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/aqweteddy/ConfigSynthFlow}
}
```
