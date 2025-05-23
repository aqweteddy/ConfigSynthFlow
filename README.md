# Config Synth Flow

A modular and extensible pipeline framework for processing, synthesizing, and generating configurations with a focus on asynchronous and parallel processing.

## Simple Usage

### Project Spotlight
Config Synth Flow provides a configurable, flexible pipeline system that allows you to:
- Create modular data processing pipelines through YAML configuration
- Process data asynchronously with high concurrency 
- Implement custom readers, writers, and executors
- Generate and process content with various LLM providers through LiteLLM integration
- Support both synchronous and asynchronous workflows

### Environment Preparation
1. This project uses Poetry for dependency management. Install Poetry first:
```bash
pip install poetry
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
run-seq path/to/your/config.yml
```

Or you can run it from the script directly:

```bash
python -m config_synth_flow.scripts.run_seq path/to/your/config.yml
```

### Writing a Configuration File
Config Synth Flow uses YAML configuration files to define the pipeline structure. Here's a simple example:

```yaml
import_path: config_synth_flow.base.executor.SimpleExecutor
init_kwargs:
  reader:
    import_path: config_synth_flow.reader.YourCustomReader
    init_kwargs:
      input_path: "path/to/input/data"
      resume: false
  
  writer:
    import_path: YourCustomWriter
    init_kwargs:
      output_path: "path/to/output"
      save_format: "jsonl"
```

Configuration files support:
- Pipelines instantiation via `import_path` and `init_kwargs`
- Recursive pipeline definition (pipelines can contain other pipelines)
- Fuzzy matching of pipeline names.
- Environment variable interpolation for sensitive values

you can see `example_configs/` for more examples.


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
Reader → [Pipes] → Writer
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
        
    def process(self, data):
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
  author = {The Config Synth Flow Team},
  title = {Config Synth Flow: A Modular Pipeline Framework},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/your-organization/config-synth-flow}
}
```
