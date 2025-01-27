# Config Synth Flow
Configurable workflows for synthetic data generation.

<!-- Icon -->
<p align="center">
  <img src="iamges/icon.webp" width="200" />
</p>


## 特點

- 使用 YAML 文件來控制數據處理和生成的流程。
- 支援常用的數據讀取和清理功能，例如：HTML 轉換、去重、數據篩選等。
- 支援 asyncio 和 multiprocessing 以加速處理。

## 安裝

使用 `poetry` 進行安裝：

```bash
poetry install
```

## 執行

使用以下命令來運行配置文件：

```bash
poetry run run-seq <config_file>
```