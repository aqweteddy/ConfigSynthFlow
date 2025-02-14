# Config Synth Flow

Config Synth Flow 是一個可配置的工作流程，用於生成合成數據。

## 特點

- 使用 YAML 文件來控制數據處理和生成的流程。
- 支援常用的文字讀取和清理功能，例如：HTML 轉換、去重、數據篩選等。
- 支援 asyncio 和 multiprocessing 以加速處理。
- 實作以下資料生成論文:
    - Magpie
    - AgentInstruct

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