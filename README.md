# LLM-RE-Benchmark: Can LLMs identify funktional requirements information?

A benchmark for evaluating LLMs regarding their ability to recognize and extract functional requirement information in natural language texts.


## 📁 Repository Content

| files                                    | Description                                                                                             |
| -----------------------------------------|---------------------------------------------------------------------------------------------------------|
| `BenchmarkRequirements.json`             | Dataset on which the LLMs run. Contains text excerpts with gold standard answers and metadata.          |
| `DataLoader.py`                          | Loads the text excerpts and reference solutions from BenchmarkRequirements.json.                        |
| `Evaluate.py`                            | Evaluates LLMs on the dataset. This file must be executed to run LLMs through the benchmark. The results of the evaluation are stored in a csv file. Furthermore, functions from MonitorResults.py are executed to analyze the results.           |
| `EvaluateJSON.py`                        | Analyzes the data set stored in the json file. The result can be used for further analysis purposes.    |
| `Metrics.py`                             | Contains functions with metrics.                                                                        |
| `Modelwrapper.py`                        | Ensures that prompts are forwarded to the configured LLMs. In our version, various interfaces are addressed that forward to models located on different hardware. The following Python files are controlled: GPTModelsAPI.py, OllamaClient.py, OllamaModels.py. The last two files are not located in this repository, as they address locally operated LLMs. Both Python files only contain functions that pass the transferred prompt to a specific LLM and return the response in the form of an unmodified string. They must be replaced with own files as needed.                        |
| `MonitorResults.py`                      | This file contains functions for analyzing the results, producing tabular output via console, and generating various diagrams. This file can also be started directly. In the main function, a csv file containing the results of past experiments can be referenced. Analyses are then performed based on the data.                      |
| `Prompts.py`                             | Contains the prompt.                        |
| `GPTModelsAPI.py`                        | Forwards the prompt to a gpt model via OpenAI API and returns the response as an unmodified string.     |
| `text_analysis_results.csv`              | Contains the results for each analyzed text excerpt during an experiment.                               |
| `cumulativ_results_per_textsegment.csv`  | Contains the results for each analyzed text excerpt from all past experiments and continuously stores all data. Must be reset manually if necessary.  |


## ⚙️ Installation

- Python ≥ 3.12
- LLMs run on different hardware (Access must be set up individually)
