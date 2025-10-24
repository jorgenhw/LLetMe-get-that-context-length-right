<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://i.imgur.com/Ktm9oiA.png">
    <img src="https://i.imgur.com/Ktm9oiA.png" alt="Logo" width=200 height=200>
  </a>
  
  <h1 align="center">LLetMe Get That Context-Length Right</h1> 
  <h2 align="center"><i>Long-Context LLM Evaluation Framework Introducing:</i> <br><br>CA-NiaH: Context-Aware Needle-in-a-Haystack</h2> 
  <h3 align="center">Cognitive Science // Master Thesis 2025</h3>
<br>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project-">About the Project 🔍</a></li>
    <li><a href="#prerequisites-">Prerequisites 📋</a></li>
    <li><a href="#supported-models-">Supported Models 🤖</a></li>
    <li><a href="#logging-information-ℹ️">Logging information ℹ️</a></li>
    <li><a href="#installation-method-1-doing-it-manually-">Installation method 1 (doing it manually) 🚀</a></li>
    <li><a href="#installation-method-2-doing-it-automatically-">Installation method 2 (doing it automatically) 🚀</a></li>
    <li><a href="#task-descriptions-and-metrics-">Task Descriptions and Metrics 📊</a>
      <ul>
        <li><a href="#1-single-needle-task">1. Single Needle Task</a></li>
        <li><a href="#2-multi-needle-task">2. Multi Needle Task</a></li>
        <li><a href="#3-multi-hop-task">3. Multi-Hop Task</a></li>
        <li><a href="#4-aggregation-task">4. Aggregation Task</a></li>
      </ul>
    </li>
    <li><a href="#project-structure-">Project Structure 📁</a></li>
    <li><a href="#results-">Results 📈</a></li>
    <li><a href="#adding-new-models-">Adding New Models 🤖</a></li>
  </ol>
</details>

## About the Project 🔍
This project provides a framework for evaluating Large Language Models' (LLMs) performance on long-context tasks. It implements four distinct evaluation tasks:

1. **Single Needle Task**: Tests the model's ability to find specific information in a document
2. **Multi Needle Task**: Evaluates finding multiple pieces of information across a document
3. **Multi-Hop Task**: Tests logical reasoning across connected statements
4. **Aggregation Task**: Assesses synthesis of independent but related information

Each task generates context-aware content, inserts it into documents, and evaluates model responses using different metrics.

The data used to calibrate the evaluation method is publicly available contracts, which can be downloaded from: https://www.atticusprojectai.org/cuad

This project has been used in a specific business case together with <a href="https://clearcontract.dk/">ClearContract</a> who used it to determine which models solved their <a href="https://clearcontract.dk/">contract review tool</a>.

This repository is part of a master thesis in Cognitive Science.

## Prerequisites 📋
- Python 3.8+
- OpenAI API key (for GPT-4)
- PDF documents for testing (place in `pdf_files` directory)
- Additional API keys for other models (optional)

**NOTE** Please be aware that using OpenAI's models from an API is a paid service - you can find their prices in the documentation at openai.com. 

## Supported Models 🤖
The framework currently supports the following models:
- GPT-4o (enabled by default)
- OpenAI o1 (disabled by default)
- DeepSeek (disabled by default)
- Claude-2 (disabled by default) - was not part of the final test
- Llama-2-70B (disabled by default) (can be accessed via huggingface API)
- Llama-2-8B (disabled by default) (can be accessed via huggingface API)

## Logging information ℹ️
Debugging is added to all of the scripts, allowing you to follow the process of your evaluation from your terminal.

## Installation method 1 (doing it manually) 🚀

1. Clone the repository:
```bash
git clone [your-repo-url]
cd long-context-llm-evaluation
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your model API keys:
Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here  # Optional
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
LLAMA_API_KEY=your_llama_key_here  # Optional
```

4. Configure models in `config.yaml`:
The `config.yaml` file controls which models are enabled and their settings. By default, only GPT-4 is enabled. To enable other models:
```yaml
models:
  gpt-4o:
    enabled: true
    # ... other settings ...
  
  deepseek:
    enabled: true  # Change to true to enable
    # ... other settings ...
```

## Installation method 2 (doing it automatically) 🚀

To make it easier to run, we have included a bash script that automatically:

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

However, before running the bash script, you still need to:

1. Place your PDF files in the `pdf_files` directory
2. Create a `.env` file with at least the OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

Thereafter, you can type the following in your terminal:

```bash
bash run.sh
```

## Task Descriptions and Metrics 📊

#### 1. Single Needle Task
- Inserts one piece of information into the document
- Generates a relevant query
- Metric: Binary accuracy (correct/incorrect)
- Results in: `results/[model_name]/single_needle_results.csv`

#### 2. Multi Needle Task
- Inserts 2-5 pieces of information
- Generates queries for each needle
- Metric: Accuracy = N_correct / N_total
- Results in: `results/[model_name]/multi_needle_results.csv`

#### 3. Multi-Hop Task
- Creates 2-4 connected logical statements
- Tests reasoning across statements
- Metrics: 
  - Success rate by hop count: Success(h) = Correct_h / Total_h
  - Hop decay: -(Success(max) - Success(1)) / (max - 1)
- Results in: `results/[model_name]/multi_hop_results.csv`

#### 4. Aggregation Task
- Inserts 2-3 related but independent statements
- Tests information synthesis
- Metric: Completeness = |Required ∩ Found| / |Required|
- Results in: `results/[model_name]/aggregation_results.csv`

## Project Structure 📁
```
.
├── main.py                 # Main execution script
├── requirements.txt        # Project dependencies
├── config.yaml            # Model and evaluation configuration
├── pdf_files/             # Directory for PDF documents
├── results/               # Evaluation results by model
└── src/
    ├── models/           # Model implementations
    │   ├── __init__.py
    │   ├── model_interface.py
    │   ├── model_registry.py
    │   ├── gpt4.py
    │   └── ...
    ├── tasks/            # Task implementations
    │   ├── single_needle_task.py
    │   ├── multi_needle_task.py
    │   ├── multi_hop_task.py
    │   └── aggregation_task.py
    ├── document_processor.py
    ├── evaluation.py
    ├── model_inference.py
    ├── pipeline.py
    ├── visualization.py
    └── data_structures.py
```

## Results 📈
Results for each model are saved in separate directories under `results/`:

A comparison report across all enabled models is generated in `results/comparison_report.csv`.

## Adding New Models 🔧
To add a new model:

1. Add configuration to `config.yaml`:
```yaml
models:
  your_model:
    enabled: false
    api_key: ${YOUR_API_KEY}
    model_name: "your-model-name"
    max_tokens: 4096
    temperature: 0.7
```

2. Create a new model implementation in `src/models/`:
```python
from .model_interface import ModelInterface

class YourModel(ModelInterface):
    def setup(self) -> None:
        # Initialize your model
        pass
    
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        # Implement response generation
        pass
    
    def get_context_window(self) -> int:
        # Return context window size
        pass
```

3. Add the model to the registry in `src/models/model_registry.py`:
```python
MODEL_IMPLEMENTATIONS = {
    "your_model": YourModel,
    # ... other models ...
}
```

## Contributing 🤝
Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄
This repository is licensed under the [MIT License](https://mit-license.org/).
It is free for everyone to use, modify, and share. If you find it useful, I'd appreciate a citation or a mention—thanks!
