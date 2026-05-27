# promptbio-bench

This repository contains the evaluation framework for [PromptBio-bench](https://doi.org/10.64898/2026.05.05.723092). Given a task and an agent's output directory, it runs a 4-step LLM-assisted pipeline to score how well the agent's output matches the reference answer.

## Installation

### Prerequisites

- Conda (Miniconda or Miniforge3)

### 1. Create the environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate eval
```

### 3. Set your API key

Create a `.env` file at the project root:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

Or copy and edit the example manually:

```
OPENAI_API_KEY=sk-...
```

The key is loaded automatically by `utils/agents.py` via `python-dotenv`.


## Evaluating a single task

```bash
python run_eval.py \
    --task-dir   <path/to/task> \
    --result-dir <path/to/agent/output> \
    --output-dir <path/to/save/results> \
    --label      <agent_name> \
    --model      gpt-5.4
```

**Arguments**

| Argument | Required | Description |
|---|---|---|
| `--task-dir` | yes | Task directory containing `task.json`, `eval.json`, and reference answer files |
| `--result-dir` | yes | Directory with the agent's output files to evaluate |
| `--output-dir` | yes | Directory where results and logs are written |
| `--label` | no | Name tag for output files (default: `agent`) |
| `--model` | no | LLM used for all pipeline steps (default: `gpt-5.4`) |

**Output files** (written to `--output-dir`)

| File | Description |
|---|---|
| `<task_id>_<label>.log` | Full pipeline log |
| `<task_id>_<label>.json` | Per-file similarity scores and status |
| `<task_id>_<label>_full.json` | Full intermediate state for debugging |

## Task directory structure

Tasks can be downloaded from the Hugging Face dataset repository `promptbio-bench-data`: https://huggingface.co/datasets/promptbio-ai/promptbio-bench-data). 

Each task is stored in a directory named by its task ID. The directory specified by --task-dir must contain the following files:

```
<task-dir>/
├── task.json        # Task definition (given to the agent)
├── eval.json        # Evaluation spec (used by run_eval.py)
├── ref_answer/      # Reference output files the agent should reproduce
│   └── A375_WGS_WMG.cnv.vcf.gz.tbi
├── ref_script/      # Reference solution scripts (not used by the evaluator)
│   └── work.sh
└── data/            # Input files provided to the agent (optional)
    └── A375_WGS_WMG.cnv.vcf.gz
```

**`task.json`** — describes the task given to the agent:

```json
{
    "id": "a-1-2",
    "question": "Please generate a tabix index for the provided VCF.gz file.",
    "input_files": ["data/A375_WGS_WMG.cnv.vcf.gz"],
    "expected_output": [
        {"file": "A375_WGS_WMG.cnv.vcf.gz.tbi", "type": "tbi", "description": ""}
    ],
    "timeout_seconds": 3600
}
```

**`eval.json`** — describes how to score the agent's output:

```json
{
    "id": "a-1-2",
    "question": "Please generate a tabix index for the provided VCF.gz file.",
    "ref_answer": ["ref_answer/A375_WGS_WMG.cnv.vcf.gz.tbi"],
    "ref_script": ["ref_script/work.sh"],
    "scoring": {
        "expected_output": [
            {
                "file": "A375_WGS_WMG.cnv.vcf.gz.tbi",
                "guidelines": "The file must be a valid tabix index for the input VCF.gz file; internal byte layout may differ."
            }
        ]
    }
}
```

`ref_answer` lists the reference files (relative to `--task-dir`) that the agent's outputs will be matched against. The optional `guidelines` field gives the LLM hints about how strictly to score each file.

## Pipeline

Each evaluation runs four steps:

1. **Match** — LLM maps the agent's output files to the reference files
2. **Detect** — identifies the format of each agent output file (FASTA, BAM, table, etc.)
3. **Recommend** — LLM picks the best comparison strategy and parameters for each file
4. **Compare** — computes a similarity score (0–1) for each file pair

The final score is the average similarity across all reference files.

## Project structure

```
run_eval.py          # Main evaluation entrypoint
utils/               # Core pipeline modules (matching, format detection, comparison)
tools/               # File-format-specific comparison tools
```
