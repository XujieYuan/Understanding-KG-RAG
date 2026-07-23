# Understanding-KG-RAG

![mindmap and pipeline](https://raw.githubusercontent.com/XujieYuan/Understanding-KG-RAG/main/figs/mindmap_and_pipeline.png)

This repository is the cleaned pilot release for the experiments described in `pilot_with_appendix.pdf`.
It is intentionally scoped to the pilot KG-RAG study only. It does not include the code layout, experiment surface, or claims for `MetaKGRAG.pdf`.

## What Is Included

- A single public entrypoint: `python run.py --config <config.yaml>`
- The pilot pipeline implementation in `KG-RAG/Pilot.py`
- Dataset-specific prompt templates in `prompts.py`
- Pre-processing scripts for building the Neo4j graph and entity embeddings
- Optional evaluation scripts under `evaluation/`

## What Is Not Included

- MetaKGRAG-specific code paths
- Historical experiment outputs
- Generated embedding files such as `entity_embeddings.pkl`
- Local cache files such as `.DS_Store` and `__pycache__`

## Repository Layout

```text
.
├── KG-RAG/
│   └── Pilot.py
├── configs/
│   ├── cmcqa.yaml
│   ├── explainpe.yaml
│   ├── commonsenseqa.yaml
│   ├── genmedgpt.yaml
│   └── cmb_exam.yaml
├── data/
│   ├── cmcqa/
│   ├── explainpe/
│   ├── commonsenseqa/
│   ├── GenMedGPT-5K/
│   └── CMB-Exam/
├── evaluation/
├── pre-processing/
│   ├── KG_Build.py
│   └── encode_entities.py
├── prompts.py
├── requirements.txt
└── run.py
```

## Environment

- Python 3.10 or newer is recommended
- A running Neo4j instance reachable through Bolt
- An OpenAI-compatible chat endpoint for the generation model
- Enough disk space for downloaded Transformer checkpoints
- A fresh virtual environment is strongly recommended to avoid dependency conflicts

Install the core runtime dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

This repository distinguishes between two kinds of files:

### Files already included

- Dataset question-answer files such as `data/cmcqa/cmcqa.json`
- Knowledge graph triples such as `data/cmcqa/knowledge_graph.txt`

### Files you must generate

- `entity_embeddings.pkl`

Generate entity embeddings from a dataset's `knowledge_graph.txt`:

```bash
python pre-processing/encode_entities.py \
  --kg data/cmcqa/knowledge_graph.txt \
  --output data/cmcqa/entity_embeddings.pkl
```

Load the same triples into Neo4j:

```bash
python pre-processing/KG_Build.py \
  --uri bolt://localhost:7687 \
  --username neo4j \
  --password <your-neo4j-password> \
  --file data/cmcqa/knowledge_graph.txt
```

## Configuration

All public runs go through `run.py` with a YAML config that follows this fixed structure:

```yaml
dataset: cmcqa

neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "replace-me"

llm:
  api_base: "http://localhost:8000/v1"
  api_key: "replace-me"

bert:
  model_path: "bert-base-chinese"

embedding:
  model_path: "distiluse-base-multilingual-cased-v1"

data:
  input: "data/cmcqa/cmcqa.json"
  entity_embeddings: "data/cmcqa/entity_embeddings.pkl"
  output: "output/cmcqa/results.csv"
```

`data.input` and `data.entity_embeddings` are resolved relative to the repository root unless you pass absolute paths.

## Quickstart

The default walkthrough uses `cmcqa`, because it is the most complete example in this release.

1. Install dependencies with `pip install -r requirements.txt`.
2. Start Neo4j and note the Bolt URI, username, and password.
3. Generate `data/cmcqa/entity_embeddings.pkl`.
4. Import `data/cmcqa/knowledge_graph.txt` into Neo4j.
5. Edit `configs/cmcqa.yaml` with your Neo4j credentials and LLM endpoint.
6. Run the pilot pipeline:

```bash
python run.py --config configs/cmcqa.yaml
```

The results CSV will be written to `output/cmcqa/results.csv`.

## Supported Datasets In This Release

- `cmcqa`
- `explainpe`
- `commonsenseqa`
- `genmedgpt`
- `cmb_exam`

For `cmb_exam`, the example config points to the `Medical_Practitioner` subset that is already present in this repository.

## Evaluation Scripts

Evaluation scripts are kept as optional utilities and are not required for reproducing the core pilot pipeline.

- `evaluation/BERTScore/localBertscore.py`
- `evaluation/ROUGE/ROUGE_en.py`
- `evaluation/ROUGE/ROUGE_zh.py`
- `evaluation/G-Eval/G_Eval_en.py`
- `evaluation/G-Eval/G_Eval_zh.py`

Some evaluation scripts expect local model files or `llama_cpp_python`. Treat them as post-processing utilities rather than part of the default quickstart.

## Common Issues

### `FileNotFoundError` for `entity_embeddings.pkl`

You need to generate embeddings first with `pre-processing/encode_entities.py`.

### Neo4j connection errors

Check `neo4j.uri`, `neo4j.username`, and `neo4j.password` in your config, and confirm that the database is reachable over Bolt.

### Model loading is slow on first run

`transformers` and `sentence-transformers` may download checkpoints the first time you run the pipeline.

### The output file is overwritten

The pilot runner recreates the target CSV at the start of each run. Change `data.output` if you want a separate file.

## Citation

If you use this repository, cite the corresponding pilot paper from your thesis materials.
