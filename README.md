# dynamic-context

Professional, minimal, and clear README for the Dynamic Context Memory Controller

## Project Overview

dynamic-context is a lightweight memory-controller and routing prototype that demonstrates an agentic workflow combining a "tagger" adapter and a "main" context-manager adapter mounted on a shared Qwen base model. The pipeline performs structured extraction (tagging), retrieval-augmented context lookups, session-context maintenance, and model-driven pruning or multi-hop retrieval requests.

This repository contains two example entry points:

- `router.py` — production-oriented routing pipeline that mounts two PEFT adapters on a single shared base model and runs a Tagger -> Memory -> Main pipeline with hybrid retrieval.
- `debug_models.py` — a quick debug harness for generating raw outputs from each adapter separately.

## Important model repositories (Hugging Face)

The code expects two adapter/LoRA repositories. These are hosted on Hugging Face and used by the examples in this repo:

- itsSHAS/qwen-tagger_model — Tagger adapter (agentic memory / structured extractor)
- itsSHAS/qwen-context-manager-lora — Context-manager adapter (main reasoning / memory controller)

You can find them on Hugging Face at the following URLs:

- https://huggingface.co/itsSHAS/qwen-tagger_model
- https://huggingface.co/itsSHAS/qwen-context-manager-lora

The code in `router.py` expects the adapters to be available locally at the paths configured by `MAIN_LORA_PATH` and `TAGGER_LORA_PATH`, or you may replace those constants with the HF repo IDs to load them directly via the Transformers/PEFT `from_pretrained` APIs.

## Requirements

- Python 3.10+
- PyTorch (matching your CUDA / CPU setup)
- transformers
- peft
- safetensors (if your LoRAs use safetensors)

Example quick install (CPU / minimal):

```bash
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft safetensors
```

For GPU installs, prefer the official PyTorch install selector at https://pytorch.org to get the correct CUDA build.

## Configuration

- `BASE_MODEL_ID` in both scripts points to the Qwen base model used for shared weights.
- `MAIN_LORA_PATH` and `TAGGER_LORA_PATH` are the local paths (or HF repo IDs) where the adapter checkpoints are stored.
- `DB_FILE` in `router.py` defaults to `real_knowledge_base.json`. The file should contain a JSON array of documents with the following (example) schema for each item:

```json
{
  "file": "filename.ext",
  "description": "short description",
  "content": "full text or extracted content...",
  "tags": ["backend", "database"]
}
```

## Quickstart — Running locally

1. Ensure model adapters are available:

   - Option A: Download or `git lfs` clone the HF adapter repos into the paths referenced by `MAIN_LORA_PATH` and `TAGGER_LORA_PATH`.
   - Option B: Keep them on Hugging Face and set the constants to the repo IDs so `PeftModel.from_pretrained` can fetch them.

2. Install Python dependencies (see Requirements above).

3. Run the router pipeline (recommended):

```bash
python router.py
```

This will initialize the tokenizer and a shared base model, mount two adapters (named `main` and `tagger`), load the knowledge base if present, and present an interactive prompt to submit queries.

4. Run the debug harness (lighter; separate processes):

```bash
python debug_models.py
```

This script loads two independent base models and attaches each LoRA separately, providing a REPL to compare raw tagger outputs and main-model outputs.

## How it works (brief)

- Tagger adapter: receives raw user text and MUST output only JSON containing `extracted_query`, `query_tags`, `extracted_data`, and `data_tags`. The tagger is trained/expected to be a passive extractor.
- Memory / Retrieval: `router.py` performs a hybrid tag-and-text search across the local `real_knowledge_base.json` to gather contextual chunks.
- Main adapter: receives an assembled prompt containing the active rolling session context and the (optionally trimmed) user query. The main model may emit special markers `<DROP_CONTEXT>` or `<REQUEST_CONTEXT>` that the pipeline interprets to prune or fetch more context respectively.

## Notes & operational considerations

- The code attempts to detect bf16 support on CUDA devices and will select an appropriate dtype. If GPU loading fails the code retries on CPU.
- `shared_base_model.resize_token_embeddings(151669)` is present in both scripts — ensure that your tokenizer and embeddings match the expected vocab size when using custom tokenizers.
- The repository is a prototype. Carefully validate model sources and adapter compatibility before running in production.

## Troubleshooting

- Out of memory on GPU: try using `device_map` configuration, reduce `max_new_tokens`, or run on CPU.
- JSON decode errors from the tagger: verify the tagger adapter outputs strict JSON and that model temperature is low for deterministic output.

## Contributing

Contributions welcome. Open an issue or PR with a clear description and reproduction steps. If you add tests or CI, include small sample data and instructions to reproduce locally.

## License & Acknowledgements

This repository is a demonstration prototype. Check the licenses of the Qwen base model and the adapter repositories on Hugging Face before commercial use.

---
 
