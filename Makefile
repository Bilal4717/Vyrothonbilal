# Pocket-Agent Makefile
# Reproducible end-to-end pipeline: install → data → train → quantize → eval → demo
#
# Usage:
#   make all          # full pipeline (train + quantize + eval)
#   make install      # install Python dependencies
#   make data         # generate training data
#   make train        # fine-tune LoRA adapter
#   make quantize     # merge + export Q4_K_M GGUF
#   make eval         # run evaluation on public_test.jsonl
#   make smoke        # 5-example smoke test (fast)
#   make demo         # launch Gradio chatbot
#   make clean        # remove generated artifacts

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL    := google/gemma-3-270m-it
ADAPTER_DIR   := artifacts/lora_adapter
GGUF_PATH     := artifacts/model_q4km.gguf
TRAIN_DATA    := data/train.jsonl
CHECKPOINT    := checkpoints
EPOCHS        := 5
LR            := 2e-4
QUANT_TYPE    := Q4_K_M
LLAMA_CPP_DIR := /tmp/llama.cpp

# ── Targets ───────────────────────────────────────────────────────────────────

.PHONY: all install data train quantize eval smoke demo clean

all: install data train quantize eval
	@echo ""
	@echo "✅  Pipeline complete. Run 'make demo' to launch the chatbot."

install:
	@echo "── Installing dependencies ──────────────────────────────────────────"
	pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install -q transformers==4.51.3 datasets accelerate evaluate trl peft \
	               bitsandbytes sentencepiece protobuf seaborn matplotlib
	pip install -q "llama-cpp-python[cuda]" || pip install -q llama-cpp-python
	pip install -q gradio gguf
	@echo "Dependencies installed."

data:
	@echo "── Generating training data ─────────────────────────────────────────"
	mkdir -p data
	python scripts/generate_data.py --out $(TRAIN_DATA) --seed 42
	@echo "Training data: $(TRAIN_DATA)"

train: $(TRAIN_DATA)
	@echo "── Fine-tuning LoRA adapter ─────────────────────────────────────────"
	mkdir -p $(ADAPTER_DIR) $(CHECKPOINT) artifacts
	python scripts/train.py \
	    --data       $(TRAIN_DATA) \
	    --base_model $(BASE_MODEL) \
	    --adapter_dir $(ADAPTER_DIR) \
	    --checkpoint_dir $(CHECKPOINT) \
	    --epochs     $(EPOCHS) \
	    --lr         $(LR)
	@echo "Adapter saved to: $(ADAPTER_DIR)"

quantize: $(ADAPTER_DIR)
	@echo "── Merging + quantizing to GGUF ────────────────────────────────────"
	python scripts/quantize.py \
	    --base_model  $(BASE_MODEL) \
	    --adapter_dir $(ADAPTER_DIR) \
	    --out         $(GGUF_PATH) \
	    --quant       $(QUANT_TYPE) \
	    --llama_cpp   $(LLAMA_CPP_DIR)
	@echo "GGUF: $(GGUF_PATH)"

eval: $(GGUF_PATH)
	@echo "── Running evaluation ───────────────────────────────────────────────"
	python scripts/evaluate.py --test starter/public_test.jsonl --out artifacts/eval_results.json
	@echo "Results: artifacts/eval_results.json"

smoke:
	@echo "── Smoke test (5 examples) ──────────────────────────────────────────"
	python scripts/evaluate.py --smoke

demo: $(GGUF_PATH)
	@echo "── Launching Gradio demo ────────────────────────────────────────────"
	python demo.py --share

clean:
	@echo "── Cleaning artifacts ───────────────────────────────────────────────"
	rm -rf data/ checkpoints/ artifacts/ results.json
	@echo "Clean complete (source files and starter/ left intact)."
