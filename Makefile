# Oracle: a small language model woven in Twill.
#
# These targets are thin convenience over the twill commands; nothing here does
# anything you cannot do by calling `twill` yourself. Run them from the repo root.

TWILL ?= twill

.PHONY: help train generate check fmt corpus quantize bench

help:
	@echo "make train              train the model and write models/oracle.bin"
	@echo "make generate           sample from the checkpoint (default prompt)"
	@echo "make generate PROMPT=.. sample from the checkpoint with your prompt"
	@echo "make generate MODEL=..  sample from a specific checkpoint (e.g. int8)"
	@echo "make quantize           pack models/oracle.bin to models/oracle-int8.bin"
	@echo "make bench              measure size, speed, memory, and quality"
	@echo "make bench MATMUL=fast  same, with the fast matmul microkernels"
	@echo "make check              static shape-check every .tw file"
	@echo "make corpus             regenerate data/corpus.txt from the source"

TW_FILES = src/model.tw src/tokenizer.tw src/quant.tw train.tw generate.tw quantize.tw bench.tw

train:
	$(TWILL) run train.tw

PROMPT ?= The
MODEL ?= models/oracle.bin
generate:
	$(TWILL) run generate.tw "$(PROMPT)" "$(MODEL)"

quantize:
	$(TWILL) run quantize.tw

# MATMUL=fast selects the hand-written matmul microkernels for this run.
MATMUL ?=
bench:
	TWILL_MATMUL=$(MATMUL) $(TWILL) run bench.tw

check:
	$(TWILL) check $(TW_FILES)

fmt:
	$(TWILL) fmt $(TW_FILES)

corpus:
	./scripts/fetch_corpus.sh
