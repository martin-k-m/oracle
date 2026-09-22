# Oracle: a small language model woven in Twill.
#
# These targets are thin convenience over the twill commands; nothing here does
# anything you cannot do by calling `twill` yourself. Run them from the repo root.

TWILL ?= twill

.PHONY: help train generate check fmt corpus

help:
	@echo "make train              train the model and write models/oracle.bin"
	@echo "make generate           sample from the checkpoint (default prompt)"
	@echo "make generate PROMPT=.. sample from the checkpoint with your prompt"
	@echo "make check              static shape-check every .tw file"
	@echo "make corpus             regenerate data/corpus.txt from the source"

train:
	$(TWILL) run train.tw

PROMPT ?= The
generate:
	$(TWILL) run generate.tw "$(PROMPT)"

check:
	$(TWILL) check src/model.tw src/tokenizer.tw train.tw generate.tw

fmt:
	$(TWILL) fmt src/model.tw src/tokenizer.tw train.tw generate.tw

corpus:
	./scripts/fetch_corpus.sh
