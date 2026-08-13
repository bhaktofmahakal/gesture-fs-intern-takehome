# Implementation Notes

## Distractor File Handling

The `data/` directory contains two files — `company_handbook.txt` and `product_faq.txt` — that are not listed in the README's documented project structure. They describe an entirely unrelated company ("Acme Corp") with HR policies (remote work, PTO, expenses) and a cloud storage product ("AcmeCloud"). Because `knowledge_base.py`'s `DirectoryLoader` uses a `**/*.txt` glob and is explicitly off-limits to modify, these distractor files are loaded and embedded into the same FAISS index as the legitimate marketing-agency content.

The filtering approach in `ask_question()` uses a source-file allowlist (`ALLOWED_FILES = {"services.txt", "pricing.txt", "faq.txt"}`), applied post-retrieval: we fetch `k=5` candidates from the vector store, then keep only chunks whose `metadata["source"]` basename matches the allowlist, taking the top 3 unique results. When no in-domain chunks survive filtering, the function returns a deterministic refusal (`"I don't have enough information to answer that."`) rather than passing empty or irrelevant context to flan-t5-base and hoping a small local model self-censors correctly.

A pure similarity-score threshold would be insufficient here because the distractor content is topically coherent, well-written text — a question like "What is your PTO policy?" genuinely matches the Acme Corp handbook at a semantic level, and the embedding model correctly identifies it as relevant. The issue is not retrieval confidence; it is retrieval from the wrong domain entirely. Only source-file filtering catches this reliably. The automated proof is in `tests/test_noise_filtering.py`, which includes an adversarial PTO-policy question that verifies no Acme Corp content leaks into the response.

## Bonus Items Completed

1. **Error handling**: Empty/whitespace input returns `"Please enter a question."` with empty sources; `KeyboardInterrupt` exits gracefully.
2. **`--query` CLI flag**: `python -m src.pipeline --query "question"` runs single-question mode and exits.
3. **Additional test cases**: `tests/test_noise_filtering.py` with 4 tests covering out-of-domain refusal, source provenance, AcmeCloud pricing isolation, and empty input.
4. **Type hints**: Full type annotations on all functions and key variables throughout `pipeline.py`.
