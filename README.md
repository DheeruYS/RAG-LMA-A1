# RAG: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

All the code is in **`rag_rerank_gdpr.ipynb`**, which is committed with its outputs.

* **Document:** GDPR, Regulation (EU) 2016/679, the official EUR-Lex PDF (`data/gdpr.pdf`, 88 pages)
* **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2`, with an exact cosine (inner-product) vector store
* **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **LLM:** `Qwen/Qwen2.5-3B-Instruct`, run locally in fp16 with greedy decoding

## Notebook contents
1. PDF cleaning (font-size based footnote and header removal) and structure-aware chunking (430 chunks)
2. Pipeline 1: bi-encoder Top-10, then Top-3 to the LLM (Answer 1)
3. Pipeline 2: cross-encoder re-ranks the same Top-10, then Top-3 to the LLM with the same prompt (Answer 2)
4. Stress tests: keyword trap, exception, negation / role limit, buried chunk, plus two counter-examples
5. Hit@3 and MRR on 20 labelled queries
6. Latency profiling (200 runs) and a cross-encoder scaling experiment

The CSV and JSON outputs are saved in `results/`.

## Running it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook rag_rerank_gdpr.ipynb
```
The LLM weights download from Hugging Face on first use, about 6 GB. A GPU with 8 GB of memory is enough.
