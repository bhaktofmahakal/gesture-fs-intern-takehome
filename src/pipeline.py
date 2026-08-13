"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
from typing import Callable

from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> Callable[[str], list[dict[str, str]]]:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt: str) -> list[dict[str, str]]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# Allowlist of legitimate knowledge-base files.  The data/ directory also
# contains company_handbook.txt and product_faq.txt — undocumented files
# describing an unrelated company ("Acme Corp").  Because knowledge_base.py's
# glob loads every *.txt file and cannot be modified, those distractor files
# get embedded into the same FAISS index.  Filtering by source filename here
# prevents out-of-domain content from leaking into answers.
ALLOWED_FILES: set[str] = {"services.txt", "pricing.txt", "faq.txt"}

# Deterministic refusal message — mirrors the prompt template's instruction
# so the response is consistent whether the LLM or the filter catches it.
NO_INFO_MSG: str = "I don't have enough information to answer that."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(
    vector_store: FAISS,
    llm: Callable[[str], list[dict[str, str]]],
    question: str,
) -> dict[str, str | list[str]]:
    """Retrieve relevant chunks, filter out-of-domain noise, and generate an answer.

    Steps:
      1. Validate input — return early on empty/whitespace-only questions.
      2. Retrieve top-5 candidates from the vector store (wider pool to
         compensate for distractor documents that will be filtered out).
      3. Keep only chunks whose source file is in ALLOWED_FILES, then take
         the top 3 unique results.
      4. If no in-domain chunks survive, return a deterministic refusal
         instead of trusting a small local LLM to self-censor.
      5. Build the prompt, call the LLM, and return answer + sources.

    Args:
        vector_store: FAISS vector store from knowledge_base.py.
        llm: Callable from get_llm().
        question: The user's question string.

    Returns:
        dict with keys:
            "answer"  -> str: the generated answer.
            "sources" -> list[str]: the chunk texts that were used.
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    # Retrieve a wider pool than needed because data/ contains two
    # undocumented files describing an unrelated company.  Filter by
    # source file here since knowledge_base.py cannot be modified.
    candidates = vector_store.similarity_search(question, k=5)

    # Deduplicate by exact page_content (chunk_overlap=50 can produce
    # near-identical chunks) and filter to allowed source files only.
    seen: set[str] = set()
    filtered: list = []
    for doc in candidates:
        basename: str = os.path.basename(doc.metadata.get("source", ""))
        if basename in ALLOWED_FILES and doc.page_content not in seen:
            seen.add(doc.page_content)
            filtered.append(doc)
        if len(filtered) == 3:
            break

    if not filtered:
        # Every top match was out-of-domain noise.  A small local LLM
        # (flan-t5-base) cannot reliably refuse on empty/irrelevant context,
        # so return the refusal deterministically.
        return {"answer": NO_INFO_MSG, "sources": []}

    context: str = "\n\n".join(doc.page_content for doc in filtered)
    prompt: str = PROMPT_TEMPLATE.format(context=context, question=question)
    result: list[dict[str, str]] = llm(prompt)
    answer: str = result[0]["generated_text"].strip()

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in filtered],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop with optional --query flag for single-question mode."""
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Marketing agency Q&A assistant",
    )
    parser.add_argument("--query", type=str, help="Ask a single question and exit")
    args: argparse.Namespace = parser.parse_args()

    print("Building knowledge base...")
    vector_store: FAISS = build_knowledge_base(data_dir)
    print("Loading LLM...")
    llm = get_llm()

    if args.query:
        result: dict[str, str | list[str]] = ask_question(vector_store, llm, args.query)
        print("\n📄 Sources:")
        for i, src in enumerate(result["sources"], 1):
            preview: str = src[:200] + "..." if len(src) > 200 else src
            print(f"  {i}. {preview}")
        print(f"\n💬 Answer: {result['answer']}")
        return

    print("\nAsk me anything about our services, pricing, or process.")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            question: str = input("> ").strip()
            if question.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if not question:
                continue

            result = ask_question(vector_store, llm, question)

            print("\n📄 Sources:")
            for i, src in enumerate(result["sources"], 1):
                preview = src[:200] + "..." if len(src) > 200 else src
                print(f"  {i}. {preview}")
            print(f"\n💬 Answer: {result['answer']}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()