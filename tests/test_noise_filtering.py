"""
Additional tests: verify out-of-domain content (undocumented distractor
files in data/) never leaks into answers about the marketing agency.
Run: pytest tests/test_noise_filtering.py -v
"""

import os

import pytest

from src.knowledge_base import build_knowledge_base
from src.pipeline import ask_question, get_llm

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def vector_store():
    """Build the vector store once for all noise-filtering tests."""
    return build_knowledge_base(DATA_DIR)


@pytest.fixture(scope="module")
def llm():
    """Load the LLM once for all noise-filtering tests."""
    return get_llm()


class TestNoiseFiltering:
    """Verify that the two undocumented distractor files in data/
    (company_handbook.txt — Acme Corp HR policy, product_faq.txt —
    AcmeCloud SaaS product) never leak into pipeline answers."""

    def test_out_of_domain_question_is_refused(
        self, vector_store, llm
    ) -> None:
        """A question that only matches the unrelated 'company_handbook.txt'
        (HR/PTO policy for a different fictional company) must not produce
        an answer sourced from that file."""
        result: dict = ask_question(vector_store, llm, "What is the remote work policy?")
        sources_text: str = " ".join(result["sources"]).lower()
        assert "remote work" not in sources_text or "acme corp employee handbook" not in sources_text

    def test_sources_never_include_handbook_or_product_faq(
        self, vector_store, llm
    ) -> None:
        """No returned source chunk should ever come from the two
        undocumented, out-of-domain distractor files."""
        for q in [
            "What is your PTO policy?",
            "How does file versioning work in AcmeCloud?",
            "How much does the Growth package cost?",
        ]:
            result: dict = ask_question(vector_store, llm, q)
            sources_text: str = " ".join(result["sources"]).lower()
            assert "acmecloud" not in sources_text
            assert "paid time off" not in sources_text

    def test_acme_cloud_pricing_is_refused(
        self, vector_store, llm
    ) -> None:
        """Questions about AcmeCloud's pricing tiers should either be
        refused or answered solely from marketing-agency pricing data."""
        result: dict = ask_question(
            vector_store, llm, "How much does the AcmeCloud Pro plan cost?"
        )
        sources_text: str = " ".join(result["sources"]).lower()
        assert "$12 per user" not in sources_text
        assert "acmecloud" not in sources_text

    def test_empty_input_is_handled(self, vector_store, llm) -> None:
        """Empty or whitespace-only input should return gracefully."""
        for q in ["", "   ", None]:
            result: dict = ask_question(vector_store, llm, q)
            assert isinstance(result["answer"], str)
            assert len(result["answer"].strip()) > 0
            assert result["sources"] == []
