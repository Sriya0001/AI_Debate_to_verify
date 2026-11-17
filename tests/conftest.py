import os
import pytest
from config import config

@pytest.fixture(autouse=True)
def mock_environment_for_tests(monkeypatch):
    """
    Ensures unit test suite runs deterministically offline using mock LLM and search fallbacks,
    avoiding network latency or API rate limits during automated testing.
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    config.llm_provider = "mock"
    yield
