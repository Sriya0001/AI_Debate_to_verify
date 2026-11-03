import time
from typing import Dict, Any


class MetricsTracker:
    """
    Centralized execution and cost metrics tracker.
    Records LLM calls, search queries, execution time, and estimated token usage.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.total_llm_calls = 0
        self.total_searches = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.agent_latencies: Dict[str, float] = {}

    def record_llm_call(self, agent_name: str, latency: float, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.total_llm_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.agent_latencies[agent_name] = self.agent_latencies.get(agent_name, 0.0) + latency

    def record_search(self):
        self.total_searches += 1

    def get_summary(self) -> Dict[str, Any]:
        total_time = round(time.time() - self.start_time, 2)
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        # Rough estimated cost based on blended pricing (~$0.50 / million tokens for small, $3.00 for large)
        estimated_cost_usd = round((self.total_prompt_tokens * 1.0 + self.total_completion_tokens * 3.0) / 1_000_000, 4)

        return {
            "execution_time_seconds": total_time,
            "total_llm_calls": self.total_llm_calls,
            "total_searches": self.total_searches,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "agent_latencies": {k: round(v, 2) for k, v in self.agent_latencies.items()},
        }


metrics = MetricsTracker()
