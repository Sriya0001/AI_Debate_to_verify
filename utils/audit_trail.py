from datetime import datetime
from typing import List


def format_audit_log(agent_name: str, action_message: str) -> str:
    """
    Formats a single standardized audit trail log line with local timestamp.
    Example: [10:32:04] Claim Analyzer created C1
    """
    ts = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] {agent_name}: {action_message}"


class AuditTrail:
    """
    Session audit trail accumulator.
    """
    def __init__(self):
        self.logs: List[str] = []

    def add(self, agent_name: str, message: str) -> str:
        entry = format_audit_log(agent_name, message)
        self.logs.append(entry)
        return entry

    def get_logs(self) -> List[str]:
        return list(self.logs)

    def clear(self):
        self.logs.clear()


audit_trail = AuditTrail()
