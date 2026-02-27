from typing import List


class AgentClient:
    """
    Base interface for agent clients.

    This class defines the standard structure that all agent
    implementations must follow.
    """
    def __init__(self, *args, **kwargs):
        pass

    def inference(self, history: List[dict]) -> str:
        raise NotImplementedError()
