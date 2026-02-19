from enum import IntEnum, Enum


class SampleStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed" # task ends normally
    AGENT_CONTEXT_LIMIT = "agent context limit" # the length of interaction history exceeds the LLM's maximum context length
    AGENT_VALIDATION_FAILED = "agent validation failed" # the agent does not follows the format instruction (?)
    AGENT_INVALID_ACTION = "agent invalid action" # the agent follows the format instruction, but its selected action is invalid
    TASK_LIMIT_REACHED = "task limit reached" # the agent does not solve the problem after reaching the predifined maximum interaction rounds
    UNKNOWN = "unknown"
    TASK_ERROR = "task error"


class WorkerStatus(IntEnum):
    ALIVE = 0
    COMA = 1
    DEAD = 2


class AgentOutputStatus(str, Enum):
    NORMAL = "normal"
    CANCELLED = "cancelled"
    AGENT_CONTEXT_LIMIT = "agent context limit"