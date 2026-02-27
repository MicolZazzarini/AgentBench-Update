import asyncio
from typing import Union, List, Dict, Any

from src.typings import (
    TaskOutput,
    AgentOutput,
    ChatHistoryItem,
    SampleIndex,
    TaskSampleExecutionResult,
)


class SessionController:
    """
    Synchronization controller coordinating interaction between
    the agent and the environment.

    This class manages bidirectional communication using asyncio
    locks and semaphores to ensure ordered execution.
    """

    def __init__(self):
        self.agent_lock = asyncio.Lock()
        self.env_lock = asyncio.Lock()
        self.agent_signal = asyncio.Semaphore(0)
        self.env_signal = asyncio.Semaphore(0)
        self.env_input: Union[None, AgentOutput] = None
        self.env_output = TaskOutput()

    async def agent_pull(
        self, env_input: Union[AgentOutput, None] = None
    ) -> TaskOutput:
        """
        Called by the agent to receive environment output.

        Optionally sends agent output back to the environment.

        Args:
            env_input (AgentOutput | None): Agent response.

        Returns:
            TaskOutput: Latest environment output.
        """
        async with self.agent_lock:
            if env_input is not None:
                self.env_input = env_input
                self.env_signal.release()
            print("acquiring agent signal")
            await self.agent_signal.acquire()
            print("pos 5")
            return self.env_output

    async def env_pull(self, history: List[ChatHistoryItem]) -> AgentOutput:
        """
        Called by the environment to request agent action.

        Args:
            history (List[ChatHistoryItem]): Filtered conversation history.

        Returns:
            AgentOutput: Agent response.
        """
        print(">> env pull waiting")
        async with self.env_lock:
            self.env_output.history = history
            self.agent_signal.release()
            await self.env_signal.acquire()
            return self.env_input

    async def env_finish(self, result: TaskOutput = None) -> None:
        """
        Signal session termination from the environment side.

        Args:
            result (TaskOutput): Final task result.
        """
        print(">> env finish waiting")
        async with self.env_lock:
            print(">> env finish done")
            history = self.env_output.history
            self.env_output = result
            if self.env_output.history is None:
                self.env_output.history = history
            self.agent_signal.release()

    def get_status(self):
        """
        Retrieve current synchronization status.

        Returns:
            dict: Debug information about session state.
        """
        waiting_for_env = self.agent_lock.locked()
        waiting_for_agent = self.env_lock.locked()
        return {
            "waiting_for_env": waiting_for_env,
            "waiting_for_agent": waiting_for_agent,
            "env_input": self.env_input,
            "env_output": self.env_output.dict(),
        }


class Session:
    """
    Represents a single interactive session between an agent
    and an execution environment.
    """
    def __init__(self) -> None:
        self.history: List[ChatHistoryItem] = []
        self.controller = SessionController()

    def inject(self, item):
        """
        Inject messages into session history.

        Supports:
        - ChatHistoryItem
        - dict representations
        - lists of items
        """
        if not item:
            return
        if isinstance(item, ChatHistoryItem):
            self.history.append(item)
        elif isinstance(item, Dict):
            self.history.append(ChatHistoryItem.parse_obj(item))
        elif isinstance(item, List):
            for sub_item in item:
                self.inject(sub_item)
        else:
            raise TypeError("Unsupported type %s" % type(item))

    def clear(self):
        self.history = []

    @staticmethod
    def _calc_segments(msg: str):
        """
        Estimate message size using heuristic segmentation.

        Used to approximate context length limits.
        """
        segments = 0
        current_segment = ""
        inside_word = False

        for char in msg:
            if char.isalpha():
                current_segment += char
                if not inside_word:
                    inside_word = True
                if len(current_segment) >= 7:
                    segments += 1
                    current_segment = ""
                    inside_word = False
            else:
                if inside_word:
                    segments += 1
                    current_segment = ""
                    inside_word = False
                if char not in [" ", "\n"]:
                    segments += 1

        if len(current_segment) > 0:
            segments += 1

        return segments

    def filter_messages(self, messages: List[ChatHistoryItem]) -> List[ChatHistoryItem]:
        """
        Trim conversation history to fit context constraints.

        Keeps only the most recent messages within a
        segment threshold.

        Args:
            messages: Full message history.

        Returns:
            List[ChatHistoryItem]: Filtered history.
        """
        assert len(messages) % 2 == 1, "Invalid message length"

        threshold_segments = 3500
        return_messages: List[ChatHistoryItem] = []
        # only include the latest {threshold_segments} segments

        segments = self._calc_segments(messages[0].content)

        for message in messages[:0:-1]:
            segments += self._calc_segments(message.content)
            if segments >= threshold_segments:
                break
            return_messages.append(message)

        if len(return_messages) > 0 and return_messages[-1].role == "user":
            return_messages.pop()

        instruction = messages[0].content

        omit = len(messages) - len(return_messages) - 1

        if omit > 0:
            instruction += f"\n\n[NOTICE] {omit} messages are omitted."
            print(f"Warning: {omit} messages are omitted.")

        return_messages.append(ChatHistoryItem(role="user", content=instruction))

        return_messages.reverse()
        return return_messages

    async def action(self, *injection) -> AgentOutput:
        """
        Execute one interaction step.

        Injects messages, requests agent action,
        and updates session history.

        Returns:
            AgentOutput: Agent response.
        """
        print("session.action")
        self.inject(list(injection))
        print("pulling env")
        agent_response = await self.controller.env_pull(
            self.filter_messages(self.history)
        )
        self.history.append(
            ChatHistoryItem(
                role="agent",
                content=agent_response.content or agent_response.status,
                reasoning_content=agent_response.reasoning_content
            )
        )
        return agent_response


class Task:
    """
    Abstract base class representing a benchmark task.
    """
    def __init__(self, name: str, concurrency: int = 1, *args, **kwargs):
        self.name = name
        self.concurrency = concurrency

    def get_indices(self) -> List[SampleIndex]:
        raise NotImplementedError()

    async def start_sample(
        self, index: SampleIndex, session: Session
    ) -> TaskSampleExecutionResult:
        raise NotImplementedError()

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        raise NotImplementedError()

    def release(self):
        pass


class VirtualTask(Task):
    """
    Example mock task used for testing session execution.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="virtual-task", *args, **kwargs)

    def get_indices(self) -> List[Any]:
        return list(range(10))

    async def start_sample(self, index, session: Session):
        print("task start sample")
        for loop_times in range(3):
            await asyncio.sleep(1)
            res = await session.action(
                {"role": "user", "content": "Loop: %d" % loop_times}
            )
            print("TASK", res)
        return {"succeed": True, "round": 10}

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        return {"score": 0.4}