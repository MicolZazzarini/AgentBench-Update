import contextlib
import time
import warnings
import requests
from urllib3.exceptions import InsecureRequestWarning
from src.typings import *
from src.utils import *
from ..agent import AgentClient

old_merge_environment_settings = requests.Session.merge_environment_settings

# ============================================================
# Context Manager: no_ssl_verification
# ------------------------------------------------------------
# Temporarily disables SSL verification for requests.
# Useful when connecting to local endpoints or self-signed
# certificates.
#
# Features:
# - Overrides merge_environment_settings to force verify=False
# - Suppresses InsecureRequestWarning
# - Cleans up adapters after use
# ============================================================
@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings["verify"] = False
        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings
        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

# ============================================================
# Prompter class
# ------------------------------------------------------------
# Responsible for formatting messages before sending them
# to the LLM.
# Provides:
# - Default prompt formatting
# - Role-based message dictionaries (user/agent)
# ============================================================
class Prompter:
    @staticmethod
    def get_prompter(prompter: Union[Dict[str, Any], None]):
        if not prompter:
            return Prompter.default()
        assert isinstance(prompter, dict)
        prompter_name = prompter.get("name", None)
        prompter_args = prompter.get("args", {})
        if hasattr(Prompter, prompter_name) and callable(
            getattr(Prompter, prompter_name)
        ):
            return getattr(Prompter, prompter_name)(**prompter_args)
        return Prompter.default()

    @staticmethod
    def default():
        return Prompter.role_content_dict()

    @staticmethod
    def role_content_dict(
        message_key: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
        user_role: str = "user",
        agent_role: str = "assistant",
    ):  
        """
        Formats messages into a role/content dictionary structure
        compatible with chat-based LLM APIs.
        """
        def prompter(messages: List[Dict[str, str]]):
            role_dict = {
                "user": user_role,
                "agent": agent_role,
            }
            prompt = []
            for item in messages:
                prompt.append(
                    {role_key: role_dict[item["role"]], content_key: item["content"]}
                )
            return {message_key: prompt}

        return prompter

# ============================================================
# Context limit checker
# ------------------------------------------------------------
# Detects whether a response indicates that the model
# exceeded its context or token limit.
# ============================================================
def check_context_limit(content: str):
    content = content.lower()
    and_words = [
        ["prompt", "context", "tokens"],
        ["limit", "exceed", "max", "long", "much", "many", "reach", "over", "up", "beyond"],
    ]
    rule = AndRule(
        [
            OrRule([ContainRule(word) for word in and_words[i]])
            for i in range(len(and_words))
        ]
    )
    return rule.check(content)

# ============================================================
# HTTPAgent class
# ------------------------------------------------------------
# Client for interacting with LLMs via HTTP endpoints.
# Supports:
# - OpenAI-compatible chat/completions
# - Ollama native /api/chat endpoints
# - Automatic prompter handling and history formatting
# - SSL verification bypass for local endpoints
# - Context-limit detection
# - Reasoning content extraction
# ============================================================
class HTTPAgent(AgentClient):
    def __init__(
        self,
        url,
        proxies=None,
        body=None,
        headers=None,
        return_format="{response}",
        prompter=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.proxies = proxies or {}
        self.headers = headers or {}
        self.body = body or {}
        self.return_format = return_format
        self.prompter = Prompter.get_prompter(prompter)

        if not self.url:
            raise Exception("Please set 'url' parameter")
    
    # Handle conversation history
    def _handle_history(self, history: List[dict]) -> Dict[str, Any]:
        return self.prompter(history)
    
    # Main inference function
    def inference(self, history: List[dict]) -> Dict[str, str]:
        """
        Sends the conversation history to the LLM endpoint and
        returns the model's response along with any reasoning trace.
        Retries up to 3 times on failure.
        """
        for attempt in range(3):
            try:
                body = self.body.copy()
                body.update(self._handle_history(history))

                with no_ssl_verification():
                    resp = requests.post(
                        self.url,
                        json=body,
                        headers=self.headers,
                        proxies=self.proxies,
                        timeout=120,
                    )

                if resp.status_code != 200:
                    if check_context_limit(resp.text):
                        raise AgentContextLimitException(resp.text)
                    else:
                        raise Exception(
                            f"Invalid status code {resp.status_code}:\n\n{resp.text}"
                        )

            except AgentClientException:
                raise
            except Exception as e:
                print("Warning:", e)
                time.sleep(attempt + 2)
                continue
            
            # Parse JSON response
            resp = resp.json()

            # ---------------------------------------------------
            # CASE 1: OpenAI-compatible format
            # ---------------------------------------------------
            if isinstance(resp, dict) and "choices" in resp and len(resp["choices"]) > 0:
                message = resp["choices"][0].get("message", {})
                content = message.get("content", "")
                reasoning = (
                    message.get("thinking")
                    or message.get("reasoning_content")
                    or ""
                )

                # Fallback extraction from <think> tags
                if not reasoning and "<think>" in content:
                    import re
                    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                    if match:
                        reasoning = match.group(1).strip()

                return {
                    "content": content,
                    "reasoning_content": reasoning,
                }

            # ---------------------------------------------------
            # CASE 2: Ollama native /api/chat format
            # ---------------------------------------------------
            if isinstance(resp, dict) and "message" in resp:
                message = resp.get("message", {})
                content = message.get("content", "")
                reasoning = message.get("thinking", "")

                return {
                    "content": content,
                    "reasoning_content": reasoning,
                }

            # ---------------------------------------------------
            # Fallback (non standard API)
            # ---------------------------------------------------
            return {
                "content": self.return_format.format(response=resp),
                "reasoning_content": "",
            }

        raise Exception("Failed after 3 attempts.")