import asyncio
import glob
import json
import os
import re
import socket
import struct
import time
from typing import List, Dict, Any, Tuple
import docker
import docker.models.containers

from src.server.task import Task, Session
from src.typings import (
    AgentOutputStatus,
    TaskOutput,
    TaskSampleExecutionResult,
    SampleStatus,
)


class Container:
    """
    Persistent interactive Docker container wrapper.

    This class creates a long-lived container running an interactive
    bash session and exposes a low-level socket interface allowing
    incremental command execution.

    The container behaves like a real OS terminal used by the agent
    during task interaction.
    """

    def __init__(self, image):
        """
        Start a detached container and initialize a persistent
        interactive bash execution session.
        """
        self.image = image
        self.client = docker.from_env()
        self.container: docker.models.containers.Container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
            labels={"created_by": "os-pipeline"},
        )

        # Create persistent bash exec session
        self.exec_id = self.client.api.exec_create(
            self.container.id, "bash --login", stdin=True, tty=True
        )["Id"]

        # Open socket connected to bash session
        sock_obj = self.client.api.exec_start(self.exec_id, socket=True)

        # Cross-platform compatibility (Linux / Windows)
        if hasattr(sock_obj, "_sock"):
            # Linux
            self.sock = sock_obj._sock
        else:
            # Windows NpipeSocket
            self.sock = sock_obj

        self.sock.settimeout(5)

        # clear buffer
        try:
            self.sock.recv(1000)
        except:
            pass

    def __del__(self):
        """Ensure container termination on destruction."""
        try:
            self.container.stop()
        except:
            pass

    def recv(self, n=4096):
        """Receive raw bytes from container socket."""
        return self.sock.recv(n)

    def send(self, data):
        """Send raw bytes to container socket."""
        self.sock.send(data)

    def execute(self, command: str):
        """
        Execute a command inside the persistent bash session.

        Returns:
            DummyOutput:
                output (bytes): command stdout/stderr
                exit_code (int)
        """
        class DummyOutput:
            output: bytes
            exit_code: int

            def __init__(self, code, o):
                self.output = o
                self.exit_code = code

        if not isinstance(command, str):
            return DummyOutput(-1, b"")
        
        # Send command to interactive shell
        self.send(command.encode("utf-8") + b"\n") 
        # ignore input line
        try:
            data = self.recv(8)
            _, n = struct.unpack(">BxxxL", data)
            _ = self.recv(n)
        except:
            pass

        time_limit = 30  # seconds
        start_time = time.time()
        output = b""

        # Read shell output until prompt reappears
        while True: 
            if time.time() - start_time > time_limit:
                break
            try:
                data = self.recv(8)
                if not data:
                    break
                _, n = struct.unpack(">BxxxL", data)
                line = self.recv(n)
                output += line
                if re.search(b"\x1b.+@.+[#|$] ", line):
                    break
            except (TimeoutError, socket.timeout):
                break

        # Remove ANSI escape sequences
        output = re.sub(b"\x1b.+@.+[#|$] ", b'', output)
        output = re.sub(b'\x1b\\[[0-9;]*[a-zA-Z]', b'', output)
        output = re.sub(b'\x1b\\][0-9]*;[^\x07]*\x07', b'', output)
        output = re.sub(b'\x1b\[\?2004[hl]', b'', output)
        output = re.sub(b'\x07', b'', output)

        return DummyOutput(0, output)

    def execute_independent(self, command, *params):
        language, command = command
        if language == "bash":
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python3", "-c", command, *params]
        elif language in ["c++", "c"]:
            compiler = "g++" if language == "c++" else "gcc"
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && {compiler} -o /tmp/a.out /tmp/main.cpp',
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError("Unsupported language")
        return self.container.exec_run(cmd)



class JudgeConfig:
    """
    Configuration describing a single OS interaction task.

    Defines:
    - container image
    - initialization scripts
    - evaluation strategy
    - validation rules
    """
    image: str = None
    init_script: List[Tuple[str, str]] = None
    start: Tuple[str, str] = None
    description: str
    check: list = None
    match: dict = None
    example_script: str = None
    
    def get_evaluation_type(self):
        """Return evaluation mode: 'check' or 'match'."""
        if self.check:
            return "check"
        elif self.match:
            return "match"

    def get_evaluation_content(self):
        """Return evaluation configuration."""
        return self.check or self.match

# few shot example
ONE_SHOT = [
    {"role": "user", "content": 'tell me how many files are in the directory "/etc"?'},
    {
        "role": "agent",
        "content": """Think: To count the files in /etc, I need to print all the files in it.

Act: bash

```bash
ls /etc
```""",
    },
    {
        "role": "user",
        "content": "The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols "
                   "selinux tmpfiles.d [truncated because the output is too long]",
    },
    {
        "role": "agent",
        "content": """Think: The output has been truncated because it is too long, so I need to count files by script directly.

Act: bash

```bash
ls -1 /etc | wc -l
```""",
    },
    {"role": "user", "content": "The output of the OS:\n220"},
    {
        "role": "agent",
        "content": """Think: Now I get the answer, it is 220.

Act: answer(220)""",
    },
]

# task 
class OSInteraction(Task):
    
    def _load_configs(self, config_path, script_root_dir=".") -> List[JudgeConfig]:
        """
        Load task definitions from JSON or JSONL configuration files.

        The method:
            1. Parses configuration files
            2. Normalizes script definitions
            3. Converts entries into JudgeConfig objects

        Returns:
            List[JudgeConfig]
        """
        def load_script(script_obj):
            """
            Normalize script specification.

            Supported formats:
                - None
                - inline string (defaults to bash)
                - {language, code}
                - {language, file}

            Returns:
                Tuple[str, str]: (language, script_content)
            """
            if script_obj is None:
                return None
            if type(script_obj) is str:
                return "bash", script_obj
            if "language" not in script_obj:
                language = "bash"
            else:
                language = script_obj["language"]
            if "file" in script_obj:
                with open(
                    os.path.join(script_root_dir, script_obj["file"]), encoding="utf-8"
                ) as f:
                    return language, f.read()
            elif "code" in script_obj:
                return language, script_obj["code"]
            else:
                raise ValueError("Invalid Script Object")

        # 1. handle input file:
        if config_path.endswith(".json"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = json.load(f)
            if isinstance(config_raw, list):
                pass
            elif isinstance(config_raw, dict):
                config_raw = [config_raw] 
            else:
                raise ValueError("Invalid Config File")
        elif config_path.endswith(".jsonl"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = [json.loads(line) for line in f.readlines()] 
        else:
            raise ValueError("Invalid Config File")

        # 2. handle configs
        configs: list[JudgeConfig] = []
        for item in config_raw:
            config = JudgeConfig()
            config.description = item["description"] # description of the task
            if "create" in item: 
                config.image = ( 
                    item["create"]["image"]
                    if ("image" in item["create"])
                    else (self.docker_config["localhost"] + "/default")
                )
                if "init" in item["create"]:
                    if type(item["create"]["init"]) is not list:
                        config.init_script = [load_script(item["create"]["init"])]
                    else:
                        config.init_script = [
                            load_script(script_obj)
                            for script_obj in item["create"]["init"]
                        ]
                else:
                    config.init_script = []
            else:
                config.image = self.docker_config["localhost"] + "/default" 
            if "start" in item:
                config.start = load_script(item["start"]) 
            evaluation = item["evaluation"] 
            if "match" in evaluation:
                if type(evaluation["match"]) is str: # case 1: match
                    config.match = {"answer": evaluation["match"], "strip": True}
                else:
                    config.match = evaluation["match"]
            elif "check" in evaluation:
                if type(evaluation["check"]) is not list: # case 2: check (verification through script)
                    config.check = [load_script(evaluation["check"])]
                else:
                    config.check = [
                        load_script(script_obj) for script_obj in evaluation["check"]
                    ]
            else:
                raise ValueError("check or match must exist.")
            if "check" in evaluation and "example" in evaluation:
                config.example_script = load_script(evaluation["example"]) 
            configs.append(config)
        return configs # ritorna la lista

    def __init__(self, data_config, docker_config, round_limit=8, **kwargs):
        """
        Initialize the OSInteraction benchmark.

        This constructor prepares the full registry of executable problems.

        Workflow:
            1. Load global benchmark configuration.
            2. Expand problem file wildcards using glob patterns.
            3. Load all JSON / JSONL problem definitions.
            4. Assign a unique index to each problem instance.
            5. Build the internal problem configuration mapping.

        The resulting `self.problem_configs` dictionary maps a unique
        sample index to its corresponding execution configuration.
        """
        super().__init__(**kwargs)
        self.round_limit: int = round_limit
        self.data_config = data_config
        self.docker_config = docker_config
        self.problem_configs: Dict[str, Dict[str, Any]] = {}  # {index: CONFIG}

        matches = []
        for item in self.data_config["files"]:
            path = item["problem_file"]
            for file in glob.glob(path):
                if file.endswith(".json") or file.endswith(".jsonl"):
                    matches.append(
                        {
                            "problem_file": file,
                            "script_dir": item["script_dir"],
                            "index_prefix": item["index_prefix"]
                            + os.path.basename(file)
                            .removesuffix(".json")
                            .removesuffix(".jsonl")
                            + "-",
                        }
                    )
        self.data_config["files"] = matches

        for item in self.data_config["files"]:
            problem_file = item["problem_file"]
            single_file_configs = self._load_configs(problem_file, item["script_dir"])
            dict_configs = {}
            for idx, config in enumerate(single_file_configs):
                dict_configs[item["index_prefix"] + "%05d" % idx] = {
                    "file": problem_file,
                    "config": config,
                    "index": idx,
                }
            self.problem_configs.update(dict_configs)

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        """
        Compute aggregated evaluation metrics across multiple executed tasks.
        """
        overall = {
            "total": len([config for config in results if config]),
            "pass": len(
                [
                    config
                    for config in results
                    if (config and config.result and config.result.get("result", False))
                ]
            ),
        }
        overall["wrong"] = overall["total"] - overall["pass"]
        overall["acc"] = overall["pass"] / overall["total"] if overall["total"] else 0
        return {
            "overall": overall,
        }

    def get_indices(self) -> List[Any]:
        return list(self.problem_configs.keys())

    def extract_action(self, raw: str):
        """
        This is the parser that transforms the agent's textual output into an action
        executable by the system. Given the full agent output (raw), the function:
        1) Extracts the "Think" section
        2) Extracts the last valid action (bash, finish, answer)
        3) Extracts any associated content (bash code or final answer)
        4) Returns a structured dictionary
        """
        think_pattern = r"Think:\s*(.+)"
        act_pattern = r"Act:\s*(.+)"

        think = re.findall(think_pattern, raw)
        act = re.findall(act_pattern, raw)

        print("EXTRACT_ACTION RAW:")
        print(raw)
        print("FOUND THINK:", think)
        print("FOUND ACT:", act)

        ret = {"thought": "\n".join(think), "action": None, "content": None}

        # reversly iterate over the action list
        for action in act[::-1]:
            if action.lower().startswith("bash"):
                ret["action"] = "bash"
                break
            if action.lower().startswith("finish"):
                ret["action"] = "commit"
                break
            if action.lower().startswith("answer"):
                content = action[6:].strip()
                left_par_pos = content.find("(")
                right_par_pos = content.rfind(")")
                if left_par_pos == -1 or right_par_pos == -1:
                    continue
                content = content[left_par_pos + 1: right_par_pos]
                ret["action"] = "commit"
                ret["content"] = content
                break

        if ret["action"] == "bash":
            # extract from ```bash to ```
            content_pattern = r"```bash\n(.*?)\n```"
            content = re.findall(content_pattern, raw, re.DOTALL)
            content = "\n\n".join(content)
            ret["content"] = content
        
        """
        Example output (ret):

        Bash case:

        {
        "thought": "I need to count the files",
        "action": "bash",
        "content": "ls -1 /etc | wc -l"
        }

        Answer case:

        {
        "thought": "Now I have the result",
        "action": "commit",
        "content": "220"
        }

        POTENTIAL ISSUES:
        - Fragile regex: does not capture multiline output
        - The bash regex is strict, requires an exact format, and fails if the agent writes bash differently
        - The answer parsing is fragile
        This is important because in the _judge_ method we do root = self.extract_action(root.content),
        and if parsing fails, the task fails.
        """
        return ret
    

    # Possible outcomes for task execution:
    #
    # 1) status = COMPLETED:
    #       - result = True  -> The task was completed successfully, and the agent's answer is correct
    #       - result = False -> The task was completed (the agent produced a final output),
    #                          but the answer is incorrect compared to the ground truth
    #
    # 2) status = AGENT_CONTEXT_LIMIT, result = False -> The agent reached the context limit
    #                                                     (max tokens/messages) and couldn't complete the task
    #
    # 3) status = TASK_LIMIT_REACHED, result = False -> The agent exceeded the maximum allowed rounds
    #                                                   without producing a final answer
    #
    # 4) status = AGENT_VALIDATION_FAILED, result = False -> The agent did not produce a valid action
    #
    # 5) status = AGENT_INVALID_ACTION, result = False -> The agent produced an unexpected action
    #
    # 6) status = UNKNOWN, result = False -> Generic error during execution
    async def start_sample(self, index, session: Session) -> TaskSampleExecutionResult:
        """
        Entry point for executing a single task/sample.
        
        Parameters:
        - index: unique key for the task
        - session: object managing the conversation with the agent
        
        Returns:
        - TaskSampleExecutionResult containing the status and result (dictionary with details)
        """
        data_item = self.problem_configs[index]
        config = data_item["config"]
        file = data_item["file"]
        index_in_file = data_item["index"]
        try:
            print("init container")
            container = Container(config.image)
            print("init container ok")
            print("start judge")
            result = await self._judge(session, config, container) 
            result.result["file"] = file
            result.result["index_in_file"] = index_in_file
            print("finish judge")
            return result
        except Exception as _:
            print("err")
            import traceback
            return TaskSampleExecutionResult(
                status=SampleStatus.UNKNOWN,
                result={"result": False, "error": traceback.format_exc()},
            )
        finally:
            try:
                container.__del__()
            except:
                pass

    async def _judge(
        self, session: Session, config: JudgeConfig, container: Container
    ) -> TaskSampleExecutionResult:
        """
        Core logic for executing and evaluating a single task.
        
        Steps:
        1) Prepare the OS environment
        2) Guide the agent through the task
        3) Execute commands in the container
        4) Validate the agent's response
        5) Determine if the task was completed successfully
        """
        print("exec start")
        if config.init_script:
            for script in config.init_script:
                init = await asyncio.to_thread(container.execute_independent, script)
                if init.exit_code != 0:
                    return TaskSampleExecutionResult(
                        status=SampleStatus.UNKNOWN, 
                        result={"result": False, "error": f'Init script {script} failed: {init}'}
                    )
        if config.start:
            start = await asyncio.to_thread(container.execute, config.start[1])
            if start.exit_code != 0:
                return TaskSampleExecutionResult(
                    status=SampleStatus.UNKNOWN, 
                    result={"result": False, "error": f'Start script {config.start} failed: {start}'}
                )
        print("exec start ok")
        
        """
        System prompt injection. The agent must:
        1) Write a "Think" section
        2) Choose exactly one action per turn: "bash", "finish", or "answer"
        
        Controlled ReAct framework
        """
        oneshot = True
        session.inject(
            {
                "role": "user",
                "content": """You are an assistant that will act like a person, I'will play the role of linux(ubuntu) operating system. Your goal is to implement the operations required by me or answer to the question proposed by me. For each of your turn, you should first think what you should do, and then take exact one of the three actions: "bash", "finish" or "answer". 

1. If you think you should execute some bash code, take bash action, and you should print like this:

Think: put your thought here.

Act: bash

```bash
# put your bash code here
```

2. If you think you have finished the task, take finish action, and you should print like this:

Think: put your thought here.

Act: finish

3. If you think you have got the answer to the question, take answer action, and you should print like this:

Think: put your thought here.

Act: answer(Your answer to the question should be put in this pair of parentheses)

If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself. Attention, your bash code should not contain any input operation. Once again, you should take only exact one of the three actions in each turn.\n\n""",
            }
        )

        if not oneshot:
            session.history[-1].content += (
                "Now, my problem is:\n\n" + config.description
            )
        else:
            session.history[-1].content += (
                "Now, my problem is:\n\n" + ONE_SHOT[0]["content"]
            )
            for item in ONE_SHOT[1:]:
                session.inject(item) 
            session.inject(
                {
                    "role": "user",
                    "content": "Now, I will start a new problem in a new OS. My problem is:\n\n"
                    + config.description,
                }
            )
        
        # Agent-OS interaction loop (default round limit = 8)
        for _ in range(self.round_limit):
            root = await session.action() # call agent

            print("\n=== RAW AGENT OUTPUT ===")
            print(root.content)
            print("========================\n")
            
            print("\n=== REASONING CONTENT ===")
            print(root.reasoning_content)
            print("========================\n")
            
            # Handle agent exceeding context limit
            if root.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT: 
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_CONTEXT_LIMIT, result={"result": False}
                )
            # Any non-normal status triggers a generic error
            if root.status != AgentOutputStatus.NORMAL: 
                return TaskSampleExecutionResult(
                    status=SampleStatus.UNKNOWN, result={"result": False}
                )
            
            # Parse the agent output into a structured action
            root = self.extract_action(root.content) 

            print("\n=== PARSED ACTION ===")
            print(root)
            print("=====================\n")

            # Ensure parsed action contains valid keys
            if "action" not in root:
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_VALIDATION_FAILED,
                    result={"result": False},
                )
            # Ensure the action is valid
            if root["action"] not in ["bash", "commit"]:
                print("INVALID ACTION DETECTED")
                print("Action value:", root["action"])
                print("Full parsed root:", root)
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_INVALID_ACTION, result={"result": False}
                )

            action = root["action"]
            content = root["content"]
            if action == "commit": 
                answer = content
                break
            elif action == "bash": 
                result = await asyncio.to_thread(container.execute, content)
                result = result.output.decode("utf-8")
                if len(result) > 800:
                    result = (
                        result[:780] + "\n[truncated because the output is too long]"
                    )
                session.inject( 
                    {
                        "role": "user",
                        "content": ("The output of the OS:\n\n" + result)
                        if result
                        else "The output of the OS is empty.",
                    }
                )
        else: 
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_LIMIT_REACHED,
                result={"result": False, "reason": "round limit"},
            )

        if isinstance(answer, str) and config.match and config.match["strip"]:
            answer = answer.strip() 
        
        # EVALUATION
        jd = False
        
        # Case 1: matching (exact string comparison)
        if config.match:
            if "answer" in config.match:
                jd = answer == config.match["answer"]
            elif "regex" in config.match:
                jd = re.search(config.match["regex"], answer) is not None
        # Case 2: validation script(s)
        elif config.check:
            params = [str(answer)]
            for script in config.check:
                if script is None:
                    script = config.example_script
                response = await asyncio.to_thread(
                    container.execute_independent, script, *params
                )
                if response.exit_code != 0:
                    jd = False
                    break
                params.append(response.output.decode("utf-8"))
            else:
                jd = True
        else:
            return TaskSampleExecutionResult(
                status=SampleStatus.UNKNOWN, result={"result": False}
            )
        
        # If evaluation succeeds (regardless of answer correctness), mark as COMPLETED
        return TaskSampleExecutionResult(
            status=SampleStatus.COMPLETED, result={"result": jd}
        )
