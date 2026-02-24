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
    def __init__(self, image):
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

        # Creazione dell'esecuzione bash persistente (sessione bash interattiva dentro il container)
        self.exec_id = self.client.api.exec_create(
            self.container.id, "bash --login", stdin=True, tty=True
        )["Id"]

        # Avvio dell'esecuzione con socket 
        sock_obj = self.client.api.exec_start(self.exec_id, socket=True)

        # Compatibilità Linux vs Windows
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
        try:
            self.container.stop()
        except:
            pass

    # Wrapper recv per uniformare Linux/Windows
    def recv(self, n=4096):
        return self.sock.recv(n)

    # Wrapper send per uniformare Linux/Windows
    def send(self, data):
        self.sock.send(data)

    def execute(self, command: str):
        class DummyOutput:
            output: bytes
            exit_code: int

            def __init__(self, code, o):
                self.output = o
                self.exit_code = code

        if not isinstance(command, str):
            return DummyOutput(-1, b"")

        self.send(command.encode("utf-8") + b"\n") # invia comando a bash
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
        while True: # loop di lettura output
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

        # Pulizia output da escape sequence
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
    image: str = None
    init_script: List[Tuple[str, str]] = None
    start: Tuple[str, str] = None
    description: str
    check: list = None
    match: dict = None
    example_script: str = None
    
    # Due modalità di valutazione: confronto diretto risposta o script di verifica
    def get_evaluation_type(self):
        if self.check:
            return "check"
        elif self.match:
            return "match"

    def get_evaluation_content(self):
        return self.check or self.match

# esempio few shot per formato azioni
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

# task vero e proprio
class OSInteraction(Task):
    
    def _load_configs(self, config_path, script_root_dir=".") -> List[JudgeConfig]:
        """
        Questo metodo:
        1) legge un file .json o .jsonl
        2) converte il contenuto in oggetti JudgeConfig
        3) Restituisce una lista di configurazioni di problemi
        Quindi, trasforma un file di configurazione in oggetti python pronti per essere usati dal giudice
        """
        def load_script(script_obj):
            """
            Questa funzione serve a normalizzare uno script. Uno script JSON può essere scritto in 3 modi:
            - se non esiste -> restituisce None
            - se lo script è una stringa, di default il lingaggio è bash
            - se lo script è un dizionario si determina in linguaggio (di default bash)
            - se script è in un file, lo carica da script root dir e restituisce (language, contenuto file)
            - se lo scirpt è inline rstituisce il codice direttamente
            - se non è valido errore
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
        # Fase 1-caricamento file JSON
        if config_path.endswith(".json"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = json.load(f)
                """
                Il file può essere:
                - Lista di problemi 
                - Singolo problema
                """
            if isinstance(config_raw, list):
                pass
            elif isinstance(config_raw, dict):
                config_raw = [config_raw] # se è un singolo dict -> lo trasforma in lista
            else:
                raise ValueError("Invalid Config File")
        elif config_path.endswith(".jsonl"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = [json.loads(line) for line in f.readlines()] # se è un jsonl un json per riga
        else:
            raise ValueError("Invalid Config File")

        # 2. handle configs
        # fase 2-Creazione JudgeConfig
        configs: list[JudgeConfig] = []
        for item in config_raw:
            config = JudgeConfig()
            config.description = item["description"] # ogni problema ha una descrizione
            if "create" in item: # se c'è un "create", definisce container e init
                config.image = ( # s enon specificata, usa immagine di default
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
                config.image = self.docker_config["localhost"] + "/default" # se create non esiste, usa immagine di default
            if "start" in item:
                config.start = load_script(item["start"]) # script eseguito prima che inizi l'interazione
            evaluation = item["evaluation"] # ogni problema ha una sezione evaluation
            if "match" in evaluation:
                if type(evaluation["match"]) is str: # caso 1: match
                    config.match = {"answer": evaluation["match"], "strip": True}
                else:
                    config.match = evaluation["match"]
            elif "check" in evaluation:
                if type(evaluation["check"]) is not list: # caso 2: check (qui la risposta viene verificata tramite script)
                    config.check = [load_script(evaluation["check"])]
                else:
                    config.check = [
                        load_script(script_obj) for script_obj in evaluation["check"]
                    ]
            else:
                raise ValueError("check or match must exist.")
            if "check" in evaluation and "example" in evaluation:
                config.example_script = load_script(evaluation["example"]) # serve come script di esempio (fallback)
            configs.append(config)
        return configs # ritorna la lista

    def __init__(self, data_config, docker_config, round_limit=8, **kwargs):
        """
        Qui viene costruita tutta la mappa dei problemi che il sistema può eseguire:
        1) Riceve configurazioni generali
        2) espande wildcard dei file problema (usando glob)
        3) carica ogni file .json/.jsonl
        4) genera un indice univoco per ogni problema
        5) costruisce self.problem_config
        Quindi prepara tutti i problemi del benchmark
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
        Questo metodo calcola le metriche aggregate finali dopo aver eseguito più task (statistiche globali)"""
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
        E' il parser che trasforma l'output testuale dell'agente in un'azione eseguibile dal sistema.
        Dato l'output completo dell'agente (raw), la funzione:
        1) Estrae il "Think"
        2) Estrae l'ultima azione valida (bash, finish, answer)
        3) Estrae eventuale contenuto (codice bash o risposta finale)
        4) restituisce un dizionario strutturato
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
        esempio output (ret):
        
        caso bash
        
        {
        "thought": "devo contare i file",
        "action": "bash",
        "content": "ls -1 /etc | wc -l"
        }

        caso answer

        {
        "thought": "ora ho il risultato",
        "action": "commit",
        "content": "220"
        }

        PROBLEMI POTENZIALI:
        - Regex fragile: non cattura multilinea
        - La regex bash è rigida, richiede esattamente un certo formato e se l'agente scrive bash diversamente non funziona
        - il parsing della answer è fragile 
        Questo è importante perchè nel metodo _judge_ viene fatto root = self.extract_action(root.content), e se il parsing fallisce, il task fallisce
        """
        return ret
    

    # Nei risultati, i casi possibili sono i seguenti:
    #
    # 1) status = completed:
    #       - result = true -> Il task è stato completato correttamente e la risposta dell'agente è corretta
    #       - result = false -> Il task è stato completato (l'agente ha prodotto un output finale), ma la risposta è sbagliata rispetto
    #                           alla ground truth
    # 
    # 2) status = agent_context_limit, result = false -> L'agente ha raggiunto il limite di contesto (numero massimo di token/messaggi)
    #                                                    e non ha potuto completare il task
    #
    # 3) status = task_limit_reached, result = false -> L'agente ha superato il numero massimo di round consentiti senza produrre una risposta
    #                                                   finale
    #
    # 4) status = agent_validation_failed, result = false -> L'agente non ha prodotto un'azione valida
    #
    # 5) status = agent_invalid_action, result = false -> L'agente ha prodotto un'azione non prevista
    #
    # 6) status = unknown, result = false -> Errore generico durante l'esecuzione
    async def start_sample(self, index, session: Session) -> TaskSampleExecutionResult:
        """
        Ponte d'ingresso dell'esecuzione di unsingolo problema.
        - index è la chiave univoca del problema
        - session è un oggetto che gestisce la conversazione con l'agente
        ritorna TaskSampleExecutionresult che contiene status e result (dict con info)"""
        data_item = self.problem_configs[index]
        config = data_item["config"]
        file = data_item["file"]
        index_in_file = data_item["index"]
        try:
            print("init container")
            container = Container(config.image)
            print("init container ok")
            print("start judge")
            # chiama il metodo judge che inietta il prompt, interagisce con l'agente, esegue comandi e valuta la risposta finale
            result = await self._judge(session, config, container) 
            result.result["file"] = file
            result.result["index_in_file"] = index_in_file
            print("finish judge")
            return result
        except Exception as _:
            print("err")
            import traceback
            # se c'è qualche errore durante l'esecuzione STATUS = UNKNOWN e result: false
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
        Prepara l'OS
        guida l'agente
        esegue comandi
        valida la risposta
        decide se il task è corretto
        """
        print("exec start")
        if config.init_script:
            for script in config.init_script:
                init = await asyncio.to_thread(container.execute_independent, script)
                if init.exit_code != 0:
                    return TaskSampleExecutionResult(
                        status=SampleStatus.UNKNOWN, # se c'è qualche errore nell'avvio stop immediato e STATUS = UNKNOWN e Result = false
                        result={"result": False, "error": f'Init script {script} failed: {init}'}
                    )
        if config.start:
            start = await asyncio.to_thread(container.execute, config.start[1])
            if start.exit_code != 0:
                return TaskSampleExecutionResult(
                    status=SampleStatus.UNKNOWN, # idem sopra
                    result={"result": False, "error": f'Start script {config.start} failed: {start}'}
                )
        print("exec start ok")
        
        # inziezione prompt di sistema
        """
        L'agente deve:
        1) scrivere Think
        2) Poi scegliere una sola azione: bash, finish, answer
        ReAct framework controllato
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
                session.inject(item) # serve a mostrare un esempio di Think, Ac, bash, Output, Answer, e poi si avvio il nuovo problema reale
            session.inject(
                {
                    "role": "user",
                    "content": "Now, I will start a new problem in a new OS. My problem is:\n\n"
                    + config.description,
                }
            )
        
        # loop di interazione agente -OS (round limit di default 8)
        for _ in range(self.round_limit):
            root = await session.action() # chiamata agente

            print("\n=== RAW AGENT OUTPUT ===")
            print(root.content)
            print("========================\n")

            if root.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT: # se supera token limit -> ritorna errore dedicato
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_CONTEXT_LIMIT, result={"result": False}
                )
            if root.status != AgentOutputStatus.NORMAL: # se non è status normal ritorna errore generico
                return TaskSampleExecutionResult(
                    status=SampleStatus.UNKNOWN, result={"result": False}
                )
            root = self.extract_action(root.content) # parsing dell'azione

            print("\n=== PARSED ACTION ===")
            print(root)
            print("=====================\n")

            """
            Converte testo LLM in:
            {
             "thought": "...",
             "action": "bash" | "commit",
             "content": ...
            }"""
            if "action" not in root:
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_VALIDATION_FAILED,
                    result={"result": False},
                )
            if root["action"] not in ["bash", "commit"]:# se LLM sbaglia formato -> invalid action
                print("INVALID ACTION DETECTED")
                print("Action value:", root["action"])
                print("Full parsed root:", root)
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_INVALID_ACTION, result={"result": False}
                )

            action = root["action"]
            content = root["content"]
            if action == "commit": # se l'azione è un commit termina il loop
                answer = content
                break
            elif action == "bash":  # esegue comando nella shell persistente
                result = await asyncio.to_thread(container.execute, content)
                result = result.output.decode("utf-8")
                if len(result) > 800:
                    result = (
                        result[:780] + "\n[truncated because the output is too long]"
                    )
                session.inject( # simula risposta dell'OS
                    {
                        "role": "user",
                        "content": ("The output of the OS:\n\n" + result)
                        if result
                        else "The output of the OS is empty.",
                    }
                )
        else: # se supera round_limit status = TASK_LIMIT_REACHED e result = false
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_LIMIT_REACHED,
                result={"result": False, "reason": "round limit"},
            )

        if isinstance(answer, str) and config.match and config.match["strip"]:
            answer = answer.strip() # normaliza la risposta
        
        # FASE DI VALUTAZIONE
        jd = False
        
        # Primo caso: match diretto (confronto stringa esatta)
        if config.match:
            if "answer" in config.match:
                # qui avviene il confronto tra la risposta prodotta e quella di ground truth che determina se il risultato è giusto o meno
                jd = answer == config.match["answer"]
            # secondo caso regex
            elif "regex" in config.match:
                jd = re.search(config.match["regex"], answer) is not None
        # altrimenti script di validazione
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
        
        # se la valutazione va a buon fine (a prescindere da risultato giusto o sbagliato) lo statu ssarà COMPLETED
        return TaskSampleExecutionResult(
            status=SampleStatus.COMPLETED, result={"result": jd}
        )