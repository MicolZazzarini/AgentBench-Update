import json
import re
from typing import Callable, Dict, List, Any
from src.server.task import Task, Session
from src.typings import TaskOutput, SampleStatus, AgentOutputStatus
from .Interaction import Container

# Big system prompt for the agent describing expected interaction
big_prompt = """
I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose to operate or to answer.
your operation should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If you are done operating, and you want to commit your final answer, then write down:
Action: Answer
Final Answer: ["ANSWER1", "ANSWER2", ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Your input will be raw MySQL response, you have to deal with it by yourself.
"""


def build_init_sql(entry):
    """
    Build the SQL statements required to initialize a MySQL database and table
    based on the input JSON entry.

    Returns:
        sql (str): Multi-statement SQL script for creation and insertion
        items_data (tuple): Flattened tuple of all row values for parameterized insertion
    """
    name = entry["table"]["table_name"]
    columns = ",".join(
        [
            f"`{column['name']}` TEXT"
            for column in entry["table"]["table_info"]["columns"]
        ]
    )
    column_names = ",".join(
        [f"`{column['name']}`" for column in entry["table"]["table_info"]["columns"]]
    )
    items = []
    items_data = ()
    for row in entry["table"]["table_info"]["rows"]:
        item = "("
        for col in row:
            item += "%s,"
            items_data += (col,)
        item = item[:-1] + ")"
        items.append(item)
    items = ",".join(items)
    sql = f"""CREATE DATABASE IF NOT EXISTS `{name}`;
USE `{name}`;
CREATE TABLE IF NOT EXISTS `{name}` ({columns});
INSERT INTO `{name}` ({column_names}) VALUES {items}; 
COMMIT;
"""
    return sql, items_data

# task

# Possible combinations of status and result in DBBench:
# 
# 1. SampleStatus.COMPLETED
#    - result = True  : agent provided a correct answer (matches ground truth for SELECT or MD5 hash for modification queries)
#    - result = False : agent provided an answer, but it does not match the ground truth
#
# 2. SampleStatus.AGENT_VALIDATION_FAILED
#    - result = False : agent did not follow the expected format (missing "Action" or "Final Answer")
#
# 3. SampleStatus.AGENT_CONTEXT_LIMIT
#    - result = False : agent exceeded the context length limit
#
# 4. SampleStatus.TASK_LIMIT_REACHED
#    - result = False : agent did not provide a final answer within max_round interactions
#
# 5. SampleStatus.UNKNOWN
#    - result = False : unexpected exception or execution error occurred
#
# Note: Other statuse like AGENT_INVALID_ACTION are not used in DBBench.
class DBBench(Task):
    """
    Task class for executing database interaction problems.
    Handles dataset loading, interaction with the agent, and evaluation.
    """
    def __init__(self, **configs):
        super().__init__(**configs)
        self.data_file = configs.pop("data_file")
        self.max_round = configs.pop("max_round", 5)
        self.dataset = []

        with open(self.data_file) as f:
            if self.data_file.endswith("json"):
                data = json.loads(f.read())
            else:
                data = [json.loads(line) for line in f.readlines()]
        
        # Normalize dataset entries into (input, answer) tuples
        for entry in data:
            """
            entry["type"] is usually a list of strings describing the kind of SQL operation this entry involves.
            ("INSERT", "DELETE", "UPDATE") are modification queries, i.e., queries that change the database rather than just select 
            data.
            If the task is a modification query, we need a hash of the resulting table (answer_md5) to check correctness later.
            For modification queries, the dataset contains an MD5 hash (ground truth answer) of the expected table content after 
            the query is executed. pop removes this field from the entry dictionary and stores it in ans.
            """
            if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
                ans = entry.pop("answer_md5")
            else:
                """
                For other types of queries (like SELECT), the expected answer is usually in label (ground truth answer to match).
                """
                ans = entry.pop("label")

            """
            After popping the answer, entry contains all information the agent needs to solve the task:
            - Table schema
            - Initial rows
            - Problem description
            """
            inp = entry

            """
            After this loop:

            self.dataset = [
                (input_for_task_1, answer_for_task_1),
                (input_for_task_2, answer_for_task_2),
                ...
            ]
            """
            self.dataset.append((inp, ans))

        self.container = Container()

    def get_indices(self) -> List[Any]:
        return list(range(len(self.dataset)))

    async def start_sample(self, index: int, session: Session) -> TaskOutput:
        """
        Run a single sample/task interaction.
        
        Args:
            index: dataset index to execute
            session: session object to manage agent interaction
        
        Returns:
            TaskOutput with status, result, and history
        """
        entry = self.dataset[index][0]
        container = self.container
        init_sql, init_data = build_init_sql(entry)
        container.execute(init_sql, data=init_data)
        db = entry["table"]["table_name"]
        session.inject({"role": "user", "content": big_prompt}) # first turn in history
        session.inject({"role": "agent", "content": "Ok."}) # second turn in history
        prompt = entry["description"] + "\n" + entry["add_description"]
        session.inject({"role": "user", "content": prompt}) # third turn in history: the real task the agent have to solve

        # start agent interaction
        res = (await session.action()).content or ""
        answer = ""
        finish_reason = SampleStatus.COMPLETED
        try:
            # searches "Action" in the content
            action = re.search(r"Action: (.*?)\n", res)
            rounds = 0

            # if there is "Action" and there is "Operation"
            while action and action.group(1) == "Operation" and rounds < self.max_round:

                # searches the query sql
                res = re.search(r"```sql\n([\s\S]*?)\n```", res)
                if not res:
                    # if there is no query sql -> agent validation failed
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                    break
                sql = res.group(1).strip()
                sql = sql.replace("\n", " ")
                # if there is the query, executes in the container
                response = container.execute(sql, db)
                if response:
                    # the result is given to the agent in the next turn of history
                    session.inject({"role": "user", "content": response}) # response could be also an error message
                else:
                    session.inject({"role": "user", "content": ""})

                # interaction with agent again
                res = await session.action()

                # handle agent context limit exceptions
                if res.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                    finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                    break

                res = res.content

                # searches again "Action" in content
                action = re.search(r"Action: (.*?)\n", res)
                rounds += 1
            else:
                # searches the answer in the content
                answer = re.search(r"\nFinal Answer:(.*)", res)
                if answer:
                    answer = answer.group(1)
                else:
                    answer = ""
                    # if there's no answer -> agent validation failed
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                if rounds >= self.max_round and not answer:
                    finish_reason = SampleStatus.TASK_LIMIT_REACHED
        except Exception as e:
            error = str(e)
            answer = ""
            # execution error
            finish_reason = SampleStatus.UNKNOWN
        else:
            error = ""

        # evaluation
        if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"): # modification of the db, md5 hash evaluation
            import hashlib

            columns = ",".join(
                [
                    f"`{column['name']}`"
                    for column in entry["table"]["table_info"]["columns"]
                ]
            )
            md5_query = (
                f"select md5(group_concat(rowhash order by rowhash)) as hash "
                f"from( SELECT substring(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash FROM `{db}`) as sub;"
            )
            answer = container.execute(md5_query, db)

            # select_query = f"SELECT {columns} FROM `{db}`.`{db}` ORDER BY {columns};"
            # rows = container.execute(select_query, db)
            # rows_str = "".join(str(row) for row in rows)
            # MD5 in Python
            # answer = hashlib.md5(rows_str.encode()).hexdigest()

        if finish_reason == SampleStatus.COMPLETED:
            if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
                # MD5 comparison of modified table
                ground_truth = self.dataset[index][1]  # answer_md5
                result = answer == ground_truth
            else:
                # query SELECT: label comparison
                ground_truth = self.dataset[index][1]  # label
                # converti l'output dell'agente in lista se serve
                try:
                    agent_answer = list(eval(answer))
                except:
                    agent_answer = [answer]
                result = agent_answer == ground_truth
        else:
            result = False
        
        container.execute(f"drop database `{db}`")

        return TaskOutput(
            status=finish_reason,
            result={
                "answer": str(answer),
                "type": entry["type"][0],
                "error": error,
                "result": result
            },
            history=session.history,
        )

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        """
        Calculate overall metrics from multiple task outputs
        """
        metrics = self.metrics
        ret = {}
        outputs = []
        answers = []
        for result in results:
            outputs.append(result.result)
            answers.append(self.dataset[result.index][1])
        for key, func in metrics.items():
            ret[key] = func(outputs, answers)
        return ret

    @property
    def metrics(self) -> Dict[str, Callable[[List[Dict[str, Any]], List[str]], float]]:
        def factory(typ):
            def acc(inp: List[Dict[str, Any]], tar: List[str]) -> float:
                correct = 0
                total = 0
                for entry, cor in zip(inp, tar):
                    if not entry:
                        continue
                    ans, t = entry["answer"], entry["type"]
                    if t != typ and not (
                        typ == "SELECT" and t not in ("INSERT", "UPDATE")
                    ):
                        continue
                    if t in ("INSERT", "DELETE", "UPDATE"):
                        correct += ans == cor # comparison between result and ground truth
                    else:
                        try:
                            ans = list(eval(ans))
                        except:
                            ans = [ans]
                        if len(ans) == 1 and len(cor) == 1:
                            try:
                                correct += float(ans[0]) == float(cor[0])
                            except (ValueError, TypeError):
                                correct += ans[0] == cor[0]
                            else:
                                print(ans, cor)
                        else:
                            try:
                                cor = set(cor)
                                ans = set(ans)
                                correct += ans == cor
                            except:
                                pass
                    total += 1
                if total == 0:
                    print(f"WARNING: {typ} does not exist!")
                    return 0
                return correct / total

            return acc

        types = [
            "other",
            "counting",
            "comparison",
            "ranking",
            "aggregation-SUM",
            "aggregation-MIN",
            "aggregation-MAX",
            "aggregation-AVG",
            "SELECT",
            "INSERT",
            "UPDATE",
        ]

        ret = {}
        for typ in types:
            ret[typ + "_accuracy"] = factory(typ)

        ret["overall_cat_accuracy"] = (
            lambda inp, tar: sum(
                [
                    ret[typ + "_accuracy"](inp, tar)
                    for typ in ("SELECT", "INSERT", "UPDATE")
                ]
            )
            / 3
        )

        return ret

    def release(self):
        self.container.delete()