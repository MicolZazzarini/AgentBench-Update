import docker
import mysql.connector
import random
import time
from typing import Optional, Union, Sequence, Dict, Any


class Container:
    password = "password"

    def __init__(self, image: str = "mysql:8.0"):
        self.deleted = False
        self.image = image
        self.client = docker.from_env()

        container_name = f"mysql_{random.randint(10000, 99999)}"

        # Avvio container con healthcheck
        self.container = self.client.containers.run(
            self.image,
            name=container_name,
            environment={
                "MYSQL_ROOT_PASSWORD": self.password,
            },
            ports={"3306/tcp": None},
            detach=True,
            remove=True,
            healthcheck={
                "test": ["CMD", "mysqladmin", "ping", "-h", "localhost", "-ppassword"],
                "interval": 2_000_000_000,   # 2s
                "timeout": 2_000_000_000,    # 2s
                "retries": 30,
                "start_period": 5_000_000_000,  # 5s grace
            },
        )

        # ---- Attendi assegnazione porta ----
        self.port = None
        for _ in range(60):
            self.container.reload()
            ports = self.container.attrs["NetworkSettings"]["Ports"]
            if ports and ports.get("3306/tcp"):
                binding = ports["3306/tcp"]
                if binding:
                    self.port = int(binding[0]["HostPort"])
                    break
            time.sleep(1)

        if self.port is None:
            self.delete()
            raise RuntimeError("MySQL container did not expose port 3306")

        # ---- Attendi stato healthy ----
        for _ in range(90):  # max ~3 minuti
            self.container.reload()
            state = self.container.attrs.get("State", {})
            health = state.get("Health", {})
            status = health.get("Status")

            if status == "healthy":
                break

            if status == "unhealthy":
                logs = self.container.logs().decode(errors="ignore")
                self.delete()
                raise RuntimeError(f"MySQL container unhealthy\n\nLogs:\n{logs}")

            time.sleep(2)
        else:
            logs = self.container.logs().decode(errors="ignore")
            self.delete()
            raise RuntimeError(f"MySQL container never became healthy\n\nLogs:\n{logs}")

        # ---- Connessione finale ----
        self.conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password=self.password,
            port=self.port,
            autocommit=True,
        )

    def delete(self):
        if not self.deleted:
            try:
                self.container.stop()
            except Exception:
                pass
            self.deleted = True

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    def execute(
        self,
        sql: str,
        database: str = None,
        data: Union[Sequence, Dict[str, Any]] = (),
    ) -> Optional[str]:

        self.conn.reconnect(attempts=3, delay=2)

        try:
            with self.conn.cursor() as cursor:

                if database:
                    cursor.execute(f"USE `{database}`;")
                    cursor.fetchall()

                rows = []

                for result in cursor.execute(sql, data, multi=True):
                    if result.with_rows:
                        rows = result.fetchall()

                result_str = str(rows)

            self.conn.commit()

        except Exception as e:
            result_str = str(e)

        if len(result_str) > 800:
            result_str = result_str[:800] + "[TRUNCATED]"

        return result_str
