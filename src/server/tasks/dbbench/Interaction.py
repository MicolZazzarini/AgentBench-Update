import docker
import mysql.connector
import random
import time
from typing import Optional, Union, Sequence, Dict, Any


class Container:
    """
    A wrapper around a MySQL Docker container that handles lifecycle, port mapping,
    health checking, and executing SQL queries.
    """
    password = "password" # default root password for MySQL

    def __init__(self, image: str = "mysql:8.0"):
        """
        Initialize and start a MySQL Docker container with healthcheck.
        
        Args:
            image (str): Docker image to use for MySQL. Defaults to "mysql:8.0".
        
        Raises:
            RuntimeError: If the container fails to expose port 3306 or never becomes healthy.
        """
        self.deleted = False
        self.image = image
        self.client = docker.from_env()

        container_name = f"mysql_{random.randint(10000, 99999)}"

        # ---- Start container with healthcheck ----
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
                "start_period": 5_000_000_000,  # 5s 
            },
        )

        # ---- Wait for container to assign a host port ----
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

        # ---- Wait for container to become healthy ----
        for _ in range(90):  
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

        # ---- Establish MySQL connection ----
        self.conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password=self.password,
            port=self.port,
            autocommit=True,
        )

    def delete(self):
        """
        Stop and remove the Docker container if it hasn't been deleted yet.
        """
        if not self.deleted:
            try:
                self.container.stop()
            except Exception:
                pass
            self.deleted = True

    def __del__(self):
        """
        Ensure the container is stopped when the Container object is garbage collected.
        """
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
        """
        Execute a SQL query (or multiple queries) in the container and return the result.

        Args:
            sql (str): SQL statement(s) to execute.
            database (str, optional): Database to use before executing query.
            data (Sequence or Dict, optional): Parameters for SQL query.

        Returns:
            str: The result of the query as a string, or the error message.
                 If the result is longer than 800 characters, it is truncated.
        """
        
        # Ensure connection is alive
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
        
        # Truncate long results
        if len(result_str) > 800:
            result_str = result_str[:800] + "[TRUNCATED]"

        return result_str
