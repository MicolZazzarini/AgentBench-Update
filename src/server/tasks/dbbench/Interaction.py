import docker
import mysql.connector
import random
import time
from typing import Optional, Union, Sequence, Dict, Any


class Container:
    password = "password"

    def __init__(self, image: str = "mysql"):
        self.deleted = False
        self.image = image
        self.client = docker.from_env()

        # Avvia il container lasciando Docker scegliere la porta host
        container_name = f"mysql_{random.randint(10000, 99999)}"
        self.container = self.client.containers.run(
            image,
            name=container_name,
            environment={"MYSQL_ROOT_PASSWORD": self.password},
            ports={"3306/tcp": None},  # Docker assegna automaticamente
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
        )

        # Aspetta che la porta sia disponibile
        self.port = None
        for _ in range(30):
            self.container.reload()  # aggiorna attrs
            ports = self.container.attrs['NetworkSettings']['Ports']
            if '3306/tcp' in ports and ports['3306/tcp']:
                self.port = int(ports['3306/tcp'][0]['HostPort'])
                break
            time.sleep(1)

        if self.port is None:
            self.delete()
            raise RuntimeError("MySQL container did not expose port 3306")

        # Attendi che MySQL sia pronto
        retry = 0
        max_retry = 30
        while True:
            try:
                self.conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password=self.password,
                    port=self.port,
                    pool_reset_session=True,
                )
            except (mysql.connector.errors.OperationalError, mysql.connector.InterfaceError):
                retry += 1
                if retry > max_retry:
                    self.delete()
                    raise RuntimeError(f"MySQL did not start after {max_retry} attempts on port {self.port}")
                time.sleep(2)
            else:
                break

    def delete(self):
        if not self.deleted:
            self.container.stop()
            self.deleted = True

    def __del__(self):
        try:
            self.delete()
        except:
            pass

    def execute(
        self,
        sql: str,
        database: str = None,
        data: Union[Sequence, Dict[str, Any]] = (),
    ) -> Optional[str]:
        self.conn.reconnect()
        try:
            with self.conn.cursor() as cursor:
                if database:
                    cursor.execute(f"USE `{database}`;")
                    cursor.fetchall()
                cursor.execute(sql, data, multi=True)
                result = cursor.fetchall()
                result = str(result)
            self.conn.commit()
        except Exception as e:
            result = str(e)
        if len(result) > 800:
            result = result[:800] + "[TRUNCATED]"
        return result