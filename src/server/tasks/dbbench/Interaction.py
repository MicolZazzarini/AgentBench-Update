import docker
import mysql.connector
import random
import socket
import time
from docker.models import containers
from typing import Optional, Union, Sequence, Dict, Any


class Container:
    port = 13000
    password = "password"

    def __init__(self, image: str = "mysql"):
        self.deleted = False
        self.image = image
        self.client = docker.from_env()

        # Trova una porta libera per il container
        p = Container.port + random.randint(0, 10000)
        while self.is_port_open(p):
            p += random.randint(0, 20)
        self.port = p

        # Avvia il container MySQL
        self.container: containers.Container = self.client.containers.run(
            image,
            name=f"mysql_{self.port}",
            environment={
                "MYSQL_ROOT_PASSWORD": self.password,
            },
            ports={"3306/tcp": self.port},  # porta container → host
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
        )

        # Attendi qualche secondo che MySQL si avvii
        time.sleep(3)

        # Prova a connetterti ripetutamente finché MySQL non è pronto
        retry = 0
        max_retry = 30  # massimo 30 tentativi
        while True:
            try:
                self.conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password=self.password,
                    port=self.port,
                    pool_reset_session=True,
                )
            except mysql.connector.errors.OperationalError:
                # MySQL non pronto, attendi
                time.sleep(2)
                retry += 1
                if retry > max_retry:
                    raise RuntimeError(f"MySQL did not start after {max_retry} attempts")
            except mysql.connector.InterfaceError:
                time.sleep(2)
                retry += 1
                if retry > max_retry:
                    raise RuntimeError(f"MySQL interface error after {max_retry} attempts")
            else:
                # Connessione avvenuta
                break

    def delete(self):
        try:
            self.container.stop()
        except Exception:
            pass
        self.deleted = True

    def __del__(self):
        try:
            if not self.deleted:
                self.delete()
        except Exception:
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
        # Limita la lunghezza della risposta
        if len(result) > 800:
            result = result[:800] + "[TRUNCATED]"
        return result

    def is_port_open(self, port) -> bool:
        # Controlla se il container esiste già
        try:
            self.client.containers.get(f"mysql_{port}")
            return True
        except Exception:
            pass

        # Controlla se la porta è occupata
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("localhost", port))
            return True
        except ConnectionRefusedError:
            return False
        finally:
            sock.close()