from typing import List

from pydantic import BaseModel

from .general import SampleIndex
from .output import AgentOutput, TaskOutput


class RegisterRequest(BaseModel):
    """
    Serve per registrare un worker 
    """
    name: str # identificativo
    address: str # host:port
    concurrency: int # numero massimo task paralleli
    indices: list # sample assegnabili


class StartSampleRequest(BaseModel):
    """
    Richiede l'avvio di un sample specifico
    """
    name: str
    index: SampleIndex


class InteractRequest(BaseModel):
    """
    Chiamata worker orchestrator per inviare riposta agent
    """
    session_id: int
    agent_response: AgentOutput


class CancelRequest(BaseModel):
    """
    Termina una sessione in corso
    """
    session_id: int


class HeartbeatRequest(BaseModel):
    """
    Liveness check
    """
    name: str
    address: str


class CalculateOverallRequest(BaseModel):
    """
    Probabilmente:worker ha completato batch, invia risultati finali, orchestratore calcola metriche globali
    """
    name: str
    results: List[TaskOutput]


class WorkerStartSampleRequest(BaseModel):
    """
    Messaggio orchestratore worker
    """
    index: SampleIndex
    session_id: int


class SampleStatusRequest(BaseModel):
    """
    Query per stato sample
    """
    session_id: int