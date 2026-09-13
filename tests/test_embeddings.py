"""Testes do embeddings.py — offline: nenhuma chamada de rede é feita.

Rode com:  .venv/bin/python tests/test_embeddings.py

Cobre a classificação dos dois tipos de erro 429, que foi o que fez uma
ingestão ficar minutos em tentativas inúteis:
  - cota DIÁRIA esgotada -> falhar na hora, com mensagem clara;
  - limite por MINUTO    -> esperar e tentar de novo, respeitando o retryDelay.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import embeddings  # noqa: E402

# Trecho real do erro devolvido pela API quando a cota diária acaba.
DAILY_QUOTA_ERROR = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota...', 'status': 'RESOURCE_EXHAUSTED', 'details': ["
    "{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaMetric': 'generativelanguage.googleapis.com/embed_content_free_tier_requests', "
    "'quotaId': 'EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier', "
    "'quotaValue': '1000'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
    "'retryDelay': '45s'}]}}"
)

# Limite por minuto: sem marcação PerDay, com retryDelay sugerido.
PER_MINUTE_ERROR = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Rate limit "
    "exceeded', 'status': 'RESOURCE_EXHAUSTED', 'details': ["
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '9s'}]}}"
)


class FakeError(Exception):
    """Erro com .code, imitando o ClientError do SDK do Gemini."""

    def __init__(self, message: str, code: int = 429):
        super().__init__(message)
        self.code = code


class StubClient:
    """Cliente falso: toda chamada de embedding levanta o erro dado."""

    def __init__(self, error: Exception):
        self.calls = 0
        self._error = error
        self.models = self

    def embed_content(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        raise self._error


def _run_with(error: Exception) -> tuple[list[float], StubClient, list[float]]:
    """Executa _embed_batch com cliente e sleep falsos; devolve o que observou."""
    sleeps: list[float] = []
    stub = StubClient(error)
    original_client = embeddings._get_client
    original_sleep = embeddings.time.sleep
    embeddings._get_client = lambda: stub
    embeddings.time.sleep = lambda seconds: sleeps.append(float(seconds))
    try:
        embeddings._embed_batch(["um texto"], embeddings.DOCUMENT_TASK)
    finally:
        embeddings._get_client = original_client
        embeddings.time.sleep = original_sleep
    return [], stub, sleeps


def test_daily_quota_fails_fast():
    """Cota diária: RuntimeError na primeira tentativa, sem nenhum sleep."""
    try:
        _run_with(FakeError(DAILY_QUOTA_ERROR))
    except RuntimeError as error:
        message = str(error)
        assert "cota diária" in message, message
        assert "meia-noite do Pacífico" in message, message
    else:
        raise AssertionError("esperava RuntimeError para cota diária")
    print("cota diaria: falha imediata OK")

    # E confirma que não dormiu nem tentou de novo.
    sleeps: list[float] = []
    stub = StubClient(FakeError(DAILY_QUOTA_ERROR))
    original_client, original_sleep = embeddings._get_client, embeddings.time.sleep
    embeddings._get_client = lambda: stub
    embeddings.time.sleep = lambda seconds: sleeps.append(float(seconds))
    try:
        try:
            embeddings._embed_batch(["x"], embeddings.DOCUMENT_TASK)
        except RuntimeError:
            pass
    finally:
        embeddings._get_client, embeddings.time.sleep = original_client, original_sleep
    assert stub.calls == 1, f"chamou a API {stub.calls} vezes; deveria ser 1"
    assert sleeps == [], f"dormiu desnecessariamente: {sleeps}"
    print("cota diaria: 1 chamada, 0 esperas OK")


def test_per_minute_retries_with_api_delay():
    """Limite por minuto: tenta de novo e usa o retryDelay informado pela API."""
    sleeps: list[float] = []
    stub = StubClient(FakeError(PER_MINUTE_ERROR))
    original_client, original_sleep = embeddings._get_client, embeddings.time.sleep
    embeddings._get_client = lambda: stub
    embeddings.time.sleep = lambda seconds: sleeps.append(float(seconds))
    try:
        try:
            embeddings._embed_batch(["x"], embeddings.DOCUMENT_TASK)
        except Exception:  # noqa: BLE001 — o erro original deve subir no fim
            pass
    finally:
        embeddings._get_client, embeddings.time.sleep = original_client, original_sleep

    assert stub.calls == embeddings.MAX_ATTEMPTS, stub.calls
    assert sleeps == [9.0] * (embeddings.MAX_ATTEMPTS - 1), sleeps
    print(f"limite por minuto: {stub.calls} chamadas, esperas {sleeps} OK")


def test_is_daily_quota_detection():
    """O detector distingue a cota diária do limite por minuto."""
    assert embeddings._is_daily_quota(FakeError(DAILY_QUOTA_ERROR))
    assert not embeddings._is_daily_quota(FakeError(PER_MINUTE_ERROR))
    print("detector de cota diaria OK")


def test_retry_delay_parsing():
    """O retryDelay é lido do erro; ausente vira None."""
    assert embeddings._retry_delay(FakeError(PER_MINUTE_ERROR)) == 9.0
    assert embeddings._retry_delay(FakeError(DAILY_QUOTA_ERROR)) == 45.0
    assert embeddings._retry_delay(FakeError("429 sem retryDelay")) is None
    print("parse do retryDelay OK")


def main():
    test_is_daily_quota_detection()
    test_retry_delay_parsing()
    test_daily_quota_fails_fast()
    test_per_minute_retries_with_api_delay()
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
