"""Testes de fumaça do retrieval.py — offline, sem API key.

Rode com:  .venv/bin/python tests/test_retrieval.py

O embedding é substituído por uma função falsa: aqui se testa a lógica de
seleção/filtros, não a qualidade do modelo (essa depende da API real).
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import embeddings  # noqa: E402
import retrieval  # noqa: E402
from test_database import (  # noqa: E402  (helpers reaproveitados)
    DOGS,
    _isolate_data,
    _maquiavel_chunks,
    _save_book,
    _sun_tzu_chunks,
)


def _fake_embed_query(text: str) -> list[float]:
    """Substitui o Gemini: devolve sempre o vetor do tema 'dogs'."""
    return DOGS


def test_connections_come_from_other_books():
    """A conexão tem que vir do OUTRO livro, não do texto/livro de origem."""
    hits = retrieval.find_connections("conhecer a si mesmo", exclude_book_id="sun-tzu")
    assert hits, "esperava conexoes do outro livro"
    assert all(h["book_id"] == "maquiavel" for h in hits)
    # Campos vindos do join com o catálogo.
    assert hits[0]["title"] == "O Principe"
    print("conexoes do outro livro OK")


def test_k_is_respected():
    """O parâmetro k limita a quantidade de chunks retornados."""
    hits = retrieval.find_connections("conhecer a si mesmo", k=1, exclude_book_id="sun-tzu")
    assert len(hits) == 1
    print("k respeitado OK")


def test_line_filter():
    """O filtro de linha de aprendizado restringe o escopo da busca."""
    _save_book(
        "outra-obra", "Outra Obra", "Autor X",
        [{"text": "texto de outra linha", "page_start": 1, "page_end": 1, "embedding": DOGS}],
    )
    # A obra extra foi salva na linha "default"; restringe a uma linha só dela.
    hits = retrieval.find_connections(
        "conhecer a si mesmo", k=3, line="linha-inexistente"
    )
    assert hits == []
    print("filtro de linha OK")


def main():
    """Roda todos os testes com dados isolados e embedding falso."""
    tmp = _isolate_data()
    embeddings.embed_query = _fake_embed_query  # corta a chamada de rede
    try:
        _save_book("sun-tzu", "A Arte da Guerra", "Sun Tzu", _sun_tzu_chunks())
        _save_book("maquiavel", "O Principe", "Maquiavel", _maquiavel_chunks())
        test_connections_come_from_other_books()
        test_k_is_respected()
        test_line_filter()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
