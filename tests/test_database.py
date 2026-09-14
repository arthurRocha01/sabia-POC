"""Testes de fumaça do database.py — sem dependências externas.

Rode com:  .venv/bin/python tests/test_database.py

Os testes usam um diretório temporário isolado: não tocam no data/ real
do projeto (nem precisam de API key — vetores são fabricados aqui).
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Garante que o módulo do projeto seja importável a partir de qualquer cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import database as db  # noqa: E402

# Vetores artificiais de dimensão 3, ortogonais entre si (espaço de teste).
DOGS = [1.0, 0.0, 0.0]
CATS = [0.0, 1.0, 0.0]
WAR = [0.0, 0.0, 1.0]


def _isolate_data():
    """Redireciona o database para um diretório temporário e limpo."""
    tmp = Path(tempfile.mkdtemp(prefix="sabia-poc-test-"))
    db.DATA_DIR = tmp / "data"
    db.CHROMA_DIR = db.DATA_DIR / "chroma"
    db.CATALOG_PATH = db.DATA_DIR / "books.json"
    # Zera o cache preguiçoso para o cliente recriar tudo no novo local.
    db._client = None
    db._collection = None
    db._catalog = None
    return tmp


def _save_book(book_id, title, author, chunks, source_path="/tmp/livro.pdf"):
    """Atalho para salvar um livro com os campos padrão de teste."""
    return db.save_book(
        book_id=book_id,
        title=title,
        author=author,
        source_path=source_path,
        chunks=chunks,
    )


def _sun_tzu_chunks():
    """Três chunks do livro de exemplo Sun Tzu."""
    return [
        {"text": "conhece o inimigo e a ti mesmo", "page_start": 1, "page_end": 1, "embedding": DOGS},
        {"text": "a arte da guerra baseia-se no engano", "page_start": 4, "page_end": 5, "embedding": CATS},
        {"text": "vencer sem combater e a excelencia", "page_start": 9, "page_end": 9, "embedding": WAR},
    ]


def _maquiavel_chunks():
    """Três chunks do livro de exemplo Maquiavel."""
    return [
        {"text": "o principe deve conhecer o povo", "page_start": 2, "page_end": 2, "embedding": DOGS},
        {"text": "parecer virtuoso e suficiente", "page_start": 7, "page_end": 7, "embedding": CATS},
        {"text": "e melhor ser temido que amado", "page_start": 12, "page_end": 12, "embedding": WAR},
    ]


def test_catalog():
    """Catálogo: existência, consulta, listagem ordenada."""
    _save_book("sun-tzu", "A Arte da Guerra", "Sun Tzu", _sun_tzu_chunks())
    _save_book("maquiavel", "O Principe", "Maquiavel", _maquiavel_chunks())

    assert db.book_exists("sun-tzu")
    assert not db.book_exists("nao-existe")
    assert db.get_book("maquiavel")["n_chunks"] == 3
    assert db.get_book("fantasma") is None

    books = db.list_books()
    assert [b["book_id"] for b in books] == ["sun-tzu", "maquiavel"]
    print("catalogo OK")


def test_ranking():
    """Busca sem filtro: top-1 é o chunk semanticamente mais próximo."""
    hits = db.search(DOGS, k=3)
    assert hits[0]["book_id"] == "sun-tzu"
    assert hits[0]["page_start"] == 1
    assert hits[0]["score"] > 0.99
    # Join com o catálogo: título e autor vêm preenchidos.
    assert hits[0]["title"] == "A Arte da Guerra"
    print("ranking OK:", hits[0]["book_id"], hits[0]["page_start"], hits[0]["score"])


def test_exclusion():
    """Excluir o livro de origem: só retornam chunks dos outros livros."""
    hits = db.search(DOGS, k=3, exclude_book_id="sun-tzu")
    assert len(hits) == 3
    assert all(h["book_id"] == "maquiavel" for h in hits)
    print("exclusao OK: so maquiavel")


def test_reingestion_replaces():
    """Re-ingerir o mesmo id substitui os chunks antigos (3 -> 1)."""
    _save_book("sun-tzu", "A Arte da Guerra", "Sun Tzu", [
        {"text": "so um chunk agora", "page_start": 3, "page_end": 3, "embedding": DOGS},
    ])
    assert db.get_book("sun-tzu")["n_chunks"] == 1
    hits = db.search(DOGS, k=5)
    assert sum(1 for h in hits if h["book_id"] == "sun-tzu") == 1
    print("re-ingestao OK: sun-tzu com 1 chunk")


def test_delete_and_empty_collection():
    """Delete limpa catálogo; busca em coleção vazia não pode quebrar."""
    db.delete_book("sun-tzu")
    db.delete_book("maquiavel")
    assert db.list_books() == []
    hits = db.search(DOGS, k=3)
    assert hits == []
    print("delete OK + busca em colecao vazia OK")


def main():
    """Roda todos os testes em ordem e reporta o resultado."""
    tmp = _isolate_data()
    try:
        test_catalog()
        test_ranking()
        test_exclusion()
        test_reingestion_replaces()
        test_delete_and_empty_collection()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
