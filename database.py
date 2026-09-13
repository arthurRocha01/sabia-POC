"""Armazenamento vetorial persistente (ChromaDB) + catálogo de livros.

data/chroma/ - vetores, textos e metadados dos chunks (ChromaDB embarcado)
data/books.json - catálogo dos livros ingeridos (fonte única de identidade
                  dos livros; os chunks no Chroma só referenciam book_id)

Este módulo nunca gera embeddings nem chama LLM: recebe e devolve dados
puros. A geração de embeddings vive em ingestion.py / retrieval.py.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).resolve().parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CATALOG_PATH = DATA_DIR / "books.json"
COLLECTION_NAME = "chunks"

_client = None
_collection = None
_catalog = None


# Internos (init preguiçoso: importar o módulo não tem efeito colateral)


def _get_client():
    """Cria o cliente persistente do ChromaDB no primeiro uso."""
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_collection():
    """Obtém (ou cria) a collection de chunks com distância por cosseno."""
    global _collection
    if _collection is None:
        # O metadata só vale na criação; se já existir uma collection com
        # outro espaço de distância, ela fica como está.
        _collection = _get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _load_catalog():
    """Carrega o catálogo do disco (preguiçoso, cacheado em memória)."""
    global _catalog
    if _catalog is None:
        if CATALOG_PATH.exists():
            _catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        else:
            _catalog = {}
    return _catalog


def _save_catalog():
    """Persiste o catálogo no disco."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(
        json.dumps(_load_catalog(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Catálogo de livros


def book_exists(book_id: str) -> bool:
    """Retorna True se o book_id já está no catálogo."""
    return book_id in _load_catalog()


def get_book(book_id: str) -> dict | None:
    """Entrada do catálogo para um book_id, ou None."""
    entry = _load_catalog().get(book_id)
    return dict(entry) if entry else None


def list_books() -> list[dict]:
    """Todas as entradas do catálogo, da ingestão mais antiga para a mais nova."""
    return sorted(
        (dict(e) for e in _load_catalog().values()),
        key=lambda e: e["ingested_at"],
    )


def all_documents() -> list[str]:
    """Todos os textos de chunks do acervo (para análises locais, sem API)."""
    result = _get_collection().get(include=["documents"])
    return [doc for doc in (result["documents"] or []) if doc]


def delete_book(book_id: str) -> None:
    """Remove os chunks do livro do armazenamento e sua entrada no catálogo."""
    if book_id not in _load_catalog():
        return
    _get_collection().delete(where={"book_id": book_id})
    del _load_catalog()[book_id]
    _save_catalog()


# Escrita


def save_book(*, book_id, title, author, source_path, line="default", chunks):
    """Persiste livro + chunks; re-ingerir o mesmo id substitui a cópia antiga.

    chunks: lista de dicts com as chaves text (str), page_start (int),
    page_end (int), embedding (list[float]).

    Os ids dos chunks são determinísticos (book_id + índice), então um
    replace reutiliza os ids. Retorna a entrada do catálogo (com n_chunks).
    """

    collection = _get_collection()
    chunks = list(chunks)
    if not chunks:
        raise ValueError(f"save_book: nenhum chunk para {book_id}")

    # Semântica de substituição: apaga chunks anteriores deste livro, se houver.
    collection.delete(where={"book_id": book_id})

    ids = [f"{book_id}:{i:06d}" for i in range(len(chunks))]
    metadatas = [
        {
            "book_id": book_id,
            "line": line,
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            # Numeração do livro (rótulo); sem ela, cai para o índice posicional.
            "label_start": c.get("label_start") or str(c["page_start"]),
            "label_end": c.get("label_end") or str(c["page_end"]),
        }
        for c in chunks
    ]
    collection.add(
        ids=ids,
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=metadatas,
    )

    entry = {
        "book_id": book_id,
        "title": title,
        "author": author,
        "source_path": source_path,
        "line": line,
        "n_chunks": len(chunks),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    _load_catalog()[book_id] = entry
    _save_catalog()
    return dict(entry)


# Busca semântica


def search(query_vector, *, k=5, exclude_book_id=None, line=None) -> list[dict]:
    """Top-k chunks mais próximos do query_vector (cosseno), melhores primeiro.

    Filtros opcionais: excluir um livro (ex.: o que o usuário está lendo)
    e/ou restringir a uma linha de aprendizado.
    Retorna um dict por chunk: book_id, title, author, source_path, line,
    page_start, page_end, text, score (similaridade do cosseno em [0, 1],
    maior = mais próximo).
    """

    collection = _get_collection()

    # Monta o filtro where dinamicamente conforme os filtros recebidos.
    conditions = []
    if line is not None:
        conditions.append({"line": {"$eq": line}})
    if exclude_book_id is not None:
        conditions.append({"book_id": {"$ne": exclude_book_id}})
    where = conditions[0] if len(conditions) == 1 else (
        {"$and": conditions} if conditions else None
    )

    if k < 1:
        raise ValueError("search: k precisa ser >= 1")

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    # Join dos metadados com o catálogo: título/autor vêm de lá, nunca
    # duplicados no chunk (evita dessincronização).
    catalog = _load_catalog()
    hits = []
    for i, chunk_id in enumerate(result["ids"][0]):
        metadata = result["metadatas"][0][i]
        book_id = metadata["book_id"]
        book_entry = catalog.get(book_id)
        # Rever tratamento de erro: chunks órfãos não podem ser silenciados
        if book_entry is None:
            print(
                f"search: chunk {chunk_id} refere-se a book_id {book_id} "
                "não presente no catálogo",
                file=sys.stderr,
            )
            continue
        hits.append({
            "book_id": book_id,
            "title": book_entry["title"],
            "author": book_entry["author"],
            "source_path": book_entry["source_path"],
            "line": metadata["line"],
            "page_start": metadata["page_start"],
            "page_end": metadata["page_end"],
            # Rótulo (numeração do livro) é o que se cita; o índice posicional
            # (page_start/page_end) serve para localizar a página no PDF.
            "label_start": metadata.get("label_start") or str(metadata["page_start"]),
            "label_end": metadata.get("label_end") or str(metadata["page_end"]),
            "text": result["documents"][0][i],
            "score": 1 - result["distances"][0][i],  # cosseno: 1 = idêntico, 0 = ortogonal
        })
    return hits
