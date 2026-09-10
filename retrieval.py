"""Busca semântica: da entrada do usuário até os chunks candidatos.

Responsabilidade única: embedar a consulta e pedir ao database os trechos
mais próximos, já com os filtros de escopo (linha de aprendizado e exclusão
do livro de origem). Explicar a conexão encontrada é papel do
interpretation.py — aqui só se selecionam os candidatos.
"""

from __future__ import annotations

import database
import embeddings

DEFAULT_K = 3


def find_connections(
    text: str,
    *,
    k: int = DEFAULT_K,
    exclude_book_id: str | None = None,
    line: str | None = None,
) -> list[dict]:
    """Retorna os k chunks de OUTROS livros mais próximos semanticamente de text.

    exclude_book_id: o livro de origem (o que o usuário está lendo). Sem essa
    exclusão, a conexão mais próxima seria o próprio texto do usuário.
    line: restringe a busca a uma linha de aprendizado.
    """
    vector = embeddings.embed_query(text)
    return database.search(
        vector,
        k=k,
        exclude_book_id=exclude_book_id,
        line=line,
    )
