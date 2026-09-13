"""Preflight: avalia um PDF ANTES de ingerir, sem gastar cota de embeddings.

Responde três perguntas:
  1. Dá para ingerir?  -> arquivo, formato, camada de texto, legibilidade.
  2. Quanto custa?     -> quantos textos contam na cota diária do free tier.
  3. Vale a pena?      -> sobreposição temática com o acervo (indício barato) e,
                          opcionalmente, uma sondagem real com embeddings.

A sondagem real (probe_connections) é a única parte que chama a API: ela embeda
algumas amostras do livro candidato e mede o que já existe no acervo para
responder isso de verdade, em vez de estimar.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import database
import embeddings
import ingestion

MIN_WORD_LEN = 5
TOP_WORDS = 300
FREE_TIER_DAILY_TEXTS = int(os.environ.get("FREE_TIER_DAILY_TEXTS", "1000"))

# Limiares da sondagem real (similaridade de cosseno do melhor trecho encontrado).
CONNECT_STRONG = 0.75
CONNECT_MODERATE = 0.60
CONNECT_TEXTS = 5    # amostras do candidato usadas na sondagem (custo em textos)

# Stopwords mínimas em português: o bastante para o indício lexical não ser
# dominado por palavras funcionais.
STOPWORDS = {
    "ainda", "algum", "alguma", "antes", "aquilo", "assim", "cada", "coisa",
    "como", "contra", "daquele", "daquilo", "depois", "desde", "dessa", "dessas",
    "desse", "desses", "desta", "destas", "deste", "destes", "deve", "dever",
    "disso", "então", "entre", "essa", "essas", "esse", "esses", "esta", "estas",
    "este", "estes", "fazer", "havia", "mesmo", "muito", "muitos", "nessa",
    "nessas", "nesse", "nesses", "nesta", "nestas", "neste", "nestes", "outra",
    "outras", "outro", "outros", "pelas", "pelos", "pode", "podem", "porque",
    "pouco", "quando", "qualquer", "quase", "sempre", "sobre", "também", "tanto",
    "temos", "tinha", "todas", "todo", "todos", "tudo", "vezes", "onde", "seja",
    "será", "seria", "forma", "parte", "pode", "fazer", "menos", "maior", "grande",
}


def _significant_words(text: str, top: int = TOP_WORDS) -> set[str]:
    """Palavras mais informativas do texto: 5+ letras, sem stopwords."""
    words = re.findall(rf"[A-Za-zÀ-ÿ]{{{MIN_WORD_LEN},}}", text.lower())
    counts = Counter(word for word in words if word not in STOPWORDS)
    return {word for word, _ in counts.most_common(top)}


def lexical_overlap(text_a: str, text_b: str) -> float:
    """Sobreposição lexical entre dois textos (coeficiente de sobreposição).

    É INDÍCIO de tema compartilhado, não prova de conexão: dois autores podem
    tratar a mesma ideia com vocabulário completamente diferente — que é
    justamente o que a busca semântica pega e este número não pega.
    """
    words_a, words_b = _significant_words(text_a), _significant_words(text_b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def corpus_text(limit_chars: int = 400_000) -> str:
    """Texto do acervo já ingerido (leitura local, sem API)."""
    return " ".join(database.all_documents())[:limit_chars]


def analyze_book(
    pdf_path: str | Path, *, title: str | None = None, author: str | None = None
) -> dict:
    """Relatório LOCAL do candidato: condições de ingestão, custo e tema.

    Não levanta exceção: problemas entram em `problems` (o objetivo é justamente
    diagnosticar antes de ingerir).
    """
    report: dict = {
        "path": str(pdf_path),
        "problems": [],
        "warnings": [],
        "ok": False,
    }

    path = Path(pdf_path)
    if not path.is_file():
        report["problems"].append(f"arquivo não encontrado: {path}")
        return report
    if path.suffix.lower() != ".pdf":
        report["problems"].append(f"formato ainda não suportado (só PDF): {path.name}")
        return report
    if title is not None and not title.strip():
        report["problems"].append("título vazio (a ficha do catálogo não pode sair incompleta)")
    if author is not None and not author.strip():
        report["problems"].append("autor vazio (a conexão é entre autores)")

    # Extração local: é o mesmo caminho da ingestão, sem embeddings.
    try:
        chunks = ingestion.extract_chunks(path)
    except ValueError as error:
        report["problems"].append(str(error))
        return report

    texts = [chunk["text"] for chunk in chunks]
    joined = " ".join(texts)
    quality = ingestion.text_quality(joined)
    report.update(
        {
            "n_chunks": len(chunks),
            "n_texts": len(texts),  # é o que conta na cota do free tier
            "chars": len(joined),
            "quality": quality,
            "first_label": chunks[0]["label_start"],
            "last_label": chunks[-1]["label_end"],
            "sample": texts[len(texts) // 2][:200],
        }
    )

    quota = len(texts) / FREE_TIER_DAILY_TEXTS
    report["quota_fraction"] = quota
    if quota > 1:
        report["problems"].append(
            f"a ingestão precisa de {len(texts)} textos e o free tier permite "
            f"{FREE_TIER_DAILY_TEXTS}/dia — precisa de mais de um dia ou billing"
        )
    elif quota > 0.5:
        report["warnings"].append(
            f"consome {quota:.0%} da cota diária do free tier ({len(texts)} textos)"
        )
    if quality["letter_ratio"] < 0.6:
        report["warnings"].append(
            f"proporção de letras baixa ({quality['letter_ratio']:.0%}) — "
            "texto possivelmente sujo"
        )

    corpus = corpus_text()
    if not corpus:
        report["warnings"].append(
            "acervo vazio: o primeiro livro não gera conexões (não há outro autor)"
        )
        report["theme_overlap"] = None
    else:
        report["theme_overlap"] = lexical_overlap(joined, corpus)

    report["ok"] = not report["problems"]
    return report


def probe_connections(
    chunks: list[dict], *, samples: int = CONNECT_TEXTS, k: int = 3
) -> list[dict]:
    """Sondagem REAL: mede o que o acervo já tem para conectar com este livro.

    Custa `samples` textos da cota (um por amostra). Compara trechos do
    candidato com os do acervo; devolve os melhores achados, do mais próximo
    ao menos próximo.
    """
    if not chunks:
        return []
    step = max(1, len(chunks) // samples)
    sample_chunks = chunks[::step][:samples]
    vectors = embeddings.embed_texts([chunk["text"] for chunk in sample_chunks])

    found: list[dict] = []
    for chunk, vector in zip(sample_chunks, vectors):
        for hit in database.search(vector, k=k):
            found.append(
                {
                    "query_page": chunk["label_start"],
                    "score": hit["score"],
                    "citation": f"{hit['author']}, {hit['title']}, pág. {hit['label_start']}",
                    "text": " ".join(hit["text"].split())[:180],
                }
            )
    return sorted(found, key=lambda item: item["score"], reverse=True)


def probe_book(
    pdf_path: str | Path, *, samples: int = CONNECT_TEXTS, k: int = 3
) -> tuple[list[dict], float | None]:
    """Extrai o candidato e sonda o acervo; devolve (achados, melhor score)."""
    chunks = ingestion.extract_chunks(Path(pdf_path))
    hits = probe_connections(chunks, samples=samples, k=k)
    return hits, (hits[0]["score"] if hits else None)


def connection_verdict(best_score: float | None) -> str:
    """Traduz a melhor similaridade encontrada numa leitura para o usuário."""
    if best_score is None:
        return "sem dados para avaliar"
    if best_score >= CONNECT_STRONG:
        return "conexões fortes: o acervo já conversa com este livro"
    if best_score >= CONNECT_MODERATE:
        return "conexões moderadas: vale ingerir e olhar com atenção"
    return "conexões fracas: provavelmente outro livro da mesma linha rende mais"
