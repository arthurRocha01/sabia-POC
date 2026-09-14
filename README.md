# Sabiá

**Conexões entre livros.** Você lê um trecho de um livro e o Sabiá mostra o que **outros autores** dizem sobre a mesma ideia — com a relação interpretada (complemento, contradição, nuance) e a fonte exata.

> **Aviso:** isto é um **POC** (prova de conceito) para validar a ideia. Não é o produto final.

## Como funciona

1. Livros em PDF são ingeridos e divididos em trechos (chunks) que sabem de quais páginas vieram.
2. Cada trecho vira um vetor (Google Gemini) e é guardado num banco vetorial local (ChromaDB).
3. O texto consultado é vetorizado com o mesmo modelo e comparado aos trechos dos outros livros.
4. Um LLM recebe o texto + os trechos encontrados e escreve o card: síntese, classificação da relação e fontes.

## Requisitos

- Python 3.14
- Chave do **Google Gemini** (embeddings): https://aistudio.google.com/apikey
- Chave do **DeepSeek** (card de insights): https://platform.deepseek.com

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # preencha GOOGLE_API_KEY e DEEPSEEK_API_KEY
```

## Uso

```bash
# lista o acervo
.venv/bin/python cli.py listar

# ingere um livro
.venv/bin/python cli.py add resources/livro.pdf --titulo "Título" --autor "Autor"

# busca conexões em outros livros e gera o card
.venv/bin/python cli.py conectar "trecho do livro que estou lendo" --de as-48-leis-do-poder

# avalia um PDF antes de ingerir (não gasta cota sem --deep)
.venv/bin/python cli.py test resources/livro.pdf --titulo "Título" --autor "Autor"
```

O `--de` indica o livro de onde o texto saiu: ele é excluído da busca, para a conexão vir de outro autor. Cada comando tem ajuda: `.venv/bin/python cli.py <comando> --help`.

## Testes

```bash
for t in database retrieval ingestion embeddings interpretation preflight cli; do
  .venv/bin/python tests/test_$t.py
done
```

Os testes rodam offline — não precisam de chave nem de rede.
