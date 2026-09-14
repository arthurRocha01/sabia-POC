# Sabiá

**Conexões entre livros.** Você lê um trecho de um livro e o Sabiá mostra o que **outros autores** dizem sobre a mesma ideia — com a relação interpretada (complemento, contradição, nuance) e a fonte exata.

> **Aviso:** isto é um **POC** (prova de conceito) para validar a ideia. Não é o produto final.

## Como funciona

1. Livros em PDF são ingeridos e divididos em trechos (chunks) que sabem de quais páginas vieram.
2. Cada trecho vira um vetor e é guardado num banco vetorial local (ChromaDB).
3. O texto consultado é vetorizado com o mesmo modelo e comparado aos trechos dos outros livros.
4. Um LLM recebe o texto + os trechos encontrados e escreve o card: síntese, classificação da relação e fontes.

## Modelos

As duas funções abaixo são independentes de provedor: qualquer modelo compatível pode ser utilizado, e cada papel pode ser atendido por um provedor diferente. Os modelos indicados são os utilizados atualmente, como sugestão de partida.

| Função | Responsabilidade | Sugerido (em uso) | Configuração |
| --- | --- | --- | --- |
| **Embeddings** | Conversão de trechos e consultas em vetores, viabilizando a busca por similaridade semântica | `gemini-embedding-001` (Google Gemini) | `EMBEDDING_MODEL` |
| **Interpretação** | Geração do card de insights a partir do texto consultado e dos trechos recuperados | `deepseek-v4-flash` (DeepSeek) | `LLM_PROVIDER`, `LLM_MODEL` |

Considerações de configuração:

- Ambos os papéis são obrigatórios. A ausência do modelo de embeddings inviabiliza a indexação e a busca; a ausência do modelo de interpretação inviabiliza a geração do card.
- A substituição do modelo de embeddings requer a reingestão do acervo, uma vez que vetores produzidos por modelos distintos não são comparáveis entre si. A substituição do modelo de interpretação não requer reingestão.
- Os dois papéis podem usar provedores distintos (é a configuração atual). Como alternativa, `LLM_PROVIDER=gemini` faz a geração do card usar o `gemini-3.6-flash`, com a mesma credencial do Google.
- O cliente de interpretação fala a API compatível com a OpenAI, com endereço configurável em `DEEPSEEK_BASE_URL` — o que permite apontar o card para outros provedores compatíveis, sem alteração de código.

## Configuração

1. **Chaves (obrigatórias)** — o POC usa uma chave para os embeddings e uma para o card:
   - `GOOGLE_API_KEY` — embeddings (e o card, se `LLM_PROVIDER=gemini`): https://aistudio.google.com/apikey
   - `DEEPSEEK_API_KEY` — card de insights: https://platform.deepseek.com

2. **Criar o `.env`:**

```bash
cp .env.example .env      # e preencha as chaves acima
```

3. **Ajustes opcionais** (todos no `.env`; o `.env.example` comenta cada um):

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | modelo de embeddings |
| `LLM_PROVIDER` / `LLM_MODEL` | `deepseek` / `deepseek-v4-flash` | provedor e modelo do card |
| `EMBEDDING_BATCH_CHARS` | `16000` | tamanho do lote enviado à API de embeddings (~4k tokens) |
| `EMBEDDING_BATCH_DELAY` | `10` | pausa entre lotes, em segundos (respeita o limite por minuto) |
| `LLM_TIMEOUT` | `60` | tempo máximo de uma chamada do card, em segundos |
| `LLM_MAX_ATTEMPTS` / `LLM_RETRY_SLEEP` | `4` / `10` | retentativas em erro temporário do provedor |

> No free tier do Gemini, mantenha os valores padrão de lote e pausa: a cota é de **1000 textos/dia** e um livro deste porte consome 250 a 600.

## Uso

```bash
# 0) instalar (uma vez)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1) avaliar o PDF ANTES de ingerir (roda local, não gasta cota)
.venv/bin/python cli.py test resources/livro.pdf --titulo "Título" --autor "Autor"

# 2) ingerir o livro (grava os vetores e a ficha no catálogo)
.venv/bin/python cli.py add resources/livro.pdf --titulo "Título" --autor "Autor"

# 3) ver o acervo (id, chunks, título e autor)
.venv/bin/python cli.py listar

# 4) pedir conexões
.venv/bin/python cli.py conectar "trecho do livro que estou lendo" --de <livro_id>
.venv/bin/python cli.py conectar "o que é a virtù?" --k 5
```

**Como ler os comandos:**

- `--de <livro_id>` — informe o livro de onde o trecho saiu (o que você está lendo). Ele é excluído da busca, para a conexão vir de **outro autor**; sem isso, o primeiro resultado é o eco do próprio texto. Em pergunta livre, não use.
- `--k N` — quantas conexões trazer (padrão 3).
- `test --deep N` — além das checagens locais, sonda conexões reais no acervo e gasta N textos de cota.
- Ajuda de qualquer comando: `.venv/bin/python cli.py <comando> --help`.

## Testes

```bash
for t in database retrieval ingestion embeddings interpretation preflight cli; do
  .venv/bin/python tests/test_$t.py
done
```

Os testes rodam offline — não precisam de chave nem de rede.
