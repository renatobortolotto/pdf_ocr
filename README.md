# pdf_ocr / txt_to_api

Ferramentas simples para enviar texto extraído de PDFs (ou arquivos `.txt` já existentes) para uma API HTTP e salvar o resultado de volta localmente ou no Google Cloud Storage (GCS).

O script principal é `txt_to_api.py`.

## O que o `txt_to_api.py` faz

1. Lê um arquivo `.txt` a partir de:
   - Caminho local, ou
   - URI do GCS (`gs://bucket/prefix/arquivo.txt`).
   - Se você passar uma pasta (local) ou um prefixo (GCS), ele resolve automaticamente o arquivo `.txt` mais recente.
2. Envia o conteúdo como JSON para um endpoint HTTP (ex.: uma API FastAPI) no formato `{ "prompt": "<conteúdo do txt>" }`.
3. Processa a resposta:
   - Tenta parsear JSON diretamente.
   - Se a resposta trouxer um bloco de código Markdown com JSON (```json { ... } ```), extrai e normaliza como `data`.
   - Caso contrário, normaliza o texto retornado, removendo marcações básicas de Markdown.
4. Salva o resultado:
   - Preferencialmente salva apenas o campo `data` (se existir). Caso contrário, salva o payload completo.
   - O destino pode ser uma pasta local ou um prefixo GCS.

## Requisitos

- Python 3.9+
- Dependências (ver `requirements.txt`):
  - `requests`
  - `google-cloud-storage` (opcional, apenas se for usar GCS)

## Instalação

Crie e ative um ambiente virtual e instale dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Autenticação GCP (se usar GCS)

Para ler/gravar no GCS, configure a credencial da conta de serviço de uma das formas:

- Defina a variável `GOOGLE_APPLICATION_CREDENTIALS` apontando para o JSON da credencial:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/caminho/para/sua-credencial.json"
```

- Ou autentique-se com o `gcloud` (para ambientes interativos):

```bash
gcloud auth application-default login
```

## Uso

Ajuda da linha de comando:

```bash
python txt_to_api.py --help
```

Parâmetros principais:
- `--src`: Caminho do `.txt` local ou `gs://bucket/prefix` ou `gs://bucket/prefix/arquivo.txt`.
- `--api-url`: Base da API (ex.: `http://localhost:8000/api`).
- `--endpoint`: Endpoint POST (padrão: `/extrator_dados_debentures`).
- `--out`: Destino onde salvar o retorno (pasta local ou `gs://bucket/prefix`).
- `--timeout`: Timeout da requisição (padrão 60s).
- `--retries`: Tentativas em caso de erro de rede (padrão 3).
- `--auth-header`: Valor para header `Authorization` (ex.: `Bearer <token>`), opcional.

### Exemplos

1) Ler `.txt` local e salvar localmente:

```bash
python txt_to_api.py \
  --src ./texto-1.txt \
  --api-url http://localhost:8000/api \
  --endpoint /extrator_dados_debentures \
  --out ./
```

2) Resolver `.txt` mais recente em uma pasta local e salvar em `./results`:

```bash
python txt_to_api.py \
  --src ./pdfs/imgs \
  --api-url http://localhost:8000/api \
  --out ./results
```

3) Ler `.txt` do GCS (específico) e salvar resultado no GCS:

```bash
python txt_to_api.py \
  --src gs://meu-bucket/prefixo/arquivo.txt \
  --api-url https://minha-api.exemplo.com/api \
  --out gs://meu-bucket/resultados
```

4) Passar um prefixo GCS e deixar o script escolher o `.txt` mais recente:

```bash
python txt_to_api.py \
  --src gs://meu-bucket/prefixo \
  --api-url https://minha-api.exemplo.com/api \
  --out gs://meu-bucket/resultados
```

5) Usando autenticação Bearer:

```bash
python txt_to_api.py \
  --src ./texto-1.txt \
  --api-url https://minha-api.exemplo.com/api \
  --auth-header "Bearer SEU_TOKEN" \
  --out ./
```

## Como funciona internamente (resumo)

- Funções utilitárias para GCS (`gs://`) permitem ler textos, listar `.txt` por prefixo e gravar JSON.
- O script tenta sempre resolver a origem (`--src`) para um único `.txt`:
  - Se for pasta local/prefixo GCS, ele encontra todos os `.txt` e escolhe o mais recente.
- A requisição HTTP é feita com `requests` e tem retentativas com backoff exponencial simples.
- A resposta é analisada:
  - Se houver JSON dentro de bloco de código Markdown (```json ... ```), ele vira `data`.
  - Caso contrário, o texto é "sanitizado" para remover marcações básicas.
- A saída é gravada como `payload-<base>-<timestamp>.json` (ou apenas o `data` quando presente).

## Dicas e resolução de problemas

- Erros de autenticação GCS: garanta que o `google-cloud-storage` esteja instalado e que as credenciais estejam configuradas.
- Falhas HTTP: use `--retries` e ajuste `--timeout`. Verifique `--api-url` e `--endpoint`.
- Conteúdo muito grande: verifique limites do endpoint e do servidor.

## Licença

Uso interno. Ajuste conforme necessário para seu projeto.

## pdf_ocr.py — extração e OCR de PDFs

O `pdf_ocr.py` busca PDFs por padrão no nome do arquivo, extrai texto de cada um e concatena tudo em um único `.txt`. Ele tenta primeiro a extração nativa do PDF e, quando identifica baixa densidade de texto, muita repetição ou "ruído", força o OCR com Tesseract.

### Quando ele força OCR?

O script calcula alguns indicadores por página e decide automaticamente:
- Poucos tokens por página (ex.: formulários digitalizados) → força OCR
- Cobertura alta de linhas repetidas (headers/rodapés se repetem muito) → força OCR
- Métricas de qualidade (impressão/alfanumérico/média de palavra) indicando "gibberish" → força OCR

### Dependências Python

Estão no `requirements.txt`. Principais bibliotecas usadas:
- PyPDF2, pdf2image, pillow
- pytesseract
- opencv-python-headless, numpy
- google-cloud-storage (apenas se for usar GCS)

### Pré-requisitos de sistema

Em Ubuntu/Debian, instale os utilitários necessários:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

Observações:
- `pdf2image` utiliza o Poppler (fornecido por `poppler-utils`).
- `pytesseract` requer o binário do Tesseract (`tesseract-ocr`).

### Uso básico

Local (busca PDFs em uma pasta e concatena o texto extraído):

```bash
python pdf_ocr.py \
  --dir ./pdfs \
  --patterns escritura "contrato de distribuição" manual
```

GCS (busca PDFs em um prefixo gs:// e escreve a saída no mesmo local):

```bash
python pdf_ocr.py \
  --dir gs://meu-bucket/minha/pasta \
  --patterns escritura "contrato de distribuição" manual
```

Parâmetros principais:
- `--dir`: pasta local ou prefixo `gs://bucket/prefix` onde procurar PDFs
- `--patterns`: lista de padrões presentes no nome do arquivo (case/acento-insensitive)
- `--non-recursive`: não desce em subpastas
- `--dpi`: DPI usado quando OCR é necessário (padrão 300)
- `--lang`: idiomas do Tesseract (padrão `por+eng`)
- `--min-tokens`, `--repeat-th`, `--repeat-pages`: controles das heurísticas de decisão de OCR

### Saída

O resultado é salvo como um único `.txt` no mesmo destino de `--dir` (local ou GCS), seguindo o padrão:

```
concat-text-YYYYMMDD-HHMMSS-<elapsed>s.txt
```

O conteúdo concatena blocos de cada PDF:

```
---- nome_arquivo.pdf ----
---- página 1 ----
<texto extraído>

---- página 2 ----
<texto extraído>
```

### Dicas

- Se o OCR ficar lento, reduza `--dpi` (ex.: 250–300) ou ajuste idiomas em `--lang`.
- Para depurar decisões de OCR, observe as mensagens `[info]` no stderr (densidade e repetição).
- Para GCS, configure as credenciais como descrito acima.
