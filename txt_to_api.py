#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Tuple, List
import re

import requests

try:
    from google.cloud import storage
except Exception:
    storage = None

# ---------------- GCS utils ----------------
def is_gcs_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("gs://")

def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key

def gcs_client() -> "storage.Client":
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage não instalado. "
            "Instale com: pip install google-cloud-storage"
        )
    return storage.Client()

def gcs_read_text(gs_path: str, encoding: str = "utf-8") -> str:
    bucket_name, key = parse_gcs_uri(gs_path)
    client = gcs_client()
    blob = client.bucket(bucket_name).blob(key)
    return blob.download_as_text(encoding=encoding)

def gcs_write_json(gs_dir: str, filename: str, payload: dict) -> str:
    bucket, prefix = parse_gcs_uri(gs_dir)
    key = f"{prefix.rstrip('/')}/{filename}" if prefix else filename
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    client = gcs_client()
    blob = client.bucket(bucket).blob(key)
    blob.upload_from_string(data, content_type="application/json; charset=utf-8")
    return f"gs://{bucket}/{key}"

# --------------- IO genéricos ---------------
def read_txt(src: str, encoding: str = "utf-8") -> str:
    if is_gcs_uri(src):
        return gcs_read_text(src, encoding=encoding)
    return Path(src).read_text(encoding=encoding)

def write_payload(out_dir: str, base_name: str, payload: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = f"payload-{base_name}-{ts}.json"
    if is_gcs_uri(out_dir):
        return gcs_write_json(out_dir, out_name, payload)
    out_path = Path(out_dir) / out_name
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)

# --------------- HTTP ---------------
def post_with_retries(url: str, json_body: dict, headers: dict, timeout: float, retries: int) -> requests.Response:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=json_body, headers=headers, timeout=timeout)
            return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    if last_exc:
        raise last_exc
    raise RuntimeError("Falha HTTP desconhecida")

# --------------- CLI ---------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lê um .txt, envia para FastAPI como {'texto': ...} e grava o retorno em um bucket/prefixo do GCS."
    )
    p.add_argument("--src", required=True, help="Caminho do .txt (local ou gs://bucket/prefix/arquivo.txt).")
    p.add_argument("--api-url", required=True, help="Base da API. Ex.: http://host:8000/api")
    p.add_argument("--endpoint", default="/extrator_dados_debentures", help="Endpoint POST (padrão: /extrator_dados_debentures)")
    p.add_argument("--out", required=True, help="Destino do payload (pasta local ou gs://bucket/prefix)")
    p.add_argument("--timeout", type=float, default=60.0, help="Timeout da requisição em segundos.")
    p.add_argument("--retries", type=int, default=3, help="Qtd de tentativas em caso de erro de rede.")
    p.add_argument("--auth-header", default=None, help="Valor opcional para Authorization. Ex.: 'Bearer <token>'")
    return p.parse_args()

def gcs_list_txts(prefix_uri: str, recursive: bool = True) -> List[str]:
    """Lista URIs gs://.../*.txt dentro do prefixo."""
    bucket, prefix = parse_gcs_uri(prefix_uri)
    client = gcs_client()
    it = client.list_blobs(bucket, prefix=(prefix.rstrip('/') + '/' if prefix else ''))
    txts = []
    base_depth = 0 if not prefix else prefix.count('/') + 1
    for b in it:
        if not b.name.lower().endswith('.txt'):
            continue
        if not recursive and b.name.count('/') != base_depth:
            continue
        txts.append((b.updated, f"gs://{bucket}/{b.name}"))
    # ordena por data desc
    txts.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in txts]

def local_list_txts(dir_path: str, recursive: bool = True) -> List[str]:
    p = Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {dir_path}")
    if p.is_file() and p.suffix.lower() == '.txt':
        return [str(p)]
    pattern = '**/*.txt' if recursive else '*.txt'
    files = [str(x) for x in p.glob(pattern)]
    files.sort(key=lambda s: Path(s).stat().st_mtime, reverse=True)
    return files

def resolve_single_txt(src: str, recursive: bool = True) -> str:
    """
    Se src é arquivo .txt -> retorna.
    Se src é prefixo/pasta -> encontra exatamente 1 .txt.
    Se >1, pega o mais recente.
    """
    if is_gcs_uri(src):
        if src.lower().endswith('.txt'):
            return src
        txts = gcs_list_txts(src, recursive=recursive)
        if not txts:
            raise FileNotFoundError(f"Nenhum .txt em {src}")
        if len(txts) > 1:
            sys.stderr.write(f"[warn] {len(txts)} .txt encontrados em {src}. Usando o mais recente.\n")
        return txts[0]
    else:
        p = Path(src)
        if p.is_file() and p.suffix.lower() == '.txt':
            return str(p)
        txts = local_list_txts(src, recursive=recursive)
        if not txts:
            raise FileNotFoundError(f"Nenhum .txt em {src}")
        if len(txts) > 1:
            sys.stderr.write(f"[warn] {len(txts)} .txt encontrados em {src}. Usando o mais recente.\n")
        return txts[0]

def gcs_read_text(gs_path: str, encoding: str = "utf-8") -> str:
    bucket_name, key = parse_gcs_uri(gs_path)
    client = gcs_client()
    blob = client.bucket(bucket_name).blob(key)
    return blob.download_as_text(encoding=encoding)

def read_txt(src: str, encoding: str = "utf-8", recursive: bool = True) -> str:
    """Resolve pasta/prefixo para 1 arquivo .txt e lê o conteúdo."""
    real_txt = resolve_single_txt(src, recursive=recursive)
    if is_gcs_uri(real_txt):
        return gcs_read_text(real_txt, encoding=encoding)
    return Path(real_txt).read_text(encoding=encoding)

def extract_json_from_text(s: str):
    """Localiza bloco ```json ...``` ou ``` ...``` e retorna dict se parsear."""
    if not isinstance(s, str):
        return None
    s = s.replace("\r\n", "\n")
    for pat in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        m = re.search(pat, s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    # fallback: se começa com "json {"
    m = re.search(r"\bjson\s*(\{.*\})\s*$", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None

def sanitize_text(s: str) -> str:
    """Remove Markdown simples e normaliza espaços."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remove fences e o literal 'json' que sobra
    s = re.sub(r"```+json", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```+", "", s)
    # negrito/itálico/crases
    s = s.replace("**", "").replace("__", "")
    s = re.sub(r"`+", "", s)
    # bullets e numeração de início de linha
    s = re.sub(r"(?m)^\s*([*\-–•]|\d+\.)\s+", "", s)
    # cabeçalhos
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    # colapsa linhas
    s = re.sub(r"\n{2,}", "\n", s).replace("\n", " ")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def main() -> None:
    args = parse_args()

    # 1) Lê o texto
    txt = read_txt(args.src, recursive=True) 


    # 2) Monta URL e headers
    base = args.api_url.rstrip("/")
    ep = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    url = f"{base}{ep}"
    headers = {"Content-Type": "application/json"}
    if args.auth_header:
        headers["Authorization"] = args.auth_header

    # 3) POST
    body = {"prompt": txt}
    resp = post_with_retries(url, body, headers, timeout=args.timeout, retries=args.retries)

    # 4) Normaliza resposta
    try:
        payload = resp.json()
    except ValueError:
        payload = {"status_code": resp.status_code, "text": resp.text}

    # Preferir JSON estruturado se vier em bloco de código
    if isinstance(payload, dict) and "text" in payload:
        raw = payload["text"]
        parsed = extract_json_from_text(raw)
        if parsed is not None:
            payload["data"] = parsed
            payload["text_clean"] = json.dumps(parsed, ensure_ascii=False, indent=2)
        else:
            payload["text_clean"] = sanitize_text(raw)
    elif not isinstance(payload, dict):
        payload = {"status_code": resp.status_code, "text": str(payload)}

    # === GRAVA APENAS "data" SE EXISTIR ===
    to_save = payload["data"] if isinstance(payload, dict) and "data" in payload else payload

    base_name = Path(args.src).name if not is_gcs_uri(args.src) else Path(parse_gcs_uri(args.src)[1]).name
    out_path = write_payload(args.out, Path(base_name).stem, to_save)
    sys.stderr.write(f"[info] POST {url} -> {resp.status_code}\n")
    sys.stderr.write(f"[info] Payload gravado em: {out_path}\n")

if __name__ == "__main__":
    main()