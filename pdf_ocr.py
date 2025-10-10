#!/usr/bin/env python3
# pdf_ocr_batch.py
# Requisitos:
#   pip install pdf2image pytesseract pillow PyPDF2 google-cloud-storage
#   Ubuntu: sudo apt-get install -y tesseract-ocr poppler-utils
# Uso:
#   Local:
#     python pdf_ocr_batch.py --dir ./pdfs/imgs
#     python pdf_ocr_batch.py --dir ./pdfs --patterns escritura "contrato de distribuição" manual
#   GCS:
#     python pdf_ocr_batch.py --dir gs://meu-bucket/minha/pasta \
#         --patterns escritura "contrato de distribuição" manual

import argparse
import os
import re
import sys
from collections import Counter
from typing import Iterable, List, Tuple
from pathlib import Path
from datetime import datetime
import time
from io import BytesIO

import cv2
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
import unicodedata

# ------------- GCS -------------
try:
    from google.cloud import storage
except Exception:  # lib opcional para uso local
    storage = None


def is_gcs_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("gs://")


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    # gs://bucket/prefix/dir -> (bucket, prefix/dir)
    assert uri.startswith("gs://")
    no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")


def gcs_client() -> "storage.Client":
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage não instalado. pip install google-cloud-storage"
        )
    return storage.Client()


def gcs_list_pdfs(dir_uri: str, recursive: bool = True) -> List[str]:
    bucket_name, prefix = parse_gcs_uri(dir_uri)
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    # list_blobs é sempre “recursivo”; simulamos não recursivo filtrando somente nível atual
    blobs = client.list_blobs(bucket, prefix=(prefix + "/" if prefix else ""))
    pdfs = []
    base_depth = 0 if not prefix else prefix.count("/") + 1
    for b in blobs:
        if not b.name.lower().endswith(".pdf"):
            continue
        if not recursive:
            # mantém apenas arquivos exatamente no "nível" do prefixo
            depth = b.name.count("/")
            if depth != base_depth:
                continue
        pdfs.append(f"gs://{bucket_name}/{b.name}")
    return pdfs


def gcs_read_bytes(gs_path: str) -> bytes:
    bucket_name, key = parse_gcs_uri(gs_path)
    client = gcs_client()
    blob = client.bucket(bucket_name).blob(key)
    return blob.download_as_bytes()  # type: ignore[attr-defined]


def gcs_write_text(dir_uri: str, filename: str, text: str) -> str:
    bucket_name, prefix = parse_gcs_uri(dir_uri)
    client = gcs_client()
    out_key = f"{prefix}/{filename}" if prefix else filename
    blob = client.bucket(bucket_name).blob(out_key)
    blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
    return f"gs://{bucket_name}/{out_key}"


# -----------------------------
# Utilidades de texto
# -----------------------------
def normalize_line(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\b([A-Z0-9]{16,})\b", "", s)
    s = re.sub(r"\bP(?:ag\.?|ágina)\s*\d+\s*/\s*\d+\b", "", s, flags=re.I)
    return s.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text, flags=re.UNICODE)


# -----------------------------
# Extração nativa (de bytes)
# -----------------------------
def extract_native_per_page_from_bytes(pdf_bytes: bytes) -> List[str]:
    pages: List[str] = []
    with BytesIO(pdf_bytes) as bio:
        reader = PdfReader(bio)
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
    return pages


# -----------------------------
# Heurísticas
# -----------------------------
def avg_tokens_per_page(pages: Iterable[str]) -> float:
    counts = [len(tokenize(p or "")) for p in pages]
    if not counts:
        return 0.0
    return sum(counts) / max(1, len(counts))


def repetition_coverage(
    pages: List[str],
    min_line_len: int = 6,
    top_k: int = 10,
    repeat_pages_frac: float = 0.6,
) -> float:
    n_pages = max(1, len(pages))
    line_to_pages = Counter()
    total_chars = 0

    for p in pages:
        lines = [normalize_line(l) for l in (p or "").splitlines()]
        lines = [l for l in lines if len(l) >= min_line_len]
        total_chars += sum(len(l) for l in lines)
        uniq = set(lines)
        for l in uniq:
            line_to_pages[l] += 1

    if total_chars == 0 or not line_to_pages:
        return 0.0

    min_pages = max(1, int(repeat_pages_frac * n_pages))
    repeated_lines = [(l, c) for l, c in line_to_pages.items() if c >= min_pages]
    if not repeated_lines:
        return 0.0

    repeated_lines.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    repeated_lines = repeated_lines[:top_k]

    rep_chars = sum(len(l) for l, _ in repeated_lines) * n_pages
    coverage = min(1.0, rep_chars / max(1, total_chars))
    return coverage


_ALLOWED_CHARS_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\s\.,;:!?()\-\/]+")


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    printable = sum(
        1 for ch in s if _ALLOWED_CHARS_RE.fullmatch(ch) or ch in ("\n", " ")
    )
    return printable / len(s)


def _alpha_num_ratio(s: str) -> float:
    if not s:
        return 0.0
    alnum = sum(ch.isalnum() for ch in s)
    return alnum / len(s)


def _mean_word_len(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def _slash_seq_frac(s: str) -> float:
    if not s:
        return 0.0
    m = re.findall(r"(?:\s|^)/(?:\d{1,3})(?=\s|$)", s)
    return min(1.0, len(" ".join(m)) / max(1, len(s)))


def _unique_chars(s: str) -> int:
    return len(set(s))


def text_quality_metrics(pages: List[str]) -> Tuple[float, float, float, float, int]:
    if not pages:
        return 0, 0, 0, 0, 0
    printable, alpha, mean_wlen, slashf = [], [], [], []
    for p in pages:
        p = p or ""
        printable.append(_printable_ratio(p))
        alpha.append(_alpha_num_ratio(p))
        toks = re.findall(r"\w+", p, flags=re.UNICODE)
        mean_wlen.append(_mean_word_len(toks))
        slashf.append(_slash_seq_frac(p))
    unique_total = _unique_chars("".join(pages))
    return (
        sum(printable) / len(printable),
        sum(alpha) / len(alpha),
        sum(mean_wlen) / len(mean_wlen),
        sum(slashf) / len(slashf),
        unique_total,
    )


def should_force_ocr(
    pages: List[str],
    min_tokens: int,
    repeat_threshold: float,
    repeat_pages_frac: float,
) -> Tuple[bool, float, float]:
    avg_tok = avg_tokens_per_page(pages)
    rep_cov = repetition_coverage(
        pages, min_line_len=6, top_k=12, repeat_pages_frac=repeat_pages_frac
    )
    pr, ar, mwl, sfrac, uniq = text_quality_metrics(pages)
    gibberish = pr < 0.85 or ar < 0.60 or mwl < 3.2 or sfrac > 0.02 or uniq < 15
    force = (avg_tok < min_tokens) or (rep_cov >= repeat_threshold) or gibberish
    return force, avg_tok, rep_cov


# -----------------------------
# Pré-processamento e OCR
# -----------------------------
def _deskew(gray: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
    if lines is None:
        return gray
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180.0 / np.pi) - 90.0
        if -max_angle <= angle <= max_angle:
            angles.append(angle)
    if not angles:
        return gray
    median = np.median(angles)
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), median, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _preprocess(img_pil: Image.Image) -> np.ndarray:
    img = np.array(img_pil.convert("L"))
    img = cv2.equalizeHist(img)
    img = _deskew(img)
    img = cv2.fastNlMeansDenoising(img, h=7, templateWindowSize=7, searchWindowSize=21)
    bin_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    blur = cv2.GaussianBlur(bin_img, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(bin_img, 1.5, blur, -0.5, 0)
    kernel = np.ones((1, 1), np.uint8)
    sharp = cv2.dilate(sharp, kernel, iterations=1)
    return sharp


def _choose_psm(text_density: float) -> int:
    if text_density < 120:
        return 3
    if text_density < 300:
        return 4
    return 6


def ocr_all_pages_from_bytes(pdf_bytes: bytes, dpi: int = 400, lang: str = "por+eng") -> str:
    images: List[Image.Image] = convert_from_bytes(
        pdf_bytes, dpi=dpi, fmt="png", thread_count=2
    )
    out_pages: List[str] = []
    for i, img in enumerate(images, start=1):
        proc = _preprocess(img)
        num_labels, _ = cv2.connectedComponents(proc)
        psm = _choose_psm(num_labels)
        config = (
            f"--oem 1 --psm {psm} -l {lang} "
            f"-c preserve_interword_spaces=1 -c tessedit_do_invert=0"
        )
        txt = pytesseract.image_to_string(proc, config=config)
        out_pages.append(f"---- página {i} ----\n{txt.strip()}")
    return "\n\n".join(out_pages).strip()


# -----------------------------
# Pipeline principal (conversão)
# -----------------------------
def load_pdf_bytes(identifier: str) -> bytes:
    if is_gcs_uri(identifier):
        return gcs_read_bytes(identifier)
    # local
    with open(identifier, "rb") as f:
        return f.read()


def extract_text(
    pdf_identifier: str,
    dpi: int,
    lang: str,
    min_tokens: int,
    repeat_th: float,
    repeat_pages_frac: float,
) -> str:
    pdf_bytes = load_pdf_bytes(pdf_identifier)
    native_pages = extract_native_per_page_from_bytes(pdf_bytes)

    force_ocr, avg_tok, rep_cov = should_force_ocr(
        native_pages,
        min_tokens=min_tokens,
        repeat_threshold=repeat_th,
        repeat_pages_frac=repeat_pages_frac,
    )

    fname = os.path.basename(
        parse_gcs_uri(pdf_identifier)[1] if is_gcs_uri(pdf_identifier) else pdf_identifier
    )

    if force_ocr:
        print(
            f"[info] OCR acionado, {fname}: avg_tokens={avg_tok:.1f} "
            f"rep_coverage={rep_cov:.2f}",
            file=sys.stderr,
        )
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        result = []
        for i, img in enumerate(images, start=1):
            gray = img.convert("L")
            bw = gray.point(lambda x: 0 if x < 200 else 255, "1")
            txt = pytesseract.image_to_string(bw, lang=lang).strip()
            result.append(f"---- página {i} ----\n{txt}")
        return "\n\n".join(result).strip()

    sys.stderr.write(
        f"[info] Extração nativa OK, {fname}: avg_tokens={avg_tok:.1f} "
        f"rep_coverage={rep_cov:.2f}\n"
    )

    result = []
    for i, page in enumerate(native_pages, start=1):
        if not page.strip():
            continue
        result.append(f"---- página {i} ----\n{page.strip()}")
    return "\n\n".join(result).strip()


# -----------------------------
# Batch
# -----------------------------
def _strip_accents_lower(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode().lower()


def find_pdfs_by_patterns(
    root_dir: str,
    patterns: List[str],
    recursive: bool = True,
) -> List[str]:
    norm_patterns = [_strip_accents_lower(p) for p in patterns]

    if is_gcs_uri(root_dir):
        candidates = gcs_list_pdfs(root_dir, recursive=recursive)
        hits: List[str] = []
        for uri in candidates:
            # base name do objeto
            name_norm = _strip_accents_lower(Path(uri).stem)
            if any(pat in name_norm for pat in norm_patterns):
                hits.append(uri)
        hits.sort(key=lambda x: _strip_accents_lower(Path(x).name))
        return hits

    # Local
    base = Path(root_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {root_dir}")

    glob_expr = "**/*.pdf" if recursive else "*.pdf"
    hits_local: List[str] = []
    for p in base.glob(glob_expr):
        name_norm = _strip_accents_lower(p.stem)
        if any(pat in name_norm for pat in norm_patterns):
            hits_local.append(str(p))
    hits_local.sort(key=lambda x: _strip_accents_lower(Path(x).name))
    return hits_local


def header_name(identifier: str) -> str:
    if is_gcs_uri(identifier):
        _, key = parse_gcs_uri(identifier)
        return key.split("/")[-1]
    return Path(identifier).name


def concat_many_pdfs_to_text(
    pdf_identifiers: List[str],
    dpi: int,
    lang: str,
    min_tokens: int,
    repeat_th: float,
    repeat_pages_frac: float,
) -> str:
    parts: List[str] = []
    for ident in pdf_identifiers:
        try:
            txt = extract_text(
                pdf_identifier=ident,
                dpi=dpi,
                lang=lang,
                min_tokens=min_tokens,
                repeat_th=repeat_th,
                repeat_pages_frac=repeat_pages_frac,
            )
        except Exception as exc:
            sys.stderr.write(f"[erro] {ident}: {exc}\n")
            continue

        header = f"---- {header_name(ident)} ----"
        parts.append(f"{header}\n{txt.strip()}")

    return "\n\n".join(parts).strip()


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai texto de vários PDFs encontrados por padrão no nome e "
            "concatena em um único TXT. Se densidade baixa ou repetição alta, "
            "força OCR. Suporta diretório local ou gs://bucket/prefix."
        )
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Pasta onde buscar PDFs. Ex.: ./pdfs ou gs://bucket/prefix",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["escritura", "contrato de distribuição", "manual"],
        help=("Padrões no nome do arquivo (case/acento-insensitive). "
              "Ex.: --patterns escritura 'contrato de distribuição' manual"),
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Não descer em subpastas.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI para OCR.")
    parser.add_argument("--lang", default="por+eng", help="Idiomas Tesseract.")
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--repeat-th", type=float, default=0.30)
    parser.add_argument("--repeat-pages", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()

    try:
        pdfs = find_pdfs_by_patterns(
            root_dir=args.dir,
            patterns=args.patterns,
            recursive=(not args.non_recursive),
        )
        if not pdfs:
            print("[info] Nenhum PDF correspondente encontrado.", file=sys.stderr)
            sys.exit(2)

        text = concat_many_pdfs_to_text(
            pdf_identifiers=pdfs,
            dpi=args.dpi,
            lang=args.lang,
            min_tokens=args.min_tokens,
            repeat_th=args.repeat_th,
            repeat_pages_frac=args.repeat_pages,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # Nome padronizado e grava na mesma origem (local ou GCS)
    elapsed = time.perf_counter() - start
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    elapsed_str = f"{elapsed:.2f}s"

    out_name = f"concat-text-{date_str}-{time_str}-{elapsed_str}.txt"

    if is_gcs_uri(args.dir):
        out_uri = gcs_write_text(args.dir, out_name, text)
        sys.stderr.write(f"[info] Saída gravada em: {out_uri}\n")
    else:
        out_path = Path(args.dir) / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        sys.stderr.write(f"[info] Saída gravada em: {out_path}\n")


if __name__ == "__main__":
    main()
