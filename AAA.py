# ===============================
# 📄 Analizador de PQR (Triple A) — Streamlit Cloud
# ===============================

import io
import os
import re
import zipfile
import base64
import json
import tempfile
from typing import List, Dict

import streamlit as st
import pandas as pd
from PIL import Image
import pypdfium2 as pdfium

# ======================
# Configuración Streamlit
# ======================
st.set_page_config(page_title="📄 Analizador PQR – Triple A", layout="wide")
st.title("📄 Analizador de Derechos de Petición (PQR) – Triple A ⚖️")
st.caption("Carga PDFs o un ZIP con varios clientes. El sistema convierte a imágenes y extrae datos + pretensiones exactas.")

# ======================
# OpenAI Client (Visión)
# ======================
try:
    from openai import OpenAI
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    oai_client = OpenAI()
    OAI_READY = True
except Exception as e:
    OAI_READY = False
    st.warning("⚠️ OpenAI SDK no disponible. Verifica requirements y secrets.")

# ======================
# Utilidades
# ======================
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 180) -> List[Image.Image]:
    """Convierte bytes de PDF a lista de PIL Images (todas las páginas)."""
    images: List[Image.Image] = []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        pdf = pdfium.PdfDocument(tmp.name)
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=dpi/72).to_pil()
            images.append(bitmap.convert("RGB"))
    return images

def image_to_data_url(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convierte PIL Image a data URL base64 (para enviar a Visión)."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"

PROMPT_CORE = (
    "Eres analista de PQR de Triple A (Barranquilla). "
    "Analiza SOLO el cuerpo del derecho de petición (ignora anexos, facturas, sellos o comprobantes).\n\n"
    "EXTRAE EXCLUSIVAMENTE si aparecen CLAROS y LITERALES estos campos (NO inventes nada):\n"
    "- CEDULA\n- NOMBRE COMPLETO\n- CORREO ELECTRÓNICO\n- TELÉFONO\n- MOTIVO_PQR (puede estar en 'Referencia', 'Ref.', 'Asunto' o en la introducción)\n"
    "- QUIEN_PRESENTA (peticionario principal, no el apoderado)\n- NOTIFICACION_A (dirección/lugar de notificación; puede aparecer como 'Notificaciones', 'Citación', 'Correspondencia')\n"
    "- RESUMEN_PQR (máx. 5 líneas, tomando frases del documento; sin inventar)\n\n"
    "PRETENSIONES (clave):\n"
    "- Captura TODAS las solicitudes/pretensiones/exigencias.\n"
    "- Incluye: títulos como PETICIONES, PRETENSIONES, SOLICITUDES, EXIGENCIAS, REQUERIMIENTOS, DEMANDAS.\n"
    "- Si están enumeradas (1., 2., 3. o viñetas), copia CADA ítem como pretensión independiente.\n"
    "- Además, detecta frases en párrafos con verbos 'Solicito', 'Pido', 'Requiero', 'Exijo', 'Se sirvan', 'Mientras tanto', etc.\n"
    "- NO resumas ni fusiones: cada frase = 1 pretensión.\n"
    "- Devuelve SIEMPRE: 'PRETENSIONES': {'TOTAL': N, 'DETALLE': ['1. ...', '2. ...', ...]}.\n\n"
    "REGLAS DE SALIDA:\n"
    "- Si un campo NO aparece o NO es claro, escribe exactamente: 'NO SE APORTÓ – VALIDAR MANUALMENTE'.\n"
    "- Devuelve SOLO un JSON válido con estas 9 claves fijas: \n"
    "  ['CEDULA','NOMBRE','CORREO','TELEFONO','MOTIVO_PQR','QUIEN_PRESENTA','NOTIFICACION_A','RESUMEN_PQR','PRETENSIONES']\n"
)

def call_vision_on_images(imgs: List[Image.Image], model: str = "gpt-4o-mini") -> Dict:
    """Envía todas las páginas de un cliente al modelo de visión con prompt blindado."""
    if not OAI_READY:
        return _manual_review_payload()

    content_blocks = [{"type": "text", "text": PROMPT_CORE}]
    for im in imgs:
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(im)}
        })

    try:
        resp = oai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0,
        )
        raw = resp.choices[0].message.content
        data = _safe_json_loads(raw)
        if not _valid_payload(data):
            return _manual_review_payload()
        return _normalize_payload(data)
    except Exception:
        return _manual_review_payload()

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        s2 = s.strip()
        if s2.startswith("```") and s2.endswith("```"):
            s2 = "\n".join(s2.splitlines()[1:-1])
        try:
            return json.loads(s2)
        except Exception:
            return None

def _valid_payload(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = [
        "CEDULA","NOMBRE","CORREO","TELEFONO","MOTIVO_PQR",
        "QUIEN_PRESENTA","NOTIFICACION_A","RESUMEN_PQR","PRETENSIONES"
    ]
    for k in keys:
        if k not in obj:
            return False
    if not isinstance(obj.get("PRETENSIONES", {}), dict):
        return False
    return True

def _manual_review_payload() -> Dict:
    return {
        "CEDULA": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "NOMBRE": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "CORREO": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "TELEFONO": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "MOTIVO_PQR": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "QUIEN_PRESENTA": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "NOTIFICACION_A": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "RESUMEN_PQR": "NO SE APORTÓ – VALIDAR MANUALMENTE",
        "PRETENSIONES": {"TOTAL": 0, "DETALLE": ["NO SE APORTÓ – VALIDAR MANUALMENTE"]},
    }

def _normalize_payload(d: Dict) -> Dict:
    def nz(v):
        v = (v or "").strip() if isinstance(v, str) else v
        return v if v else "NO SE APORTÓ – VALIDAR MANUALMENTE"

    pret = d.get("PRETENSIONES", {}) or {}
    detalle = pret.get("DETALLE", []) or []
    detalle_norm = []
    for i, item in enumerate(detalle, start=1):
        if isinstance(item, dict):
            item = item.get("texto") or item.get("detalle") or json.dumps(item, ensure_ascii=False)
        s = str(item).strip()
        if not s:
            s = "NO SE APORTÓ – VALIDAR MANUALMENTE"
        if not re.match(r"^\d+\.\s", s):
            s = f"{i}. {s}"
        detalle_norm.append(s)
    total = len(detalle_norm)

    return {
        "CEDULA": nz(d.get("CEDULA")),
        "NOMBRE": nz(d.get("NOMBRE")),
        "CORREO": nz(d.get("CORREO")),
        "TELEFONO": nz(d.get("TELEFONO")),
        "MOTIVO_PQR": nz(d.get("MOTIVO_PQR")),
        "QUIEN_PRESENTA": nz(d.get("QUIEN_PRESENTA")),
        "NOTIFICACION_A": nz(d.get("NOTIFICACION_A")),
        "RESUMEN_PQR": nz(d.get("RESUMEN_PQR")),
        "PRETENSIONES": {"TOTAL": total, "DETALLE": detalle_norm},
    }

# ======================
# Ingesta de archivos
# ======================
mode = st.radio("Modo de carga", ["Subir PDFs/Imágenes", "Subir ZIP"], horizontal=True)

client_docs: Dict[str, List[Image.Image]] = {}

if mode == "Subir ZIP":
    up = st.file_uploader("Sube un ZIP con varios PDFs/imagenes (cada PDF = un cliente)", type=["zip"])
    if up is not None:
        with zipfile.ZipFile(up) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                with zf.open(info) as f:
                    data = f.read()
                ext = os.path.splitext(name)[1].lower()
                base = os.path.splitext(os.path.basename(name))[0]
                if ext in [".pdf"]:
                    imgs = pdf_bytes_to_images(data)
                    client_docs[base] = client_docs.get(base, []) + imgs
                elif ext in [".jpg", ".jpeg", ".png"]:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    group = base.split("_")[0]
                    client_docs[group] = client_docs.get(group, []) + [img]

else:
    ups = st.file_uploader(
        "Sube 1..N PDFs o imágenes (PNG/JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    if ups:
        for up in ups:
            ext = os.path.splitext(up.name)[1].lower()
            base = os.path.splitext(os.path.basename(up.name))[0]
            if ext == ".pdf":
                imgs = pdf_bytes_to_images(up.read())
                client_docs[base] = client_docs.get(base, []) + imgs
            else:
                img = Image.open(up).convert("RGB")
                group = base.split("_")[0]
                client_docs[group] = client_docs.get(group, []) + [img]

# ======================
# Procesamiento
# ======================
if client_docs:
    st.subheader("📦 Clientes detectados")
    st.write(f"Total clientes: **{len(client_docs)}**")

    model_name = st.selectbox("Modelo de visión", ["gpt-4o-mini", "gpt-4o"], index=0)
    run = st.button("🚀 Procesar todo (extraer datos + pretensiones)")

    if run:
        rows = []
        qc_rows = []

        progress = st.progress(0)
        for idx, (client_key, pages_imgs) in enumerate(client_docs.items(), start=1):
            result = call_vision_on_images(pages_imgs, model=model_name)

            pret = result.get("PRETENSIONES", {})
            total = pret.get("TOTAL", 0)
            detalle_str = "\n".join(pret.get("DETALLE", []))

            row = {
                "CLIENTE_ID": client_key,
                "CEDULA": result.get("CEDULA"),
                "NOMBRE": result.get("NOMBRE"),
                "CORREO": result.get("CORREO"),
                "TELEFONO": result.get("TELEFONO"),
                "MOTIVO_PQR": result.get("MOTIVO_PQR"),
                "QUIEN_PRESENTA": result.get("QUIEN_PRESENTA"),
                "NOTIFICACION_A": result.get("NOTIFICACION_A"),
                "RESUMEN_PQR": result.get("RESUMEN_PQR"),
                "PRETENSIONES_TOTAL": total,
                "PRETENSIONES_DETALLE": detalle_str,
            }
            rows.append(row)

            qc_rows.append({
                "CLIENTE_ID": client_key,
                "PRETENSIONES_TOTAL": total,
                "ALERTA": "⚠️ REVISAR MANUALMENTE" if total == 0 else "OK",
            })

            progress.progress(min(idx/len(client_docs), 1.0))

        df = pd.DataFrame(rows)
        qc = pd.DataFrame(qc_rows)

        st.success("✔️ Procesamiento completado.")
        st.subheader("📊 Resultado consolidado")
        st.dataframe(df, use_container_width=True)

        st.subheader("🧪 Control de calidad (conteo de pretensiones)")
        st.dataframe(qc, use_container_width=True)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="PQR", index=False)
            qc.to_excel(writer, sheet_name="QC_Pretensiones", index=False)
        st.download_button(
            "📥 Descargar Excel consolidado",
            data=out.getvalue(),
            file_name="pqr_consolidado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Sube PDFs, imágenes o un ZIP para comenzar.")
