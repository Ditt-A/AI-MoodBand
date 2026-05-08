import os, sys, json, datetime as dt, traceback
from dotenv import load_dotenv

try:
    from google import genai
    _GENAI_IMPORT_ERROR = None
except Exception as e:
    genai = None
    _GENAI_IMPORT_ERROR = e

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "API.env")
load_dotenv(dotenv_path=ENV_PATH)

def _read_api_key(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    if not value or value == "-":
        return None
    return value

API_KEY = _read_api_key("API_KEY_SUMMARIZER")

MODEL = "gemini-2.5-flash"
client = None
if API_KEY and genai is not None:
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

def _summarizer_unavailable_reason() -> str | None:
    if genai is None:
        base = (
            "Library 'google-genai' tidak bisa diimport pada interpreter ini. "
            f"Python aktif: {sys.executable}. "
            "Aktifkan virtualenv proyek lalu install: python -m pip install google-genai."
        )
        if _GENAI_IMPORT_ERROR:
            return f"{base} Detail: {_GENAI_IMPORT_ERROR}"
        return base
    if not API_KEY:
        return "Ringkasan tidak tersedia (API key summarizer belum diset)."
    return None

SYSTEM_PROMPT = """
Kamu adalah peringkas harian untuk percakapan dukungan emosional non-klinis.
Tujuan: buat ringkasan harian singkat yang cukup kaya konteks agar esok hari coach dapat melanjutkan tanpa kehilangan benang merah.

ATURAN:
- Bahasa Indonesia, ringkas, hangat, TANPA PII (nama, nomor kontak, alamat).
- Fokus pada: (1) masalah/tema utama, (2) emosi dominan & trennya, (3) beban/konsekuensi (deadline, ujian, konflik), (4) gangguan fungsi (tidur/fokus/makan), (5) keputusan/komitmen pengguna, (6) hal yang perlu ditindaklanjuti besok.
- Jika ada sinyal keselamatan (risk ≥ high): tandai safety_flag=true dan tulis saran tindak lanjut singkat (tanpa detail berbahaya).

INPUT (sebagai JSON):
{
  "messages": [ { "role": "user|coach", "text": "...", "timestamp": "ISO8601" }, ... ],
  "analytics": { "avg_stress": 0-100, "max_stress": 0-100, "top_emotions": ["..."], "top_topics": ["..."] },
  "carry_over_notes": "<opsional ringkasan kemarin>"
}

OUTPUT (WAJIB hanya JSON):
{
  "daily_summary": "<≤180 kata, tanpa PII>",
  "key_points": ["...", "..."],
  "follow_up_tomorrow": ["pertanyaan/cek-lanjutan/aksi kecil"],
  "safety_flag": false
}
"""

# Tambahan di file summarizer.py

def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    # fallback minimal agar tidak 500
    return {
        "daily_summary": "Ringkasan tidak tersedia karena kendala sistem.",
        "key_points": [],
        "follow_up_tomorrow": [],
        "safety_flag": False
    }

def _normalize_messages(messages: list[dict]) -> list[dict]:
    def _role(r):
        return "user" if r.lower() not in ("user","coach") else r.lower()
    out = []
    for m in messages:
        t = (m.get("timestamp") or dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).isoformat())
        out.append({
            "role": _role(m.get("role","user")),
            "text": (m.get("text") or "").strip(),
            "timestamp": t
        })
    # sort by time
    out.sort(key=lambda x: x["timestamp"])
    return out

def _redact_pii(s: str) -> str:
    # redaksi ringan: email & nomor panjang
    import re
    s = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[email]', s)
    s = re.sub(r'\+?\d[\d\-\s]{8,}\d', '[nomor]', s)
    return s

def _trim_to_words(s: str, max_words: int = 180) -> str:
    words = s.split()
    return " ".join(words[:max_words])

def summarize_day(messages: list[dict], analytics: dict | None = None, carry_over_notes: str | None = None) -> dict:
    """
    messages: [{"role":"user|coach", "text":"...", "timestamp":"ISO"}, ...]
    analytics: {"avg_stress": int, "max_stress": int, "top_emotions": [...], "top_topics": [...]}
    carry_over_notes: ringkasan kemarin (opsional)
    """
    unavailable_reason = _summarizer_unavailable_reason()
    if unavailable_reason or not client:
        return {
            "daily_summary": unavailable_reason or "Ringkasan tidak tersedia.",
            "key_points": [],
            "follow_up_tomorrow": [],
            "safety_flag": False
        }

    msgs = _normalize_messages(messages)
    # redaksi ringan (tanpa PII)
    msgs = [{"role": m["role"], "text": _redact_pii(m["text"]), "timestamp": m["timestamp"]} for m in msgs]

    payload = {
        "messages": msgs,
        "analytics": analytics or {"avg_stress": None, "max_stress": None, "top_emotions": [], "top_topics": []},
        "carry_over_notes": _redact_pii(carry_over_notes) if carry_over_notes else None
    }

    try:
        chat = client.chats.create(
            model=MODEL,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",  # PENTING: di top-level config
            }
        )
        r = chat.send_message(json.dumps(payload, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
        return {
            "daily_summary": "Ringkasan tidak tersedia karena kendala koneksi ke model.",
            "key_points": [],
            "follow_up_tomorrow": [],
            "safety_flag": False
        }

    # ambil teks dari parts
    parts = r.candidates[0].content.parts if getattr(r, "candidates", None) else []
    text_chunks = []
    for p in parts or []:
        if getattr(p, "text", None):
            text_chunks.append(p.text)
        elif getattr(p, "executable_code", None):
            text_chunks.append(p.executable_code.code)
    text = "\n".join(text_chunks).strip()

    out = _safe_json_loads(text)

    # validasi ringan + trimming 180 kata
    out["daily_summary"] = _trim_to_words((out.get("daily_summary") or "").strip(), 180)
    out["key_points"] = list(out.get("key_points") or [])
    out["follow_up_tomorrow"] = list(out.get("follow_up_tomorrow") or [])
    out["safety_flag"] = bool(out.get("safety_flag", False))

    return out
