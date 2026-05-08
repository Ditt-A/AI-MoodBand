# memory_store.py
import os, sys, json, time, uuid, pickle, numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "API.env")
load_dotenv(ENV_PATH)

def _read_api_key(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    if not value or value == "-":
        return None
    return value

try:
    from google import genai
    _GENAI_IMPORT_ERROR = None
except Exception as e:
    genai = None
    _GENAI_IMPORT_ERROR = e

EMB_API_KEY = _read_api_key("API_KEY_TODATABASE")
EMB_MODEL = "text-embedding-004"
client = genai.Client(api_key=EMB_API_KEY, http_options={'api_version':'v1alpha'}) if (EMB_API_KEY and genai is not None) else None

def embed_texts(texts: List[str], task_type="retrieval_document") -> np.ndarray:
    if genai is None:
        base = (
            "Library 'google-genai' tidak bisa diimport pada interpreter ini. "
            f"Python aktif: {sys.executable}. "
            "Aktifkan virtualenv proyek lalu install: python -m pip install google-genai."
        )
        detail = f" Detail: {_GENAI_IMPORT_ERROR}" if _GENAI_IMPORT_ERROR else ""
        raise RuntimeError(base + detail)

    if not client:
        raise RuntimeError("API_KEY_TODATABASE tidak ditemukan")

    if not isinstance(texts, list):
        texts = [texts]

    res = client.models.embed_content(
        model=EMB_MODEL,
        contents=texts,
    )

    # Ambil vektor dari res.embeddings
    vecs = [e.values for e in res.embeddings]
    X = np.array(vecs, dtype="float32")
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return X


# --- FAISS index (local) ---
try:
    import faiss
    _FAISS_IMPORT_ERROR = None
except ImportError as e:
    faiss = None
    _FAISS_IMPORT_ERROR = e

DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "memory.index")
META_PATH  = os.path.join(DATA_DIR, "memory_meta.pkl")

# in-memory structures
index = None  # faiss.IndexFlatIP
metas: Dict[int, Dict[str, Any]] = {}  # row_id -> payload
row_counter = 0

def _ensure_store():
    global index, metas, row_counter
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            metas = pickle.load(f)
        if faiss is not None and os.path.exists(INDEX_PATH):
            index = faiss.read_index(INDEX_PATH)
        else:
            index = None
        row_counter = max(metas.keys())+1 if metas else 0
    else:
        index = None
        metas = {}
        row_counter = 0


def _persist():
    os.makedirs(DATA_DIR, exist_ok=True)
    if faiss is not None and index is not None:
        faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

def upsert_memory(user_id: str, role: str, text: str, meta: Optional[Dict[str, Any]]=None):
    """Simpan satu potongan memori percakapan (user/coach)."""
    _ensure_store()

    global index, row_counter
    use_vector_index = False
    if faiss is not None and client is not None:
        try:
            # boleh simpan ringkasan pendek kalau tidak consent teks mentah
            vec = embed_texts([text])[0].reshape(1, -1)
            if index is None:
                # Hindari mapping index/metas tidak sinkron jika data lama sudah ada tanpa index.
                if row_counter == 0:
                    index = faiss.IndexFlatIP(vec.shape[1])
                else:
                    vec = None
            if vec is not None and index is not None:
                if index.d != vec.shape[1]:
                    raise RuntimeError(
                        f"Dimensi embedding ({vec.shape[1]}) != dimensi index ({index.d}). "
                        f"Hapus folder data/ agar index rebuild dgn dimensi baru, "
                        f"atau pastikan model embedding konsisten."
                    )
                index.add(vec)
                use_vector_index = True
        except Exception:
            # Jika index sudah aktif, tambahkan vektor nol agar row index tetap selaras dengan metas.
            if index is not None:
                filler = np.zeros((1, index.d), dtype="float32")
                index.add(filler)

    metas[row_counter] = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "role": role,
        "text": text,
        "ts": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "indexed": use_vector_index,
        **(meta or {})
    }
    row_counter += 1
    _persist()

def upsert_daily_summary(user_id: str, summary_text: str, summary_date: str | None = None):
    """Simpan ringkasan harian + tanggal (YYYY-MM-DD, WIB)."""
    if summary_date is None:
        summary_date = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    upsert_memory(
        user_id,
        role="daily_summary",
        text=summary_text,
        meta={
            "is_summary": True,
            "summary_date": summary_date  # ← tanggal disimpan di meta
        }
    )


def get_daily_summary(user_id: str, date: str | None = None) -> Optional[Dict[str, Any]]:
    """Jika date=None → ambil ringkasan terbaru. Jika date='YYYY-MM-DD' → ambil ringkasan utk tanggal tsb."""
    _ensure_store()
    cand = [m for m in metas.values() if m["user_id"] == user_id and m.get("is_summary")]
    if not cand:
        return None

    if date:
        # pakai summary_date kalau ada; fallback ke ts[:10]
        cand = [m for m in cand if (m.get("summary_date") or m["ts"][:10]) == date]
        if not cand:
            return None

    cand.sort(key=lambda m: m["ts"], reverse=True)
    return cand[0]


def retrieve_memories(user_id: str, query_text: str, k: int = 4) -> List[Dict[str, Any]]:
    _ensure_store()
    if index is None or len(metas) == 0:
        return []

    try:
        qv = embed_texts([query_text])[0].reshape(1,-1)
    except Exception:
        return []

    if qv.shape[1] != index.d:
        # Model/dimensi berubah dibanding index yg sudah ada
        # Lebih aman kembalikan kosong dan arahkan user untuk rebuild
        return []

    scores, idxs = index.search(qv, min(k*8, len(metas)))
    now = datetime.now(timezone(timedelta(hours=7)))
    results = []
    for i, s in zip(idxs[0], scores[0]):
        if i < 0:
            continue
        m = metas.get(i)
        if not m or m["user_id"] != user_id:
            continue
        age_days = max(0.0, (now - datetime.fromisoformat(m["ts"])).total_seconds()/86400.0)
        rec_boost = max(0.2, 1.0 - min(age_days/7.0, 1.0)*0.5)
        results.append((float(s)*rec_boost, m))
    results.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in results[:k]]

def list_recent_messages(user_id: str, limit: int = 8) -> list[dict]:
    _ensure_store()
    rows = [m for m in metas.values() if m["user_id"]==user_id and m["role"] in ("user","coach")]
    rows.sort(key=lambda m: m["ts"], reverse=True)
    rows = rows[:limit]
    rows.reverse()
    return [{"role": r["role"], "text": r["text"], "timestamp": r["ts"]} for r in rows]

def list_today_messages(user_id: str) -> list[dict]:
    _ensure_store()
    today = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    rows = [m for m in metas.values() if m["user_id"]==user_id and m["role"] in ("user","coach") and m["ts"][:10]==today]
    rows.sort(key=lambda m: m["ts"])
    return [{"role": r["role"], "text": r["text"], "timestamp": r["ts"]} for r in rows]

def today_has_summary(user_id: str) -> bool:
    _ensure_store()
    today = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    return any(
        m.get("is_summary")
        and m["user_id"] == user_id
        and (m.get("summary_date") or m["ts"][:10]) == today
        for m in metas.values()
    )


