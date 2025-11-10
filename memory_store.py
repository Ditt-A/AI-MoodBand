# memory_store.py
import os, json, time, uuid, pickle, numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv("APAAC/API.env")

from google import genai
EMB_API_KEY = os.getenv("API_KEY_TODATABASE")
EMB_MODEL = "text-embedding-004"
client = genai.Client(api_key=EMB_API_KEY, http_options={'api_version':'v1alpha'}) if EMB_API_KEY else None

def embed_texts(texts: List[str], task_type="retrieval_document") -> np.ndarray:
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
except ImportError:
    raise RuntimeError("Install FAISS: pip install faiss-cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # -> .../APAAC
DATA_DIR = os.path.join(BASE_DIR, "data")                 # -> .../APAAC/data
INDEX_PATH = os.path.join(DATA_DIR, "memory.index")
META_PATH  = os.path.join(DATA_DIR, "memory_meta.pkl")

# in-memory structures
index = None  # faiss.IndexFlatIP
metas: Dict[int, Dict[str, Any]] = {}  # row_id -> payload
row_counter = 0

def _ensure_store():
    global index, metas, row_counter
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metas = pickle.load(f)
        row_counter = max(metas.keys())+1 if metas else 0
    else:
        index = None
        metas = {}
        row_counter = 0


def _persist():
    os.makedirs(DATA_DIR, exist_ok=True)
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

def upsert_memory(user_id: str, role: str, text: str, meta: Optional[Dict[str, Any]]=None):
    """Simpan satu potongan memori percakapan (user/coach)."""
    _ensure_store()
    # boleh simpan ringkasan pendek kalau tidak consent teks mentah
    vec = embed_texts([text])[0].reshape(1,-1)
    global index, row_counter
    if index is None:
        # pertama kali — bangun index sesuai dimensi embedding
        index = faiss.IndexFlatIP(vec.shape[1])
    elif index.d != vec.shape[1]:
        raise RuntimeError(
            f"Dimensi embedding ({vec.shape[1]}) ≠ dimensi index ({index.d}). "
            f"Hapus folder data/ agar index rebuild dgn dimensi baru, "
            f"atau pastikan model embedding konsisten."
        )
    index.add(vec)
    metas[row_counter] = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "role": role,
        "text": text,
        "ts": datetime.now(timezone(timedelta(hours=7))).isoformat(),
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

    qv = embed_texts([query_text])[0].reshape(1,-1)
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


