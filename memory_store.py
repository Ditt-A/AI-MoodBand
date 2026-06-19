# memory_store.py
import os, sys, json, time, uuid, pickle, numpy as np
from collections import Counter
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
EMB_MODEL = "gemini-embedding-001"
client = genai.Client(api_key=EMB_API_KEY, http_options={'api_version':'v1beta'}) if (EMB_API_KEY and genai is not None) else None

def embed_texts(texts: List[str]) -> np.ndarray:
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

    # Panggil per teks agar setiap memory row selalu memiliki tepat satu vektor.
    vecs = []
    for text in texts:
        res = client.models.embed_content(
            model=EMB_MODEL,
            contents=text,
        )
        vecs.append(res.embeddings[0].values)

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
CONVERSATION_PATH = os.path.join(DATA_DIR, "conversations.pkl")

# in-memory structures
index = None  # faiss.IndexFlatIP
metas: Dict[int, Dict[str, Any]] = {}  # row_id -> payload
conversations: Dict[str, Dict[str, Any]] = {}
row_counter = 0

def _ensure_store():
    global index, metas, conversations, row_counter
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

    if os.path.exists(CONVERSATION_PATH):
        with open(CONVERSATION_PATH, "rb") as f:
            conversations = pickle.load(f)
    else:
        conversations = {}

    # Migrasikan percakapan lama yang belum memiliki thread tanpa menghapus data.
    legacy_groups: Dict[str, list[dict]] = {}
    for row in metas.values():
        if row.get("role") not in {"user", "coach"}:
            continue
        if not row.get("conversation_id"):
            legacy_id = f"legacy-{row.get('user_id', 'demo_user')}"
            row["conversation_id"] = legacy_id
            legacy_groups.setdefault(legacy_id, []).append(row)
    for legacy_id, rows in legacy_groups.items():
        if legacy_id in conversations:
            continue
        rows.sort(key=lambda item: item.get("ts", ""))
        user_id = rows[0].get("user_id", "demo_user")
        first_user = next((item.get("text", "") for item in rows if item.get("role") == "user"), "")
        title = _conversation_title(first_user) if first_user else "Percakapan sebelumnya"
        conversations[legacy_id] = {
            "id": legacy_id,
            "user_id": user_id,
            "title": title,
            "created_at": rows[0].get("ts", ""),
            "updated_at": rows[-1].get("ts", ""),
        }


def _persist():
    os.makedirs(DATA_DIR, exist_ok=True)
    if faiss is not None and index is not None:
        faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)
    with open(CONVERSATION_PATH, "wb") as f:
        pickle.dump(conversations, f)


def _conversation_title(text: str) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return "Percakapan baru"
    return clean if len(clean) <= 44 else clean[:43].rstrip() + "…"


def create_conversation(user_id: str, title: str = "Percakapan baru") -> dict:
    _ensure_store()
    conversation_id = str(uuid.uuid4())
    now = datetime.now(timezone(timedelta(hours=7))).isoformat()
    conversations[conversation_id] = {
        "id": conversation_id,
        "user_id": user_id,
        "title": _conversation_title(title),
        "created_at": now,
        "updated_at": now,
    }
    _persist()
    return dict(conversations[conversation_id])


def get_conversation(user_id: str, conversation_id: str | None) -> Optional[dict]:
    if not conversation_id:
        return None
    _ensure_store()
    conversation = conversations.get(conversation_id)
    if not conversation or conversation.get("user_id") != user_id:
        return None
    return dict(conversation)


def list_conversations(user_id: str, limit: int = 30) -> list[dict]:
    _ensure_store()
    rows = [item for item in conversations.values() if item.get("user_id") == user_id]
    rows.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return [dict(item) for item in rows[:max(1, min(int(limit), 100))]]


def _touch_conversation(user_id: str, conversation_id: str, role: str, text: str, timestamp: str):
    conversation = conversations.get(conversation_id)
    if not conversation or conversation.get("user_id") != user_id:
        conversations[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "title": "Percakapan baru",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        conversation = conversations[conversation_id]
    conversation["updated_at"] = timestamp
    if role == "user" and conversation.get("title") == "Percakapan baru":
        conversation["title"] = _conversation_title(text)

def upsert_memory(
    user_id: str,
    role: str,
    text: str,
    meta: Optional[Dict[str, Any]]=None,
    conversation_id: str | None = None,
):
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

    timestamp = datetime.now(timezone(timedelta(hours=7))).isoformat()
    if role in {"user", "coach"}:
        conversation_id = conversation_id or f"legacy-{user_id}"
        _touch_conversation(user_id, conversation_id, role, text, timestamp)

    metas[row_counter] = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "role": role,
        "text": text,
        "ts": timestamp,
        "indexed": use_vector_index,
        **({"conversation_id": conversation_id} if conversation_id else {}),
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


def list_daily_summaries(user_id: str, limit: int = 14) -> list[dict]:
    """Daftar ringkasan terbaru untuk halaman ringkasan harian."""
    _ensure_store()
    rows = [
        row for row in metas.values()
        if row.get("user_id") == user_id and row.get("is_summary")
    ]
    rows.sort(key=lambda row: row.get("ts", ""), reverse=True)
    return [
        {
            "text": row.get("text", ""),
            "date": row.get("summary_date") or row.get("ts", "")[:10],
            "timestamp": row.get("ts", ""),
        }
        for row in rows[:limit]
    ]


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

def list_recent_messages(
    user_id: str,
    limit: int = 8,
    conversation_id: str | None = None,
) -> list[dict]:
    _ensure_store()
    rows = [
        m for m in metas.values()
        if m["user_id"] == user_id
        and m["role"] in ("user", "coach")
        and (conversation_id is None or m.get("conversation_id") == conversation_id)
    ]
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


def list_memory_records(user_id: str, limit: int = 60) -> list[dict]:
    """Percakapan terbaru beserta metadata yang aman ditampilkan di UI lokal."""
    _ensure_store()
    rows = [
        row for row in metas.values()
        if row.get("user_id") == user_id and row.get("role") in {"user", "coach"}
    ]
    rows.sort(key=lambda row: row.get("ts", ""), reverse=True)
    return [
        {
            "role": row.get("role", "user"),
            "text": row.get("text", ""),
            "timestamp": row.get("ts", ""),
            "mood": row.get("mood"),
            "emotions": row.get("emotions") or [],
            "stress": row.get("stress"),
            "topics": row.get("topics") or [],
            "opening": bool(row.get("opening")),
            "conversation_id": row.get("conversation_id"),
        }
        for row in rows[:max(1, min(int(limit), 200))]
    ]


def get_analytics(user_id: str, days: int = 7) -> dict:
    """Agregasi metadata lokal untuk dashboard, tanpa panggilan model tambahan."""
    _ensure_store()
    days = max(1, min(int(days), 30))
    tz_wib = timezone(timedelta(hours=7))
    today = datetime.now(tz_wib).date()
    dates = [today - timedelta(days=offset) for offset in reversed(range(days))]
    date_keys = {day.isoformat() for day in dates}

    rows = [
        row for row in metas.values()
        if row.get("user_id") == user_id and row.get("ts", "")[:10] in date_keys
    ]
    coach_rows = [row for row in rows if row.get("role") == "coach"]
    user_rows = [row for row in rows if row.get("role") == "user"]

    emotion_counts = Counter()
    topic_counts = Counter()
    mood_counts = Counter()
    risk_counts = Counter()
    all_stress = []

    for row in coach_rows:
        emotion_counts.update(str(item) for item in (row.get("emotions") or []))
        topic_counts.update(str(item) for item in (row.get("topics") or []))
        risk_counts[str(row.get("risk") or "none")] += 1
        stress = row.get("stress")
        if isinstance(stress, (int, float)):
            all_stress.append(float(stress))

    for row in user_rows:
        mood_counts[str(row.get("mood") or "normal")] += 1

    stress_trend = []
    for day in dates:
        key = day.isoformat()
        daily_rows = [row for row in coach_rows if row.get("ts", "")[:10] == key]
        values = [
            float(row["stress"]) for row in daily_rows
            if isinstance(row.get("stress"), (int, float))
        ]
        stress_trend.append({
            "date": key,
            "label": day.strftime("%d/%m"),
            "day": ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"][day.weekday()],
            "average": round(sum(values) / len(values)) if values else 0,
            "maximum": round(max(values)) if values else 0,
            "count": len(daily_rows),
        })

    def _breakdown(counter: Counter, limit: int = 6) -> list[dict]:
        total = sum(counter.values())
        return [
            {
                "name": name.replace("_", " ").title(),
                "count": count,
                "percent": round((count / total) * 100) if total else 0,
            }
            for name, count in counter.most_common(limit)
        ]

    return {
        "days": days,
        "total_messages": len(user_rows),
        "total_responses": len(coach_rows),
        "average_stress": round(sum(all_stress) / len(all_stress)) if all_stress else 0,
        "highest_stress": round(max(all_stress)) if all_stress else 0,
        "top_emotion": emotion_counts.most_common(1)[0][0].title() if emotion_counts else "Belum ada",
        "top_topic": topic_counts.most_common(1)[0][0].replace("_", " ").title() if topic_counts else "Belum ada",
        "stress_trend": stress_trend,
        "emotions": _breakdown(emotion_counts),
        "topics": _breakdown(topic_counts),
        "moods": _breakdown(mood_counts, limit=4),
        "risk_alerts": sum(count for risk, count in risk_counts.items() if risk in {"high", "critical"}),
        "has_data": bool(user_rows or coach_rows),
    }

def today_has_summary(user_id: str) -> bool:
    _ensure_store()
    today = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
    return any(
        m.get("is_summary")
        and m["user_id"] == user_id
        and (m.get("summary_date") or m["ts"][:10]) == today
        for m in metas.values()
    )


