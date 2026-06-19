# model.py
import os, sys, json, datetime as dt, traceback
from html import escape
from dotenv import load_dotenv

try:
    from google import genai
    _GENAI_IMPORT_ERROR = None
except Exception as e:
    genai = None
    _GENAI_IMPORT_ERROR = e

try:
    from google.genai import types as genai_types
except Exception:
    genai_types = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "API.env")
load_dotenv(dotenv_path=ENV_PATH)

def _read_api_key(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    if not value or value == "-":
        return None
    return value

API_KEY = _read_api_key("API_KEY_GENERATOR")

MODEL = "gemini-3.1-flash-lite"
client = None
if API_KEY and genai is not None:
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1beta'})

def _model_unavailable_reason() -> str | None:
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
        return "API_KEY tidak ditemukan. Cek file API.env (API_KEY_GENERATOR=...)."
    return None

SYSTEM_PROMPT = """
Kamu adalah pendamping emosional non-klinis untuk remaja/mahasiswa Indonesia.
Gaya: hangat, empatik, tidak menghakimi, ringkas, Bahasa Indonesia sehari-hari, hindari diagnosis, hindari janji-janji medis.

=== MODE PEMBUKAAN (OPENING) ===
- Jika payload.meta.opening == true:
  - Anggap ini sapaan awal berbasis mood_emoji (normal/marah/sedih/senang).
  - Gunakan nada pembuka sesuai mood (lihat bagian MOOD EMOJI AWAL).
  - Utamakan DENGARKAN: validasi singkat + 1 pertanyaan ajakan cerita.
  - Set default stress_score=0 KECUALI SEC terpenuhi jelas.
  - JANGAN berikan suggested_actions pada pembukaan.
  - conversation_control: need_clarification=true, offer_suggestions=false, phase="listen".
  - REFERESI RESPON(untuk marah): "ada apa hari ini? kok kamu bisa marah hm? kalau mau cerita aku dengerin ya!"

=== MEMORI & KONTEKS (PENTING) ===
- Abaikan percakapan sebelumnya kecuali yang diberi pada input: memory_context dan/or daily_summary.
- Gunakan HANYA konteks yang disuntikkan (memory_context/daily_summary/mood_emoji). Jangan mengklaim “ingat” hal lain.
- Jika ada konflik, prioritaskan data terbaru pada memory_context.

=== MODE INTERAKSI (LISTEN-FIRST) ===
- Fase awal: DENGARKAN. Validasi perasaan, ringkas balik (reflective listening), dan AJUKAN 1 pertanyaan klarifikasi.
- JANGAN memberi suggested_actions dulu, kecuali:
  (a) pengguna meminta saran/ide, atau
  (b) pengguna tampak bingung/minta arahan, atau
  (c) risiko ≥ MODERATE.
- Jika user memberi sinyal “ya, mau saran”, baru beri maksimal 3 aksi yang TERSTRUKTUR (lihat daftar aksi tetap).
- Tawarkan izin dulu: “Mau aku kasih 2–3 ide kecil yang bisa dicoba?”

=== MOOD EMOJI AWAL ===
- Gunakan “mood_emoji” (normal/marah/sedih/senang) untuk menyesuaikan nada pembuka:
  - marah: validasi kemarahan singkat, hindari menggurui.
  - sedih: hangat & pelan, normalisasi perasaan.
  - senang: apresiasi singkat, jangan berlebihan.
  - normal: netral hangat.

=== DISAMBIGUASI SARKAS/SLANG & EMOJI ===
- Tawa (“wkwk”, “wk”, “haha”, “ngakak”, “lol”, 😭/🤣) sering bermakna canda/eksagerasi. Jangan menaikkan skor tanpa bukti lain.
- “capek/lelah” + tawa TANPA bukti beban/konsekuensi/impairment → boleh stress_score=0.
- Sarkas positif (“mantap dapat E”, “love this for me”) → beri skor >0 hanya jika ada BUKTI (nilai jelek, dimarahi, dll.). Tanpa bukti → 0–20.

=== SEC: STRESS EVIDENCE CHECKLIST (harus ≥1 untuk skor >0) ===
1) Distres eksplisit: “cemas”, “takut/panik”, “tertekan”, “overthinking (serius)”, “sedih banget”, “gak kuat”.
2) Beban/konsekuensi jelas: “deadline/tugas numpuk”, “remed/nilai jelek”, “skripsi/UTS”, konflik relasi/keluarga/keuangan.
3) Gangguan fungsi: tidur/makan/fokus/absen terganggu; gejala tubuh (berdebar, pusing).
4) Persistensi/waktu: “akhir-akhir ini/terus-terusan/berhari-hari”.

=== KALIBRASI SKOR ===
- 0: SEC=0 (tidak ada bukti stres).
- 1–25: ringan (≥1 SEC, dampak kecil).
- 25–50: sedang (≥2 SEC atau 1 SEC + gangguan fungsi ringan).
- 50–80: berat (≥2 SEC + gangguan fungsi nyata/konsekuensi kuat).
- 80–100: krisis (niat/rencana melukai diri/bunuh diri/kekerasan).

=== RISIKO & KRISIS ===
- risk_flag=high: putus asa kuat / “pengen hilang” berulang TANPA rencana spesifik.
- risk_flag=critical: ada rencana/alat/waktu/lokasi, atau bahaya langsung. Respon aman + rujukan (<hotline_lokal>/<kontak_kampus>), TANPA detail berbahaya.

=== DAFTAR EMOSI TETAP (gunakan salah satu/lebih) ===
["senang","tenang","lega","bingung","cemas","sedih","lelah","kesepian","marah","malu","frustrasi","overwhelmed","kecewa","khawatir"]

=== DAFTAR TOPIK TETAP (multi-label) ===
["akademik","pertemanan","relasi","keluarga","keuangan","kesehatan","online/sosmed","aktivitas_sosial","pekerjaan","spiritualitas","kelelahan_emosional","lainnya"]

=== DAFTAR AKSI TETAP (maks 3 saat diizinkan) ===
- breathing: {"protocol":"4-7-8","duration_min":2} atau {"protocol":"box-4-4-4-4","duration_min":2} //duration_min tergantung dari seberapa berat permasalahan dia
- journaling: {"template":"3 prioritas + 1 langkah mudah","duration_min":5} //duration_min tergantung dari seberapa berat permasalahan dia
- break: {"timer_min":15,"before_burnout_tip":true} //duration_min tergantung dari seberapa berat permasalahan dia
- grounding(menyentuh objek, melihat pemandangan): {"method":"5-4-3-2-1","duration_min":3} //duration_min tergantung dari seberapa berat permasalahan dia
- sleep: {"routine":"wind-down 30 menit","duration_min":30} //duration_min tergantung dari seberapa berat permasalahan dia
- prioritization: {"method":"prioritas 3 hal","duration_min":5}
- pomodoro: {"cycle":"25-5","rounds":1}
- stretching: {"routine":"neck-shoulder","duration_min":2}
- hydration: {"amount":"1 gelas","duration_min":1}
- reach_out: {"target":"teman/keluarga tepercaya","duration_min":3}
- safety_planning (khusus risiko tinggi): {"steps":"singkat 3 langkah","duration_min":5}
- call_emergency (khusus critical): {"channel":"darurat/lokal","duration_min":1}

=== TUGAS UTAMA ===
1) Saring SEC: jika SEC=0 → stress_score=0.
2) Analisis: kembalikan emotions (dari daftar tetap), stress_score (0–100), topics (dari daftar tetap), risk_flag.
3) Respon:
   - Jika stress_score=0 → coach_reply hangat + 1 pertanyaan klarifikasi; suggested_actions = [].
   - Jika >0 dan BELUM ada izin → coach_reply hangat + 1 pertanyaan, serta "offer_suggestions": true (tawarkan).
   - Jika >0 dan SUDAH diizinkan → berikan ≤3 suggested_actions dari daftar tetap (durasi jelas).
4) Sarankan istirahat sebelum burnout bila skor ≥26 (boleh sebagai bagian dari aksi saat diizinkan).
5) Jika krisis → risk_flag sesuai (high/critical) + respons aman + rujukan (<hotline_lokal>/<kontak_kampus>). Jangan berikan detail berbahaya.

=== FORMAT OUTPUT (WAJIB HANYA JSON) ===
{
  "analysis": {
    "emotions": ["..."],           // dari daftar EMOSI TETAP
    "stress_score": 0-100,
    "topics": ["..."],             // dari daftar TOPIK TETAP
    "risk_flag": "none|low|moderate|high|critical"
  },
  "conversation_control": {
    "need_clarification": true|false,
    "clarify_question": "<1 kalimat tanya>",
    "offer_suggestions": true|false,     // true jika boleh tawarkan aksi
    "phase": "listen"|"suggest"          // set "listen" jika belum ada izin
  },
  "coach_reply": "<≤120 kata, hangat, non-judgmental>",
  "suggested_actions": [ /* 0..3 item, hanya dari DAFTAR AKSI TETAP */ ],
  "gamification": {"streak_increment": true, "potential_badge": "Calm Starter"}
}

EDGE CASE:
- Input hanya emoji/tawa → stress_score 0–10, ajukan klarifikasi.
- Sarkas positif atas kejadian negatif (mis. “mantap nilai E”) → minta konfirmasi bukti; beri skor sedang/berat hanya jika ada beban/konsekuensi.
- Jangan diagnosis/label klinis; hindari bahasa menghakimi; tetap aman.
Jawab SELALU dalam Bahasa Indonesia.
"""

def _build_curhat_payload(
    curhat_text: str,
    profile: dict | None = None,
    mood_emoji: str = "normal",
    memory_context: dict | None = None,
    opening: bool = False
) -> str:
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    payload = {
        "meta": {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S %z"),
            "timezone": "Asia/Jakarta",
            "profile": profile or {"age_group": "mahasiswa", "language": "id", "context": "akademik"},
            "mood_emoji": (mood_emoji or "normal"),
            "opening": bool(opening)
        },
        # hanya konteks yang kamu suntikkan yang boleh dipakai model
        "memory_context": memory_context or {
            "recent_turns": [],
            "daily_summary": None,
            "salient_facts": [],
            "safety_note": None
        },
        "curhat": (curhat_text or "").strip()
    }
    return json.dumps(payload, ensure_ascii=False)


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {
        "analysis": {"emotions": [], "stress_score": None, "topics": [], "risk_flag": "none"},
        "coach_reply": "Maaf, ada kendala saat memproses. Coba kirim ulang ya.",
        "suggested_actions": [],
        "gamification": {"streak_increment": False}
    }

def _detect_image_mime(image_data: bytes, declared_mime: str | None = None) -> str:
    allowed = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    if declared_mime in allowed:
        return declared_mime
    if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
        return "image/gif"
    if image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"

def _extract_response_text(response) -> str:
    if getattr(response, "text", None):
        return response.text.strip()
    parts = response.candidates[0].content.parts if getattr(response, "candidates", None) else []
    return "\n".join(
        [
            (getattr(p, "text", None) or getattr(getattr(p, "executable_code", None) or {}, "code", ""))
            for p in (parts or [])
        ]
    ).strip()

def _request_model(prompt: str, image_data: bytes | None = None, image_mime_type: str | None = None):
    config = {"system_instruction": SYSTEM_PROMPT, "response_mime_type": "application/json"}
    try:
        if image_data:
            mime_type = _detect_image_mime(image_data, image_mime_type)
            if genai_types is not None:
                image_part = genai_types.Part.from_bytes(data=image_data, mime_type=mime_type)
            else:
                image_part = {"inline_data": {"mime_type": mime_type, "data": image_data}}
            contents = [prompt, image_part]
        else:
            contents = prompt
        return client.models.generate_content(model=MODEL, contents=contents, config=config)
    except Exception:
        # Fallback SDK path for text-only requests.
        if image_data:
            raise
        chat = client.chats.create(model=MODEL, config=config)
        return chat.send_message(prompt)
    
def _coerce_schema(d: dict) -> dict:
    # analysis
    d.setdefault("analysis", {})
    a = d["analysis"]
    a.setdefault("emotions", [])
    a.setdefault("stress_score", 0 if a.get("stress_score") is None else a.get("stress_score"))
    a.setdefault("topics", [])
    a.setdefault("risk_flag", "none")

    # conversation_control
    d.setdefault("conversation_control", {})
    cc = d["conversation_control"]
    cc.setdefault("need_clarification", True)
    cc.setdefault("clarify_question", "")
    cc.setdefault("offer_suggestions", False)
    cc.setdefault("phase", "listen")

    # suggested_actions harus kosong kalau stress_score == 0
    if (a.get("stress_score") or 0) == 0:
        d["suggested_actions"] = []

    # gamification default
    d.setdefault("gamification", {"streak_increment": True, "potential_badge": "Calm Starter"})
    return d


def analyze_raw(
    query: str,
    mood_emoji: str = "normal",
    memory_context: dict | None = None,
    profile: dict | None = None,
    image_data: bytes | None = None,
    image_mime_type: str | None = None,
) -> dict:
    unavailable_reason = _model_unavailable_reason()
    if unavailable_reason or not client:
        return {
            "analysis": {"emotions": [], "stress_score": None, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": unavailable_reason or "Model belum siap digunakan.",
            "suggested_actions": [],
            "gamification": {"streak_increment": False}
        }
    prompt = _build_curhat_payload(query, profile, mood_emoji, memory_context)
    try:
        r = _request_model(prompt, image_data=image_data, image_mime_type=image_mime_type)
    except Exception:
        traceback.print_exc()
        return {
            "analysis": {"emotions": [], "stress_score": None, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": "Maaf, ada kendala koneksi ke model. Coba kirim ulang ya.",
            "suggested_actions": [],
            "gamification": {"streak_increment": False}
        }

    text = _extract_response_text(r)
    return _coerce_schema(_safe_json_loads(text))

def generate_opening(
    mood_emoji: str = "normal",
    memory_context: dict | None = None,
    profile: dict | None = None,
) -> dict:
    unavailable_reason = _model_unavailable_reason()
    if unavailable_reason or not client:
        return _coerce_schema({
            "analysis": {"emotions": [], "stress_score": 0, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": unavailable_reason or "Model belum siap digunakan.",
            "suggested_actions": []
        })
    prompt = _build_curhat_payload("", profile, mood_emoji, memory_context, opening=True)
    try:
        r = _request_model(prompt)
    except Exception:
        traceback.print_exc()
        return _coerce_schema({
            "analysis": {"emotions": [], "stress_score": 0, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": "Maaf, ada kendala koneksi ke model. Coba kirim ulang ya.",
            "suggested_actions": []
        })

    text = _extract_response_text(r)
    return _coerce_schema(_safe_json_loads(text))


def _format_action_card(action) -> str:
    action_ui = {
        "breathing": ("Tarik napas pelan", "Luangkan sebentar untuk mengatur ritme napas.", "🌬️"),
        "journaling": ("Tuangkan ke tulisan", "Rapikan isi kepala tanpa harus sempurna.", "✍️"),
        "break": ("Ambil jeda", "Beri tubuh dan pikiran ruang untuk berhenti sebentar.", "☕"),
        "grounding": ("Kembali ke sekitar", "Arahkan perhatian ke hal-hal yang ada di dekatmu.", "🌿"),
        "sleep": ("Siapkan waktu istirahat", "Turunkan ritme agar tubuh lebih siap beristirahat.", "🌙"),
        "prioritization": ("Pilih yang paling penting", "Kecilkan beban menjadi beberapa langkah yang jelas.", "🎯"),
        "pomodoro": ("Fokus satu sesi", "Kerjakan satu bagian kecil, lalu beri diri waktu jeda.", "⏱️"),
        "stretching": ("Regangkan tubuh", "Lepaskan sedikit ketegangan yang tersimpan di tubuh.", "🙆"),
        "hydration": ("Minum air", "Mulai dari kebutuhan tubuh yang paling sederhana.", "💧"),
        "reach_out": ("Hubungi orang tepercaya", "Kamu tidak harus membawa semuanya sendirian.", "🤝"),
        "safety_planning": ("Susun langkah aman", "Buat rencana pendek untuk menjaga dirimu tetap aman.", "🛟"),
        "call_emergency": ("Cari bantuan segera", "Hubungi bantuan lokal atau orang terdekat sekarang.", "☎️"),
    }
    detail_labels = {
        "protocol": "Pola",
        "duration_min": "Durasi",
        "timer_min": "Jeda",
        "template": "Panduan",
        "method": "Metode",
        "routine": "Rutinitas",
        "cycle": "Siklus",
        "rounds": "Putaran",
        "amount": "Target",
        "target": "Hubungi",
        "steps": "Langkah",
        "channel": "Saluran",
        "before_burnout_tip": "Pengingat",
    }

    def _display_value(key, value) -> str:
        if key in {"duration_min", "timer_min"}:
            return f"{value} menit"
        if isinstance(value, bool):
            return "Sebelum terlalu lelah" if value else "Tidak"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            return " · ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    if isinstance(action, dict) and len(action) == 1:
        action_name, payload = next(iter(action.items()))
        action_name = str(action_name)
        fallback_label = action_name.replace("_", " ").title()
        label, description, icon = action_ui.get(
            action_name,
            (fallback_label, "Coba langkah kecil ini saat kamu sudah siap.", "✨"),
        )
        if isinstance(payload, dict) and payload:
            details = []
            for key, value in payload.items():
                pretty_key = escape(detail_labels.get(str(key), str(key).replace("_", " ").title()))
                pretty_value = escape(_display_value(str(key), value))
                details.append(
                    f"<span class='action-detail'><small>{pretty_key}</small><b>{pretty_value}</b></span>"
                )
            return (
                "<button type='button' class='action-card'>"
                f"<div class='action-icon' aria-hidden='true'>{icon}</div>"
                "<div class='action-content'>"
                f"<h4>{escape(label)}</h4>"
                f"<p>{escape(description)}</p>"
                f"<div class='action-details'>{''.join(details)}</div>"
                "</div>"
                "<span class='action-open-icon' aria-hidden='true'>↗</span>"
                "</button>"
            )
        return (
            "<button type='button' class='action-card'>"
            f"<div class='action-icon' aria-hidden='true'>{icon}</div>"
            f"<div class='action-content'><h4>{escape(label)}</h4>"
            f"<p>{escape(description if payload in (None, '') else str(payload))}</p></div>"
            "<span class='action-open-icon' aria-hidden='true'>↗</span>"
            "</button>"
        )

    if isinstance(action, str):
        return (
            "<button type='button' class='action-card'>"
            "<div class='action-icon' aria-hidden='true'>✨</div>"
            f"<div class='action-content'><h4>Langkah kecil</h4><p>{escape(action)}</p></div>"
            "<span class='action-open-icon' aria-hidden='true'>↗</span>"
            "</button>"
        )

    if isinstance(action, dict):
        action_text = " · ".join(
            f"{str(key).replace('_', ' ').title()}: {_display_value(str(key), value)}"
            for key, value in action.items()
        )
    else:
        action_text = str(action)

    return (
        "<button type='button' class='action-card'>"
        "<div class='action-icon' aria-hidden='true'>✨</div>"
        f"<div class='action-content'><h4>Langkah kecil</h4><p>{escape(action_text)}</p></div>"
        "<span class='action-open-icon' aria-hidden='true'>↗</span>"
        "</button>"
    )


def _html_from_result(data: dict) -> str:
    cc = data.get("conversation_control", {}) or {}
    reply = escape(data.get("coach_reply", ""))
    acts = data.get("suggested_actions", []) or []
    raw_reply = data.get("coach_reply", "") or ""
    raw_question = cc.get("clarify_question", "") or ""
    question = escape(raw_question)

    html = []
    html.append("<div class='chat-response'>")
    html.append(f"<p class='reply-copy'>{reply}</p>")
    if question and raw_question.casefold() not in raw_reply.casefold():
        html.append(f"<p class='assistant-question'>{question}</p>")
    if acts:
        html.append(
            "<section class='action-section'>"
            "<div class='action-heading'>"
            "<span class='action-heading-icon'>✦</span>"
            "<div><strong>Langkah kecil yang bisa dicoba</strong>"
            "<small>Pilih satu saja yang terasa paling ringan.</small></div>"
            "</div><div class='action-grid'>"
        )
        for act in acts[:3]:
            html.append(_format_action_card(act))
        html.append("</div></section>")
    html.append("</div>")
    return "".join(html)

def render_html(data: dict) -> str:
    """
    Menerima dict hasil analyze_raw() atau generate_opening()
    dan mengubahnya jadi HTML siap-tampil.
    """
    return _html_from_result(data)


def get_model_result(
    query: str,
    image_data: bytes | None = None,
    mood_emoji: str = "normal",
    memory_context: dict | None = None,
    image_mime_type: str | None = None,
) -> str:
    data = analyze_raw(
        query,
        mood_emoji=mood_emoji,
        memory_context=memory_context,
        image_data=image_data,
        image_mime_type=image_mime_type,
    )
    return _html_from_result(data)
