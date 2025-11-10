# model.py
import os, json, datetime as dt, traceback
from dotenv import load_dotenv

try:
    from google import genai
except Exception as e:
    raise RuntimeError("Library 'google-genai' belum terpasang atau konflik versi. "
                       "Install: pip install google-genai") from e

load_dotenv(dotenv_path="APAAC/API.env")
API_KEY = os.getenv("API_KEY_GENERATOR")

MODEL = "gemini-2.5-flash"
client = None
if API_KEY:
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

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
) -> dict:
    if not API_KEY or not client:
        return {
            "analysis": {"emotions": [], "stress_score": None, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": "API_KEY tidak ditemukan. Cek APAAC/API.env (API_KEY_GENERATOR=...).",
            "suggested_actions": [],
            "gamification": {"streak_increment": False}
        }
    prompt = _build_curhat_payload(query, profile, mood_emoji, memory_context)
    try:
        chat = client.chats.create(
            model=MODEL,
            config={"system_instruction": SYSTEM_PROMPT, "response_mime_type": "application/json"}
        )
        r = chat.send_message(prompt)
    except Exception:
        traceback.print_exc()
        return {
            "analysis": {"emotions": [], "stress_score": None, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": "Maaf, ada kendala koneksi ke model. Coba kirim ulang ya.",
            "suggested_actions": [],
            "gamification": {"streak_increment": False}
        }

    parts = r.candidates[0].content.parts if getattr(r, "candidates", None) else []
    text = "\n".join([(getattr(p,"text",None) or getattr(getattr(p,"executable_code",None) or {}, "code","")) for p in (parts or [])]).strip()
    return _coerce_schema(_safe_json_loads(text))

def generate_opening(
    mood_emoji: str = "normal",
    memory_context: dict | None = None,
    profile: dict | None = None,
) -> dict:
    if not API_KEY or not client:
        return _coerce_schema({
            "analysis": {"emotions": [], "stress_score": 0, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": "API_KEY tidak ditemukan. Cek APAAC/API.env (API_KEY_GENERATOR=...).",
            "suggested_actions": []
        })
    prompt = _build_curhat_payload("", profile, mood_emoji, memory_context, opening=True)
    try:
        chat = client.chats.create(
            model=MODEL,
            config={"system_instruction": SYSTEM_PROMPT, "response_mime_type": "application/json"}
        )
        r = chat.send_message(prompt)
    except Exception:
        traceback.print_exc()
        return _coerce_schema({
            "analysis": {"emotions": [], "stress_score": 0, "topics": [], "risk_flag": "none"},
            "conversation_control": {"need_clarification": True, "clarify_question":"", "offer_suggestions": False, "phase":"listen"},
            "coach_reply": "Maaf, ada kendala koneksi ke model. Coba kirim ulang ya.",
            "suggested_actions": []
        })

    parts = r.candidates[0].content.parts if getattr(r, "candidates", None) else []
    text = "\n".join([(getattr(p,"text",None) or getattr(getattr(p,"executable_code",None) or {}, "code","")) for p in (parts or [])]).strip()
    return _coerce_schema(_safe_json_loads(text))


def _html_from_result(data: dict) -> str:
    a = data.get("analysis", {})
    reply = data.get("coach_reply", "")
    acts = data.get("suggested_actions", []) or []
    emos = ", ".join(a.get("emotions", [])) or "-"
    topik = ", ".join(a.get("topics", [])) or "-"
    skor = a.get("stress_score", "-")
    risiko = a.get("risk_flag", "none")

    html = []
    html.append("<div class='result' style='line-height:1.5'>")
    html.append(f"<p><b>Emosi:</b> {emos}<br><b>Skor Stres:</b> {skor}<br><b>Topik:</b> {topik}<br><b>Risiko:</b> {risiko}</p>")
    html.append(f"<p><b>Coach:</b> {reply}</p>")
    if acts:
        html.append("<p><b>Aksi yang disarankan:</b></p><ul>")
        for act in acts:
            html.append(f"<li><code>{json.dumps(act, ensure_ascii=False)}</code></li>")
        html.append("</ul>")
    html.append("</div>")
    return "".join(html)

def render_html(data: dict) -> str:
    """
    Menerima dict hasil analyze_raw() atau generate_opening()
    dan mengubahnya jadi HTML siap-tampil.
    """
    return _html_from_result(data)


def get_model_result(query: str, image_data: bytes | None = None, mood_emoji: str = "normal", memory_context: dict | None = None) -> str:
    data = analyze_raw(query, mood_emoji=mood_emoji, memory_context=memory_context)
    return _html_from_result(data)
