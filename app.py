from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, traceback

from model import analyze_raw, generate_opening, render_html
import memory_store as mem
import summarizer as sumz

USER_ID = "demo_user"

app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('main.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        mood = request.form.get('mood', 'normal')
        query = request.form['query']
        file = request.files.get('image')
        opening = request.form.get('opening') == '1'

        image_data = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

        # 1) rakit memory_context
        recent = mem.list_recent_messages(USER_ID, limit=8)
        daily  = mem.get_daily_summary(USER_ID)
        memory_context = {
            "recent_turns": recent,
            "daily_summary": daily["text"] if daily else None,
            "salient_facts": [],
            "safety_note": None
        }

        if opening or not query:
            # MODE PEMBUKAAN: sapaan awal berdasarkan emoji
            data = generate_opening(mood_emoji=mood, memory_context=memory_context)
            # simpan hanya balasan coach (belum ada input user)
            mem.upsert_memory(
                USER_ID, role="coach",
                text=data.get("coach_reply",""),
                meta={
                    "emotions": data.get("analysis",{}).get("emotions",[]),
                    "stress":   data.get("analysis",{}).get("stress_score"),
                    "topics":   data.get("analysis",{}).get("topics",[]),
                    "opening":  True,
                    "mood":     mood
                }
            )
        else:
            # MODE BIASA: user -> model
            data = analyze_raw(query, mood_emoji=mood, memory_context=memory_context)
            # 3) simpan memori (user + coach)
            mem.upsert_memory(USER_ID, role="user",  text=query, meta={"mood": mood})
            mem.upsert_memory(
                USER_ID, role="coach",
                text=data.get("coach_reply",""),
                meta={
                    "emotions": data.get("analysis",{}).get("emotions",[]),
                    "stress":   data.get("analysis",{}).get("stress_score"),
                    "topics":   data.get("analysis",{}).get("topics",[])
                }
            )
            # 4) ringkasan harian (opsional — hanya kalau ada input user)
            if not mem.today_has_summary(USER_ID):
                todays = mem.list_today_messages(USER_ID)
                analytics = {
                    "avg_stress": data.get("analysis",{}).get("stress_score"),
                    "max_stress": data.get("analysis",{}).get("stress_score"),
                    "top_emotions": data.get("analysis",{}).get("emotions",[]),
                    "top_topics": data.get("analysis",{}).get("topics",[])
                }
                daily_sum = sumz.summarize_day(
                    todays,
                    analytics=analytics,
                    carry_over_notes=(daily["text"] if daily else None)
                )
                mem.upsert_daily_summary(USER_ID, daily_sum.get("daily_summary",""))

        # 5) render ke HTML
        banner = (
            f"<div class='alert'>Konteks: {len(recent)} memori"
            f"{' + ringkasan' if daily else ''}</div>"
        )
        result_html = banner + render_html(data)
        return render_template('main.html', result=result_html, prev_mood=mood)

    except Exception:
        print("=== FLASK ROUTE ERROR ===")
        traceback.print_exc()
        return render_template(
            'main.html',
            result="<div class='result'>Terjadi error di server. Cek log terminal untuk detail.</div>"
        )

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        mood = request.form.get('mood', 'normal')

        # ambil percakapan hari ini + carry-over ringkasan sebelumnya (jika ada)
        todays = mem.list_today_messages(USER_ID)
        prev = mem.get_daily_summary(USER_ID)
        carry = prev["text"] if prev else None

        # analytics opsional (boleh None untuk uji awal)
        out = sumz.summarize_day(todays, analytics=None, carry_over_notes=carry)

        # simpan ke memori vektor sebagai daily summary baru
        mem.upsert_daily_summary(USER_ID, out.get("daily_summary", ""))

        summary_html = _render_summary_card(out)
        return render_template('main.html', summary=summary_html, prev_mood=mood)
    except Exception:
        traceback.print_exc()
        return render_template(
            'main.html',
            summary="<div class='result'>Gagal membuat ringkasan. Cek log terminal.</div>"
        )

def _render_summary_card(out: dict) -> str:
    kp = "".join(f"<li>{x}</li>" for x in (out.get("key_points") or []))
    fu = "".join(f"<li>{x}</li>" for x in (out.get("follow_up_tomorrow") or []))
    safe = "Ya" if out.get("safety_flag") else "Tidak"
    return (
        "<div class='result'>"
        "<h2>Ringkasan Harian</h2>"
        f"<p>{out.get('daily_summary','(kosong)')}</p>"
        "<h3>Poin Kunci</h3>"
        f"<ul>{kp or '<li>(tidak ada)</li>'}</ul>"
        "<h3>Follow-up Besok</h3>"
        f"<ul>{fu or '<li>(tidak ada)</li>'}</ul>"
        f"<p><b>Safety flag:</b> {safe}</p>"
        "</div>"
    )


if __name__ == '__main__':
    app.run(debug=True, port=8000)
