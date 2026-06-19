from flask import Flask, jsonify, redirect, render_template, request, url_for
from html import escape
import traceback

from model import analyze_raw, generate_opening, render_html
import memory_store as mem
import summarizer as sumz

USER_ID = "demo_user"

app = Flask(__name__, template_folder="templates")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_MOODS = {'normal', 'sedih', 'marah', 'senang'}
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _is_async_request():
    return request.headers.get('X-Requested-With') == 'fetch'


@app.context_processor
def inject_sidebar_conversations():
    return {
        'sidebar_conversations': mem.list_conversations(USER_ID, limit=20),
    }

@app.route('/')
def home():
    return render_template('landing.html')


@app.route('/chat')
def chat():
    mood = request.args.get('mood', 'normal')
    if mood not in ALLOWED_MOODS:
        mood = 'normal'
    requested_id = request.args.get('conversation')
    if request.args.get('new') == '1':
        conversation = mem.create_conversation(USER_ID)
        return redirect(url_for('chat', conversation=conversation['id'], mood=mood))

    conversation = mem.get_conversation(USER_ID, requested_id)
    if conversation is None:
        available = mem.list_conversations(USER_ID, limit=30)
        conversation = available[0] if available else mem.create_conversation(USER_ID)
        return redirect(url_for(
            'chat',
            conversation=conversation['id'],
            mood=mood,
            **({'checkin': '1'} if request.args.get('checkin') == '1' else {}),
        ))

    recent = mem.list_recent_messages(
        USER_ID,
        limit=30,
        conversation_id=conversation['id'],
    )
    return render_template(
        'main.html',
        recent_messages=recent,
        prev_mood=mood,
        active_conversation=conversation,
        conversations=mem.list_conversations(USER_ID, limit=30),
    )


@app.route('/analytics')
def analytics():
    return render_template(
        'analytics.html',
        analytics=mem.get_analytics(USER_ID, days=7),
    )


@app.route('/image-analysis', methods=['GET', 'POST'])
def image_analysis():
    if request.method == 'GET':
        return render_template('tools.html', tool='image')

    mood = request.form.get('mood', 'normal')
    context = (request.form.get('context') or '').strip()
    file = request.files.get('image')
    if not file or not file.filename:
        return render_template(
            'tools.html',
            tool='image',
            tool_error="Pilih gambar terlebih dahulu.",
            prev_mood=mood,
            submitted_context=context,
        ), 400
    if not allowed_file(file.filename):
        return render_template(
            'tools.html',
            tool='image',
            tool_error="Format belum didukung. Gunakan PNG, JPG, GIF, atau WEBP.",
            prev_mood=mood,
            submitted_context=context,
        ), 400

    image_data = file.read()
    prompt = context or "Bantu aku merefleksikan konteks emosional dari gambar ini dengan lembut."
    recent = mem.list_recent_messages(USER_ID, limit=8)
    daily = mem.get_daily_summary(USER_ID)
    data = analyze_raw(
        prompt,
        mood_emoji=mood,
        memory_context={
            "recent_turns": recent,
            "daily_summary": daily["text"] if daily else None,
            "salient_facts": [],
            "safety_note": None,
        },
        image_data=image_data,
        image_mime_type=file.mimetype,
    )
    mem.upsert_memory(
        USER_ID,
        role="user",
        text=context or "Mengirim gambar untuk direfleksikan.",
        meta={"mood": mood, "has_image": True},
    )
    mem.upsert_memory(
        USER_ID,
        role="coach",
        text=data.get("coach_reply", ""),
        meta={
            "emotions": data.get("analysis", {}).get("emotions", []),
            "stress": data.get("analysis", {}).get("stress_score"),
            "topics": data.get("analysis", {}).get("topics", []),
            "risk": data.get("analysis", {}).get("risk_flag", "none"),
            "from_image": True,
        },
    )
    return render_template(
        'tools.html',
        tool='image',
        image_result=render_html(data),
        image_name=file.filename,
        prev_mood=mood,
        submitted_context=context,
    )


@app.route('/daily-summary')
def daily_summary():
    history = mem.list_daily_summaries(USER_ID, limit=14)
    return render_template(
        'tools.html',
        tool='summary',
        summary_history=history,
        latest_summary=history[0] if history else None,
        today_message_count=len(mem.list_today_messages(USER_ID)),
    )


@app.route('/memories')
def memories():
    records = mem.list_memory_records(USER_ID, limit=80)
    return render_template(
        'tools.html',
        tool='memory',
        memories=records,
        user_memory_count=sum(1 for item in records if item["role"] == "user"),
        coach_memory_count=sum(1 for item in records if item["role"] == "coach"),
    )


@app.route('/search', methods=['POST'])
def search():
    try:
        mood = request.form.get('mood', 'normal')
        query = (request.form.get('query') or '').strip()
        mode = (request.form.get('mode') or '').strip().lower()
        conversation = mem.get_conversation(
            USER_ID,
            request.form.get('conversation_id'),
        ) or mem.create_conversation(USER_ID)
        conversation_id = conversation['id']
        file = request.files.get('image')
        opening = mode == 'open' or not query

        image_data = None
        image_mime_type = None
        if file and file.filename:
            if not allowed_file(file.filename):
                if _is_async_request():
                    return jsonify(
                        ok=False,
                        error="Format gambar tidak didukung. Gunakan PNG, JPG, GIF, atau WEBP."
                    ), 400
                return render_template(
                    'main.html',
                    result="<div class='result'>Format gambar tidak didukung. Gunakan png/jpg/jpeg/gif/webp.</div>",
                    prev_mood=mood
                )
            image_data = file.read()
            image_mime_type = file.mimetype

        # 1) rakit memory_context
        recent = mem.list_recent_messages(
            USER_ID,
            limit=12,
            conversation_id=conversation_id,
        )
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
                conversation_id=conversation_id,
                meta={
                    "emotions": data.get("analysis",{}).get("emotions",[]),
                    "stress":   data.get("analysis",{}).get("stress_score"),
                    "topics":   data.get("analysis",{}).get("topics",[]),
                    "risk":     data.get("analysis",{}).get("risk_flag", "none"),
                    "opening":  True,
                    "mood":     mood
                }
            )
        else:
            # MODE BIASA: user -> model
            data = analyze_raw(
                query,
                mood_emoji=mood,
                memory_context=memory_context,
                image_data=image_data,
                image_mime_type=image_mime_type,
            )
            # 3) simpan memori (user + coach)
            mem.upsert_memory(
                USER_ID,
                role="user",
                text=query,
                meta={"mood": mood},
                conversation_id=conversation_id,
            )
            mem.upsert_memory(
                USER_ID, role="coach",
                text=data.get("coach_reply",""),
                conversation_id=conversation_id,
                meta={
                    "emotions": data.get("analysis",{}).get("emotions",[]),
                    "stress":   data.get("analysis",{}).get("stress_score"),
                    "topics":   data.get("analysis",{}).get("topics",[]),
                    "risk":     data.get("analysis",{}).get("risk_flag", "none")
                }
            )
            # Ringkasan dibuat hanya saat pengguna memilih tool ringkasan.
        result_html = render_html(data)
        conversation = mem.get_conversation(USER_ID, conversation_id)
        if _is_async_request():
            return jsonify(
                ok=True,
                html=result_html,
                reply=data.get("coach_reply", ""),
                mood=mood,
                conversation=conversation,
            )

        return render_template(
            'main.html',
            result=result_html,
            prev_mood=mood,
            submitted_query=query,
            submitted_image=bool(image_data),
            recent_messages=recent,
            active_conversation=conversation,
            conversations=mem.list_conversations(USER_ID, limit=30),
        )

    except Exception:
        print("=== FLASK ROUTE ERROR ===")
        traceback.print_exc()
        if _is_async_request():
            return jsonify(
                ok=False,
                error="Terjadi kendala saat memproses pesan. Coba kirim ulang, ya."
            ), 500
        return render_template(
            'main.html',
            result="<div class='result'>Terjadi error di server. Cek log terminal untuk detail.</div>"
        )


@app.errorhandler(413)
def upload_too_large(_error):
    message = "Ukuran gambar terlalu besar. Maksimum 6 MB."
    if _is_async_request():
        return jsonify(ok=False, error=message), 413
    if request.path == '/image-analysis':
        return render_template(
            'tools.html',
            tool='image',
            tool_error=message,
        ), 413
    return render_template(
        'main.html',
        result=f"<div class='result'>{message}</div>",
    ), 413

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
        history = mem.list_daily_summaries(USER_ID, limit=14)
        return render_template(
            'tools.html',
            tool='summary',
            summary_result=summary_html,
            summary_history=history,
            latest_summary=history[0] if history else None,
            today_message_count=len(todays),
            prev_mood=mood,
        )
    except Exception:
        traceback.print_exc()
        return render_template(
            'tools.html',
            tool='summary',
            tool_error="Gagal membuat ringkasan. Periksa koneksi model lalu coba lagi.",
            summary_history=mem.list_daily_summaries(USER_ID, limit=14),
            today_message_count=len(mem.list_today_messages(USER_ID)),
        )

def _render_summary_card(out: dict) -> str:
    def _render_items(items: list[str]) -> str:
        if not items:
            return "<li class='muted-item'>(tidak ada)</li>"
        return "".join(f"<li>{escape(str(item))}</li>" for item in items)

    key_points = _render_items(out.get("key_points") or [])
    follow_up = _render_items(out.get("follow_up_tomorrow") or [])
    safe = "Ya" if out.get("safety_flag") else "Tidak"
    safety_class = "summary-safe" if not out.get("safety_flag") else "summary-alert"
    summary_text = escape(out.get("daily_summary", "(kosong)"))
    return (
        "<div class='summary-card'>"
        "<div class='summary-lede'>"
        f"<p class='summary-copy'>{summary_text}</p>"
        f"<span class='summary-badge {safety_class}'>Safety flag: {safe}</span>"
        "</div>"
        "<div class='summary-grid'>"
        "<section class='summary-section'>"
        "<h4>Poin kunci</h4>"
        f"<ul class='clean-list'>{key_points}</ul>"
        "</section>"
        "<section class='summary-section'>"
        "<h4>Follow-up besok</h4>"
        f"<ul class='clean-list'>{follow_up}</ul>"
        "</section>"
        "</div>"
        "</div>"
    )


if __name__ == '__main__':
    app.run(debug=True, port=8000)
