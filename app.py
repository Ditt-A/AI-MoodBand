import os
import secrets
from datetime import timedelta
from functools import wraps

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for
from html import escape
import traceback

import auth_store as auth
from model import analyze_raw, generate_opening, render_html
import memory_store as mem
import summarizer as sumz
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates")


def _load_secret_key() -> str:
    configured = (os.getenv("FLASK_SECRET_KEY") or "").strip()
    if configured:
        return configured
    os.makedirs(mem.DATA_DIR, exist_ok=True)
    secret_path = os.path.join(mem.DATA_DIR, "flask_secret.key")
    if os.path.exists(secret_path):
        with open(secret_path, "r", encoding="utf-8") as file:
            value = file.read().strip()
            if value:
                return value
    generated = secrets.token_hex(32)
    with open(secret_path, "w", encoding="utf-8") as file:
        file.write(generated)
    return generated


app.config.update(
    SECRET_KEY=_load_secret_key(),
    SESSION_COOKIE_NAME="moodband_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("FLASK_COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_MOODS = {'normal', 'sedih', 'marah', 'senang'}
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _store_chat_image(
    user_id: str,
    original_name: str,
    image_data: bytes,
    image_mime_type: str | None,
) -> dict:
    safe_name = secure_filename(original_name or "gambar")
    ext = safe_name.rsplit(".", 1)[1].lower() if "." in safe_name else "jpg"
    image_id = f"{secrets.token_hex(16)}.{ext}"
    user_dir = os.path.join(mem.DATA_DIR, "chat_images", user_id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, image_id)
    with open(path, "wb") as file:
        file.write(image_data)
    return {
        "id": image_id,
        "name": safe_name or f"gambar.{ext}",
        "mime": image_mime_type or "application/octet-stream",
        "url": url_for("chat_image", image_id=image_id),
    }


def _is_async_request():
    return request.headers.get('X-Requested-With') == 'fetch'


def _safe_next(default: str = "chat") -> str:
    target = (request.values.get("next") or "").strip()
    if target.startswith("/") and not target.startswith("//"):
        return target
    return url_for(default)


def _current_user() -> dict | None:
    user = auth.get_user(session.get("user_id"))
    if not user and session.get("user_id"):
        session.clear()
    return user


def _current_user_id() -> str:
    user = _current_user()
    if not user:
        raise RuntimeError("User belum login.")
    return user["id"]


def _login_user(user: dict, remember: bool = True):
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = remember
    _csrf_token()


def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _validate_csrf() -> bool:
    sent = request.form.get("_csrf_token") or request.headers.get("X-CSRF-Token")
    return bool(sent and secrets.compare_digest(sent, session.get("_csrf_token", "")))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if _current_user():
            return view(*args, **kwargs)
        if _is_async_request():
            return jsonify(ok=False, error="Sesi berakhir. Masuk lagi untuk melanjutkan."), 401
        next_url = request.full_path if request.query_string else request.path
        return redirect(url_for("signin", next=next_url))
    return wrapped


@app.before_request
def protect_post_requests():
    if request.method != "POST":
        return None
    if request.endpoint in {"static"}:
        return None
    if request.endpoint not in {"signin", "signup"} and not _current_user():
        return None
    if _validate_csrf():
        return None
    if _is_async_request():
        return jsonify(ok=False, error="Sesi keamanan kadaluwarsa. Muat ulang halaman lalu coba lagi."), 400
    mode = request.endpoint if request.endpoint in {"signin", "signup"} else "signin"
    return render_template(
        "auth.html",
        mode=mode,
        error="Sesi keamanan kadaluwarsa. Muat ulang halaman lalu coba lagi.",
        next_url=_safe_next(),
        current_user=_current_user(),
    ), 400


@app.context_processor
def inject_sidebar_conversations():
    user = _current_user()
    return {
        'sidebar_conversations': mem.list_conversations(user["id"], limit=20) if user else [],
        'current_user': user,
        'csrf_token': _csrf_token,
    }

@app.route('/')
def home():
    return render_template('landing.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    next_url = _safe_next()
    if not (request.values.get("next") or "").strip():
        next_url = url_for("chat", new=1, checkin=1)
    if request.method == "POST":
        user = auth.authenticate(
            request.form.get("email", ""),
            request.form.get("password", ""),
        )
        if not user:
            return render_template(
                "auth.html",
                mode="signin",
                error="Email atau password belum cocok.",
                next_url=next_url,
                form_email=request.form.get("email", ""),
            ), 400
        _login_user(user, remember=request.form.get("remember") == "1")
        return redirect(next_url)
    return render_template("auth.html", mode="signin", next_url=next_url)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    next_url = _safe_next()
    if not (request.values.get("next") or "").strip():
        next_url = url_for("chat", new=1, checkin=1)
    if request.method == "POST":
        password = request.form.get("password", "")
        if password != request.form.get("password_confirm", ""):
            return render_template(
                "auth.html",
                mode="signup",
                error="Konfirmasi password belum sama.",
                next_url=next_url,
                form_name=request.form.get("name", ""),
                form_email=request.form.get("email", ""),
            ), 400
        try:
            user = auth.create_user(
                request.form.get("name", ""),
                request.form.get("email", ""),
                password,
            )
        except auth.AuthError as error:
            return render_template(
                "auth.html",
                mode="signup",
                error=str(error),
                next_url=next_url,
                form_name=request.form.get("name", ""),
                form_email=request.form.get("email", ""),
            ), 400
        _login_user(user, remember=True)
        return redirect(next_url)
    return render_template("auth.html", mode="signup", next_url=next_url)


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route('/chat')
@login_required
def chat():
    user_id = _current_user_id()
    mood = request.args.get('mood', 'normal')
    if mood not in ALLOWED_MOODS:
        mood = 'normal'
    requested_id = request.args.get('conversation')
    if request.args.get('new') == '1':
        conversation = mem.get_reusable_empty_conversation(user_id) or mem.create_conversation(user_id)
        mem.prune_empty_conversation_duplicates(user_id, keep_id=conversation["id"])
        redirect_args = {'conversation': conversation['id'], 'mood': mood}
        if request.args.get('checkin') == '1':
            redirect_args['checkin'] = '1'
        return redirect(url_for('chat', **redirect_args))

    conversation = mem.get_conversation(user_id, requested_id)
    if conversation is None:
        available = mem.list_conversations(user_id, limit=30)
        conversation = available[0] if available else mem.create_conversation(user_id)
        return redirect(url_for(
            'chat',
            conversation=conversation['id'],
            mood=mood,
            **({'checkin': '1'} if request.args.get('checkin') == '1' else {}),
        ))

    recent = mem.list_recent_messages(
        user_id,
        limit=30,
        conversation_id=conversation['id'],
    )
    return render_template(
        'main.html',
        recent_messages=recent,
        prev_mood=mood,
        active_conversation=conversation,
        conversations=mem.list_conversations(user_id, limit=30),
    )


@app.route('/analytics')
@login_required
def analytics():
    user_id = _current_user_id()
    return render_template(
        'analytics.html',
        analytics=mem.get_analytics(user_id, days=7),
    )


@app.route('/chat-image/<image_id>')
@login_required
def chat_image(image_id):
    attachment = mem.get_chat_image(_current_user_id(), image_id)
    if not attachment:
        abort(404)
    return send_file(
        attachment["path"],
        mimetype=attachment["mime"],
        download_name=attachment["name"],
        conditional=True,
    )


@app.route('/image-analysis', methods=['GET', 'POST'])
@login_required
def image_analysis():
    if request.method == 'GET':
        return redirect(url_for("chat", attach=1))
    user_id = _current_user_id()

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
    recent = mem.list_recent_messages(user_id, limit=8)
    daily = mem.get_daily_summary(user_id)
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
        user_id,
        role="user",
        text=context or "Mengirim gambar untuk direfleksikan.",
        meta={"mood": mood, "has_image": True},
    )
    mem.upsert_memory(
        user_id,
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
@login_required
def daily_summary():
    user_id = _current_user_id()
    history = mem.list_daily_summaries(user_id, limit=14)
    return render_template(
        'tools.html',
        tool='summary',
        summary_history=history,
        latest_summary=history[0] if history else None,
        today_message_count=len(mem.list_today_messages(user_id)),
        chat_data_count=mem.count_chat_data(user_id),
        tool_success=(
            "Semua data chat akun ini sudah dihapus."
            if request.args.get("cleared") == "1" else None
        ),
    )


@app.route('/chat-data/clear', methods=['POST'])
@login_required
def clear_chat_data():
    mem.delete_chat_data(_current_user_id(), include_summaries=True)
    return redirect(url_for("daily_summary", cleared="1"))


@app.route('/memories')
@login_required
def memories():
    user_id = _current_user_id()
    records = mem.list_memory_records(user_id, limit=80)
    return render_template(
        'tools.html',
        tool='memory',
        memories=records,
        user_memory_count=sum(1 for item in records if item["role"] == "user"),
        coach_memory_count=sum(1 for item in records if item["role"] == "coach"),
    )


@app.route('/search', methods=['POST'])
@login_required
def search():
    try:
        user_id = _current_user_id()
        mood = request.form.get('mood', 'normal')
        query = (request.form.get('query') or '').strip()
        mode = (request.form.get('mode') or '').strip().lower()
        conversation = mem.get_conversation(
            user_id,
            request.form.get('conversation_id'),
        ) or mem.create_conversation(user_id)
        conversation_id = conversation['id']
        file = request.files.get('image')

        image_data = None
        image_mime_type = None
        image_attachment = None
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
            image_attachment = _store_chat_image(
                user_id,
                file.filename,
                image_data,
                image_mime_type,
            )

        opening = (mode == 'open' and image_data is None) or (not query and image_data is None)

        # 1) rakit memory_context
        recent = mem.list_recent_messages(
            user_id,
            limit=12,
            conversation_id=conversation_id,
        )
        daily  = mem.get_daily_summary(user_id)
        memory_context = {
            "recent_turns": recent,
            "daily_summary": daily["text"] if daily else None,
            "salient_facts": [],
            "safety_note": None
        }

        if opening:
            # MODE PEMBUKAAN: sapaan awal berdasarkan emoji
            data = generate_opening(mood_emoji=mood, memory_context=memory_context)
            # simpan hanya balasan coach (belum ada input user)
            mem.upsert_memory(
                user_id, role="coach",
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
            user_text = query or "Mengirim gambar untuk direfleksikan."
            data = analyze_raw(
                user_text,
                mood_emoji=mood,
                memory_context=memory_context,
                image_data=image_data,
                image_mime_type=image_mime_type,
            )
            # 3) simpan memori (user + coach)
            user_meta = {"mood": mood}
            if image_attachment:
                user_meta.update({
                    "has_image": True,
                    "image_id": image_attachment["id"],
                    "image_name": image_attachment["name"],
                    "image_mime": image_attachment["mime"],
                })
            mem.upsert_memory(
                user_id,
                role="user",
                text=user_text,
                meta=user_meta,
                conversation_id=conversation_id,
            )
            mem.upsert_memory(
                user_id, role="coach",
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
        conversation = mem.get_conversation(user_id, conversation_id)
        if _is_async_request():
            return jsonify(
                ok=True,
                html=result_html,
                reply=data.get("coach_reply", ""),
                mood=mood,
                conversation=conversation,
                attachment=(
                    {
                        "url": image_attachment["url"],
                        "name": image_attachment["name"],
                    }
                    if image_attachment else None
                ),
            )

        return render_template(
            'main.html',
            result=result_html,
            prev_mood=mood,
            submitted_query=query,
            submitted_image=(
                {
                    "url": image_attachment["url"],
                    "name": image_attachment["name"],
                }
                if image_attachment else None
            ),
            recent_messages=recent,
            active_conversation=conversation,
            conversations=mem.list_conversations(user_id, limit=30),
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
@login_required
def summarize():
    try:
        user_id = _current_user_id()
        mood = request.form.get('mood', 'normal')

        # ambil percakapan hari ini + carry-over ringkasan sebelumnya (jika ada)
        todays = mem.list_today_messages(user_id)
        prev = mem.get_daily_summary(user_id)
        carry = prev["text"] if prev else None

        # analytics opsional (boleh None untuk uji awal)
        out = sumz.summarize_day(todays, analytics=None, carry_over_notes=carry)

        # simpan ke memori vektor sebagai daily summary baru
        mem.upsert_daily_summary(user_id, out.get("daily_summary", ""))

        summary_html = _render_summary_card(out)
        history = mem.list_daily_summaries(user_id, limit=14)
        return render_template(
            'tools.html',
            tool='summary',
            summary_result=summary_html,
            summary_history=history,
            latest_summary=history[0] if history else None,
            today_message_count=len(todays),
            chat_data_count=mem.count_chat_data(user_id),
            prev_mood=mood,
        )
    except Exception:
        traceback.print_exc()
        user_id = session.get("user_id")
        return render_template(
            'tools.html',
            tool='summary',
            tool_error="Gagal membuat ringkasan. Periksa koneksi model lalu coba lagi.",
            summary_history=mem.list_daily_summaries(user_id, limit=14) if user_id else [],
            today_message_count=len(mem.list_today_messages(user_id)) if user_id else 0,
            chat_data_count=mem.count_chat_data(user_id) if user_id else 0,
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
