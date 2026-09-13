from flask import Flask, request, jsonify, send_file, send_from_directory
from pathlib import Path

from model import ask_rawiyah
from tts import speak_rawiyah

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_DIR = BASE_DIR / "audio"

AUDIO_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


# الواجهة
@app.route("/")
def home():
    return send_file(BASE_DIR / "index.html")


# الصور واللوقو
@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


# ملفات الصوت
@app.route("/audio/<path:filename>")
def audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# سؤال راوية
@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400

    try:
        # المودل
        answer = ask_rawiyah(question)

        # تحويل الإجابة إلى صوت
        audio_path = speak_rawiyah(
            answer,
            str(AUDIO_DIR / "answer.mp3")
        )

        return jsonify({
            "answer": answer,
            "audio_url": "/audio/answer.mp3"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)