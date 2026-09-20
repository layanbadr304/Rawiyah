from flask import Flask, request, jsonify, send_file, send_from_directory
from pathlib import Path

from model import ask_rawiyah
from tts import speak_rawiyah

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_DIR = BASE_DIR / "audio"

AUDIO_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


def detect_topic(question: str):
    """
    يحدد الرواية الأقرب من سؤال المستخدم.
    نستخدم مطابقة كلمات بسيطة للـMVP، ويمكن لاحقًا نقلها إلى model.py.
    """
    q = question.strip().lower()

    topics = {
        "masmak": [
            "المصمك",
            "قصر المصمك",
            "حصن المصمك",
        ],
        "diriyah": [
            "الدرعية",
            "الطريف",
            "حي الطريف",
            "الدولة السعودية الأولى",
            "محمد بن سعود",
        ],
        "sadu": [
            "السدو",
            "نسيج السدو",
            "حرفة السدو",
            "النقوش",
        ],
    }

    for topic, keywords in topics.items():
        if any(keyword in q for keyword in keywords):
            return topic

    return None


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
    data = request.get_json() or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "السؤال فارغ"}), 400

    try:
        # تحديد الرواية المناسبة
        topic = detect_topic(question)

        # المودل
        answer = ask_rawiyah(question)

        # تحويل الإجابة إلى صوت
        audio_path = speak_rawiyah(
            answer,
            str(AUDIO_DIR / "answer.mp3")
        )

        return jsonify({
            "answer": answer,
            "audio_url": "/audio/answer.mp3",
            "topic": topic
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
