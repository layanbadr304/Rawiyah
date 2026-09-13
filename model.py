import json
import os
import re
from pathlib import Path
from difflib import SequenceMatcher

from llama_cpp import Llama

# TTS is optional. A TTS failure must never break the text answer.
try:
    from tts import speak_rawiyah
except ImportError:
    speak_rawiyah = None


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "heritage.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "ALLaM-7B-Instruct-preview-Q2_K.gguf"
MODEL_DIR.mkdir(exist_ok=True)

SAFE_FALLBACK = "لا أملك معلومات موثوقة كافية عن هذا الموضوع حاليًا."

# Accuracy-first settings.
MAX_RETRIEVED_ITEMS = 2
MIN_RETRIEVAL_SCORE = 18.0
MIN_SCORE_MARGIN = 4.0
MAX_GENERATION_ATTEMPTS = 2
DEBUG_RETRIEVAL = os.getenv("RAWIYAH_DEBUG", "0") == "1"

llm = None
_cached_items = None


# ============================================================
# Model
# ============================================================

def load_model():
    global llm

    if llm is not None:
        return llm

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "لم أجد مودل ALLaM هنا:\n"
            f"{MODEL_FILE}\n\n"
            "تأكدي أن ملف GGUF موجود داخل مجلد models وبنفس الاسم."
        )

    print("Loading Rawiyah model (ALLaM)...")

    llm = Llama(
        model_path=str(MODEL_FILE),
        n_gpu_layers=-1,
        n_ctx=3072,
        n_batch=256,
        verbose=False,
    )

    print("Model ready.")
    return llm


# ============================================================
# Data loading
# ============================================================

def load_heritage_data():
    global _cached_items

    if _cached_items is not None:
        return _cached_items

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"لم أجد ملف البيانات هنا:\n{DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(
            "صيغة heritage.json غير متوقعة. "
            "المفروض تكون قائمة أو تحتوي على المفتاح items."
        )

    # Keep only valid records with at least one human-readable identifier.
    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(isinstance(item.get(k), str) and item.get(k).strip()
               for k in ("name", "title", "topic", "id")):
            cleaned.append(item)

    _cached_items = cleaned
    return cleaned


# ============================================================
# Arabic normalization and tokenization
# ============================================================

def normalize_arabic(text):
    if not text:
        return ""

    text = str(text).lower().strip()

    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
        "ـ": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^0-9a-zA-Z\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


STOP_WORDS = {
    "احكي", "حكي", "تكلمي", "كلمني", "قولي", "حدثيني", "عرفيني", "عرفني",
    "وش", "ما", "هو", "هي", "عن", "في", "من", "على", "الى", "إلى",
    "ابي", "ابغى", "اريد", "قصة", "قصه", "حكاية", "حكايه", "رواية", "روايه",
    "لي", "لنا", "معلومات", "معلومه", "اعطني", "عطني", "لو", "عنها", "عنه",
    "هذا", "هذه", "ذلك", "وشي", "ايش", "مين", "وين", "متى", "كيف",
}


def tokens_list(text):
    return [
        token for token in normalize_arabic(text).split()
        if len(token) > 1 and token not in STOP_WORDS
    ]


def tokens_set(text):
    return set(tokens_list(text))


def clean_user_topic(question):
    """Remove conversational filler so 'وش قصة الدرعية؟' becomes roughly 'الدرعيه'."""
    return " ".join(tokens_list(question)).strip()


def text_values(value):
    values = []

    if isinstance(value, str):
        if value.strip():
            values.append(value.strip())

    elif isinstance(value, list):
        for item in value:
            values.extend(text_values(item))

    elif isinstance(value, dict):
        preferred = (
            "title", "name", "text", "description", "label", "value",
            "source_name", "source_type", "region"
        )
        used = set()
        for key in preferred:
            if key in value:
                values.extend(text_values(value[key]))
                used.add(key)
        for key, val in value.items():
            if key not in used and isinstance(val, str):
                values.extend(text_values(val))

    return values


# ============================================================
# Accuracy-first retrieval
# ============================================================

def item_name(item):
    return (
        item.get("name")
        or item.get("title")
        or item.get("topic")
        or item.get("id")
        or ""
    )


def item_aliases(item):
    aliases = []
    for field in ("aliases", "tags"):
        values = item.get(field, [])
        if isinstance(values, list):
            aliases.extend(str(x).strip() for x in values if str(x).strip())
    return aliases


def searchable_text(item):
    parts = []
    for field in (
        "name", "title", "topic", "id", "aliases", "tags", "region", "type",
        "short_description", "summary", "description", "facts"
    ):
        parts.extend(text_values(item.get(field)))
    # Intentionally exclude story/chapters from retrieval. They can contain broad prose
    # and make a related record outrank the exact entity the user asked for.
    return " ".join(parts)


def _phrase_match_score(query_topic, candidate):
    """High score for exact/contained entity matches; lower score for fuzzy similarity."""
    q = normalize_arabic(query_topic)
    c = normalize_arabic(candidate)

    if not q or not c:
        return 0.0

    if q == c:
        return 120.0

    # Candidate appears as a full phrase in the user's cleaned query.
    if c in q:
        return 70.0

    # User asked for a shorter entity that appears inside a longer candidate.
    # This is useful, but deliberately much weaker than exact matching.
    if q in c:
        length_penalty = max(0, len(c.split()) - len(q.split())) * 4
        return max(12.0, 32.0 - length_penalty)

    # Fuzzy matching helps small spelling mistakes without overriding exact matches.
    ratio = SequenceMatcher(None, q, c).ratio()
    if ratio >= 0.90:
        return 28.0 * ratio
    if ratio >= 0.80:
        return 18.0 * ratio

    return 0.0


def score_item(question, item):
    q_topic = clean_user_topic(question)
    q_tokens = tokens_set(question)
    if not q_tokens:
        return 0.0

    name = item_name(item)
    score = _phrase_match_score(q_topic, name)

    # Aliases/tags help variants such as "مدائن صالح" -> "الحِجر".
    best_alias = 0.0
    for alias in item_aliases(item):
        best_alias = max(best_alias, _phrase_match_score(q_topic, alias) * 0.80)
    score += best_alias

    # Token overlap is supporting evidence, not the main decision-maker.
    searchable_tokens = tokens_set(searchable_text(item))
    overlap = q_tokens & searchable_tokens
    score += len(overlap) * 4.0

    # Region and type are mild supporting signals.
    region = item.get("region")
    if isinstance(region, str) and normalize_arabic(region) in normalize_arabic(question):
        score += 3.0

    item_type = item.get("type")
    if isinstance(item_type, str):
        type_tokens = tokens_set(item_type)
        score += len(q_tokens & type_tokens) * 1.5

    return score


def retrieve_items(question, items, limit=MAX_RETRIEVED_ITEMS):
    scored = [(score_item(question, item), item) for item in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = [(score, item) for score, item in scored if score > 0]

    if not scored:
        return [], {"reason": "no_match"}

    best_score = scored[0][0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    if DEBUG_RETRIEVAL:
        print("\n[Retrieval debug]")
        for score, item in scored[:5]:
            print(f"{score:6.1f}  {item_name(item)}")

    # Refuse weak matches rather than guessing.
    if best_score < MIN_RETRIEVAL_SCORE:
        return [], {
            "reason": "low_confidence",
            "best_score": best_score,
            "second_score": second_score,
        }

    # If two very different candidates are nearly tied and neither is an exact entity match,
    # answering is riskier than refusing.
    q_topic = clean_user_topic(question)
    best_exact = normalize_arabic(item_name(scored[0][1])) == normalize_arabic(q_topic)
    if (
        not best_exact
        and second_score > 0
        and (best_score - second_score) < MIN_SCORE_MARGIN
        and best_score < 65
    ):
        return [], {
            "reason": "ambiguous",
            "best_score": best_score,
            "second_score": second_score,
        }

    # For a direct entity question, one record is safest.
    if best_exact or best_score >= 65:
        return [scored[0][1]], {
            "reason": "high_confidence",
            "best_score": best_score,
            "second_score": second_score,
        }

    # Otherwise allow at most one extra record, and only if it is genuinely close.
    results = [scored[0][1]]
    if limit > 1 and len(scored) > 1:
        if scored[1][0] >= best_score * 0.78:
            results.append(scored[1][1])

    return results[:limit], {
        "reason": "confident",
        "best_score": best_score,
        "second_score": second_score,
    }


# ============================================================
# Evidence extraction (FACTS ONLY — no story copying)
# ============================================================

def _fact_strings(item):
    facts = []
    raw = item.get("facts", [])

    if isinstance(raw, list):
        for fact in raw:
            if isinstance(fact, str) and fact.strip():
                facts.append(fact.strip())
            elif isinstance(fact, dict):
                label = fact.get("label") or fact.get("title") or fact.get("name") or ""
                value = fact.get("value") or fact.get("text") or fact.get("description") or ""
                if isinstance(value, str) and value.strip():
                    if label:
                        facts.append(f"{label}: {value.strip()}")
                    else:
                        facts.append(value.strip())

    return facts


def _source_records(item):
    sources = []
    raw = item.get("sources", [])
    if not isinstance(raw, list):
        return sources

    for source in raw:
        if isinstance(source, str) and source.strip():
            sources.append({"name": source.strip(), "url": ""})
        elif isinstance(source, dict):
            name = (
                source.get("source_name")
                or source.get("name")
                or source.get("title")
                or source.get("source")
                or ""
            )
            url = source.get("source_url") or source.get("url") or ""
            if name or url:
                sources.append({"name": str(name).strip(), "url": str(url).strip()})
    return sources


def build_evidence(items):
    """
    Build a compact evidence ledger. We intentionally DO NOT pass story/chapters to ALLaM.
    This forces the model to synthesize from structured facts instead of copying prose.
    """
    blocks = []
    all_facts = []
    all_sources = []

    for idx, item in enumerate(items, start=1):
        name = item_name(item)
        region = item.get("region") if isinstance(item.get("region"), str) else ""
        item_type = item.get("type") if isinstance(item.get("type"), str) else ""
        short_description = (
            item.get("short_description")
            if isinstance(item.get("short_description"), str)
            else ""
        )
        facts = _fact_strings(item)
        sources = _source_records(item)

        # If a record has no facts at all, the short description can act as one conservative fact.
        evidence_facts = list(facts)
        if not evidence_facts and short_description.strip():
            evidence_facts.append(short_description.strip())

        if not evidence_facts:
            continue

        lines = [f"[ENTITY {idx}]", f"الاسم: {name}"]
        if item_type:
            lines.append(f"النوع: {item_type}")
        if region:
            lines.append(f"المنطقة: {region}")

        lines.append("الحقائق المسموح استخدامها:")
        for fidx, fact in enumerate(evidence_facts, start=1):
            fact_id = f"E{idx}F{fidx}"
            lines.append(f"{fact_id}: {fact}")
            all_facts.append({"id": fact_id, "text": fact})

        if sources:
            lines.append("المصادر المرجعية:")
            for source in sources:
                label = source["name"] or "مصدر"
                if source["url"]:
                    lines.append(f"- {label}: {source['url']}")
                else:
                    lines.append(f"- {label}")
                all_sources.append(source)

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks).strip(), all_facts, all_sources


# ============================================================
# Validation helpers
# ============================================================

def extract_numbers(text):
    # Arabic-Indic + Western digits, including years and simple decimal forms.
    return set(re.findall(r"[0-9٠-٩]+(?:[.,٫][0-9٠-٩]+)?", text or ""))


def normalize_digits(text):
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return (text or "").translate(trans)


def validate_numeric_grounding(answer, evidence_text):
    answer_nums = extract_numbers(normalize_digits(answer))
    evidence_nums = extract_numbers(normalize_digits(evidence_text))
    unsupported = answer_nums - evidence_nums
    return len(unsupported) == 0, sorted(unsupported)


def sentence_content_overlap(sentence, evidence_text):
    """
    Conservative lexical support check. It is intentionally not perfect Arabic NLI,
    but it catches many sentences that drift far away from the supplied facts.
    """
    s_tokens = tokens_set(sentence)
    if not s_tokens:
        return 1.0

    e_tokens = tokens_set(evidence_text)
    if not e_tokens:
        return 0.0

    supported = s_tokens & e_tokens
    return len(supported) / max(1, len(s_tokens))


def validate_lexical_grounding(answer, evidence_text):
    sentences = [s.strip() for s in re.split(r"[.!؟?\n]+", answer) if s.strip()]
    if not sentences:
        return False, []

    weak = []
    for sentence in sentences:
        ratio = sentence_content_overlap(sentence, evidence_text)
        # Paraphrasing creates some new function words, so this threshold is deliberately moderate.
        if ratio < 0.38:
            weak.append((sentence, ratio))

    return len(weak) == 0, weak


def validate_forbidden_meta(answer):
    forbidden = (
        "قاعدة البيانات", "البرومبت", "التعليمات", "المعلومات الموثوقة",
        "لا استطيع الوصول", "كنموذج", "كمودل"
    )
    norm = normalize_arabic(answer)
    bad = [p for p in forbidden if normalize_arabic(p) in norm]
    return len(bad) == 0, bad


def validate_answer(answer, evidence_text):
    if not answer or len(answer.strip()) < 8:
        return False, {"reason": "empty_or_too_short"}

    numeric_ok, unsupported_numbers = validate_numeric_grounding(answer, evidence_text)
    lexical_ok, weak_sentences = validate_lexical_grounding(answer, evidence_text)
    meta_ok, forbidden_meta = validate_forbidden_meta(answer)

    ok = numeric_ok and lexical_ok and meta_ok
    return ok, {
        "numeric_ok": numeric_ok,
        "unsupported_numbers": unsupported_numbers,
        "lexical_ok": lexical_ok,
        "weak_sentences": weak_sentences,
        "meta_ok": meta_ok,
        "forbidden_meta": forbidden_meta,
    }


# ============================================================
# Generation
# ============================================================

def _call_model(system_message, user_message, max_tokens=150, temperature=0.15):
    model = load_model()

    try:
        result = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.75,
            repeat_penalty=1.12,
        )
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        prompt = f"{system_message}\n\n{user_message}\n\nراوية:"
        result = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.7,
            repeat_penalty=1.1,
            stop=["المستخدم:", "سؤال المستخدم:", "SYSTEM:"],
            echo=False,
        )
        return result["choices"][0]["text"].strip()


def generate_story(question, evidence_text):
    system_message = """
أنتِ راوية.
أجيبي فقط اعتمادًا على المعلومات المرفقة.

ممنوع إضافة أي معلومة أو وصف أو علاقة غير موجودة صراحة في المعلومات.
حافظي على معنى العلاقات كما وردت دون تقويتها أو تغييرها.
مثال: "في عهد" تبقى "في عهد"، و"ارتبط بـ" لا تتحول إلى "لعب دورًا في".

إذا لم تكفِ المعلومات، قولي:
"لا أملك معلومات موثوقة كافية عن هذا الموضوع حاليًا."

- عند الشك، استخدمي صياغة أقرب ما يمكن إلى wording الحقائق الأصلية.
""" .strip()

  
    user_message = f"""
المعلومات الموثوقة:
{evidence_text}

السؤال:
{question}


""".strip()

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        temperature = 0.12 if attempt == 0 else 0.0
        answer = _call_model(
            system_message,
            user_message,
            max_tokens=150,
            temperature=temperature,
        )

        if answer == SAFE_FALLBACK:
            return answer

        ok, details = validate_answer(answer, evidence_text)
        if ok:
            return answer

        if DEBUG_RETRIEVAL:
            print(f"[Validation failed attempt {attempt + 1}] {details}")

        # Retry with validation feedback but without exposing it to the user.
        user_message = f"""
الأدلة:
{evidence_text}

سؤال المستخدم:
{question}

المحاولة السابقة لم تجتز فحص الدقة لأنها احتوت صياغة غير مدعومة بشكل كافٍ.
أعيدي الإجابة بجملتين قصيرتين جدًا، مستخدمةً فقط ما هو مذكور صراحة في الأدلة.
لا تضيفي أي كلمة معلوماتية جديدة غير مدعومة.
""".strip()

    # Accuracy beats fluency: if the answer cannot be validated, refuse safely.
    return SAFE_FALLBACK


# ============================================================
# Public question function
# ============================================================

def ask_rawiyah(question):
    question = (question or "").strip()
    if not question:
        return SAFE_FALLBACK

    items = load_heritage_data()
    matches, retrieval_info = retrieve_items(question, items)

    if DEBUG_RETRIEVAL:
        print(f"[Retrieval status] {retrieval_info}")

    if not matches:
        return SAFE_FALLBACK

    evidence_text, facts, sources = build_evidence(matches)
    if not evidence_text or not facts:
        return SAFE_FALLBACK

    return generate_story(question, evidence_text)


# Optional helper for the Flask/UI layer: answer + trusted sources separately.
def ask_rawiyah_with_metadata(question):
    question = (question or "").strip()
    if not question:
        return {"answer": SAFE_FALLBACK, "sources": [], "entities": []}

    items = load_heritage_data()
    matches, retrieval_info = retrieve_items(question, items)
    if not matches:
        return {
            "answer": SAFE_FALLBACK,
            "sources": [],
            "entities": [],
            "retrieval": retrieval_info,
        }

    evidence_text, facts, sources = build_evidence(matches)
    if not evidence_text or not facts:
        return {
            "answer": SAFE_FALLBACK,
            "sources": [],
            "entities": [item_name(x) for x in matches],
            "retrieval": retrieval_info,
        }

    answer = generate_story(question, evidence_text)
    return {
        "answer": answer,
        "sources": sources,
        "entities": [item_name(x) for x in matches],
        "retrieval": retrieval_info,
    }


# ============================================================
# TTS + terminal demo
# ============================================================

def maybe_speak(answer):
    if speak_rawiyah is None or answer == SAFE_FALLBACK:
        return

    try:
        audio_file = speak_rawiyah(answer, "answer.mp3")
        print("تم إنشاء الصوت:")
        print(audio_file)
    except Exception as exc:
        print(f"ملاحظة: تعذر إنشاء الصوت: {exc}")


if __name__ == "__main__":
    print("\nراوية جاهزة ✦")
    print("اكتبي سؤالك، أو اكتبي خروج لإنهاء البرنامج.\n")

    while True:
        question = input("أنتِ: ").strip()

        if question.lower() in {"خروج", "exit", "quit", "q"}:
            print("راوية: إلى لقاء قريب ✦")
            break

        if not question:
            continue

        try:
            answer = ask_rawiyah(question)
            print("\nراوية:")
            print(answer)
            maybe_speak(answer)
            print()
        except Exception as exc:
            print("\nحدث خطأ:")
            print(exc)
            print()
