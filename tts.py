import asyncio
from pathlib import Path
import edge_tts


VOICE = "ar-EG-SalmaNeural"
RATE = "-12%"
PITCH = "-3Hz"
VOLUME = "+0%"

OUTPUT_DIR = Path(__file__).parent / "audio"
OUTPUT_DIR.mkdir(exist_ok=True)


async def text_to_speech(text, filename="rawiyah.mp3"):
    output_path = OUTPUT_DIR / filename

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        pitch=PITCH,
        volume=VOLUME
    )

    await communicate.save(str(output_path))

    return output_path


def speak_rawiyah(text, filename="rawiyah.mp3"):
    return asyncio.run(
        text_to_speech(
            text=text,
            filename=filename
        )
    )


if __name__ == "__main__":
    text = "لكل مكان رواية، وراوية ترويها لك."

    file_path = speak_rawiyah(
        text,
        "intro.mp3"
    )

    print("تم إنشاء الصوت بنجاح:")
    print(file_path)