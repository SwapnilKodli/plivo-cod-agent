"""Synthesize the scripted caller's lines with Koyal (batch API) as mu-law 8 kHz,
the exact format Plivo streams from a real phone call.

    python -m bench.make_fixtures
"""

import audioop  # stdlib through Python 3.12
import base64
import io
import json
import os
import wave
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

OUT = Path(__file__).parent / "fixtures"
CALLER_VOICE = os.getenv("CALLER_VOICE_ID", "OmziUEKO4d83QuGMo3bP")  # Aman (hi, M): a different voice from the agent

# Code-mixed caller turns for a COD confirmation call. "barge_in" turns are spoken
# over the agent to measure interruption handling.
SCRIPTS = {
    "happy_path": [
        {"text": "हाँ, मैं रोहन बोल रहा हूँ।"},
        {"text": "हाँ हाँ, order sahi hai, paneer butter masala aur naan."},
        {"text": "Address bhi correct hai, बस landmark add कर दो, Total petrol pump के पास।"},
        {"text": "नहीं, cash ही दूँगा, UPI नहीं चाहिए।"},
        {"text": "हाँ, confirm कर दो। Thank you."},
    ],
    "cancel_dnd": [
        {"text": "Haan bolo, kaun?"},
        {"text": "Maine koi order nahi kiya, ye galti se ho gaya, cancel कर दो।", "barge_in": True},
        {"text": "और please मुझे दोबारा call मत करना।"},
    ],
    "reschedule_prepaid": [
        {"text": "हाँ जी, बोलिए।"},
        {"text": "Aaj ghar pe koi nahi hoga, kal subah dus baje bhej sakte ho?"},
        {"text": "UPI se pay karne pe discount milega? Theek hai, link bhej do."},
        {"text": "Bas itna hi, thanks."},
    ],
}


def synth(text: str) -> bytes:
    if not os.getenv("PLIVO_AUTH_ID"):
        return synth_sarvam(text)
    auth_id, token = os.environ["PLIVO_AUTH_ID"], os.environ["PLIVO_AUTH_TOKEN"]
    r = requests.post(
        f"https://api.plivo.com/v1/Account/{auth_id}/TTS/",
        auth=(auth_id, token),
        json={
            "voice_id": CALLER_VOICE,
            "text": text,
            "language": "hi",
            "output_format": {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.content


def synth_sarvam(text: str) -> bytes:
    """Fallback when there is no Plivo account: Sarvam bulbul, converted to mu-law 8 kHz."""
    r = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]},
        json={"text": text, "target_language_code": "hi-IN", "model": "bulbul:v3",
              "speaker": os.getenv("CALLER_SARVAM_VOICE", "rohan"), "speech_sample_rate": 8000},
        timeout=60,
    )
    r.raise_for_status()
    with wave.open(io.BytesIO(base64.b64decode(r.json()["audios"][0]))) as w:
        pcm, width, rate = w.readframes(w.getnframes()), w.getsampwidth(), w.getframerate()
    if rate != 8000:
        pcm, _ = audioop.ratecv(pcm, width, 1, rate, 8000, None)
    return audioop.lin2ulaw(pcm, width)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, turns in SCRIPTS.items():
        manifest[name] = []
        for i, turn in enumerate(turns):
            path = OUT / f"{name}_{i}.ulaw"
            if not path.exists():
                path.write_bytes(synth(turn["text"]))
            manifest[name].append({**turn, "file": path.name})
            print(f"{path.name}: {turn['text']}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
