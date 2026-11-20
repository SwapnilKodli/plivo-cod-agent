"""Replay scripted caller audio into the agent over Plivo's Audio Stream protocol
and measure what a caller would actually experience on the wire.

It impersonates Plivo: sends a `start` event, then a continuous stream of 20 ms
mu-law `media` frames (caller speech or line silence), and listens for the
agent's `playAudio` / `clearAudio` events.

Metrics per turn:
  response_ms   end of caller speech -> first agent audio frame
  barge_in_ms   caller starts talking over agent -> agent sends clearAudio

    python -m bench.replay --pipeline sarvam-koyal --script happy_path --runs 3
"""

import argparse
import asyncio
import audioop  # stdlib through Python 3.12
import base64
import json
import statistics
import time
import uuid
import wave
from pathlib import Path

import websockets

FIX = Path(__file__).parent / "fixtures"
RESULTS = Path(__file__).parent / "results"
FRAME = 160  # 20 ms of 8 kHz mu-law
SILENCE = b"\xff" * FRAME
AGENT_DONE_GAP = 1.2  # seconds without agent audio = agent finished its turn


class Call:
    def __init__(self, ws):
        self.ws = ws
        self.queue: list[bytes] = []  # pending caller frames
        self.last_agent_audio = 0.0
        self.first_audio_after: float | None = None
        self.awaiting_since: float | None = None
        self.clear_at: float | None = None
        self.agent_audio_seen = asyncio.Event()
        self.closed = asyncio.Event()  # agent hung up (or the socket dropped)
        # Recording: both sides laid on one 8 kHz timeline, mixed to a WAV for listening.
        self.t0: float | None = None
        self.caller_track = bytearray()
        self.agent_track = bytearray()

    def _place(self, track: bytearray, ulaw: bytes, at: float):
        pos = max(len(track), int((at - self.t0) * 8000)) if track is self.agent_track else len(track)
        track.extend(SILENCE[:1] * (pos - len(track)))
        track.extend(ulaw)

    def save_wav(self, path: Path):
        n = max(len(self.caller_track), len(self.agent_track))
        a = audioop.ulaw2lin(bytes(self.caller_track.ljust(n, SILENCE[:1])), 2)
        b = audioop.ulaw2lin(bytes(self.agent_track.ljust(n, SILENCE[:1])), 2)
        mixed = audioop.add(audioop.mul(a, 2, 0.8), audioop.mul(b, 2, 0.8), 2)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(mixed)

    async def sender(self, stream_id: str):
        t0 = self.t0 = time.perf_counter()
        n = 0
        while True:
            chunk = self.queue.pop(0) if self.queue else SILENCE
            self._place(self.caller_track, chunk, time.perf_counter())
            if self.closed.is_set():
                return
            try:
                await self._send_media(stream_id, n, chunk)
            except websockets.ConnectionClosed:
                return
            if chunk is not SILENCE and not self.queue:
                self.awaiting_since = time.perf_counter()  # caller just finished speaking
                self.first_audio_after = None
            n += 1
            # Real-time pacing against an absolute clock so jitter doesn't accumulate.
            await asyncio.sleep(max(0, t0 + n * 0.02 - time.perf_counter()))

    async def _send_media(self, stream_id: str, n: int, chunk: bytes):
            await self.ws.send(json.dumps({
                "event": "media",
                "streamId": stream_id,
                "media": {"track": "inbound", "timestamp": str(n * 20), "chunk": n,
                          "payload": base64.b64encode(chunk).decode()},
            }))

    async def receiver(self):
        try:
            await self._receive()
        except websockets.ConnectionClosed:
            pass
        finally:
            self.closed.set()

    async def _receive(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            now = time.perf_counter()
            if msg.get("event") == "playAudio":
                if self.t0 is not None:
                    self._place(self.agent_track, base64.b64decode(msg["media"]["payload"]), now)
                self.last_agent_audio = now
                self.agent_audio_seen.set()
                if self.awaiting_since and self.first_audio_after is None:
                    self.first_audio_after = now
            elif msg.get("event") == "clearAudio":
                self.clear_at = now

    def say(self, audio: bytes):
        self.queue.extend(audio[i:i + FRAME].ljust(FRAME, b"\xff") for i in range(0, len(audio), FRAME))

    async def wait_agent_done(self, timeout=20.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end and not self.closed.is_set():
            if self.last_agent_audio and time.perf_counter() - self.last_agent_audio > AGENT_DONE_GAP:
                return
            await asyncio.sleep(0.05)


async def run_once(url: str, pipeline: str, script: str, turns: list[dict]) -> dict:
    run_id = f"{script}-{pipeline}-{uuid.uuid4().hex[:6]}"
    body = base64.urlsafe_b64encode(json.dumps(
        {"pipeline": pipeline, "order_id": "ORD-4821", "run_id": run_id}).encode()).decode()
    stream_id, call_id = str(uuid.uuid4()), str(uuid.uuid4())
    out = {"run_id": run_id, "pipeline": pipeline, "script": script, "turns": []}
    async with websockets.connect(f"{url}?body={body}", max_size=None) as ws:
        await ws.send(json.dumps({"event": "start", "start": {
            "streamId": stream_id, "callId": call_id, "accountId": "bench",
            "tracks": ["inbound"], "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}}}))
        call = Call(ws)
        tasks = [asyncio.create_task(call.sender(stream_id)), asyncio.create_task(call.receiver())]
        try:
            # Agent speaks first on an outbound call.
            t_open = time.perf_counter()
            await asyncio.wait_for(call.agent_audio_seen.wait(), 40)
            out["greeting_ms"] = round((time.perf_counter() - t_open) * 1000)
            for turn in turns:
                if call.closed.is_set():
                    out["ended_early"] = True  # agent hung up before the script finished
                    break
                audio = (FIX / turn["file"]).read_bytes()
                call.awaiting_since = call.first_audio_after = None  # never reuse the last turn's timing
                if turn.get("barge_in"):
                    # Wait until the agent is mid-sentence, then talk over it.
                    call.clear_at = None
                    t_barge = time.perf_counter()
                    call.say(audio)
                    await asyncio.sleep(2.0)
                    barge = round((call.clear_at - t_barge) * 1000) if call.clear_at else None
                else:
                    await call.wait_agent_done()
                    call.say(audio)
                    barge = None
                while call.queue and not call.closed.is_set():
                    await asyncio.sleep(0.02)
                deadline = time.perf_counter() + 8.0
                while call.first_audio_after is None and time.perf_counter() < deadline and not call.closed.is_set():
                    await asyncio.sleep(0.01)
                start = call.awaiting_since
                resp = round((call.first_audio_after - start) * 1000) if call.first_audio_after and start else None
                out["turns"].append({"text": turn["text"], "response_ms": resp, "barge_in_ms": barge})
            await call.wait_agent_done()
        finally:
            for t in tasks:
                t.cancel()
            RESULTS.mkdir(exist_ok=True)
            call.save_wav(RESULTS / f"{run_id}.wav")
    return out


def summarize(rows: list[dict]) -> dict:
    resp = sorted(t["response_ms"] for r in rows for t in r["turns"] if t["response_ms"])
    barge = [t["barge_in_ms"] for r in rows for t in r["turns"] if t["barge_in_ms"]]

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None

    return {
        "turns": len(resp),
        "response_p50_ms": pct(resp, 50),
        "response_p90_ms": pct(resp, 90),
        "response_mean_ms": round(statistics.mean(resp)) if resp else None,
        "barge_in_median_ms": statistics.median(barge) if barge else None,
        "greeting_median_ms": statistics.median([r["greeting_ms"] for r in rows]) if rows else None,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:7860/ws")
    ap.add_argument("--pipeline", default="sarvam-koyal")
    ap.add_argument("--script", default="all")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    manifest = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
    scripts = manifest if a.script == "all" else {a.script: manifest[a.script]}
    rows = []
    for name, turns in scripts.items():
        for i in range(a.runs):
            r = await run_once(a.url, a.pipeline, name, turns)
            print(json.dumps(r, ensure_ascii=False))
            rows.append(r)
    summary = {"pipeline": a.pipeline, **summarize(rows)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{a.pipeline}.json").write_text(
        json.dumps({"summary": summary, "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
