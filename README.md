# plivo-cod-agent

A Hinglish voice agent that confirms **Cash-on-Delivery orders** over a real phone call on
**Plivo**. It speaks with **Koyal**, Plivo's new Indian-language TTS model, and includes an open
benchmark that measures response latency the way a caller hears it.

COD orders that never get accepted (RTO) are a major cost for Indian food-delivery and D2C
brands. A 45-second confirmation call before dispatch is one of the highest-volume voice-AI
use cases on Plivo today.

```
Plivo call ──(Audio Stream, μ-law 8k, bidirectional WS)──▶ FastAPI /ws
   ▲                                                         │
   │                      Pipecat pipeline                   ▼
   └── playAudio ◀── Koyal TTS ◀── Groq LLM + tools ◀── Sarvam STT (codemix)
                                      │
                         confirm / cancel / reschedule / update address
                         prepaid UPI link (Plivo SMS) / do-not-call / hang up
```

## What's in here

| Path | What it is |
|---|---|
| `pipecat_koyal/tts.py` | **Koyal TTS service for Pipecat.** Pipecat doesn't ship one yet. It streams over Koyal's realtime WebSocket, maps one turn to one Koyal context, and cancels server-side on barge-in so interrupted speech isn't streamed. Written to be upstreamed. |
| `app/agent.py` | Hinglish prompt (Devanagari + English, as Indian support agents actually speak) and 7 tools |
| `app/orders.py` | Guardrails **in code, not in the prompt**: 9am–9pm IST calling window and a do-not-call list checked before dialing, plus an outcome log |
| `app/server.py` | `POST /start` (outbound dial), `/answer` (Plivo XML, works for inbound too), `WS /ws` |
| `app/bot.py` | Pipeline with switchable speech stacks, plus a per-turn latency breakdown logged to `data/latency.jsonl` |
| `bench/` | Replay harness that **impersonates Plivo's stream protocol**, so every stack hears the same caller audio, at no cost and repeatably |

## Speech stacks under test

| `PIPELINE` | STT | LLM | TTS |
|---|---|---|---|
| `sarvam-koyal` (default) | Sarvam saaras, codemix | Groq | **Plivo Koyal** |
| `deepgram-koyal` | Deepgram nova-3 `multi` | Groq | **Plivo Koyal** |
| `sarvam-sarvam` | Sarvam saaras | Groq | Sarvam bulbul (TTS baseline) |
| `gemini-live` | Gemini Live native audio, speech-to-speech | | |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in keys
python -m app.server        # :7860
cloudflared tunnel --url http://localhost:7860   # put the host in PUBLIC_HOST
```

Place a call:

```bash
curl -X POST localhost:7860/start -H "Content-Type: application/json" \
  -d '{"phone_number": "+91XXXXXXXXXX", "order_id": "ORD-4821"}'
```

Notes:
- Trial accounts can only call **verified** numbers.
- Indian SMS needs DLT registration, so the prepaid-link SMS is logged instead of sent when it can't go out.
- If outbound dialing is blocked on your trial, point a Plivo number's answer URL at `https://<host>/answer` and call it.

### Without a Plivo account

Everything except the phone leg and Koyal runs without Plivo credentials:

- `PIPELINE=*-koyal` falls back to Sarvam bulbul TTS with a warning.
- `bench.make_fixtures` synthesizes the caller with Sarvam instead of Koyal.
- Auto hang-up (which needs Plivo's REST API) is turned off, so `end_call` just closes the stream.

The replay harness speaks Plivo's stream protocol, so the server can't tell it from a real call.

## Benchmark

```bash
python -m bench.make_fixtures                                # caller lines → μ-law 8k (Koyal batch API, a male hi voice)
python -m bench.replay --pipeline sarvam-koyal --runs 5
python -m bench.replay --pipeline deepgram-koyal --runs 5
python -m bench.replay --pipeline gemini-live --runs 5
```

Three scripted calls: happy path, cancel + do-not-call (with a barge-in), and reschedule +
prepaid. The harness measures on the wire, as the caller experiences it:

- **response_ms**: end of caller speech → first agent audio frame
- **barge_in_ms**: caller talks over the agent → agent sends `clearAudio`

The server side also logs Pipecat's per-service TTFB breakdown for each turn.

### Results

First run, 2026-11-20: one run of each script per stack (two for deepgram-sarvam), run one stack at a time, without a Plivo account.
Koyal stacks are pending Plivo credentials.

| Stack | p50 response | p90 response | barge-in | greeting | turns |
|---|---|---|---|---|---|
| gemini-live | 1452 ms | 1587 ms | 428 ms | 2672 ms | 8 |
| deepgram-sarvam | 1778 ms | 2839 ms | 674 ms | 3008 ms | 18 |
| sarvam-sarvam | 2573 ms | 2969 ms | 651 ms | 2093 ms | 10 |
| sarvam-koyal | needs Plivo | | | | |
| deepgram-koyal | needs Plivo | | | | |

Server-side median TTFB per service (cascades share the same Groq LLM and Sarvam voice):

| Stack | STT | LLM | TTS |
|---|---|---|---|
| sarvam-sarvam | 1405 ms | 435 ms | 193 ms |
| deepgram-sarvam | 507 ms | 372 ms | 199 ms |

Speech-to-text dominates. The TTS slot, where Koyal goes, is about 0.2 s of a 2.6 s reply.

## Limits

- The harness skips the PSTN leg, so it reports pipeline latency. Real calls add carrier
  latency on top. A few real calls are used to sanity-check the numbers.
- The caller audio is synthetic (Koyal). Real callers are noisier. Recorded human clips can
  go in `bench/fixtures/` with the same manifest format.
