"""Builds and runs one call's pipeline. PIPELINE selects the speech stack:

  sarvam-koyal    Sarvam saaras (codemix STT) -> Groq LLM -> Plivo Koyal TTS   (default)
  deepgram-koyal  Deepgram nova-3 multi       -> Groq LLM -> Plivo Koyal TTS
  sarvam-sarvam   Sarvam saaras               -> Groq LLM -> Sarvam bulbul TTS (TTS baseline)
  gemini-live     Gemini Live native audio (speech-to-speech baseline)
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

from app.agent import build_tools, register_tools, system_prompt
from app.orders import DATA_DIR, get_order, record_event
from pipecat_koyal.tts import KoyalTTSService

load_dotenv(override=True)

PIPELINES = ("sarvam-koyal", "deepgram-koyal", "sarvam-sarvam", "deepgram-sarvam", "gemini-live")
LATENCY_FILE = DATA_DIR / "latency.jsonl"

# LLMs occasionally write tool names or stage directions as text ("(end_call)",
# "(Waiting for customer response)"). Nothing in parentheses is meant to be spoken on a call.
_TOOL_NAMES = "|".join(t.name for t in build_tools().standard_tools)
_UNSPOKEN = re.compile(r"\([^)]*\)|\b(?:" + _TOOL_NAMES + r")\b")


async def _strip_tool_names(text: str, _text_type) -> str:
    return _UNSPOKEN.sub("", text)


SPEECH_TRANSFORMS = [("*", _strip_tool_names)]


def _stt(kind: str):
    if kind == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramSTTService.Settings(model="nova-3", language="multi"),
        )
    from pipecat.services.sarvam.stt import SarvamRealtimeSTTService

    return SarvamRealtimeSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamRealtimeSTTService.Settings(mode="codemix", language_code="hi-IN"),
    )


def _tts(kind: str):
    if kind == "koyal" and not os.getenv("PLIVO_AUTH_ID"):
        logger.warning("No PLIVO_AUTH_ID set: Koyal unavailable, falling back to Sarvam bulbul TTS")
        kind = "sarvam"
    if kind == "sarvam":
        from pipecat.services.sarvam.tts import SarvamTTSService

        return SarvamTTSService(
            api_key=os.environ["SARVAM_API_KEY"],
            text_transforms=SPEECH_TRANSFORMS,
            settings=SarvamTTSService.Settings(
                voice=os.getenv("SARVAM_VOICE", "priya"), model="bulbul:v3", language=Language.HI
            ),
        )
    return KoyalTTSService(
        text_transforms=SPEECH_TRANSFORMS,
        auth_id=os.environ["PLIVO_AUTH_ID"],
        auth_token=os.environ["PLIVO_AUTH_TOKEN"],
        settings=KoyalTTSService.Settings(
            voice=os.getenv("KOYAL_VOICE_ID", "Hiqcy02wuf8a4sCMcdNO"),  # Ritika (hi, F)
            language=Language.HI,
        ),
    )


def _llm():
    from pipecat.services.groq.llm import GroqLLMService

    model = os.getenv("GROQ_MODEL")
    return GroqLLMService(api_key=os.environ["GROQ_API_KEY"], **({"model": model} if model else {}))


async def run_bot(transport, body: dict, handle_sigint: bool = False):
    pipeline_name = body.get("pipeline") or os.getenv("PIPELINE", "sarvam-koyal")
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown PIPELINE {pipeline_name!r}; choose from {PIPELINES}")
    order = get_order(body.get("order_id"))
    phone = body.get("phone", "")
    run_id = body.get("run_id") or f"{order.order_id}-{int(time.time())}"
    logger.info(f"Call {run_id}: order={order.order_id} pipeline={pipeline_name}")

    prompt = system_prompt(order)
    tools = build_tools()

    if pipeline_name == "gemini-live":
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

        llm = GeminiLiveLLMService(
            api_key=os.environ["GOOGLE_API_KEY"], system_instruction=prompt, tools=tools
        )
        # Realtime models need a turn to answer before they speak first.
        context = LLMContext(messages=[{"role": "user", "content": "(call connected) Greet the customer now."}])
        stt = tts = None
    else:
        stt_kind, tts_kind = pipeline_name.split("-")
        stt, llm, tts = _stt(stt_kind), _llm(), _tts(tts_kind)
        context = LLMContext(messages=[{"role": "system", "content": prompt}], tools=tools)
    register_tools(llm, order, phone)

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    procs = [transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg]
    procs = [p for p in procs if p is not None]

    latency = UserBotLatencyObserver()

    @latency.event_handler("on_latency_breakdown")
    async def on_breakdown(_obs, breakdown):
        row = {"run_id": run_id, "pipeline": pipeline_name, "total_secs": breakdown.total_secs,
               "ttfb": [{"processor": t.processor, "secs": t.duration_secs} for t in breakdown.ttfb]}
        LATENCY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with Path(LATENCY_FILE).open("a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(f"turn latency {breakdown.total_secs * 1000:.0f} ms")

    worker = PipelineWorker(
        Pipeline(procs),
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[latency],
    )
    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_connected(_t, _c):
        record_event(order, "call_started", pipeline=pipeline_name, run_id=run_id)
        # Outbound call: the agent speaks first.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_t, _c):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    from pipecat.runner.utils import parse_telephony_websocket
    from pipecat.serializers.plivo import PlivoFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

    _, call_data = await parse_telephony_websocket(runner_args.websocket)
    auth_id, auth_token = os.getenv("PLIVO_AUTH_ID", ""), os.getenv("PLIVO_AUTH_TOKEN", "")
    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=auth_id,
        auth_token=auth_token,
        # Hanging up the call needs Plivo's REST API; without credentials (local
        # benchmark runs) end_call just closes the stream.
        params=PlivoFrameSerializer.InputParams(auto_hang_up=bool(auth_id and auth_token)),
    )
    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer
        ),
    )
    await run_bot(transport, runner_args.body or {}, runner_args.handle_sigint)
