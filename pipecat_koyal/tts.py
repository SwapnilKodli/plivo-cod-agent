"""Plivo Koyal text-to-speech service for Pipecat.

Koyal is Plivo's streaming TTS model (GA Nov 2026), tuned for Indian languages
and code-mixed speech. This service speaks its realtime WebSocket protocol:

    wss://api.plivo.com/v1/Account/{auth_id}/TTS/Realtime/

Protocol notes (from Plivo's TTS API reference):

- Auth is HTTP Basic (Plivo Auth ID / Auth Token) on the upgrade request.
- Every message is scoped by a ``context_id``. The *first* ``text`` message of a
  context carries the synthesis options (voice, language, output format, pace);
  later messages for that context may only carry ``text`` and ``finish``.
- A ``context_id`` can be used once per connection, and at most 8 may be open
  on a socket at a time. ``finish: true`` closes a context; the server answers
  with a ``done`` event once its audio is fully sent.
- Audio arrives as binary frames: ``[u16 LE id length][context_id][audio]``.
- ``cancel`` stops a context immediately (used for barge-in).
"""

import base64
import json
import struct
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

KOYAL_REALTIME_URL = "wss://api.plivo.com/v1/Account/{auth_id}/TTS/Realtime/"

# Koyal supports raw PCM at these rates (plus mu-law/a-law at 8 kHz).
_SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000)

# Languages listed in Plivo's TTS docs. Each Koyal voice speaks exactly one of them.
_LANGUAGE_MAP: dict[Language, str] = {
    Language.EN: "en-IN",
    Language.EN_US: "en-US",
    Language.EN_GB: "en-GB",
    Language.EN_IN: "en-IN",
    Language.EN_AU: "en-AU",
    Language.HI: "hi",
    Language.HI_IN: "hi",
    Language.BN: "bn",
    Language.BN_IN: "bn",
    Language.TA: "ta",
    Language.TA_IN: "ta",
    Language.TE: "te",
    Language.TE_IN: "te",
    Language.MR: "mr",
    Language.MR_IN: "mr",
    Language.GU: "gu",
    Language.GU_IN: "gu",
    Language.KN: "kn",
    Language.KN_IN: "kn",
}


def language_to_koyal_language(language: Language) -> str | None:
    """Map a Pipecat ``Language`` to a Koyal language code, or None if unsupported."""
    return _LANGUAGE_MAP.get(language)


@dataclass
class KoyalTTSSettings(TTSSettings):
    """Runtime-updatable settings for :class:`KoyalTTSService`.

    Parameters:
        pace: Speaking rate, 0.7 to 1.2 (Koyal default 1.0).
        volume: Output gain, 0.5 to 2.0 (Koyal default 1.0).
        lexicon_id: ID of a saved Koyal lexicon of custom pronunciations.
    """

    pace: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    lexicon_id: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class KoyalTTSService(WebsocketTTSService):
    """Streaming TTS over Plivo Koyal's realtime WebSocket API.

    One Pipecat turn maps to one Koyal context: sentences from the LLM are sent
    as they are aggregated, and the context is finished when the turn ends.
    Interruptions cancel the context server-side so no further audio is billed
    or streamed.

    Example::

        tts = KoyalTTSService(
            auth_id=os.environ["PLIVO_AUTH_ID"],
            auth_token=os.environ["PLIVO_AUTH_TOKEN"],
            settings=KoyalTTSService.Settings(voice="<voice_id>", language=Language.HI),
        )
    """

    Settings = KoyalTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        auth_id: str,
        auth_token: str,
        url: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize the Koyal TTS service.

        Args:
            auth_id: Plivo Auth ID.
            auth_token: Plivo Auth Token.
            url: Override for the realtime endpoint (defaults to Plivo's).
            sample_rate: Output sample rate; one of 8000, 16000 or 24000. If None,
                the pipeline's output rate is used.
            settings: Voice, language, pace, volume and lexicon.
            text_aggregation_mode: How LLM text is aggregated before synthesis.
            **kwargs: Passed to :class:`WebsocketTTSService`.
        """
        default_settings = self.Settings(
            model=None,
            voice=None,
            language="auto",
            pace=None,
            volume=None,
            lexicon_id=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=True,
            pause_frame_processing=False,
            sample_rate=sample_rate,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._auth_id = auth_id
        self._auth_token = auth_token
        self._url = url or KOYAL_REALTIME_URL.format(auth_id=auth_id)
        # Contexts whose options message has already been sent on this socket.
        self._opened_contexts: set[str] = set()
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Koyal reports TTFB and usage, so metrics are supported."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Pipecat ``Language`` to Koyal's language code."""
        return language_to_koyal_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Validate the output rate and open the socket."""
        await super().setup(setup)
        if self.sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Koyal supports sample rates {_SUPPORTED_SAMPLE_RATES}, got {self.sample_rate}"
            )
        await self._connect()

    # -- connection ---------------------------------------------------------

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Plivo Koyal TTS")
            token = base64.b64encode(f"{self._auth_id}:{self._auth_token}".encode()).decode()
            self._websocket = await self._websocket_connect(
                self._url, additional_headers={"Authorization": f"Basic {token}"}
            )
            # context_ids are scoped to a connection, so a fresh socket starts clean.
            self._opened_contexts.clear()
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Koyal connection failed: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: Disconnecting from Koyal")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Koyal disconnect error: {e}", exception=e)
        finally:
            self._websocket = None
            self._opened_contexts.clear()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    # -- messages -----------------------------------------------------------

    def _options(self) -> dict[str, Any]:
        """Synthesis options sent on the first message of each context."""
        opts: dict[str, Any] = {
            "voice_id": self._settings.voice,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        }
        if self._settings.language:
            opts["language"] = self._settings.language
        # Koyal rejects explicit nulls, so unset options are omitted entirely.
        for key in ("pace", "volume", "lexicon_id"):
            value = assert_given(getattr(self._settings, key))
            if value is not None:
                opts[key] = value
        return opts

    def _build_text_msg(self, text: str, context_id: str, finish: bool = False) -> str:
        msg: dict[str, Any] = {
            "type": "text",
            "context_id": context_id,
            "text": text,
            "finish": finish,
        }
        if context_id not in self._opened_contexts:
            msg.update(self._options())
            self._opened_contexts.add(context_id)
        return json.dumps(msg)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the Koyal context on barge-in so it stops streaming audio."""
        await self.stop_all_metrics()
        if context_id and self._websocket and context_id in self._opened_contexts:
            await self._websocket.send(json.dumps({"type": "cancel", "context_id": context_id}))
        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Finish the current context so Koyal synthesizes any buffered text."""
        ctx = context_id or self.get_active_audio_context_id()
        if not ctx or not self._websocket or ctx not in self._opened_contexts:
            return
        logger.trace(f"{self}: finishing Koyal context {ctx}")
        await self._websocket.send(self._build_text_msg("", ctx, finish=True))

    async def _update_settings(self, delta: KoyalTTSSettings) -> dict[str, Any]:
        """Apply a settings delta; voice/language changes start a new context."""
        changed = await super()._update_settings(delta)
        if changed.keys() & {"voice", "language", "pace", "volume", "lexicon_id"}:
            if self._turn_context_id and self.audio_context_available(self._turn_context_id):
                await self.flush_audio(context_id=self._turn_context_id)
            if self._turn_context_id:
                self._turn_context_id = self.create_context_id()
        return changed

    async def _handle_binary(self, data: bytes):
        if len(data) < 2:
            return
        (id_len,) = struct.unpack_from("<H", data, 0)
        ctx_id = data[2 : 2 + id_len].decode("utf-8")
        audio = data[2 + id_len :]
        if not audio or not self.audio_context_available(ctx_id):
            return
        await self.stop_ttfb_metrics()
        await self.append_to_audio_context(
            ctx_id,
            TTSAudioRawFrame(
                audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=ctx_id
            ),
        )

    async def _handle_event(self, msg: dict[str, Any]):
        kind = msg.get("type")
        ctx_id = msg.get("context_id")
        if kind == "done":
            self._opened_contexts.discard(ctx_id)
            if ctx_id and self.audio_context_available(ctx_id):
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                await self.remove_audio_context(ctx_id)
            logger.debug(
                f"{self}: Koyal context {ctx_id} {msg.get('reason')} "
                f"(server ttfb={msg.get('ttfb_ms')}ms, chars={msg.get('characters')})"
            )
        elif kind == "error":
            logger.error(f"{self}: Koyal error: {msg}")
            if ctx_id:
                self._opened_contexts.discard(ctx_id)
                await self.push_frame(TTSStoppedFrame(context_id=ctx_id))
            await self.stop_all_metrics()
            await self.push_error(error_msg=f"Koyal error {msg.get('code')}: {msg.get('message')}")
            self.reset_active_audio_context()
        elif kind == "warning":
            logger.warning(f"{self}: Koyal warning: {msg}")
        # "timestamps" and "flush_done" need no handling here.

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                await self._handle_binary(message)
            else:
                await self._handle_event(json.loads(message))
        # Socket closed by the server (idle timeout or 1013 retry): reconnect.
        logger.debug(f"{self}: Koyal connection closed, reconnecting")
        await self._connect_websocket()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send one aggregated chunk of text into the turn's Koyal context."""
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()
            try:
                await self._get_websocket().send(self._build_text_msg(text, context_id))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Koyal send failed: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Koyal error: {e}")
