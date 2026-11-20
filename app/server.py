"""FastAPI server: trigger outbound calls, serve Plivo answer XML, host the media WebSocket.

  POST /start   {"phone_number": "+91...", "order_id": "ORD-4821", "pipeline": "sarvam-koyal"}
  GET  /answer  Plivo fetches this when the call connects (also usable as the inbound answer URL)
  WS   /ws      Plivo Audio Stream (bidirectional, mu-law 8 kHz)
"""

import base64
import json
import os
from contextlib import asynccontextmanager
from html import escape
from urllib.parse import quote

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import Response
from loguru import logger

from app.orders import can_call, get_order

load_dotenv(override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = aiohttp.ClientSession()
    yield
    await app.state.session.close()


app = FastAPI(lifespan=lifespan)


def _public_host(request: Request) -> str:
    return os.getenv("PUBLIC_HOST") or request.headers["host"]


@app.post("/start")
async def start_call(request: Request):
    data = await request.json()
    phone = str(data.get("phone_number", "")).strip()
    if not phone:
        raise HTTPException(400, "phone_number is required")
    order = get_order(data.get("order_id"))

    # Guardrail runs before any money is spent on a dial.
    allowed, reason = can_call(phone)
    if not allowed and not data.get("force"):
        raise HTTPException(409, f"Call blocked: {reason}")

    body = {"order_id": order.order_id, "phone": phone, "pipeline": data.get("pipeline")}
    answer_url = f"https://{_public_host(request)}/answer?body={quote(json.dumps(body))}"
    auth_id, token = os.environ["PLIVO_AUTH_ID"], os.environ["PLIVO_AUTH_TOKEN"]
    async with request.app.state.session.post(
        f"https://api.plivo.com/v1/Account/{auth_id}/Call/",
        json={"to": phone, "from": os.environ["PLIVO_PHONE_NUMBER"], "answer_url": answer_url, "answer_method": "GET"},
        auth=aiohttp.BasicAuth(auth_id, token),
    ) as r:
        if r.status != 201:
            raise HTTPException(502, f"Plivo error {r.status}: {await r.text()}")
        result = await r.json()
    return {"status": "call_initiated", "request_uuid": result.get("request_uuid"), "order_id": order.order_id}


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(request: Request, body: str | None = Query(None)):
    payload = json.loads(body) if body else {}
    if not payload.get("phone"):
        # Inbound calls: Plivo sends the caller as From.
        form = await request.form() if request.method == "POST" else {}
        payload["phone"] = request.query_params.get("From") or form.get("From", "")
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    ws_url = escape(f"wss://{_public_host(request)}/ws?body={encoded}")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">{ws_url}</Stream>
</Response>"""
    return Response(content=xml, media_type="application/xml")


@app.websocket("/ws")
async def ws(websocket: WebSocket, body: str | None = Query(None)):
    await websocket.accept()
    body_data = {}
    if body:
        try:
            body_data = json.loads(base64.urlsafe_b64decode(body))
        except Exception as e:
            logger.warning(f"Bad body param: {e}")
    from pipecat.runner.types import WebSocketRunnerArguments

    from app.bot import bot

    try:
        await bot(WebSocketRunnerArguments(websocket=websocket, body=body_data))
    except Exception as e:
        logger.exception(f"Bot error: {e}")
        await websocket.close()


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
