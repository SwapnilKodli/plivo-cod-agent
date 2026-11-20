"""Prompt and tools for the Hinglish COD-confirmation agent."""

import json
import os

import aiohttp
from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import FunctionCallResultProperties, EndWorkerFrame, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

from app.orders import Order, add_to_dnd, order_summary, record_event


def system_prompt(order: Order) -> str:
    return f"""Aap "Riya" hain, ek delivery partner ki taraf se outbound call kar rahi hain.
Kaam: Cash-on-Delivery (COD) order confirm karna, taaki fake/RTO orders kam hon.

Order details (sirf yahi facts use karein, kuch bhi invent mat karein):
{json.dumps(order_summary(order), ensure_ascii=False, indent=2)}

Kaise baat karni hai:
- Natural Hinglish bolein, jaise Indian customer support bolta hai. Hindi words Devanagari me likhein,
  English words (order, delivery, address, cash, UPI, link) Latin script me. Example:
  "नमस्ते रोहन जी, मैं Riya बोल रही हूँ, आपके order के बारे में call किया था।"
- Har reply chhota rakhein: 1-2 sentences. Phone call hai, essay nahi.
- Koi markdown, emoji, bullet, ya special characters nahi. Amount ko "चार सौ छियासी रुपये" jaise bolein.
- Agar customer English me bole to English me reply karein.

Flow:
1. Greet karein, naam confirm karein, batayein ki unke {order.amount_inr} rupaye ke COD order ke liye call hai.
2. Items aur delivery slot confirm karein.
3. Address padh kar confirm karein. Badlav ho to update_address call karein.
4. Ek baar politely offer karein: UPI se abhi pay karein to {order.prepaid_discount_inr} rupaye off. Haan bole to send_prepaid_link.
   Mana kare to push mat karein.
5. Final decision ke liye confirm_order, cancel_order, ya reschedule_delivery call karein.
6. Thank you bol kar end_call call karein.

Guardrails:
- Customer bole "call mat karo" / "don't call me" -> turant mark_do_not_call, sorry bolein, end_call.
- Price, discount, refund ke baare me sirf upar diye facts. Pata na ho to bolein "main team se confirm karwa deti hoon".
- Kisi bhi haal me OTP, card number, ya UPI PIN mat maangiye.
"""


def build_tools() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema("confirm_order", "Customer ne COD order confirm kar diya.", {}, []),
            FunctionSchema(
                "cancel_order",
                "Customer order cancel karna chahta hai.",
                {"reason": {"type": "string", "description": "Cancel ka reason, customer ke shabdon me"}},
                ["reason"],
            ),
            FunctionSchema(
                "reschedule_delivery",
                "Customer delivery ka time/din badalna chahta hai.",
                {"new_slot": {"type": "string", "description": "Naya slot, e.g. 'kal subah 10-12'"}},
                ["new_slot"],
            ),
            FunctionSchema(
                "update_address",
                "Sirf tab jab customer ne address me correction ya landmark bataya ho. Address sahi ho to mat call karo.",
                {"correction": {"type": "string"}},
                ["correction"],
            ),
            FunctionSchema(
                "send_prepaid_link",
                "Customer UPI se prepay karne ko taiyaar hai; payment link SMS bhejo.",
                {},
                [],
            ),
            FunctionSchema("mark_do_not_call", "Customer ne aage se call na karne ko kaha.", {}, []),
            FunctionSchema("end_call", "Goodbye bolne ke BAAD call khatam karo. Iske baad kuch mat bolo.", {}, []),
        ]
    )


async def send_sms(to: str, text: str) -> bool:
    """Send an SMS via Plivo. Trial accounts can only reach verified numbers, and
    Indian destinations need DLT registration, so failure is logged, not fatal."""
    auth_id, token, src = (os.getenv(k) for k in ("PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN", "PLIVO_PHONE_NUMBER"))
    if not (auth_id and token and src and to):
        logger.info(f"SMS skipped (missing config or number): {text}")
        return False
    url = f"https://api.plivo.com/v1/Account/{auth_id}/Message/"
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json={"src": src, "dst": to, "text": text}, auth=aiohttp.BasicAuth(auth_id, token)) as r:
            ok = r.status in (200, 202)
            if not ok:
                logger.warning(f"Plivo SMS failed ({r.status}): {await r.text()}")
            return ok


def register_tools(llm, order: Order, phone: str) -> None:
    async def confirm_order(p: FunctionCallParams):
        record_event(order, "confirmed")
        await p.result_callback({"status": "confirmed"})

    async def cancel_order(p: FunctionCallParams):
        record_event(order, "cancelled", reason=p.arguments.get("reason"))
        await p.result_callback({"status": "cancelled"})

    async def reschedule_delivery(p: FunctionCallParams):
        record_event(order, "rescheduled", new_slot=p.arguments.get("new_slot"))
        await p.result_callback({"status": "rescheduled", "new_slot": p.arguments.get("new_slot")})

    async def update_address(p: FunctionCallParams):
        correction = (p.arguments.get("correction") or "").strip()
        if not correction:
            # Models sometimes fire this on "address is fine". Refuse instead of logging a no-op change.
            await p.result_callback({"status": "rejected", "reason": "no correction given; address unchanged"})
            return
        record_event(order, "address_updated", correction=correction)
        await p.result_callback({"status": "address_updated"})

    async def send_prepaid_link(p: FunctionCallParams):
        link = f"https://pay.example.com/{order.order_id}"
        sent = await send_sms(phone, f"Pay Rs {order.amount_inr - order.prepaid_discount_inr} for {order.order_id}: {link}")
        record_event(order, "prepaid_link", sms_sent=sent)
        await p.result_callback({"status": "link_sent" if sent else "link_queued"})

    async def mark_do_not_call(p: FunctionCallParams):
        if phone:
            add_to_dnd(phone)
        record_event(order, "do_not_call")
        await p.result_callback({"status": "added_to_dnd"})

    async def end_call(p: FunctionCallParams):
        record_event(order, "call_ended")
        # run_llm=False: the goodbye was already said; don't let the model speak again.
        await p.result_callback({"status": "ending"}, properties=FunctionCallResultProperties(run_llm=False))
        # Flush any speech still queued, then hang up (serializer calls Plivo's hangup API).
        await p.llm.push_frame(EndWorkerFrame(reason="agent ended call"))

    for fn in (confirm_order, cancel_order, reschedule_delivery, update_address,
               send_prepaid_link, mark_do_not_call, end_call):
        llm.register_function(fn.__name__, fn)


__all__ = ["system_prompt", "build_tools", "register_tools", "TTSSpeakFrame"]
