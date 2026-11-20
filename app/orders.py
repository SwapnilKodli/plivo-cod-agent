"""Mock order store, calling guardrails and call-outcome log.

Guardrails live here, in code, not in the prompt: the LLM cannot talk its way
past a calling-hours window or a do-not-call entry.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path


IST = timezone(timedelta(hours=5, minutes=30), "IST")  # no DST, no tzdata needed
# Conservative window for promotional/service calls in India.
CALL_WINDOW = (time(9, 0), time(21, 0))

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DND_FILE = DATA_DIR / "dnd.json"
OUTCOMES_FILE = DATA_DIR / "outcomes.jsonl"


@dataclass
class Order:
    order_id: str
    customer_name: str
    phone: str
    items: list[str]
    amount_inr: int
    address: str
    delivery_slot: str
    prepaid_discount_inr: int = 20
    events: list[dict] = field(default_factory=list)


ORDERS: dict[str, Order] = {
    o.order_id: o
    for o in [
        Order(
            order_id="ORD-4821",
            customer_name="Rohan",
            phone="",
            items=["Paneer Butter Masala", "2 Butter Naan", "Gulab Jamun"],
            amount_inr=486,
            address="Flat 302, Sai Residency, HSR Layout Sector 2, Bengaluru",
            delivery_slot="aaj shaam 7 se 8 baje",
        ),
        Order(
            order_id="ORD-7310",
            customer_name="Priya",
            phone="",
            items=["boAt Airdopes 141", "USB-C cable"],
            amount_inr=1349,
            address="B-14, Shanti Nagar, Andheri East, Mumbai",
            delivery_slot="kal dopahar 12 se 3 baje",
        ),
    ]
}

DEFAULT_ORDER_ID = "ORD-4821"


def get_order(order_id: str | None) -> Order:
    return ORDERS.get(order_id or DEFAULT_ORDER_ID, ORDERS[DEFAULT_ORDER_ID])


def _load_dnd() -> set[str]:
    try:
        return set(json.loads(DND_FILE.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def add_to_dnd(phone: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dnd = _load_dnd() | {phone}
    DND_FILE.write_text(json.dumps(sorted(dnd)))


def can_call(phone: str, now: datetime | None = None) -> tuple[bool, str]:
    """Pre-dial check. Returns (allowed, reason)."""
    now = now or datetime.now(IST)
    if phone in _load_dnd():
        return False, "number is on the do-not-call list"
    if not (CALL_WINDOW[0] <= now.time() <= CALL_WINDOW[1]):
        return False, f"outside calling window {CALL_WINDOW[0]:%H:%M}-{CALL_WINDOW[1]:%H:%M} IST"
    return True, "ok"


def record_event(order: Order, kind: str, **details) -> dict:
    event = {"ts": datetime.now(IST).isoformat(), "order_id": order.order_id, "event": kind, **details}
    order.events.append(event)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTCOMES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def order_summary(order: Order) -> dict:
    d = asdict(order)
    d.pop("events")
    d.pop("phone")
    return d
