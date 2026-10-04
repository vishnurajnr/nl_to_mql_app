"""Seed the `orders` collection with realistic sample data for demos/tests.

Generates orders spread across the last ~14 days so that analytical /
root-cause questions ("Why is revenue low today?") have a baseline to compare
against. Intentionally skews *today* toward more returns/cancellations and
lower revenue so the diagnostic path has something interesting to explain.

Usage:
    python seed_data.py            # ensure indexes + insert sample orders
    python seed_data.py --drop     # drop the collection first, then seed

This is the ONLY module in this app that writes to `orders`. The runtime
query engine (agent.py / db.py) is strictly read-only.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient

from config import MONGODB_URI, DB_NAME, ORDERS_COLLECTION
from db import ensure_indexes, PAYMENT_METHODS, STATUS_VALUES

# A dedicated client with writes enabled (the shared db.py client is read-only).
_client = MongoClient(MONGODB_URI, appname="da680-nl-to-mql-seed")
_orders = _client[DB_NAME][ORDERS_COLLECTION]

CATALOG = [
    ("Wireless Mouse", "electronics", 25.0),
    ("Mechanical Keyboard", "electronics", 89.0),
    ("USB-C Cable", "accessories", 12.5),
    ("Running Shoes", "footwear", 74.0),
    ("Yoga Mat", "fitness", 30.0),
    ("Coffee Beans 1kg", "grocery", 22.0),
    ("Water Bottle", "fitness", 18.0),
    ("Desk Lamp", "home", 45.0),
]

CITIES = [
    ("Zurich", "ZH", "CH"),
    ("Geneva", "GE", "CH"),
    ("Bern", "BE", "CH"),
    ("Berlin", "BE", "DE"),
    ("Vienna", "W", "AT"),
]


def _rand_items() -> tuple[list[dict], float]:
    n = random.randint(1, 3)
    items = []
    total = 0.0
    for _ in range(n):
        name, category, price = random.choice(CATALOG)
        qty = random.randint(1, 4)
        total += price * qty
        items.append(
            {
                "item_id": f"itm_{random.randint(1000, 9999)}",
                "item_name": name,
                "category": category,
                "quantity": qty,
                "price": price,
            }
        )
    return items, round(total, 2)


def _make_order(created_at: datetime, skew_negative: bool) -> dict:
    items, total = _rand_items()
    if skew_negative:
        # Today skews toward returns/cancellations and cheaper baskets.
        status = random.choices(
            STATUS_VALUES, weights=[2, 5, 4, 1, 1], k=1
        )[0]
        total = round(total * 0.6, 2)
    else:
        status = random.choices(
            STATUS_VALUES, weights=[6, 1, 1, 1, 1], k=1
        )[0]
    city, state, country = random.choice(CITIES)
    return {
        "order_id": str(random.randint(10_000_000, 99_999_999)),
        "customer_id": f"cust_{random.randint(100, 999)}",
        "created_at": created_at,
        "status": status,
        "total_amount": total,
        "payment_method": random.choice(PAYMENT_METHODS),
        "items": items,
        "shipping_address": {"city": city, "state": state, "country": country},
    }


def seed(drop: bool = False, days: int = 14, per_day: int = 20) -> int:
    if drop:
        _orders.drop()
        print(f"Dropped collection {DB_NAME}.{ORDERS_COLLECTION}")

    ensure_indexes()
    print("Indexes ensured.")

    now = datetime.now(timezone.utc)
    docs: list[dict] = []
    for d in range(days):
        day_start = (now - timedelta(days=d)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        is_today = d == 0
        # Fewer, more negative orders today to create a diagnosable dip.
        count = max(6, per_day // 2) if is_today else per_day
        for _ in range(count):
            ts = day_start + timedelta(
                hours=random.randint(0, 23), minutes=random.randint(0, 59)
            )
            docs.append(_make_order(ts, skew_negative=is_today))

    # order_id is unique; regenerate on the rare collision by using ordered=False.
    result = _orders.insert_many(docs, ordered=False)
    print(f"Inserted {len(result.inserted_ids)} orders across {days} days.")

    # Insert one well-known order id for easy Path A demos.
    demo_id = "37126471"
    if not _orders.find_one({"order_id": demo_id}):
        demo = _make_order(now - timedelta(hours=3), skew_negative=False)
        demo["order_id"] = demo_id
        demo["status"] = "fulfilled"
        _orders.insert_one(demo)
        print(f"Inserted demo order_id={demo_id}.")

    return _orders.count_documents({})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the orders collection.")
    parser.add_argument("--drop", action="store_true", help="Drop collection first.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--per-day", type=int, default=20)
    args = parser.parse_args()

    total = seed(drop=args.drop, days=args.days, per_day=args.per_day)
    print(f"Done. Collection now holds {total} orders.")
