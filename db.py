"""MongoDB Atlas connection client, schema definition, and index setup.

This module centralises everything related to the database layer:

* A ``MongoClient`` configured with restricted, read-oriented connection
  settings (timeouts, retry policy, read preference).
* The authoritative ``orders`` collection schema, passed to the LLM so it can
  ground generated MQL in real field names and enum values.
* An idempotent index-creation routine that provisions the three indexes the
  spec requires on Atlas setup.
* A safe read-execution wrapper that enforces a ``maxTimeMS`` timeout and a
  result-size cap on every query.

Nothing in this module performs writes to the ``orders`` collection.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId, json_util
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from config import (
    DB_NAME,
    KB_AUTO_COLLECTION,
    KB_AUTO_INDEX,
    KB_AUTO_PATH,
    KB_DB_NAME,
    KB_MANUAL_COLLECTION,
    KB_MANUAL_INDEX,
    KB_MANUAL_PATH,
    KB_TOP_K,
    KB_VOYAGE_MODEL,
    MAX_RESULT_DOCS,
    MAX_TIME_MS,
    MONGODB_URI,
    ORDERS_COLLECTION,
    VOYAGE_API_KEY,
)

# ---------------------------------------------------------------------------
# Connection client with restricted settings.
# ---------------------------------------------------------------------------
# * serverSelectionTimeoutMS / connectTimeoutMS keep the app responsive if the
#   cluster is unreachable rather than hanging indefinitely.
# * retryWrites=False — this application is read-only; disabling write retries
#   makes accidental writes fail fast instead of being silently retried.
# * readPreference=secondaryPreferred offloads analytical reads from the
#   primary when a replica is available.
mongo_client: MongoClient = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
    retryWrites=False,
    readPreference="secondaryPreferred",
    appname="da680-nl-to-mql",
)

db = mongo_client[DB_NAME]
orders = db[ORDERS_COLLECTION]


# ---------------------------------------------------------------------------
# Authoritative schema description (fed to the LLM).
# ---------------------------------------------------------------------------
STATUS_VALUES = ["fulfilled", "returned", "cancelled", "pending", "processing"]
PAYMENT_METHODS = ["credit_card", "paypal", "apple_pay", "bank_transfer"]

# A compact, human/LLM-readable schema. Kept as a plain dict so it can be
# serialised straight into a system prompt.
ORDERS_SCHEMA: dict[str, Any] = {
    "collection": ORDERS_COLLECTION,
    "description": "E-commerce orders. One document per placed order.",
    "fields": {
        "_id": "ObjectId",
        "order_id": "String (unique, indexed, e.g. '37126471')",
        "customer_id": "String",
        "created_at": "ISODate (order creation timestamp, UTC)",
        "status": f"String enum, one of {STATUS_VALUES}",
        "total_amount": "Double (order total in currency units)",
        "payment_method": f"String enum, one of {PAYMENT_METHODS}",
        "items": [
            {
                "item_id": "String",
                "item_name": "String",
                "category": "String",
                "quantity": "Integer",
                "price": "Double",
            }
        ],
        "shipping_address": {
            "city": "String",
            "state": "String",
            "country": "String",
        },
    },
    "notes": [
        "created_at is stored as a BSON date; build date filters with real "
        "date objects, not strings.",
        "items is an array; use $unwind before grouping on item fields.",
        "Amounts are Doubles; revenue = sum of total_amount.",
    ],
}


def schema_as_prompt() -> str:
    """Return the orders schema as a formatted JSON string for prompts."""
    return json.dumps(ORDERS_SCHEMA, indent=2, default=str)


# ---------------------------------------------------------------------------
# Index provisioning.
# ---------------------------------------------------------------------------
def ensure_indexes() -> list[str]:
    """Create the indexes required by the spec. Idempotent.

    Returns the list of index names that exist after the call.

    Indexes:
      1. Unique single-field index on ``order_id`` — fast, unique lookups.
      2. Compound ``{created_at: -1, status: 1}`` — time-ranged analytics
         filtered/grouped by status.
      3. Multikey compound ``{"items.category": 1, status: 1}`` — nested item
         category analysis by order status.
    """
    orders.create_index([("order_id", ASCENDING)], unique=True, name="uniq_order_id")
    orders.create_index(
        [("created_at", DESCENDING), ("status", ASCENDING)],
        name="created_at_status",
    )
    orders.create_index(
        [("items.category", ASCENDING), ("status", ASCENDING)],
        name="items_category_status",
    )
    return [ix["name"] for ix in orders.list_indexes()]


# ---------------------------------------------------------------------------
# Safe read helpers used by the agent's execution layer.
# ---------------------------------------------------------------------------
def _to_jsonable(value: Any) -> Any:
    """Convert BSON (ObjectId, datetime, etc.) into JSON-serialisable data."""
    return json.loads(json_util.dumps(value))


def find_one_order(order_id: str) -> dict | None:
    """Path A primitive: fetch a single order by its business ``order_id``.

    Read-only. Enforces the shared maxTimeMS ceiling.
    """
    doc = orders.find_one({"order_id": str(order_id)}, max_time_ms=MAX_TIME_MS)
    return _to_jsonable(doc) if doc else None


def _decode_extended_json(pipeline: list[dict]) -> list[dict]:
    """Convert MongoDB extended-JSON literals into real BSON types.

    LLMs express dates as ``{"$date": "..."}`` (and ObjectIds as
    ``{"$oid": "..."}``) in the JSON pipeline. pymongo does NOT interpret these
    automatically — passed as plain dicts they become literal sub-documents and
    match nothing. Round-tripping through ``bson.json_util`` decodes them into
    ``datetime`` / ``ObjectId`` so comparisons work as intended.
    """
    return json_util.loads(json_util.dumps(pipeline))


def run_aggregation(pipeline: list[dict]) -> list[dict]:
    """Path B primitive: execute a validated read-only aggregation pipeline.

    The caller is responsible for validating/sanitising ``pipeline`` first
    (see agent.validate_pipeline). This function still defends the database by:
      * decoding extended-JSON literals ({"$date": ...}) into real BSON,
      * forcing ``maxTimeMS`` so a pathological pipeline cannot run unbounded,
      * appending a ``$limit`` cap so result sets stay bounded.
    """
    safe_pipeline = _decode_extended_json(list(pipeline))
    # Cap output size unless the pipeline already ends in a small $limit.
    safe_pipeline.append({"$limit": MAX_RESULT_DOCS})
    cursor = orders.aggregate(
        safe_pipeline,
        maxTimeMS=MAX_TIME_MS,
        allowDiskUse=False,
    )
    return [_to_jsonable(doc) for doc in cursor]


def current_date_context() -> dict[str, str]:
    """Return date anchors the LLM needs to build 'today' vs baseline queries.

    All timestamps are UTC ISO-8601 strings. The LLM is instructed to wrap
    these in ``$date`` extended-JSON so they parse back into BSON dates.
    """
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # 7 full days before today, used as the comparison baseline window.
    baseline_days = 7
    baseline_start = start_of_day - timedelta(days=baseline_days)
    return {
        "now_utc": now.isoformat(),
        "today_start_utc": start_of_day.isoformat(),
        "baseline_start_utc": baseline_start.isoformat(),
        "baseline_days": str(baseline_days),
        "weekday": now.strftime("%A"),
    }


def health_check() -> bool:
    """Ping the cluster; returns True if reachable."""
    try:
        mongo_client.admin.command("ping")
        return True
    except PyMongoError:
        return False


# ---------------------------------------------------------------------------
# Knowledge-base retrieval (RAG) — vector search over rag_demo.
# ---------------------------------------------------------------------------
# Separate database handle; the restricted (read-only) client is reused.
kb_db = mongo_client[KB_DB_NAME]

# The fields we return as retrieval context. Deliberately excludes the raw
# embedding vector so large arrays never bloat the LLM context or the API
# response.
_KB_PROJECTION = {
    "_id": 0,
    "title": 1,
    "content": 1,
    "source": 1,
    "score": {"$meta": "vectorSearchScore"},
}


def _embed_query_voyage(text: str) -> list[float]:
    """Embed a query string with Voyage AI for the manual-embedding collection.

    Imported lazily so the app runs without the voyage client when auto-embed
    mode is used. Raises if no valid key is configured.
    """
    if not VOYAGE_API_KEY:
        raise RuntimeError(
            "KB manual mode requires VOYAGE_API_KEY, but none is set."
        )
    import voyageai  # lazy import

    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    # input_type='query' tells Voyage to encode this as a search query rather
    # than a stored document, which improves retrieval quality.
    result = client.embed([text], model=KB_VOYAGE_MODEL, input_type="query")
    return result.embeddings[0]


def vector_search_auto(query_text: str, k: int = KB_TOP_K) -> list[dict]:
    """Semantic search against the auto-embed collection.

    Atlas embeds ``query_text`` server-side (no client Voyage key needed) using
    the model declared in the auto-embed index. Read-only; result count capped.
    """
    coll = kb_db[KB_AUTO_COLLECTION]
    k = min(k, MAX_RESULT_DOCS)
    pipeline = [
        {
            "$vectorSearch": {
                "index": KB_AUTO_INDEX,
                "path": KB_AUTO_PATH,
                "query": query_text,          # auto-embed: text, not a vector
                "numCandidates": max(50, k * 10),
                "limit": k,
            }
        },
        {"$project": _KB_PROJECTION},
    ]
    cursor = coll.aggregate(pipeline, maxTimeMS=MAX_TIME_MS)
    return [_to_jsonable(doc) for doc in cursor]


def vector_search_manual(query_text: str, k: int = KB_TOP_K) -> list[dict]:
    """Semantic search against the manual-embedding collection.

    The query is embedded client-side with Voyage AI, then matched against the
    stored ``embedding`` vectors. Requires a valid VOYAGE_API_KEY. Read-only;
    result count capped.
    """
    coll = kb_db[KB_MANUAL_COLLECTION]
    k = min(k, MAX_RESULT_DOCS)
    query_vector = _embed_query_voyage(query_text)
    pipeline = [
        {
            "$vectorSearch": {
                "index": KB_MANUAL_INDEX,
                "path": KB_MANUAL_PATH,
                "queryVector": query_vector,  # manual: a precomputed vector
                "numCandidates": max(50, k * 10),
                "limit": k,
            }
        },
        {"$project": _KB_PROJECTION},
    ]
    cursor = coll.aggregate(pipeline, maxTimeMS=MAX_TIME_MS)
    return [_to_jsonable(doc) for doc in cursor]
