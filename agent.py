"""Agentic NL-to-MQL engine implementing the dual-path routing architecture.

    User text
        |
        v
    AI Intent Router  (route_intent)
      |            |
   Path A        Path B
 direct lookup  analytical / root-cause
      |            |
 tool calling   text-to-MQL pipeline + validator + execution
      |            |
      +-----> natural-language synthesis <-----+

Public entry point: ``answer(user_text, history=None) -> ChatResult``.

Design notes
------------
* The router is a small structured-output LLM call. It is intentionally cheap
  and deterministic (it only chooses a path and, for Path A, extracts the
  order_id) so that the expensive reasoning happens inside each path.
* Path A never generates a pipeline — it extracts an identifier and calls
  ``db.find_one_order``.
* Path B generates an aggregation pipeline as JSON, which is then run through
  ``validate_pipeline`` before it ever touches the database. Only an
  allow-listed set of read-only stages is permitted.
* All raw MongoDB output is sent back to the LLM for a final natural-language
  synthesis so the user gets an insightful answer, not a JSON dump.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from config import llm, MAX_TIME_MS, KB_EMBED_MODE, KB_TOP_K
from db import (
    current_date_context,
    find_one_order,
    run_aggregation,
    schema_as_prompt,
    vector_search_auto,
    vector_search_manual,
)

# ---------------------------------------------------------------------------
# Rate-limit (HTTP 429) retry handling.
# ---------------------------------------------------------------------------
# Free-tier LLM providers (notably Gemini) enforce per-minute request quotas.
# When exceeded they return HTTP 429 (RESOURCE_EXHAUSTED). These helpers wrap
# any runnable's .invoke() and retry with backoff, honouring the provider's
# suggested retry delay when it is present in the error message.
MAX_LLM_RETRIES = 5
DEFAULT_RETRY_DELAY_S = 8.0
MAX_RETRY_DELAY_S = 60.0


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if the exception looks like a 429 / quota-exhausted error."""
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "rate limit" in text
        or "quota" in text
        or "too many requests" in text
    )


def _is_daily_quota_error(exc: Exception) -> bool:
    """True if the 429 is a *per-day* free-tier cap (not worth retrying now).

    Per-minute limits recover within a minute and are worth retrying; a daily
    cap will not recover until the quota window resets, so we surface it
    immediately rather than sleeping through pointless retries.
    """
    text = str(exc).lower()
    return "perday" in text or "per day" in text or "requestsperday" in text


def _suggested_delay(exc: Exception, attempt: int) -> float:
    """Extract the provider's suggested retry delay, else exponential backoff.

    Gemini includes a ``retryDelay`` like '7s' or 'retry in 7.29s' in the 429
    payload. We parse the first such value and add a small safety margin; if
    none is found we fall back to exponential backoff capped at
    ``MAX_RETRY_DELAY_S``.
    """
    text = str(exc)
    match = re.search(r"retry(?:\s*in|delay)?['\":\s]*([0-9]+(?:\.[0-9]+)?)\s*s", text, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1.0, MAX_RETRY_DELAY_S)
    return min(DEFAULT_RETRY_DELAY_S * (2 ** attempt), MAX_RETRY_DELAY_S)


def _rate_limit_message(exc: Exception) -> str:
    """User-facing message for a rate-limit failure."""
    if _is_rate_limit_error(exc):
        if _is_daily_quota_error(exc):
            return (
                "The LLM's free-tier daily quota is exhausted. Please try again "
                "after the quota resets, switch GEMINI_MODEL to "
                "gemini-2.5-flash-lite (higher daily limit), or use a paid key."
            )
        return (
            "The LLM is rate limited right now. Please wait a moment and try again."
        )
    return "Sorry, I couldn't process that request due to an LLM error."


def invoke_with_retry(runnable, payload):
    """Invoke a runnable, retrying on 429/quota errors with backoff.

    Non-rate-limit errors propagate immediately. After ``MAX_LLM_RETRIES``
    exhausted attempts, the final rate-limit error is re-raised so callers can
    surface it to the user.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            return runnable.invoke(payload)
        except Exception as exc:  # noqa: BLE001 - inspect then re-raise/retry
            if not _is_rate_limit_error(exc):
                raise
            # A per-day free-tier cap won't recover until the quota resets, so
            # retrying is futile — surface it right away.
            if _is_daily_quota_error(exc):
                print("[llm] daily free-tier quota exhausted; not retrying.", flush=True)
                raise
            last_exc = exc
            if attempt == MAX_LLM_RETRIES - 1:
                break
            delay = _suggested_delay(exc, attempt)
            print(f"[llm] rate limited (429); retrying in {delay:.1f}s "
                  f"(attempt {attempt + 1}/{MAX_LLM_RETRIES})", flush=True)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc

# ---------------------------------------------------------------------------
# Sanitization guardrails.
# ---------------------------------------------------------------------------
# Only these aggregation stages may appear in a generated pipeline. Everything
# else — especially anything that writes or manipulates indexes — is rejected.
ALLOWED_STAGES: set[str] = {
    "$match",
    "$group",
    "$project",
    "$sort",
    "$limit",
    "$unwind",
    "$facet",
    "$lookup",
}

# Stages that must never appear. Kept explicit for clear error messages and as
# defence-in-depth alongside the allow-list.
FORBIDDEN_STAGES: set[str] = {
    "$out",
    "$merge",
    "$currentOp",
    "$collStats",
    "$indexStats",
    "$planCacheStats",
    "$listSessions",
    "$listLocalSessions",
    "$function",
    "$accumulator",
}


class PipelineValidationError(ValueError):
    """Raised when a generated pipeline fails the safety checks."""


# Accumulator operators that some LLMs (notably Gemini via structured output)
# occasionally emit as a bare string instead of the required object form.
# e.g. {"count": "$sum"} instead of {"count": {"$sum": 1}}. We repair these
# deterministically before validation/execution.
_ACCUMULATORS = {
    "$sum",
    "$avg",
    "$min",
    "$max",
    "$count",
    "$first",
    "$last",
    "$push",
    "$addToSet",
    "$stdDevPop",
    "$stdDevSamp",
}


def repair_pipeline(pipeline: Any) -> Any:
    """Deterministically fix common, safe LLM malformations in a pipeline.

    Currently repairs bare-string accumulators inside ``$group`` stages:
    a value like ``"$sum"`` becomes ``{"$sum": 1}`` (count) and any other
    accumulator becomes ``{"<op>": 1}`` as a best-effort default. Recurses
    into ``$facet`` and ``$lookup`` sub-pipelines.

    This does not change stage safety — ``validate_pipeline`` still runs
    afterwards. It only corrects syntax the LLM got wrong.
    """
    if not isinstance(pipeline, list):
        return pipeline

    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            continue
        (op,) = stage.keys()
        body = stage[op]
        if op == "$group" and isinstance(body, dict):
            for field, val in list(body.items()):
                if field == "_id":
                    continue
                # Bare-string accumulator, e.g. "count": "$sum".
                if isinstance(val, str) and val in _ACCUMULATORS:
                    body[field] = {val: 1}
        elif op == "$facet" and isinstance(body, dict):
            for name, sub in body.items():
                body[name] = repair_pipeline(sub)
        elif op == "$lookup" and isinstance(body, dict) and "pipeline" in body:
            body["pipeline"] = repair_pipeline(body["pipeline"])
    return pipeline


def validate_pipeline(pipeline: Any) -> list[dict]:
    """Validate and sanitise a candidate aggregation pipeline.

    Rules enforced:
      * The pipeline must be a list of single-key stage objects.
      * Every stage operator must be in ``ALLOWED_STAGES``.
      * No stage from ``FORBIDDEN_STAGES`` may appear anywhere, including
        inside ``$facet`` sub-pipelines and ``$lookup.pipeline``.
      * ``$where`` / ``$function`` / ``$accumulator`` (JS execution) are
        rejected recursively.

    Returns the pipeline unchanged if valid; raises PipelineValidationError
    otherwise. This runs *before* the query reaches MongoDB.
    """
    if not isinstance(pipeline, list) or not pipeline:
        raise PipelineValidationError("Pipeline must be a non-empty list of stages.")

    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            raise PipelineValidationError(
                "Each stage must be an object with exactly one operator key."
            )
        (op,) = stage.keys()
        if op in FORBIDDEN_STAGES:
            raise PipelineValidationError(f"Forbidden stage: {op}")
        if op not in ALLOWED_STAGES:
            raise PipelineValidationError(f"Stage not allowed: {op}")

        # Recurse into nested pipelines that can smuggle in forbidden ops.
        if op == "$facet":
            for sub in stage[op].values():
                validate_pipeline(sub)
        elif op == "$lookup":
            sub = stage[op].get("pipeline")
            if sub is not None:
                validate_pipeline(sub)

    # Reject any JavaScript-execution operators lurking in stage bodies.
    _reject_js(pipeline)
    return pipeline


def _reject_js(node: Any) -> None:
    """Recursively reject $where / $function / $accumulator / mapReduce JS."""
    banned = {"$where", "$function", "$accumulator", "$expr$function"}
    if isinstance(node, dict):
        for k, v in node.items():
            if k in banned:
                raise PipelineValidationError(f"JavaScript execution operator not allowed: {k}")
            _reject_js(v)
    elif isinstance(node, list):
        for item in node:
            _reject_js(item)


# ---------------------------------------------------------------------------
# Router.
# ---------------------------------------------------------------------------
class RouteDecision(BaseModel):
    """Structured router output."""

    path: Literal["direct_lookup", "analytical", "knowledge_base"] = Field(
        description=(
            "'direct_lookup' when the user asks for a specific record by an "
            "exact identifier (e.g. an order_id). 'analytical' for aggregated, "
            "comparative, diagnostic, or root-cause questions about the orders "
            "data. 'knowledge_base' for general/conceptual or how-to questions "
            "answered from documentation (not from the orders data)."
        )
    )
    order_id: str | None = Field(
        default=None,
        description="The exact order_id to look up, when path is direct_lookup.",
    )
    reason: str = Field(description="One short sentence explaining the choice.")


ROUTER_SYSTEM_PROMPT = """You are the intent router for an assistant that can
either query an e-commerce `orders` database or answer general questions from a
documentation knowledge base.

Classify the user's latest message into exactly one path:

- "direct_lookup": the user wants a specific order record identified by an
  exact identifier (an order_id, e.g. "order 37126471", "status of
  #10294"). Extract that identifier into `order_id`.
- "analytical": the user asks an aggregated, comparative, diagnostic, or
  root-cause question about the ORDERS DATA (e.g. "Why is revenue low today?",
  "top payment method for cancelled orders", "how many returns this week?").
- "knowledge_base": the user asks a general, conceptual, definitional, or
  how-to question that is answered from DOCUMENTATION rather than from the
  orders data (e.g. "What is MongoDB Atlas?", "How does vector search work?",
  "Explain embeddings", "What are the benefits of sharding?").

Decision guide:
- A specific order id present -> direct_lookup.
- A question about numbers/trends/causes in the orders data -> analytical.
- A "what is / how does / explain / why would" question about concepts,
  products, or best practices (not about this store's orders) -> knowledge_base.
When unsure between analytical and knowledge_base, ask: does answering this
require reading the store's order records? If yes -> analytical; if it is
general knowledge -> knowledge_base.
"""


def route_intent(user_text: str) -> RouteDecision:
    """Classify user intent and (for Path A) extract the order_id."""
    router = llm.with_structured_output(RouteDecision)
    return invoke_with_retry(
        router, [("system", ROUTER_SYSTEM_PROMPT), ("user", user_text)]
    )


# ---------------------------------------------------------------------------
# Path B: MQL generation.
# ---------------------------------------------------------------------------
class GeneratedPipeline(BaseModel):
    """Structured MQL-generation output: a pipeline plus a short rationale."""

    pipeline: list[dict] = Field(
        description="A read-only MongoDB aggregation pipeline as a JSON array of stage objects."
    )
    explanation: str = Field(
        description="One or two sentences describing what the pipeline computes."
    )


def _mql_generation_prompt() -> str:
    date_ctx = current_date_context()
    return f"""You are an expert MongoDB aggregation author. Given a user's
analytical question, produce a single READ-ONLY aggregation pipeline for the
`orders` collection.

CURRENT DATE CONTEXT (UTC):
  now:            {date_ctx['now_utc']}
  today start:    {date_ctx['today_start_utc']}
  baseline start: {date_ctx['baseline_start_utc']}  ({date_ctx['baseline_days']} full days before today)
  weekday:        {date_ctx['weekday']}

COLLECTION SCHEMA:
{schema_as_prompt()}

HARD RULES:
- Output ONLY stages from this allow-list: {sorted(ALLOWED_STAGES)}.
- NEVER use $out, $merge, index stages, $where, $function, or any JavaScript.
- The pipeline must be read-only. Do not attempt to modify data.
- `created_at` is a BSON date. Express date literals as MongoDB extended JSON,
  e.g. {{"$date": "2026-09-26T00:00:00Z"}}. Do NOT use plain strings for dates.
- For root-cause / "why" questions, use a $facet stage to compute BOTH the
  current window (e.g. today) AND a baseline (e.g. prior 7-day daily average)
  in one pass, so the answer can compare them.
- DIAGNOSTIC DEPTH (important): a revenue/sales drop can have MANY causes, not
  just lower order volume. Do NOT filter to a single status and stop. To
  diagnose "why is revenue/sales low/down", break the numbers down BY `status`
  so the answer can distinguish between competing explanations:
    * fewer fulfilled orders  -> genuine demand / volume drop
    * more cancelled orders   -> checkout, fraud-block, or pricing issues
    * more returned orders     -> product/quality or fulfillment issues
    * more pending/processing  -> payment-gateway or technical glitch (orders
                                  started but not completing)
  Compute, per status, BOTH a count of orders AND the summed total_amount, for
  today and for the baseline, so the analysis can attribute the revenue change
  to the specific status(es) responsible. Realized revenue should be based on
  completed sales (fulfilled), while cancelled/returned/pending represent
  potential revenue that did NOT convert — surface all of them. Prefer
  expressing the baseline as a PER-DAY AVERAGE per status so it is directly
  comparable to today (see the $facet example below).
- Enum fields: status in {['fulfilled','returned','cancelled','pending','processing']};
  payment_method in {['credit_card','paypal','apple_pay','bank_transfer']}.
- To analyse fields inside `items`, $unwind "$items" first.
- Keep result sets small: group/aggregate rather than returning raw documents,
  and add a $sort + $limit when returning ranked lists.

ACCUMULATOR SYNTAX (critical — malformed accumulators are rejected by MongoDB):
- Inside $group, every field value MUST be an accumulator OBJECT, never a bare
  string. Count with {{"$sum": 1}}, sum a field with {{"$sum": "$total_amount"}},
  average with {{"$avg": "$total_amount"}}.
- CORRECT:   {{"$group": {{"_id": "$payment_method", "n": {{"$sum": 1}}}}}}
- WRONG:     {{"$group": {{"_id": "$payment_method", "n": "$sum"}}}}   (bare string)
- Field references in expressions start with "$", e.g. "$status", "$items.price".

WORKED EXAMPLE — "top payment method for cancelled orders":
[
  {{"$match": {{"status": "cancelled"}}}},
  {{"$group": {{"_id": "$payment_method", "count": {{"$sum": 1}}}}}},
  {{"$sort": {{"count": -1}}}},
  {{"$limit": 1}}
]

$facet EXAMPLE — root-cause "why is revenue low today?" with a FULL status
breakdown so every possible cause is visible (each facet value is an ARRAY OF
STAGE OBJECTS, never sentences or strings). The "today" facet is a direct
per-status total; the "baseline" facet computes a true PER-DAY AVERAGE per
status (group per day+status first, then average across days) so the two sides
are directly comparable WITHOUT the reader having to divide anything:
[
  {{"$facet": {{
    "today_by_status": [
      {{"$match": {{"created_at": {{"$gte": {{"$date": "{date_ctx['today_start_utc']}"}}}}}}}},
      {{"$group": {{"_id": "$status", "orders": {{"$sum": 1}}, "amount": {{"$sum": "$total_amount"}}}}}}
    ],
    "baseline_daily_avg_by_status": [
      {{"$match": {{"created_at": {{"$gte": {{"$date": "{date_ctx['baseline_start_utc']}"}}, "$lt": {{"$date": "{date_ctx['today_start_utc']}"}}}}}}}},
      {{"$group": {{
        "_id": {{"status": "$status", "day": {{"$dateToString": {{"format": "%Y-%m-%d", "date": "$created_at"}}}}}},
        "orders": {{"$sum": 1}}, "amount": {{"$sum": "$total_amount"}}
      }}}},
      {{"$group": {{"_id": "$_id.status", "avg_orders_per_day": {{"$avg": "$orders"}}, "avg_amount_per_day": {{"$avg": "$amount"}}}}}}
    ]
  }}}}
]
Compare today's per-status totals against the baseline's per-day averages
directly. Name it "baseline_daily_avg_by_status" so the values are unambiguous.

OUTPUT FORMAT (STRICT):
Return ONLY a single JSON object, no prose or markdown fences, of the form:
{{"pipeline": [ ...stage objects... ], "explanation": "one or two sentences"}}
The "pipeline" value MUST be a JSON array whose elements are stage objects.
Every $facet sub-pipeline value MUST also be an array of stage objects.
"""


def _extract_json_object(text: str) -> dict:
    """Pull the first top-level JSON object out of an LLM text response.

    Handles models that wrap JSON in ```json fences or add stray prose. Raises
    ValueError if no parseable object is found.
    """
    # Strip common markdown code fences.
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    # Find the outermost {...} span.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(candidate[start : end + 1])


MAX_GENERATION_ATTEMPTS = 3


def generate_pipeline(user_text: str) -> GeneratedPipeline:
    """Ask the LLM to author a read-only aggregation pipeline.

    We request a raw JSON object (not provider structured output) and parse it
    ourselves. This proved far more reliable with Gemini for deeply nested
    pipelines (e.g. ``$facet``), where structured-output coercion tended to
    emit malformed sub-pipelines. The parsed pipeline is then passed through
    ``repair_pipeline`` to correct common, safe syntax mistakes (e.g.
    bare-string accumulators) before validation and execution.

    Smaller models occasionally emit slightly malformed JSON for complex
    pipelines. Rather than fail the whole request on a single bad emission, we
    retry generation a few times, feeding the parse error back to the model so
    it can correct itself. Rate-limit errors are not retried here (they bubble
    up from ``invoke_with_retry``, which already handles backoff).
    """
    messages = [("system", _mql_generation_prompt()), ("user", user_text)]
    last_err: Exception | None = None

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        msg = invoke_with_retry(llm, messages)
        raw = _text_of(msg)
        try:
            data = _extract_json_object(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            # Feed the malformed output and the error back for a self-correction.
            last_err = exc
            messages = [
                ("system", _mql_generation_prompt()),
                ("user", user_text),
                ("assistant", raw),
                ("user",
                 f"That response was not valid JSON ({exc}). Reply again with "
                 "ONLY a single valid JSON object of the form "
                 '{"pipeline": [...stage objects...], "explanation": "..."} '
                 "and nothing else."),
            ]
            continue

        pipeline = repair_pipeline(data.get("pipeline", []))
        return GeneratedPipeline(
            pipeline=pipeline,
            explanation=str(data.get("explanation", "")),
        )

    raise ValueError(
        f"Could not obtain valid pipeline JSON after {MAX_GENERATION_ATTEMPTS} "
        f"attempts. Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Synthesis.
# ---------------------------------------------------------------------------
LOOKUP_SYNTHESIS_PROMPT = """You are a helpful order-support assistant. The
user asked about a specific order. Below is the raw order document (JSON) that
was retrieved from the database, or a note that it was not found. Write a
concise, friendly natural-language answer. Mention the order status, total,
payment method, and key items when available. Do not invent fields.
"""

ANALYTICAL_SYNTHESIS_PROMPT = """You are a data analyst assistant. The user
asked an analytical/root-cause question about e-commerce orders. Below is the
JSON output of the aggregation that was executed. Interpret the numbers and
write an insightful, natural-language answer. If the data compares a current
window to a baseline, explain the difference and a likely explanation. Be
specific with figures. Do not fabricate data beyond what is shown.

For "why is revenue/sales low/down" questions, the data is broken down by order
status. When a facet is named like "baseline_daily_avg_by_status", its values
are ALREADY per-day averages — compare today's totals to them directly and do
NOT divide again. Do NOT jump to a single conclusion. Weigh ALL the statuses
and identify which one(s) actually explain the change:
- Fewer FULFILLED orders vs baseline  -> genuine demand / volume drop.
- More CANCELLED orders                -> checkout friction, fraud blocks, or
                                          pricing problems.
- More RETURNED orders                 -> product quality or fulfillment issues.
- More PENDING / PROCESSING orders     -> a likely payment-gateway or technical
                                          glitch: orders are being started but
                                          not completing.
Point out the specific status(es) driving the drop, quantify them, and only
name a cause the data supports. If several factors contribute, say so and rank
them. If the data does not clearly indicate a cause, say that plainly rather
than guessing.
"""


KB_SYNTHESIS_PROMPT = """You are a knowledge-base assistant. The user asked a
general/conceptual question. Below are the most relevant documentation
passages retrieved from the knowledge base, each with a title and source.

Answer the user's question USING ONLY the information in these passages. Rules:
- Ground every claim in the retrieved passages; do not add outside knowledge.
- If the passages do not contain the answer, say so plainly rather than
  guessing.
- Be concise and clear. Cite the source title(s) you relied on, e.g.
  "(source: MongoDB Atlas Overview)".
- If passages conflict or are only partially relevant, say what is and isn't
  covered.
"""


def _text_of(message: AIMessage) -> str:
    """Extract plain text from an AIMessage whose content may be block list."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") in ("output_text", "text"):
                parts.append(block.get("text", "") or "")
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts).strip()


def _synthesize(system_prompt: str, user_text: str, data: Any) -> str:
    # Pass pre-formatted text (e.g. a retrieval context block) through
    # verbatim; JSON-encode structured data (dicts/lists) for the model.
    data_block = data if isinstance(data, str) else json.dumps(data, indent=2, default=str)
    try:
        msg = invoke_with_retry(llm, [
            ("system", system_prompt),
            ("user", f"User question: {user_text}\n\nData:\n{data_block}"),
        ])
        return _text_of(msg)
    except Exception as exc:  # noqa: BLE001 - data is already fetched; degrade gracefully
        if _is_rate_limit_error(exc):
            # The query succeeded; only the natural-language summary failed.
            return (
                _rate_limit_message(exc)
                + " The raw query results are included below."
            )
        raise


# ---------------------------------------------------------------------------
# Result container + public entry point.
# ---------------------------------------------------------------------------
@dataclass
class ChatResult:
    """Everything the API returns for one turn."""

    response: str                       # natural-language answer
    path: str                           # "direct_lookup" | "analytical"
    router_reason: str = ""
    generated_mql: Any = None           # dict (Path A filter) or list (pipeline)
    raw_results: Any = None             # raw JSON returned from MongoDB
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "path": self.path,
            "router_reason": self.router_reason,
            "generated_mql": self.generated_mql,
            "raw_results": self.raw_results,
            "error": self.error,
            "meta": self.meta,
        }


def _retrieve_kb(query_text: str, k: int = KB_TOP_K) -> list[dict]:
    """Retrieve the top-k knowledge-base passages for a query.

    Dispatches to the configured embedding mode: 'manual' uses client-side
    Voyage embeddings; anything else uses Atlas auto-embed (the default, which
    needs no Voyage key).
    """
    if KB_EMBED_MODE == "manual":
        return vector_search_manual(query_text, k=k)
    return vector_search_auto(query_text, k=k)


def _handle_knowledge_base(user_text: str, decision: RouteDecision) -> ChatResult:
    """Path C (RAG): retrieve relevant docs by vector search, then augment the
    LLM's answer with them. This is a classic Retrieval-Augmented Generation
    flow: retrieve -> stuff context -> generate a grounded answer.
    """
    try:
        docs = _retrieve_kb(user_text)
    except Exception as exc:  # noqa: BLE001 - surface retrieval failures cleanly
        # A missing/invalid Voyage key only affects 'manual' mode; guide the user.
        msg = (
            "I couldn't search the knowledge base. If it is configured for "
            "manual Voyage embeddings, a valid VOYAGE_API_KEY is required; "
            "the Atlas auto-embed collection needs no key."
        )
        return ChatResult(
            response=msg,
            path="knowledge_base",
            router_reason=decision.reason,
            error=f"retrieval_error: {exc}",
        )

    if not docs:
        return ChatResult(
            response="I couldn't find anything relevant in the knowledge base for that question.",
            path="knowledge_base",
            router_reason=decision.reason,
            raw_results=[],
        )

    # Build the retrieval context block the LLM will be grounded on.
    context = "\n\n".join(
        f"[{i+1}] Title: {d.get('title','(untitled)')}\n"
        f"Source: {d.get('source','?')}\n"
        f"Content: {d.get('content','')}"
        for i, d in enumerate(docs)
    )

    response = _synthesize(KB_SYNTHESIS_PROMPT, user_text, context)
    return ChatResult(
        response=response,
        path="knowledge_base",
        router_reason=decision.reason,
        generated_mql=None,          # not a database query; no MQL to audit
        raw_results=docs,            # the retrieved passages (with scores)
        meta={"retrieved": len(docs), "embed_mode": KB_EMBED_MODE},
    )


def _handle_direct_lookup(user_text: str, decision: RouteDecision) -> ChatResult:
    """Path A: extract order_id (already done by router) and fetch the record."""
    order_id = (decision.order_id or "").strip()
    generated_mql = {"order_id": order_id} if order_id else None

    if not order_id:
        return ChatResult(
            response="I couldn't find an order ID in your message. Please share the exact order_id you want to look up.",
            path="direct_lookup",
            router_reason=decision.reason,
            generated_mql=generated_mql,
        )

    try:
        doc = find_one_order(order_id)
    except PyMongoError as exc:
        return ChatResult(
            response="Sorry, the lookup failed because of a database error.",
            path="direct_lookup",
            router_reason=decision.reason,
            generated_mql=generated_mql,
            error=str(exc),
        )

    if doc is None:
        return ChatResult(
            response=f"I couldn't find any order with ID {order_id}.",
            path="direct_lookup",
            router_reason=decision.reason,
            generated_mql=generated_mql,
            raw_results=None,
        )

    response = _synthesize(LOOKUP_SYNTHESIS_PROMPT, user_text, doc)
    return ChatResult(
        response=response,
        path="direct_lookup",
        router_reason=decision.reason,
        generated_mql=generated_mql,
        raw_results=doc,
    )


def _handle_analytical(user_text: str, decision: RouteDecision) -> ChatResult:
    """Path B: generate -> validate -> execute -> synthesise."""
    try:
        gen = generate_pipeline(user_text)
    except Exception as exc:  # noqa: BLE001 - surface generation failures to the user
        # Distinguish rate-limit failures (transient/quota) from genuine
        # translation problems so the user gets an actionable message.
        if _is_rate_limit_error(exc):
            msg = _rate_limit_message(exc)
        else:
            msg = "I wasn't able to translate that question into a query. Could you rephrase it?"
        return ChatResult(
            response=msg,
            path="analytical",
            router_reason=decision.reason,
            error=f"generation_error: {exc}",
        )

    pipeline = gen.pipeline

    # Sanitisation guardrail: validate BEFORE touching the database.
    try:
        validate_pipeline(pipeline)
    except PipelineValidationError as exc:
        return ChatResult(
            response="I generated a query but it didn't pass safety validation, so I didn't run it.",
            path="analytical",
            router_reason=decision.reason,
            generated_mql=pipeline,
            error=f"validation_error: {exc}",
        )

    try:
        results = run_aggregation(pipeline)
    except PyMongoError as exc:
        return ChatResult(
            response="The query was valid but failed to execute against the database.",
            path="analytical",
            router_reason=decision.reason,
            generated_mql=pipeline,
            error=str(exc),
            meta={"max_time_ms": MAX_TIME_MS},
        )

    response = _synthesize(ANALYTICAL_SYNTHESIS_PROMPT, user_text, results)
    return ChatResult(
        response=response,
        path="analytical",
        router_reason=decision.reason,
        generated_mql=pipeline,
        raw_results=results,
        meta={"pipeline_explanation": gen.explanation, "result_count": len(results)},
    )


def answer(user_text: str, history: list | None = None) -> ChatResult:
    """Route the request and return a fully-formed ChatResult.

    ``history`` is accepted for API symmetry / future multi-turn support; the
    router and generators are single-turn by design for determinism, so it is
    currently unused beyond being echoed through if provided.
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return ChatResult(response="Please enter a question.", path="analytical")

    try:
        decision = route_intent(user_text)
    except Exception as exc:  # noqa: BLE001 - router LLM call can hit rate limits
        return ChatResult(
            response=_rate_limit_message(exc),
            path="analytical",
            error=f"router_error: {exc}",
        )

    if decision.path == "direct_lookup":
        return _handle_direct_lookup(user_text, decision)
    if decision.path == "knowledge_base":
        return _handle_knowledge_base(user_text, decision)
    return _handle_analytical(user_text, decision)
