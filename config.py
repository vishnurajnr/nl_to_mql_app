"""Central configuration for the NL-to-MQL web application.

Loads environment variables and constructs the shared LLM client. Two LLM
providers are supported behind a single ``llm`` object so the rest of the app
(agent.py, server.py) never needs to know which one is active:

* **azure**  — Azure OpenAI, the default used by the other DA680 lab modules.
* **google** — Google Gemini via ``langchain-google-genai``. Gemini has a free
  daily quota, which makes it a good zero-cost option for this app.

Select the provider with the ``LLM_PROVIDER`` env var ("azure" or "google").
If ``LLM_PROVIDER`` is not set, we auto-detect: Google is chosen when a
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` is present and no Azure key is, else
Azure.

Environment variables (see README.md and the repo-root .env):
    MONGODB_URI            Atlas connection string.
    LLM_PROVIDER           "azure" | "google" (optional; auto-detected).

    # Azure OpenAI provider:
    AZURE_OPENAI_API_KEY   Azure OpenAI key.
    AZURE_OPENAI_ENDPOINT  Azure OpenAI endpoint URL.
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION (optional).

    # Google Gemini provider:
    GOOGLE_API_KEY         Gemini API key (GEMINI_API_KEY also accepted).
    GEMINI_MODEL           Model name (default "gemini-2.5-flash").

The spec references a generic ``LLM_API_KEY``; per provider that role is filled
by ``AZURE_OPENAI_API_KEY`` or ``GOOGLE_API_KEY``. The README documents both.
"""

import os

from dotenv import load_dotenv

# Resolve the repo-root .env regardless of the working directory.
load_dotenv(f"{os.path.dirname(os.path.abspath(__file__))}/../.env")

# --- Environment variables -------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI or MONGODB_URI == "<your-atlas-connection-string>":
    raise ValueError("MONGODB_URI environment variable is required")

# Database and collection targeted by this application.
DB_NAME = os.getenv("MONGODB_DB", "da680_shop")
ORDERS_COLLECTION = "orders"

# Hard ceiling for any database read, enforced on every query/aggregation.
MAX_TIME_MS = int(os.getenv("MQL_MAX_TIME_MS", "5000"))

# Maximum number of documents any single read may return.
MAX_RESULT_DOCS = int(os.getenv("MQL_MAX_RESULT_DOCS", "200"))

# --- Knowledge-base (RAG) configuration ------------------------------------
# A separate database/collection holding documents for semantic retrieval.
# Two retrieval modes are supported:
#   * "auto"   -> the collection uses an Atlas auto-embed vector index. Atlas
#                 embeds the query text server-side, so NO client-side Voyage
#                 key is needed. This is the default and works out of the box.
#   * "manual" -> the collection stores precomputed Voyage embeddings. The app
#                 must embed the query with the Voyage API before searching,
#                 which requires a valid VOYAGE_API_KEY.
KB_DB_NAME = os.getenv("KB_DB", "rag_demo")

KB_EMBED_MODE = (os.getenv("KB_EMBED_MODE") or "auto").strip().lower()

# Auto-embed collection (default): query by text, Atlas embeds server-side.
KB_AUTO_COLLECTION = os.getenv("KB_AUTO_COLLECTION", "knowledge_base_auto")
KB_AUTO_INDEX = os.getenv("KB_AUTO_INDEX", "vector_index_auto")
KB_AUTO_PATH = os.getenv("KB_AUTO_PATH", "content")

# Manual-embedding collection: query vector built client-side via Voyage.
KB_MANUAL_COLLECTION = os.getenv("KB_MANUAL_COLLECTION", "knowledge_base")
KB_MANUAL_INDEX = os.getenv("KB_MANUAL_INDEX", "vector_index")
KB_MANUAL_PATH = os.getenv("KB_MANUAL_PATH", "embedding")
# Voyage model whose output dimensions match the stored embeddings (768 dims
# here -> voyage-3-lite). Only used in "manual" mode.
KB_VOYAGE_MODEL = os.getenv("KB_VOYAGE_MODEL", "voyage-3-lite")

# How many documents to retrieve as context for each KB question.
KB_TOP_K = int(os.getenv("KB_TOP_K", "4"))

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Gemini API key can be provided under either common name.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _resolve_provider() -> str:
    """Decide which provider to use.

    Explicit ``LLM_PROVIDER`` wins. Otherwise prefer Google when a Gemini key
    is present but no Azure key is; fall back to Azure.
    """
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("azure", "google"):
        return explicit
    has_azure = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    if GOOGLE_API_KEY and not has_azure:
        return "google"
    return "azure"


LLM_PROVIDER = _resolve_provider()


# --- LLM client factory ----------------------------------------------------
# The web API returns a single JSON payload, so streaming is off by default.
# Function/tool calling and structured output are used by agent.py; both the
# Azure and Gemini clients support `.with_structured_output(...)` and
# `.bind_tools(...)`, so the agent code is provider-agnostic.
def _build_azure(streaming: bool):
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        use_responses_api=True,
        output_version="responses/v1",
        streaming=streaming,
        reasoning={"effort": "medium", "summary": "auto"},
        model_kwargs={"parallel_tool_calls": False},
    )


def _build_google(streaming: bool):
    # Imported lazily so the app runs without langchain-google-genai installed
    # when the Azure provider is selected.
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not GOOGLE_API_KEY:
        raise ValueError(
            "LLM_PROVIDER=google but no GOOGLE_API_KEY / GEMINI_API_KEY is set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    # gemini-2.5-flash: free-tier friendly, supports tool calling + structured
    # output, 1M-token context. Use gemini-2.5-flash-lite for a higher daily
    # request quota if you hit rate limits.
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        streaming=streaming,
        # Keep tool calls sequential to mirror the Azure client's behaviour.
        model_kwargs={},
    )


def _build_llm(streaming: bool = False):
    if LLM_PROVIDER == "google":
        return _build_google(streaming)
    return _build_azure(streaming)


llm = _build_llm(streaming=False)


def model_info() -> dict:
    """Report the active LLM provider and model for display in the UI/API.

    Reads the real configured values rather than hardcoding, so the badge the
    user sees always matches what is actually answering their questions.
    """
    if LLM_PROVIDER == "google":
        provider_label = "Google Gemini"
        model = GEMINI_MODEL
    else:
        provider_label = "Azure OpenAI"
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-nano")
    return {
        "provider": LLM_PROVIDER,        # machine value: "azure" | "google"
        "provider_label": provider_label,  # human label
        "model": model,                  # model / deployment name
    }


def kb_info() -> dict:
    """Report the active knowledge-base retrieval configuration."""
    if KB_EMBED_MODE == "manual":
        collection, index = KB_MANUAL_COLLECTION, KB_MANUAL_INDEX
        embed = f"Voyage ({KB_VOYAGE_MODEL})"
    else:
        collection, index = KB_AUTO_COLLECTION, KB_AUTO_INDEX
        embed = "Atlas auto-embed"
    return {
        "db": KB_DB_NAME,
        "mode": KB_EMBED_MODE,     # "auto" | "manual"
        "collection": collection,
        "index": index,
        "embedding": embed,
        "top_k": KB_TOP_K,
    }
