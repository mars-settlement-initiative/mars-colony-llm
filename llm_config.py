"""Decision settings only. Keep world/benchmark settings in config.py fixed."""

import os

# Gemini is the default online provider. Use --mode mock without credentials.
# Free access depends on the API key's Google project remaining on Free Tier.
LLM_MODE = os.getenv("MARS_LLM_MODE", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_API_CALLS = int(os.getenv("MARS_MAX_API_CALLS", "120"))
# Conservative teaching defaults, NOT a claim about Google's current quotas.
# Requests are sequential already. Set MARS_GEMINI_REQUEST_SECONDS when a
# provider account needs explicit pacing between calls.
GEMINI_MIN_REQUEST_SECONDS = float(os.getenv("MARS_GEMINI_REQUEST_SECONDS", "1"))
GEMINI_DECISION_INTERVAL = int(os.getenv("MARS_GEMINI_DECISION_INTERVAL", "10"))
API_TIMEOUT_SECONDS = 20
# Gemini counts internal thinking tokens toward this limit. Keep enough room for
# thinking plus the small structured decision response.
MAX_OUTPUT_TOKENS = 1024


def default_model(mode):
    return GEMINI_MODEL if mode == "gemini" else OPENAI_MODEL
