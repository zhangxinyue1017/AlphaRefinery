#!/usr/bin/env bash

# Copy this file to `llm_refine_provider_env.sh`, then fill in real credentials.
# Usage:
#   cp ./llm_refine_provider_env.example.sh ./llm_refine_provider_env.sh
#   source ./llm_refine_provider_env.sh
#
# The copied `llm_refine_provider_env.sh` stays local and is ignored by git.


# ===== OpenAI-compatible default example =====
# Replace the values below with the provider and model you actually use.
# For OpenAI, keep the default base_url.
# For an OpenAI-compatible relay, change `LLM_PROVIDER_NAME` and `LLM_BASE_URL`.
export LLM_PROVIDER_NAME="openai"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="your-api-key-here"
export LLM_MODEL="gpt-5.4"
export LLM_TEMPERATURE="0.4"
export LLM_MAX_TOKENS="2000"
export LLM_TIMEOUT="600"


echo "[llm-refine] provider=${LLM_PROVIDER_NAME} model=${LLM_MODEL}"
