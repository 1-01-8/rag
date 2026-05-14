#!/bin/bash
# Project-local Qwen 3.5 9B vLLM launcher with tool-calling enabled.
# Differs from /home/xxm/models/qwen3.5-9b/serve_vllm.sh by adding:
#   --enable-auto-tool-choice
#   --tool-call-parser qwen3_xml
#
# The chat_template_no_think.jinja uses the Qwen3 XML-style tool call format
# (<tool_call><function=...><parameter=...>), so we must use the qwen3_xml
# parser, NOT hermes (which expects JSON inside <tool_call> tags).
#
# Required for multi_agent tests that exercise real tool use.
# Assumes the qwen35 conda env is on PATH (source it before running).

cd /home/xxm/models/qwen3.5-9b

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model ./model_weights \
    --served-model-name qwen3.5-9b \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --gdn-prefill-backend triton \
    --chat-template ./chat_template_no_think.jinja \
    --language-model-only \
    --port 8000 \
    --enable-log-requests \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
