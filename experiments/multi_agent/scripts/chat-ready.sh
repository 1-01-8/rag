#!/usr/bin/env bash
# 已验证可用的快速启动: SiliconFlow + DeepSeek-V3.1 + ma_statutes (13722 chunks)
# + 无 Supervisor (省时) + 全量索引 (秒级 search)
#
# 用法:
#     conda activate qwen35  && bash scripts/chat-ready.sh
#
# 或加专业方向 (民事/劳动/交通/婚姻/房产/通用):
#     bash scripts/chat-ready.sh --specialty 劳动
#
# 继续上次会话:
#     bash scripts/chat-ready.sh --session-id chat_xxxxxx
#
# 此脚本透传所有额外参数给 chat.py, 所以可以叠加任意 --max-* / --no-supervisor 等
set -euo pipefail

# 防止用户传 --provider/--model 跟脚本硬编码冲突 (会导致 model 名串到错的端点)
for arg in "$@"; do
    case "$arg" in
        --provider|--provider=*|--model|--model=*)
            echo "❌ chat-ready.sh 是 SiliconFlow + DeepSeek-V3.1 专用包装"
            echo "   不要传 --provider / --model, 这两个已经写死."
            echo ""
            echo "   想换 provider/model? 直接调 chat.py:"
            echo "     python scripts/chat.py --provider local \\"
            echo "         --statutes-collection ma_statutes \\"
            echo "         --statutes-sparse data/indexes/statutes_sparse.json"
            exit 1
            ;;
    esac
done

# 前置检查
if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
    echo "❌ SILICONFLOW_API_KEY 未设置"
    echo "   export SILICONFLOW_API_KEY=sk-xxx  (https://cloud.siliconflow.cn)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 已知可用的索引文件 — 若缺失提示用户先 build
if [[ ! -f "data/indexes/statutes_sparse.json" ]]; then
    echo "❌ data/indexes/statutes_sparse.json 不存在 — ma_statutes 索引未建"
    echo "   先跑: python scripts/build_statutes_index.py \\"
    echo "       --corpus-dir /home/xxm/rag/Chinese-Laws/extracted \\"
    echo "       --collection ma_statutes \\"
    echo "       --sparse-out data/indexes/statutes_sparse.json \\"
    echo "       --batch-size 8"
    exit 1
fi

exec python scripts/chat.py \
    --provider siliconflow \
    --model deepseek-ai/DeepSeek-V3.1 \
    --statutes-collection ma_statutes \
    --statutes-sparse data/indexes/statutes_sparse.json \
    --no-supervisor \
    "$@"
