export HF_HOME=/mnt/hdd/wyt/hf
experiments/optimize/4bit.sh Qwen/Qwen3-4B

python3 scripts/pseudo_quant.py \
    --model Qwen/Qwen3-4B \
    --result-dir output/Qwen3-4B \
    --output-path models/Qwen3-4B-PARO-pseudo


./experiments/tasks/non_reasoning.sh models/Qwen3-4B-PARO-pseudo


python scripts/eval_ppl.py \
    --model models/Qwen3-4B-PARO-pseudo \
    --seed 0 \
    --seqlen 2048