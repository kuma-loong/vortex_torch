export CUDA_VISIBLE_DEVICES=4

python run_ruler.py \
    --request_len 8191 \
    --model_name Qwen/Qwen3-4B \
    --dataset "niah_single_1" \
    --dataset_dir "/home/haizhonz/Zhaofeng/RULER/scripts/benchmark_root/Qwen3-4B/synthetic/4096/data" \
    --num_pages 16
