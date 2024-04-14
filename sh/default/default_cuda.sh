selected_gpu=""
count=0
#加载CUDA序号自动填充
gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
# 检查是否有可用的 GPU
if [ -n "$gpu_info" ]; then
    # 遍历每个 GPU 的信息
    while IFS=',' read -r index used_memory; do
        used_memory=$(echo $used_memory | grep -oE '[0-9]+')
         if [ "$used_memory" -lt 20 ]; then
            # 如果满足条件，将 GPU 序号添加到 selected_gpu 变量中
            if [ -z "$selected_gpu" ]; then
                selected_gpu="$index"
            else
                selected_gpu="$selected_gpu,$index"
            fi
            ((count++))
            #最多加载几个GPU
            if [ "$count" -ge 2 ]; then
                break
            fi
        fi
    done <<< "$gpu_info"
fi

export CUDA=${selected_gpu}
echo "加载显卡:${CUDA}"