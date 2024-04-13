#!/bin/bash
# -*- coding: utf-8 -*-

#训练标题
file_name=$(basename "$0")
name=${file_name%.*}
# 默认参数
default_op=1
#根目录
root_path=$(readlink -f "../")
if [ ! -e "$root_path" ]; then
    echo "错误: 根目录不存在"
    exit 1
fi

#日志文件夹
current_time=$(date "+%m%d%H")
log_root_path="${root_path}/log"
mkdir -p "${log_root_path}"
log_path="${log_root_path}/${name}_${current_time}.log"

#数据集根目录
data_root_path=${root_path}"/datasets"
#图片处理代码目录
image_process_root_path=${root_path}"/image_process"
#图片输出路径
output_path=${data_root_path}/${name}
#识别类别个数
num_classes=$(ls ${output_path}/train/ | wc -l)
#目标路径更改文件名称
bash ./default/rename_file.sh  ${output_path}
#训练集execl路径
train_execl_path="${data_root_path}/${name}/训练样本.xlsx"
#验证集execl路径
valida_execl_path="${data_root_path}/${name}/验证样本.xlsx"

# ############################数据处理#############################
echo "是否处理数据,会重新删除,生成文件!  0-不操作(默认) 1-切割数据 2-下载&切割数据 "
read -t 3 -p "输入你的操作:" op
op=${op:-0}
if [ $op -gt 0 ]; then
  if [ $op -gt 1 ]; then
 #训练数据加载
 python3  ${image_process_root_path}/load_image_data.py \
        --train_execl_path ${train_execl_path} \
        --valida_execl_path ${valida_execl_path} \
        --max_threads_num 200 \
        --output_path ${output_path}  |  tee -a "${log_path}"
  fi
 #数据切割
 python3  ${image_process_root_path}/divide_data.py \
          --divide_path ${output_path} |  tee -a "${log_path}"
fi
#############################数据处理#############################


############################训练###################################
model_path=${output_path}'/model'
mkdir -p ${model_path}  &&    chmod -R 777  ${model_path}

master_port=20001
epochs=200

# model='tf_efficientnet_b8.ap_in1k'
# img_size=672
# batch_size=6

model='efficientnet_b5'
img_size=416
batch_size=32

#############################训练#################################
echo "是否重新训练 1-是  0-不操作 (默认-${default_op})"
read -t 3 -p "输入你的操作:" op
op=${op:-${default_op}}
if [ $op -gt 0 ]; then
# python3 -u \
CUDA=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=${CUDA} python3 -m torch.distributed.launch --nproc_per_node=$(echo "$CUDA" | awk -F',' '{print NF}') --master_port=${master_port} --use_env \
  ${root_path}"/train"/train_add_loss.py ${output_path}\
      --num-classes "${num_classes}"\
      --model ${model} \
      -b ${batch_size}\
      --img-size ${img_size} \
      --sched step --epochs ${epochs}  \
      --pretrained --lr .001 -j 8  --opt adamp  --weight-decay 1e-5  --decay-epochs 2  --decay-rate .95  --warmup-lr 1e-6 \
      --warmup-epochs 1      --remode pixel       --reprob 0.15     --amp \
      --output ${model_path} | tee -a "${log_path}"
fi
#############################训练#################################


#############################验证#################################
index_path=${output_path}"/index.txt"
model_file_path=${model_path}'/'$(ls -l ${model_path} | grep ${model} | tail -n 1 | awk '{print $9}')
echo '模型完整路径:' "${model_file_path}"
#############################验证#################################
echo "是否启动验证脚本 1-是  0-不操作 (默认-${default_op})"
read -t 3 -p "输入你的操作:" op
op=${op:-${default_op}}
if [ $op -gt 0 ]; then
python3 ${root_path}"/valid"/validate.py ${output_path}"/valida"\
       --class_map ${index_path} \
       --num-classes "${num_classes}" \
       --img-size 416 \
       --model ${model}\
       --split val2 -b 128 \
       --checkpoint "${model_file_path}" | tee -a "${log_path}"
fi
#############################验证#################################



#############################模型导出#################################
checkpointName='last'
model_export_path="${model_file_path}/${name}_${checkpointName}_model_best_one.onnx"
if [ "$checkpointName" == 'model_best' ]; then
    model_export_path="${model_file_path}/${name}_${checkpointName}_one.onnx"
fi
echp "模型导出路径:${model_export_path}"
#############################模型导出#################################
echo "是否导出模型 1-是  0-不操作 (默认-${default_op})"
read -t 3 -p "输入你的操作:" op
op=${op:-${default_op}}
if [ $op -gt 0 ]; then
python3 ${root_path}/valid/onnx_export_vit.py  \
       --num_classes ${num_classes}  \
       --input_dim ${img_size} \
       --folder_path ${model_file_path}/${checkpointName}'.pth.tar' \
       --single_model --model_name ${model} | tee -a "${log_path}"
mv ${model_file_path}/${checkpointName}'.onnx' ${model_export_path}
fi
#############################模型导出#################################


#############################验证服务#################################
echo "是否验证execl 1-是  0-不操作 (默认-${default_op})"
read -t 3 -p "输入你的操作:" op
op=${op:-${default_op}}
if [ $op -gt 0 ]; then
python3 ${root_path}/"valid"/predict_class_excel.py \
        --valida_execl_path ${valida_execl_path} \
        --model_path ${model_export_path} | tee -a "${log_path}"
fi
#############################验证服务###########################
