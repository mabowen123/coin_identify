#根目录
source ./env.sh

url=""
model_export_path=""

python3 ${valid_root_path}"/predict_class_excel.py \
        --valida_execl_path ${valida_execl_path} \
        --url ${url} \
        --model_path ${model_export_path} | tee -a "${log_path}"