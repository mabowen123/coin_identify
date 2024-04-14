echo "加载环境变量"
export root_path="xxxxxxxxxxxxx"
if [ ! -e "$root_path" ]; then
    echo "错误: 根目录不存在"
    exit 1
fi

echo "根目录:${root_path}"
#日志文件夹
current_time=$(date "+%m%d%H%M")
export log_root_path="${root_path}/log"
#默认日志路径
export log_path="${log_root_path}/${current_time}.log"
#数据集目录
export data_root_path=${root_path}"/datasets"
echo "数据集路径:${data_root_path}"
#图片代码目录
export image_process_root_path=${root_path}"/image_process"
echo "图片代码路径:${image_process_root_path}"
#训练代码路径
export train_process_root_path=${root_path}"/train"
echo "训练代码路径:${train_process_root_path}"
#sh脚本路径
export sh_root_path=${root_path}"/sh"
echo "sh脚本路径:${sh_root_path}"
#验证代码路径
export valid_root_path=${root_path}"/valid"
echo "验证代码路径:${valid_root_path}"
if [ $# -ne  1 ]; then
  echo "未输入name"
fi

name=$1
# 检查 name 是否非空
if [ -n "$name" ]; then
       mkdir -p "${log_root_path}/${name}"
       export log_path="${log_root_path}/${name}/${current_time}.log"
       echo "日志路径:${log_path}"
       export output_path=${data_root_path}/${name}
       echo "目标输出路径:${output_path}"
       #训练图片文件夹路径
       export train_image_path=${output_path}"/train"
       echo "训练图片文件夹路径:${train_image_path}"
       export  model_path=${output_path}'/model'
       mkdir -p ${model_path}  &&    chmod -R 777  ${model_path}
       echo "模型存放路径:${model_path}"
       #训练集execl路径
       export train_execl_path="${data_root_path}/${name}/训练样本.xlsx"
       echo "训练集execl路径:${train_execl_path}"
       #验证集execl路径
       export valida_execl_path="${data_root_path}/${name}/验证样本.xlsx"
       echo "验证集execl路径:${valida_execl_path}"
       #index文件路径
       index_path=${output_path}"/index.txt"
       echo "index文件路径:${index_path}"
fi

chmod -R 777 ${root_path}
cd ${sh_root_path}