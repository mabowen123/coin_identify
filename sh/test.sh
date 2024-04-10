#根目录
root_path="/Users/mabowen/Documents/www/coin_Identify"
#数据集根目录
data_root_path=${root_path}"/datasets/origin"
#图片处理代码目录
image_process_root_path=${root_path}"/image_process"

#训练标题
name="淳化元宝"
#训练集execl路径
train_execl_path="/Users/mabowen/Documents/www/coin_Identify/datasets/淳化元宝/4.3淳化元宝训练样本.xlsx"
#测试集execl路径
test_execl_path="/Users/mabowen/Documents/www/coin_Identify/datasets/淳化元宝/4.3淳化元宝测试样本.xlsx"

#图片输出路径
output_path=${data_root_path}/${name}


#训练数据加载
python3  ${image_process_root_path}/load_image_data.py --train_execl_path ${train_execl_path} --test_execl_path ${test_execl_path} --output_path ${output_path}