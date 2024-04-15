## 训推一体权重转换工具

推理场景基于Huggingface的权重进行推理，Huggingface主要基于GPU训练，而昇腾主要在NPU上进行训练，不同硬件平台对应的模型权重格式存在差异。需要支持在NPU和GPU上训练好的模型转成Huggingface格式safetensors文件，用于推理使用。


#### 前提条件
准备以下权重：
1. 复训前huggingface权重
2. 复训后权重

##### 依赖安装

如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：**pip3 install xxx** **--user**，安装命令可在任意路径下执行。

```shell
# python=3.8
pip install torch-2.1.0-cp38-cp38m-linux_aarch64.whl
pip install torch_npu-2.1.0.post5_XXXXXX-cp38-cp38m-linux_aarch64.whl
source /path/to/Ascend/ascend-toolkit/set_env.sh

git clone https://gitee.com/ascend/ModelLink.git 
git clone https://gitee.com/ascend/AscendSpeed.git

cd AscendSpeed
pip3 install -r requirements.txt
pip3 install -e .
cd ..
cd ModelLink
pip3 install -r requirements.txt
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..
```

##### 代码获取

```shell
git clone https://gitee.com/Ascend/att.git
cd att
git checkout develop

cd ../ModelLink
git reset --hard c566ce4fa99cf3ea179b163355fca2c2aedfc471
cp ../att/debug/weight_convert/diff.patch .
git apply --check  diff.patch
git apply diff.patch
cd ../att/debug/weight_convert/
```

#### 启动工具

1. 参考表1 参数说明配置信息，执行如下命令启动分析任务。转换后权重会保存在`原始huggingface权重存放位置/mg2hf`下

```shell
python3 convert_ckpt.py -i 待转换权重路径 -o 原始huggingface权重存放位置 -m 模型类型，可选项：llama/bloom\
                        [--target-tensor-parallel-size 张量并行数 \
                         --target-pipeline-parallel-size 流水线并行数\
                         --embed-layernorm]
```

   **表1 参数说明**

   | 参数                | 参数说明                                                     | 取值示例                                                |
   | ---------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
   | -i <br> --input-model-dir          | **必选** 待转换权重文件的存放位置         | /home/*xxx*/*input_weight*     |
   | -o <br> --output-model-dir         |  **必选** 导出权重文件的存放位置（要求目录下有原始huggingface权重） | /home/*xxx*/*output_weight* |
   | -m <br> --model                    |  **必选** 转换的模型类型 | llama（默认）<br>bloom |
   | --target-tensor-parallel-size      | 转换后张量并行数 | 1 |
   | --target-pipeline-parallel-size    | 转换后流水线并行数 | 1 |
   | --embed-layernorm    | 模型中是否存在embedding layernorm结构 | False(默认)<br>True |
   | -h <br>--help       | 显示帮助信息。                        | -                 |

   
2. 模型转换命令参考

 **Llama 7/13/65B**、 **Llama2 7/13/70B**
```shell
python3 convert_ckpt.py -o "your huggingface checkpoint output path" \
                        -i "your megatron checkpoint path" \
                        --model llama
```

 **Bloom 7B**
```shell
python3 convert_ckpt.py -o "your huggingface checkpoint output path" \
                        -i "your megatron checkpoint path" \
                        --model bloom
```


3. 分析完成后，进入输出路径，查看转换结果。