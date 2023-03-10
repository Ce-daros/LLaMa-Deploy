# Windows下LLaMa的本地部署

## 速查 Todo List

1. 安装 Python 和 Anaconda
2. 安装 CUDA Toolkit 和 cuDNN
3. 下载模型和创建虚拟环境
4. 安装 Pytorch
5. 运行模型

## 安装 Python 和 Anaconda 
### [Python](https://www.python.org/ftp/python/3.11.2/python-3.11.2-amd64.exe)

验证：`python --version` 返回类似 `Python 3.11.2`
### [Conda](https://repo.anaconda.com/archive/Anaconda3-2022.10-Windows-x86_64.exe)

验证：`conda --version` 返回类似 `conda 22.9.0`

## 安装 [Cuda Toolkit](https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_516.94_windows.exe) 与 [cuDNN](https://developer.nvidia.com/downloads/c118-cudnn-windows-8664-880121cuda11-archivezip)

安装前请先[更新驱动](https://www.nvidia.cn/geforce/drivers/)

cuDNN 需注册 Nvidia 账号后下载。

验证：
```
C:\Users\xxx>nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:59:34_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

## 下载模型和创建虚拟环境

[模型下载](magnet:?xt=urn:btih:b8287ebfa04f879b048d4d4404108cf3e8014352)

### 创建与激活虚拟环境

```
C:\Users\xxx>conda create --name llama_env python=3.9

C:\Users\xxx>conda activate llama_env

(llama_env) C:\Users\xxx>
```

## 安装 Pytorch

在上一步的虚拟环境中

`(llama_env) C:\Users\xxx>conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

验证：
```
(llama) C:\Users\xxx>python
Python 3.9.16 (main, Mar  1 2023, 18:30:21) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
```

## 运行模型

```
git clone https://github.com/facebookresearch/llama
cd llama
pip install -r requirements.txt
pip install -e .
```

`python -m torch.distributed.launch example.py --ckpt_dir E:/model/7B --tokenizer_path E:/model/tokenizer.model --max_batch_size=1`

`E:/model/7B` `E:/model/tokenizer.model` 自行替换。

```
(llama) E:\LLaMA>python -m torch.distributed.launch example.py --ckpt_dir E:/llama/model/7B --tokenizer_path E:/llama/model/tokenizer.model --max_batch_size=1
NOTE: Redirects are currently not supported in Windows or MacOs.
D:\Users\xxx\anaconda3\envs\llama\lib\site-packages\torch\distributed\launch.py:180: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See
https://pytorch.org/docs/stable/distributed.html#launch-utility for
further instructions

> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{"seed": 1, "temp": 0.7, "top_p": 0.0, "top_k": 40, "repetition_penalty": 1.1764705882352942, "max_seq_len": 1024, "max_gen_len": 512}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loading
Loaded in 8.75 seconds
```

非专业级显卡不建议尝试 7B 以外的模型

## 修改参数

皆为`example.py`

+ `prompts = [...` 对话内容
+ `max_seq_len: int = 1024,` 单次问答长度
+ `count: int = 3,` 每个问题尝试生成的答案个数
