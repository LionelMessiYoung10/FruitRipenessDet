# -*- coding: utf-8 -*-

__version__ = '1.0'
__author__ = 'Lai Mengyang'
__email__ = 'lmy_sdufe@foxmail.com'
__url__ = 'https://github.com/LionelMessiYoung10?tab=repositories'

AUTHOR_INFO = ("基于机器学习和图像处理的苹果识别系统v1.0\n"
               "作者：来孟阳\n")

ENV_CONFIG = ("[配置环境]\n"
              "(1)使用anaconda新建python3.10环境:\n"
              "conda create -n env_rec python=3.10\n"
              "(2)激活创建的环境:\n"
              "conda activate env_rec\n"
              "(3)使用pip安装所需的依赖，可通过requirements.txt:\n"
              "pip install -r requirements.txt\n")

with open('./环境配置.txt', 'w', encoding='utf-8') as f:
    f.writelines(ENV_CONFIG + "\n\n" + AUTHOR_INFO)
