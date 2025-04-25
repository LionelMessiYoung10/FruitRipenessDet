# -*- coding: utf-8 -*-
"""
@Auth ： 来孟阳

@IDE ：PyCharm
运行本项目需要python3.8及以下依赖库（完整库见requirements.txt）：
    opencv-python==4.5.5.64
    tensorflow==2.9.1
    PyQt5==5.15.6
    scikit-image==0.19.3
    torch==1.8.0
    keras==2.9.0
    Pillow==9.0.1
    scipy==1.8.0
点击运行主程序runMain.py，程序所在文件夹路径中请勿出现中文
"""
Chinese_name = {"Apple Fresh": "新鲜苹果", "Apple Rotten": "腐烂苹果", "Apple Semifresh": "半熟苹果",
                "Apple Semirotten": "半腐烂苹果", "Banana Fresh": "新鲜香蕉", "Banana Rotten": "腐烂香蕉",
                "Banana Semifresh": "半熟香蕉", "Banana Semirotten": "半腐烂香蕉",
                "Mango Fresh": "新鲜芒果", "Mango Rotten": "腐烂芒果", "Mango Semifresh": "半熟芒果",
                "Mango Semirotten": "半腐烂芒果", "Melon Fresh": "新鲜瓜类", "Melon Rotten": "腐烂瓜类",
                "Melon Semifresh": "半熟瓜类", "Melon Semirotten": "半腐烂瓜类",
                "Orange Fresh": "新鲜橙子", "Orange Rotten": "腐烂橙子", "Orange Semifresh": "半熟橙子",
                "Orange Semirotten": "半腐烂橙子", "Peach Fresh": "新鲜桃子", "Peach Rotten": "腐烂桃子",
                "Peach Semifresh": "半熟桃子", "Peach Semirotten": "半腐烂桃子",
                "Pear Fresh": "新鲜梨子", "Pear Rotten": "腐烂梨子", "Pear Semifresh": "半熟梨子",
                "Pear Semirotten": "半腐烂梨子", "Ripe_Grape": "成熟葡萄", "Unripe_Grape": "未成熟葡萄"}
