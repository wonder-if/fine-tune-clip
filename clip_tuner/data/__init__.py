from .restructure_datasets import add_train_dir
from .loading import DatasetManager, load_dataset

"""
项目架构————数据模块

我们的项目开发基于huggingface的dataset和transformers库, 其作用是实现clip类的模型的
- 微调
- zero-shot
- few-shot
- 域自适应等功能。

因此，数据部分就是要求在实验时, 尽可能的方便的加载和预处理数据集, 数据集包括:
- 域自适应数据集：
    - office-31:
        - amazon, A, amazon/
        - dslr, D, dslr/
        - webcam, W, webcam/
    - office-home:
        - Art, Ar, art/
        - Clipart, Cl, clipart/
        - Product, Pr, product/
        - Real World, Rw, real_world/
    - visda-2017:
        - Synthetic, Syn, synthetic/
        - Real, Rel, real/
    - domainnet:
        - Clipart, clp, clipart/
        - Infograph, inf, infograph/
        - Painting, pnt, painting/
        - Quickdraw, qdr, quickdraw/
        - Real, rel, real/
        - Sketch, skt, sketch/

1. 需求分析：

    a. 我们对数据集的需求主要是本地数据集, 由于dataset库可以直接加载本地数据集, 因此我们只需要管理路径即可。
    b. 我们希望通过一个json配置文件来管理数据集的路径, 方便后续的修改和扩展。
    c. 我们的任务包括域自适应任务, 因此数据集可能包括多个域, 我们需要能够加载多个域的数据集。

2. 功能设计:

    a. 用例设计:
        1) 输入路径加载数据集
            输入: 数据集路径
            输出: 数据集
        2) 输入数据集名加载数据集
            输入: 配置文件、数据集名
            输出: 数据集
        3) 输入数据集名和域名加载数据集
            输入: 配置文件、数据集名、域名
            输出: 数据集
        4) 数据集预处理
            输入: 数据集、预处理方式
            输出: 预处理后的数据集

    b. 架构设计：
        1) 数据集管理器：管理数据集的路径和加载
        2) 数据集加载函数：根据数据集路径和数据集名(域名)加载数据集
        3) 数据集预处理方法：对数据集进行预处理, 包括prompt生成, tokenize, padding等操作

3. 详细设计:


"""
