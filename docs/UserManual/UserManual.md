---
title: FyTok
subtitle: 用户手册
Author: 于治 yuzhi@ipp.ac.cn
toc-title: 目录
number-section: True
mainfont: Noto Sans Mono CJK SC
---

# 引言

## 标识

- 软件名称 : FyTok 托卡马克模拟器
- 版本号 : 1.0.0
- 完成时间 : 2021.8.26

| 缩写 | 描述                                             |
| ---- | ------------------------------------------------ |
| ITER | International Thermonuclear Experimental Reactor |
| IMAS | Integrated Modeling & Analysis Suite             |
| DD   | Data Dictionary 数据字典                         |
| IM   | Integrated Modeling                              |
| URI  | Uniform Resource Identifier                      |
| DAG  | Directed acyclic graph 有向无环图                |
| CLI  | Command Line Interface                           |
| DOM  | Document Object Model                            |

## 系统概述

**FyTok**是一个托卡马克（Tokamak）集成建模(Integrated Modeling,IM)和分析框架

**FyTok**是由中国科学院等离子体物理研究所开发。主要贡献人员包括：

- 代码编写：于 治，yuzhi@ipp.ac.cn
- 文档管理：刘晓娟，lxj@ipp.ac.cn

本程序开发受到下列项目支持：

- 国家磁约束核聚变能发展研究专项（National MCF Energy R&D Program under Contract），氘氚聚变等离子体中 alpha 粒子过程对等离子体约束 性能影响的理论模拟研究，Alpha 粒子密度和能谱分布的集成建模研究，编号 2018YFE0304102
- 聚变堆主机关键系统综合研究设施 （ Comprehensive Research Facility for Fusion Technology Program of China）总控系统， 集成数值建模和数据分析系统框架开发，编号 No. 2018-000052-73-01-001228.

## 文档概述

本文档为 FyTok 的《用户手册》。

# 引用文件

| 文件名称                                | 来源           |
| --------------------------------------- | -------------- |
|“Design and First Applications of the ITER Integrated Modelling & Analysis Suite.” Nuclear Fusion 55, no. 12 (2015). | doi:10.1088/0029-5515/55/12/123006|
| The ITER Integrated Modelling Programme | ITER IDM UID 2EFR4K |

# 软件综述

## 软件应用

**FyTok** 是一个 Python module，主要功能是将数据源根据预先定义的映射关系转换为层次化树状（Hierarchical Tree）数据结构，并通过统一的路径标识（Uniform Resource Identifier ,URI）形式访问。

## 软件清单

```bash
FyTok/python/
└── spdm
    ├── data
    │   ├── Collection.py
    │   ├── Combiner.py
    │   ├── Document.py
    │   ├── Entry.py
    │   ├── Field.py
    │   ├── File.py
    │   ├── Function.py
    │   ├── Mapping.py
    │   ├── Node.py
    │   ├── db
    │   │   └── FileCollection.py
    │   └── file
    │       ├── PluginHDF5.py
    │       ├── PluginJSON.py
    │       ├── PluginNAMELIST.py
    │       ├── PluginNumPy.py
    │       ├── PluginXML.py
    │       └── PluginYAML.py
    └── util
        ├── Alias.py
        ├── BiMap.py
        ├── Factory.py
        ├── LazyProxy.py
        ├── Multimap.py
        ├── PathTraverser.py
        ├── RefResolver.py
        ├── SpObject.py
        ├── dict_util.py
        ├── io.py
        ├── logger.py
        ├── sp_export.py
        ├── urilib.py
        └── utilities.py

```

## 软件环境

- 硬件环境：X86 兼容架构平台环境
- 语言环境：Python >= 3.8
- 依赖 Python 库（必须）
  - numpy >= 1.18.3
- 依赖 Python 库（可选）：
  - f90nml >=1.2
  - h5py >= 2.10.0
  - pyyaml >= 5.3.1
  - lxml >= 4.5.2
  - scipy >= 1.4.1

<!--  ## 意外事故以及运行的备用状态和方式 -->

<!--  ## 保密性隐私性 -->

<!--# 访问软件-->

<!--  ## 软件的首次用户 -->

<!--  ### 熟悉设备 -->

<!-- ## 访问控制 -->

# 安装和设置

安装方式：

```{Bash}
$git@gitee.com:SimPla/FyTok.git <SPDB_INSTALL_DIR>
$export PYTHON_PATH=<SPDB_INSTALL_DIR>/python:${PYTHON_PATH}

```

<!--## 启动过程 -->

<!-- ## 停止和挂起工作 -->

# 使用软件

|                                 |
| :-----------------------------: |
| ![FyTok2](./figures/FyTok2.svg) |
|       图 1 FyTok 软件架构       |

**FyTok** 中提供的数据类型可划分为四种基本形式。其中两种属于“Tree”节点，可以拥有子节点 ，分别为

- **Sequence 序列** : 一组数据的有序排列，数据类型可以不一致，在不同场合下也被称为 list，tuple，array（不同于前面 的单一类型构成的数组）
- **Mapping 映射** : 一组 key-value 组成的集合，在不同场合下也被称为 dict，object，structure 等，FyTok 中 key 默认为字符串类型。

另两种为“Leaf”节点，不具有“子节点”，分别为

- **Scalar 标量** : 对应于整型（int），浮点（float）和 字符串（string）等元数据类型；
- **nd-array 多维数组**: 由单一元数据类型（scalar）构成的紧密规则排列的阵列，在不同场景下也被称为 tensor、matrix 或者 array，对应为 numpy.ndarray。

这四种数据类型嵌套构成 FyTok 的树状结构。

## 用法：访问原生混和数据结构

```python
>>>from spdm.data import Dict
>>>cache= {
        "a": [
            "hello world {name}!",
            "hello world2 {name}!",
            1, 2, 3, 4
        ],
        "c": "I'm {age}!",
        "d": {
            "e": "Just a test!",
            "f": "address"
        }
    }
    d = Dict(cache)

>>>d["d","e"]
Just a test!
>>>d["d","f"]==cache["d"]["f"]
True
>>>d["a",2]
1
>>>d["a"][2:6]==[1.0, 2, 3, 4]
True
>>>d["d","g"]="Hello world!"
>>>cache["d"]["g"]
Hello world!
```

## 用法：访问特定数据文件格式，如 [GEQDsk](https://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf)

```python
>>>from spdm.data.File import File
   from spdm.data import Dict
>>>eqdsk_file = File("g900003.gfile", format="geqdsk")
>>>d=Dict(equdsk_file.entry)
>>>d["profiles_1d","psi"]
array([-13.464095  , -13.41590565, -13.36771631, -13.31952696,
       -13.27133761, -13.22314827, -13.17495892, -13.12676957,
       -13.07858022, -13.03039088, -12.98220153, -12.93401218,
show more (open the raw output data in a text editor) ...
        -1.70589433,  -1.65770499,  -1.60951564,  -1.56132629,
        -1.51313695,  -1.4649476 ,  -1.41675825,  -1.3685689 ,
        -1.32037956,  -1.27219021,  -1.22400086,  -1.17581152,
        -1.12762217])

```

## 用法：List 追加元素和遍历

```python
>>>from spdm.data.File import File
   from spdm.data import Dict
>>>cache=[]
>>>d=List(cache)
>>>d[_next_]=1
   d[_next_]=2
   d[_next_]=3
   d[_next_]=4
>>>print(cache)
[1, 2, 3, 4]
>>>for i in d:
       print(i)
1
2
3
4

```

<!--## 约定  -->

<!--## 处理过程 -->

<!--## （软件使用的方面） -->

<!--## 相关处理 -->

<!--## 数据备份 -->

<!--## 错误、故障和紧急情况恢复 -->

<!--## 消息 -->

<!--## 快速引用指南 -->

<!--# 注解 -->
