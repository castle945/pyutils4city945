# city945 的 Python 工具包

![PyPI - Version](https://img.shields.io/pypi/v/pu4c)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pu4c)

### 介绍

#### 设计思路

1. 模块化设计，为每份代码库编写的工具函数汇总作为模块，模块里可以编写特定于该代码库的代码
2. 模块内文件组织，包括 `utils/common_utils.py` `app.py` `config.py` ，工具函数写到 `utils` 目录的文件中，应用函数写到 `app.py` 中，参数配置写到 `config.py` 中
3. 弹性依赖和快速导包，保证安装时尽量不依赖其他库，使用时安装所需依赖库即可正常运行，保证 `import pu4c` 快速执行，这要求：
   - `app.py` 中不能导入大型依赖库如 `open3d` ，而是在具体函数中导入
   - `utils/__init__.py` 中可以导入同级文件如 `common_utils.py` 中的工具函数，从而支持链式调用这些工具函数如 `pu4c.common.utils.funcxxx()` 。快速导包要求 `utils/__init__.py` 中导入的同级文件中不能导入大型依赖库

#### 相关说明

- Python 导包有缓存机制，多次导入与单次导入耗时一致，意味着可以把所有的 import 语句写在函数中
- 可以通过 `help(pu4c.xxx.xxx)` 或 `pu4c.xxx.xxx.__doc__` 来查看函数注释
- 注意查看所用模块 `config.py` 中的配置进行个性化修改
- 遵循语义化版本控制规则，v.主版本号.次版本号.修订号，主版本号进行不兼容 API 修改，次版本号新增、改进向后兼容的功能，修订号只修复错误，预发布版本加后缀 -alpha/beta

### 安装

```bash
# pip 安装
pip install pu4c

# 打包上传 pypi
pip install setuptools wheel twine
python3 setup.py sdist bdist_wheel
twine upload dist/*
pip install pu4c -i https://pypi.org/simple
```

### 快速演示

点云目标检测可视化
![demo_cloud_viewer.png](docs/demo_cloud_viewer.png)
三维占据预测可视化
![demo_voxel_viewer.png](docs/demo_voxel_viewer.png)
更多演示参考 [DEMO.md](docs/DEMO.md)

### 快速入门

#### 服务器端数据在本地界面中可视化

```bash
# 本地计算机作为 RPC 服务端
python -c "import pu4c; pu4c.common.app.start_rpc_server()"
ssh user@ip -R 30570:localhost:30570 # SSH 转发并在使用过程中保持终端 ssh 连接不断开，端口配置位于 `pu4c/common/config.py` ，参数 -R remote_port:localhost:local_port

# 服务器作为 RPC 客户端，可在交互式终端（如调试终端）中使用
import pu4c
pu4c.det3d.app.cloud_viewer(filepath="/datasets/KITTI/object/training/velodyne/000000.bin", num_features=4, rpc=True) # 置 rpc=True 进行远程函数调用

# 注意
# 如果远程函数调用时使用了文件路径作为参数需要确保 RPC 服务器上存在该文件，对于数据集等可以通过 nfs 挂载到相同路径，或者修改 rpc 装饰器修改路径前缀
```

更多 API 使用参考 `test/*.ipynb` 如 [test_det3d.ipynb](test/test_det3d.ipynb)

### 许可证

本代码采用 [GPL-3.0](LICENSE) 许可发布，这意味着你可以自由复制和分发软件，无论个人使用还是商业用途，但修改后的源码不可闭源且必须以相同的许可证发布

<!-- Changelog, Introduction, Model Zoo, Installation, Quick Demo, Getting Started, API Reference, FAQ, License, Acknowledgement, Citation, Contribution -->

<!-- 更新日志、简介、模型库、安装、快速演示、快速入门、API 参考、常见问题、许可证、致谢、引用、贡献 -->
