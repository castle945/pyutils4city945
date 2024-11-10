### city945 的 Python 工具包

- 设计思路
    1. 模块化设计，为每份代码库编写的工具函数汇总作为模块，模块里可以编写特定于该代码库的代码
    2. 模块内文件组织，包含 `utils/common_utils.py` `app.py` `config.py` ，本模块暴露的可被调用的工具函数接口见 `utils/__init__.py`，应用函数接口写到 `app.py` 中，一些参数配置写到 `config.py` 中，不限制模块间的调用
    3. 弹性依赖，`utils/__init__.py` 中暴露的工具函数文件（如 `common_utils.py`） 以及 `app.py` 中禁止直接导入 `open3d` 等大型依赖库

- 注意事项
    - Python 导包有缓存机制，多次导入与单次导入耗时一致
    - 可以通过 `help(pu4c.xxx.xxx)` 或 `pu4c.xxx.xxx.__doc__` 来查看函数注释
    - 代码库缓存目录 `/tmp/pu4c/`
    - 遵循语义化版本控制规则，v.主版本号.次版本号.修订号，主版本号进行不兼容 API 修改，次版本号新增、改进向后兼容的功能，修订号只修复错误，预发布版本加后缀 -alpha/beta

- 快捷命令
    ```bash
    pip install setuptools wheel twine
    python3 setup.py sdist bdist_wheel
    twine upload dist/*
    pip install pu4c -i https://pypi.org/simple
    ```

### 使用说明

- 服务器端数据在本地界面中可视化
    ```bash
    # 本地计算机作为 RPC 服务端
    python -c "import pu4c; pu4c.common.app.start_rpc_server()"
    ssh user@ip -R 30570:localhost:30570 # SSH 转发并在使用过程中保持终端 ssh 连接不断开，端口配置位于 `pu4c/common/config.py` ，参数 -R remote_port:localhost:local_port
    # 服务器作为 RPC 客户端，可在交互式终端（如调试终端）中使用
    import pu4c
    pu4c.det3d.app.cloud_viewer(filepath="/datasets/KITTI/object/training/velodyne/000000.bin", num_features=4, rpc=True) # 置 rpc=True 进行远程函数调用
    ```
