from .config import rpc_server_ip, rpc_server_port

def start_rpc_server():
    import rpyc, pickle
    from rpyc.utils.server import ThreadedServer
    from pu4c.det3d.app import cloud_viewer, voxel_viewer, cloud_viewer_panels, cloud_player, plot_tsne2d, plot_umap
    class RPCService(rpyc.Service):
        def exposed_cloud_viewer(self, serialized_args, serialized_kwargs):
            args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
            return pickle.dumps(cloud_viewer(*args, **kwargs))
        def exposed_voxel_viewer(self, serialized_args, serialized_kwargs):
            args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
            return pickle.dumps(voxel_viewer(*args, **kwargs))
        def exposed_cloud_viewer_panels(self, serialized_args, serialized_kwargs):
            args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
            return pickle.dumps(cloud_viewer_panels(*args, **kwargs))
        def exposed_cloud_player(self, serialized_args, serialized_kwargs):
            args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
            return pickle.dumps(cloud_player(*args, **kwargs))

        def exposed_plot_tsne2d(self, serialized_args, serialized_kwargs):
            args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
            return pickle.dumps(plot_tsne2d(*args, **kwargs))
        def exposed_plot_umap(self, serialized_args, serialized_kwargs):
            args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
            return pickle.dumps(plot_umap(*args, **kwargs))

    server = ThreadedServer(RPCService, port=rpc_server_port, auto_register=True)
    server.start()

def deep_equal(var1, var2, tol=None, ignore_keys=[], verbose=True, has_complex_type=False):
    """
    比较两个复杂变量是否相等，支持 dict, list, ndarray 等类型的嵌套
    Args:
        ignore_keys: list, 忽略字典中的键
        tol: tuple, (atol, rtol) 数值比较的容忍度
        has_complex_type: 是否包含复杂数据类型，即非 Python 内置类型或 numpy 类型
    """
    import numpy as np
    def convert_type(var):
        import torch
        if isinstance(var, dict):
            for k in var.keys():
                var[k] = convert_type(var[k])
        if isinstance(var, (list, tuple)):
            for i in range(len(var)):
                var[i] = convert_type(var[i])
        # 如果是张量，转 ndarray
        if isinstance(var, torch.Tensor):
            var = var.detach().cpu().numpy()
        # 如果是类实例，则转成字典继续转换，常规用法定义的类都有该方法并可以通过 vars 转成字典
        if hasattr(var, '__dict__'):
            var = convert_type(vars(var))
        return var

    def deep_equal_with_reason(var1, var2, reason):
        # 比较数据类型
        if type(var1) != type(var2):
            return False, f"{reason}: 类型不一致"
        
        # 如果是字典，则比较键值对
        if isinstance(var1, dict):
            if len(ignore_keys) > 0:
                var1 = {k: v for k, v in var1.items() if k not in ignore_keys}
                var2 = {k: v for k, v in var2.items() if k not in ignore_keys}
            elif var1.keys() != var2.keys():
                return False, f"{reason}: 键值不匹配"
            
            for k in var1.keys():
                is_equal, reason = deep_equal_with_reason(var1[k], var2[k], reason)
                if not is_equal:
                    return False, f"['{k}']{reason}"
            return True, reason
        
        # 如果是列表或元组，则逐元素比较
        if isinstance(var1, (list, tuple)):
            if len(var1) != len(var2):
                return False, f"{reason}: 列表长度不相等"
            
            for i, (x, y) in enumerate(zip(var1, var2)):
                is_equal, reason = deep_equal_with_reason(x, y, reason)
                if not is_equal:
                    return False, f"[{i}]{reason}"
            return True, reason
        
        # 如果是 NumPy 数组，则根据数组内容选择不同的比较方法
        if isinstance(var1, np.ndarray):
            if var1.shape != var2.shape:
                return False, f"{reason}: 数组形状不相等"
            
            if var1.dtype.kind in ['i', 'f']:
                # 数值数组，使用 np.isclose 进行比较 np.isclose: if abs(var1-var2) <= atol + rtol*abs(b)
                mask = np.isclose(var1, var2, atol=tol[0], rtol=tol[1]) if tol is not None else np.isclose(var1, var2) # rtol 相对误差容忍度
                if not np.all(mask):
                    idx = np.argwhere(~mask)[0] # 所有不相等值的索引，这里只打印第一个不相等的索引
                    return False, f"{reason}: 数组值在索引 {tuple(idx)} 处不相等"
            else:
                # 非数值数组（如字符串数组），使用 np.array_equal 进行比较
                if not np.array_equal(var1, var2):
                    idx = np.argwhere(var1 != var2)[0]
                    return False, f"{reason}: 数组值在索引 {tuple(idx)} 处不相等"
            return True, reason

        # 其他类型，直接比较
        return (True, reason) if var1 == var2 else (False, f"{reason}: 值不相等")

    if has_complex_type:
        var1, var2 = convert_type(var1), convert_type(var2)
    is_equal, reason = deep_equal_with_reason(var1, var2, reason='')
    if not is_equal and verbose:
        print(f"reason: {reason}")
    return is_equal