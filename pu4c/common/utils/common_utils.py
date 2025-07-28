import rpyc
import pickle, json, os
import pu4c.config as cfg

def rpc_func(func):
    def wrapper(*args, **kwargs):
        if ('rpc' in kwargs) and kwargs['rpc']:
            kwargs['rpc'] = False
            conn = rpyc.connect(cfg.rpc_server_ip, cfg.rpc_server_port)
            remote_method = getattr(conn.root, func.__name__, None)
            if remote_method:
                serialized_rets = remote_method(pickle.dumps(args), pickle.dumps(kwargs))
                conn.close()
                return pickle.loads(serialized_rets)
            else:
                raise AttributeError(f"Remote object has no attribute '{func.__name__}'")
        else:
            return func(*args, **kwargs) # python 中会为无返回值的函数返回 None
    return wrapper

def read_pickle(filepath=cfg.cache_pkl):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
def write_pickle(data, filepath=cfg.cache_pkl):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
def write_json(filepath, data, indent=None):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

class TestDataDB:
    """
    1. 简单的键值数据库，可用于存储测试数据，其中使用 json 文件存储数据的键值-路径对，使用 pickle 文件存储键值-数据对
    2. json 文件可编辑，删除条目并运行 gc 方法进行垃圾回收
    3. 建议压缩保存为 .pkl.tgz 文件以进行文件传输
    """
    def __init__(self, dbname="pu4c_test_data", root=cfg.cache_dir):
        self.dbname = dbname
        self.root = root

        keys_file = dbname + '.keys.json'   # 键值对文件
        main_file = dbname + '.pkl'         # 主数据文件
        keys_path = os.path.join(root, keys_file)
        main_path = os.path.join(root, main_file)
        if os.path.exists(os.path.join(root, dbname + '.pkl.tgz')) and not os.path.exists(keys_path):
            self.restore()
        if not os.path.exists(keys_path):
            write_json(keys_path, {}, indent=4)
        if not os.path.exists(main_path):
            write_pickle({}, main_path)
        self.main_file = main_file
        self.main_path = main_path
        self.keys_path = keys_path
        self.keys_dict = read_json(keys_path)
        self.filesize = 1 * 1024**3
    def get(self, key, default=None):
        return read_pickle(os.path.join(self.root, self.keys_dict[key]))[key] if key in self.keys_dict else default # 如果某个测试需要一批多个数据，则将其打包作为数据库的一项
    def set(self, key, data):
        # 键值存在则更新数据
        if key in self.keys_dict:
            filepath = os.path.join(self.root, self.keys_dict[key])
            filedata = read_pickle(filepath)
            filedata[key] = data
            write_pickle(filedata, filepath)
            print(f"update {key}, data at {filepath}")
            return

        # 键值不存在则添加数据
        self.keys_dict[key] = self.main_file
        write_json(self.keys_path, self.keys_dict, indent=4)
        maindata = read_pickle(self.main_path)
        maindata[key] = data
        write_pickle(maindata, self.main_path)

        if os.path.getsize(self.main_path) > self.filesize:
            # 主文件过大时则将主文件重命名为新文件，并更新 keys_dict
            from datetime import datetime
            now = datetime.now()
            timestamp = f"-{now.year % 100}{now.month:02d}{now.day:02d}"
            newfile = self.main_file[:-len('.pkl')] + timestamp + '.pkl'
            
            if os.system(f"cp {os.path.join(self.root, self.main_file)} {os.path.join(self.root, newfile)}") != 0:
                raise Exception(f"create new file {newfile} failed")
            print(f"create new file {newfile}")
            # udpate key
            self.keys_dict.update({key:newfile for key, val in self.keys_dict.items() if val == self.main_file})
            write_json(self.keys_path, self.keys_dict, indent=4)
            write_pickle({}, self.main_path)
            

    def remove(self, key):
        assert key in self.keys_dict
        filepath = os.path.join(self.root, self.keys_dict[key])
        # remove key
        self.keys_dict.pop(key)
        write_json(self.keys_path, self.keys_dict, indent=4)
        # remove data
        filedata = read_pickle(filepath)
        filedata.pop(key)
        write_pickle(filedata, filepath)
        print(f"remove {key}, data at {filepath}")
    def rename(self, key, new_key):
        assert key in self.keys_dict
        filepath = os.path.join(self.root, self.keys_dict[key])
        # rename key
        self.keys_dict[new_key] = self.keys_dict[key]
        self.keys_dict.pop(key)
        write_json(self.keys_path, self.keys_dict, indent=4)
        # rename data
        filedata = read_pickle(filepath)
        filedata[new_key] = filedata[key]
        filedata.pop(key)
        write_pickle(filedata, filepath)
        print(f"rename {key} to {new_key}, data at {filepath}")
    def gc(self):
        import glob
        files = glob.glob(f'{self.root}/{self.dbname}*.pkl')
        deleted_files = []
        for filepath in files:
            filedata = read_pickle(filepath)
            keys = list(filedata.keys())
            [filedata.pop(key) for key in keys if key not in self.keys_dict]
            if filedata:
                write_pickle(filedata, filepath)
            else:
                if os.system(f"rm -f {filepath}") != 0:
                    raise Exception(f"remove file {filepath} failed")
                deleted_files.append(filepath)
                print(f"remove file {filepath} successed")
        write_json(self.keys_path, self.keys_dict, indent=4) # 写键，键和数据必须同时操作
        return deleted_files
    def archive(self):
        if os.name == 'posix':
            cmd = f"cd {self.root} && tar -zcf {self.dbname}.pkl.tgz {self.dbname}.keys.json {self.dbname}*.pkl"
            if os.system(cmd) != 0:
                raise Exception(f"{cmd} failed")
        elif os.name == 'nt':
            raise Exception("Windows is not supported")
    def restore(self):
        if os.name == 'posix':
            cmd = f"cd {self.root} && tar -zxf {self.dbname}.pkl.tgz"
            if os.system(cmd) != 0:
                raise Exception(f"{cmd} failed")
        elif os.name == 'nt':
            raise Exception("Windows is not supported")
    def cat(self, dbname, root, keys: list = None):
        """拼接另外的数据库到当前数据库"""
        datadb = TestDataDB(dbname=dbname, root=root)
        if keys is None:
            cat_keys_dict = datadb.keys_dict
        else:
            cat_keys_dict = {k:v for k,v in datadb.keys_dict.items() if k in keys}
        # 检查键是否冲突
        conflict_keys = set(self.keys_dict.keys()) & set(cat_keys_dict.keys())
        if conflict_keys:
            raise Exception(f"Conflict keys: {conflict_keys}")
        for key, filepath in cat_keys_dict.items():
            data = read_pickle(os.path.join(root, filepath))[key]
            self.set(key, data)

def convert_type(data, typeinfo=False):
    """
    将复杂数据类型转至多包含 ndarray 的简单数据类型
    Args:
        typeinfo: 是否保留原始类型信息
    """
    if isinstance(data, dict):
        return {k:convert_type(v, typeinfo) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_type(i, typeinfo) for i in data]
    elif isinstance(data, tuple):
        return tuple([convert_type(i, typeinfo) for i in list(data)]) # 转成 list 才支持遍历
    elif 'torch.Tensor' in str(type(data)):
        newdata = data.detach().cpu().numpy()
        return ({'typeinfo': str(type(data))}, newdata) if typeinfo else newdata
    elif 'DataContainer' in str(type(data)):
        # 见于 mmlab1.0 的 mmcv 中的数据结构
        newdata = ({'typeinfo': str(type(data))}, data.data) if typeinfo else data.data
        return convert_type(newdata, typeinfo) # 还没结束，其数据可能是 dict 等，故仍需递归
    elif 'BaseDataElement' in str(data.__class__.__mro__): # 该类的继承关系，即数据是 BaseDataElement 及其子类的实例
        # 见于 mmlab2.0 的 mmengine 中的数据结构
        newdata = ({'typeinfo': str(type(data))}, data.to_dict()) if typeinfo else data.to_dict()
        return convert_type(newdata, typeinfo)
    elif hasattr(data, '__dict__'):
        # 未知的类实例，则转成字典继续转换，常规用法定义的类都有该方法并可以通过 vars 转成字典
        newdata = ({'typeinfo': str(type(data))}, vars(data)) if typeinfo else vars(data)
        return convert_type(newdata, typeinfo)
    return data

def remove_typeinfo(data):
    if isinstance(data, dict):
        return {k:remove_typeinfo(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [remove_typeinfo(i) for i in data]
    elif isinstance(data, tuple):
        if len(data) == 2 and isinstance(data[0], dict) and 'typeinfo' in data[0]:
            return data[1]
    return data
