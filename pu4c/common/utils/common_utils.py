import rpyc, pickle
import os
from pu4c.common.config import rpc_server_ip, rpc_server_port, cache_dir

def rpc_func(func):
    def wrapper(*args, **kwargs):
        if ('rpc' in kwargs) and kwargs['rpc']:
            kwargs['rpc'] = False
            conn = rpyc.connect(rpc_server_ip, rpc_server_port)
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


def read_pickle(filepath):
    with open(filepath, 'rb') as f:
       data = pickle.load(f)
    return data
def write_pickle(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
class TestDataDB:
    def __init__(self, dbname="pu4c_test_data", root=cache_dir):
        mainfile = dbname + '.pkl'
        mainpath = os.path.join(root, mainfile)
        if not os.path.exists(mainpath):
            write_pickle(mainpath, data={"keys_dict": {}})
        
        self.root = root
        self.mainfile = mainfile
        self.mainpath = mainpath
        self.filesize = 1 * 1024**3
        self.keys_dict = read_pickle(mainpath)["keys_dict"]
    def get(self, key, default=None):
        return read_pickle(os.path.join(self.root, self.keys_dict[key]))[key] if key in self.keys_dict else default # 如果某个测试需要一批多个数据，则将其打包作为数据库的一项
    def add(self, key, data):
        if key in self.keys_dict: return
        maindata = read_pickle(self.mainpath)
        maindata["keys_dict"][key] = self.keys_dict[key] = self.mainfile
        maindata[key] = data
        write_pickle(self.mainpath, data=maindata)

        if os.path.getsize(self.mainpath) > self.filesize:
            # 主文件过大时则将主文件重命名为新文件，并更新 keys_dict
            # 新文件中不删除 keys_dict 则在不进行删改数据库的情况下，也可以视作当前时间的备份
            from datetime import datetime
            now = datetime.now()
            timestamp = f"-{now.year % 100}{now.month:02d}{now.day:02d}"
            newfile = self.mainfile[:-len('.pkl')] + timestamp + '.pkl'
            
            if os.system(f"cp {os.path.join(self.root, self.mainfile)} {os.path.join(self.root, newfile)}") != 0:
                raise Exception(f"create new file {newfile} failed")
            print(f"create new file {newfile}")
            # udpate key
            maindata["keys_dict"].update({key:newfile for key, val in maindata["keys_dict"].items() if val == self.mainfile})
            self.keys_dict = new_keys_dict = maindata["keys_dict"]
            write_pickle(self.mainpath, data={"keys_dict": new_keys_dict})
    def update(self, key, data):
        assert key in self.keys_dict
        filepath = os.path.join(self.root, self.keys_dict[key])
        filedata = read_pickle(filepath)
        filedata[key] = data
        write_pickle(filepath, filedata)
        print(f"update {key}, data at {filepath}")
    def remove(self, key):
        assert key in self.keys_dict
        filepath = os.path.join(self.root, self.keys_dict[key])
        # remove key
        maindata = read_pickle(self.mainpath)
        maindata["keys_dict"].pop(key)
        self.keys_dict.pop(key)
        # remove data
        if filepath == self.mainpath:
            maindata.pop(key)
            write_pickle(self.mainpath, maindata)
        else:
            write_pickle(self.mainpath, maindata)
            filedata = read_pickle(filepath)
            filedata.pop(key)
            write_pickle(filepath, filedata)
        print(f"remove {key}, data at {filepath}")
    def rename(self, key, new_key):
        assert key in self.keys_dict
        filepath = os.path.join(self.root, self.keys_dict[key])
        # rename key
        maindata = read_pickle(self.mainpath)
        self.keys_dict[new_key] = maindata["keys_dict"][new_key] = maindata["keys_dict"][key]
        maindata["keys_dict"].pop(key)
        self.keys_dict.pop(key)
        # rename data
        if filepath == self.mainpath:
            maindata[new_key] = maindata[key]
            maindata.pop(key)
            write_pickle(self.mainpath, maindata)
        else:
            write_pickle(self.mainpath, maindata)
            filedata = read_pickle(filepath)
            filedata[new_key] = filedata[key]
            filedata.pop(key)
            write_pickle(filepath, filedata)
        print(f"rename {key} to {new_key}, data at {filepath}")
