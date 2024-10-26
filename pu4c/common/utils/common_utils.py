import rpyc, pickle
from pu4c.common.config import rpc_server_ip, rpc_server_port

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
