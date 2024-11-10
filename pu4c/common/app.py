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
