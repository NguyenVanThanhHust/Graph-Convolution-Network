import open3d as o3d
import numpy as np
import os
if __name__ == "__main__":
    print("Testing IO for meshes ...")
    if os.path.isfile("../data/ModelNet10/raw/monitor/train/monitor_0002.off"):
        mesh = o3d.io.read_triangle_mesh("../data/ModelNet10/raw/chair/train/chair_0001.off")
        print(mesh)
        o3d.io.write_triangle_mesh("monitor_0002.ply", mesh)
        pcd = o3d.io.read_point_cloud("./monitor_0002.ply")
        print(pcd)
        print(np.asarray(pcd.points))
        o3d.visualization.draw_geometries([pcd])
    else:
        print("Check")
