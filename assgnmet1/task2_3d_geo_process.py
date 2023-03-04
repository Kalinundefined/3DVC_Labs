import numpy as np
import trimesh
import trimesh.sample

def uniform_sample(mesh, cnt):
    trimesh.sample.sample_surface(mesh=mesh,count=cnt)

def main():
    saddle = trimesh.load_mesh("/home/karin/assignment/3dvc_1/assgnmet1/saddle.obj")
    pts_100k = uniform_sample(saddle, 100000)

if __name__ == "__main__":
    print(main())