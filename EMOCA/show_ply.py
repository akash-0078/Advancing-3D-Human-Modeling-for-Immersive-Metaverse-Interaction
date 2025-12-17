import subprocess
import os
import glob
import open3d as o3d

# Step 1: Run your script to generate the PLY
cmd = [
    "python", "run.py",
    "--input", "sample.png",
    "--output", "output",
    "--save_results", "false",
    "--smooth", "True",
    "--save_ply", "true"
]

print("🧠 Running model generation...")
subprocess.run(cmd, check=True)

# Step 2: Find the generated .ply file
ply_files = glob.glob("output/*.ply")

if not ply_files:
    print(" No .ply file found in output directory.")
    exit(1)

ply_path = sorted(ply_files)[-1]  # pick the latest one
print(f"Found PLY file: {ply_path}")

# Step 3: Load and show in 3D viewer
mesh = o3d.io.read_triangle_mesh(ply_path)

if not mesh.has_triangles():
    print("Warning: PLY file has no triangle mesh. It might be a point cloud.")
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])
else:
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

print("👀 Viewer closed. Done!")
