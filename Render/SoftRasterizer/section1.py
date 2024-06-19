import pycuda.autoinit
from pycuda import driver
from pycuda import gpuarray
from pycuda import compiler
device = driver.Device(0)

import numpy as np
import cv2
height = 600
width = 800
color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
color_buffer[:,:,2] = 254 # R
color_buffer[:,:,1] = 67  # G
color_buffer[:,:,0] = 101 # B
while True:
    cv2.imshow('pycuda', color_buffer)
    if (cv2.waitKey(1) & 0xFF) == 27: # esc退出程序
        break
exit()


gpu_color_buffer = gpuarray.zeros((height, width, 3), dtype=np.uint8)
    # color_buffer = gpu_color_buffer.get()




# 绘制渲染结果
# 读取obj文件格式  
import open3d as o3d 
mesh = o3d.io.read_triangle_mesh('bunny.obj')
# cpu data
# o3d.visualization.draw(mesh)
vertices = np.asarray(mesh.vertices)
# mesh.compute_vertex_normals() # 如果没有，则计算顶点法向量
normals = np.asarray(mesh.vertex_normals)
triangles = np.asarray(mesh.triangles)

height = 600
width = 800
triangle_number = triangles.shape[0]

# gpu data
vertices_gpu = gpuarray.to_gpu(vertices.astype(np.float32))
normals_gpu = gpuarray.to_gpu(normals.astype(np.float32))
triangles_gpu = gpuarray.to_gpu(triangles.astype(np.int32))
render_target_gpu = gpuarray.to_gpu(np.zeros((height, width, 3)).astype(np.uint8))

# rander
with open('vertex_shader.cu', 'r') as f:
    mod = compiler.SourceModule(f.read())
vertex_shader = mod.get_function("vertex_shader")
with open('fragment_shader.cu', 'r') as f:
    mod = compiler.SourceModule(f.read())
fragment_shader = mod.get_function("fragment_shader")

# loop
import cv2, time
start_time = time.time()  # 记录开始时间
frame_count = 0
while True:
    # vs
    vertex_shader(
        np.int32(triangle_number),
        vertices_gpu, 
        triangles_gpu, 
        grid=(int(np.ceil(triangles_gpu/device.max_threads_per_block)),), 
        block=(device.max_threads_per_block, 1, 1)
        )
    # fs
    fragment_shader(
        np.int32(height),
        np.int32(width),
        render_target_gpu, 
        vertices_gpu, 
        normals_gpu, 
        triangles_gpu, 
        grid=(int(np.ceil(height*width/device.max_threads_per_block)),), 
        block=(device.max_threads_per_block, 1, 1)
        )
    # show
    frame_count += 1
    current_time = time.time()
    time_diff = current_time - start_time
    if time_diff >= 0.1:
        fps = frame_count / time_diff
        frame_count = 0
        start_time = time.time()
    render_target = render_target_gpu.get()
    cv2.putText(render_target, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Image', render_target)
    if cv2.waitKey(1) == ord('q'):
        break