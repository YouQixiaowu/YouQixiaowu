import pycuda.autoinit
from pycuda import driver
from pycuda import gpuarray
from pycuda import compiler
device = driver.Device(0)

with open('code5.cu', 'r') as f:
    mod = compiler.SourceModule(f.read())
ClearColor = mod.get_function("ClearColor")
VertexProcess = mod.get_function("VertexProcess")
RasterizeTriangle = mod.get_function("RasterizeTriangle")

import numpy as np
import cv2
height = 600
width = 800
vertices = np.array([
    0.5, 0.5, 0.0,
    0.5, -0.5, 0.0,
    -0.5, 0.5, 0.0,
], dtype=np.float32)
vertice_number = np.int32(vertices.shape[0]/3)
triangle_number = np.int32(vertice_number/3)
background_color = np.void((254, 67, 101),  dtype='u1,u1,u1') # r,g,b
resolution = np.void((width, height),  dtype='u4,u4') # w,h
viewport = np.void((300, 200, int(width/2), int(height/2)),  dtype='u4,u4,u4,u4') # x,y,w,h
# gpu buffer
gpu_color_buffer = gpuarray.zeros((height, width, 4), dtype=np.uint8)
gpu_vertices_buffer = gpuarray.to_gpu(vertices)
viewport_transform = np.array([
    [0.5*viewport[2], 0.0, 0.0, viewport[0] + 0.5*viewport[2]],
    [0.0, 0.5*viewport[3], 0.0, viewport[1] + 0.5*viewport[3]],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

# loop
while True:
    # clear color
    ClearColor(
        gpu_color_buffer, 
        resolution,
        background_color,
        grid=(int((height*width + device.max_threads_per_block - 1) / device.max_threads_per_block), 1), 
        block=(device.max_threads_per_block, 1, 1))
    # vertex shader
    gpu_vertex_shader_out = gpuarray.zeros((vertice_number, 4), dtype=np.float32)
    VertexProcess(
        gpu_vertex_shader_out,
        gpu_vertices_buffer,
        np.int32(vertice_number),
        grid=(int((vertice_number + device.max_threads_per_block - 1) / device.max_threads_per_block), 1), 
        block=(device.max_threads_per_block, 1, 1))
    # primitive assembly
    vertex_shader_out = gpu_vertex_shader_out.get()
    for i in range(triangle_number):
        a = np.dot(viewport_transform , vertex_shader_out[i*3+0])
        b = np.dot(viewport_transform , vertex_shader_out[i*3+1])
        c = np.dot(viewport_transform , vertex_shader_out[i*3+2])
        a = np.void((a[0], a[1], a[2], 0.0),  dtype='f4,f4,f4,f4')
        b = np.void((b[0], b[1], b[2], 0.0),  dtype='f4,f4,f4,f4')
        c = np.void((c[0], c[1], c[2], 0.0),  dtype='f4,f4,f4,f4')
        # rasterize
        RasterizeTriangle(
            gpu_color_buffer, 
            resolution,
            viewport,
            a, b, c,
            grid=(int((viewport[2]*viewport[3] + device.max_threads_per_block - 1) / device.max_threads_per_block), 1), 
            block=(device.max_threads_per_block, 1, 1))
    # show
    color_buffer = gpu_color_buffer.get()
    cv2.imshow('pycuda', color_buffer)
    if (cv2.waitKey(1) & 0xFF) == 27: # esc退出程序
        break

