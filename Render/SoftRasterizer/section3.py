import pycuda.autoinit
from pycuda import driver
from pycuda import gpuarray
from pycuda import compiler
device = driver.Device(0)
mod = compiler.SourceModule("""
void __global__ ClearColor(
    uchar4* color_buffer, 
    const uint2 resolution, 
    const uchar3 color)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= resolution.x * resolution.y)
        return;
    color_buffer[tid].z = color.x; // r
    color_buffer[tid].y = color.y; // g
    color_buffer[tid].x = color.z; // b
    color_buffer[tid].w = 255; // b
}
""")
clear_color = mod.get_function("ClearColor")
import numpy as np
import cv2
height = 600
width = 800
background_color = np.void((254, 67, 101),  dtype='u1, u1, u1') # r,g,b
resolution = np.void((width, height),  dtype='u4, u4') # w,h
gpu_color_buffer = gpuarray.zeros((height, width, 4), dtype=np.uint8)
gpu_background_color = gpuarray.empty((3,), dtype=np.uint8)
while True:
    # clear color
    clear_color(
        gpu_color_buffer, 
        resolution,
        background_color,
        grid=(int((height*width + device.max_threads_per_block - 1) / device.max_threads_per_block), 1), 
        block=(device.max_threads_per_block, 1, 1))
    # show
    color_buffer = gpu_color_buffer.get()
    cv2.imshow('pycuda', color_buffer)
    if (cv2.waitKey(1) & 0xFF) == 27: # esc退出程序
        break

