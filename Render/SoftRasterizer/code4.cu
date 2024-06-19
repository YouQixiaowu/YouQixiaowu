void __global__ ClearColor(
    uchar4* color_buffer, 
    const uint2 resolution, 
    const uchar3 color)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= resolution.x * resolution.y)
        return;
    color_buffer[tid].z = color.x;  // r
    color_buffer[tid].y = color.y;  // g
    color_buffer[tid].x = color.z;  // b
    color_buffer[tid].w = 255;      // a
}

void __device__ VertexShader(float4& position, const float3& vertice)
{
    position = float4{vertice.x, vertice.y, vertice.z, 1.0f};
}

uchar4 __device__ FragmentShader(float4 frag_coord)
{
    return uchar4{255,255,255,255};
}

void __global__ VertexProcess(
    float4 *vertex_shader_out, 
    const float3 *vertices, 
    unsigned int triangle_number)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= triangle_number)
        return;
    VertexShader(vertex_shader_out[tid], vertices[tid]);
}

void __global__ RasterizePoint(
    uchar4* color_buffer, 
    const uint2 resolution, 
    const uint4 viewport, 
    float4 p,
    float r)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= viewport.z * viewport.w) // w * h
        return;
    uint2 vo = {tid % viewport.z + viewport.x, viewport.w - tid / viewport.z + viewport.y};
    float2 fp{p.x, p.y};
    float2 fo{float(vo.x)+0.5f, float(vo.y)+0.5f};
    if((fp.x-fo.x)*(fp.x-fo.x) + (fp.y-fo.y)*(fp.y-fo.y) < r*r)
    {
        uchar4 color = FragmentShader(float4{fo.x, fo.y, 0, 0});
        int index = (resolution.y-vo.y)*resolution.x + vo.x;
        color_buffer[index].z = color.x;  // r
        color_buffer[index].y = color.y;  // g
        color_buffer[index].x = color.z;  // b
        color_buffer[index].w = color.w;  // a
    }
}