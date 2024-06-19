__device__ __host__ bool IsInTriangle(const float2& o, const float2& a, const float2& b, const float2& c) {
    float o_ab = (o.x - a.x) * (b.y - a.y) - (o.y - a.y) * (b.x - a.x);
    float o_bc = (o.x - b.x) * (c.y - b.y) - (o.y - b.y) * (c.x - b.x);
    float o_ca = (o.x - c.x) * (a.y - c.y) - (o.y - c.y) * (a.x - c.x);
    return ((o_ab >= 0 && o_bc >= 0 && o_ca >= 0) || (o_ab <= 0 && o_bc <= 0 && o_ca <= 0));
}

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

void __global__ RasterizeTriangle(
    uchar4* color_buffer, 
    const uint2 resolution, 
    const uint4 viewport, 
    float4 a,
    float4 b,
    float4 c)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= viewport.z * viewport.w) // w * h
        return;
    uint2 vo = {tid % viewport.z + viewport.x, viewport.w - tid / viewport.z + viewport.y};
    float2 fa{a.x, a.y};
    float2 fb{b.x, b.y};
    float2 fc{c.x, c.y};
    float2 fo{float(vo.x)+0.5f, float(vo.y)+0.5f};
    if (IsInTriangle(fo, fa, fb, fc))
    {
        uchar4 color = FragmentShader(float4{fo.x, fo.y, 0, 0});
        int index = (resolution.y-vo.y)*resolution.x + vo.x;
        color_buffer[index].z = color.x;  // r
        color_buffer[index].y = color.y;  // g
        color_buffer[index].x = color.z;  // b
        color_buffer[index].w = color.w;  // a
    }
}