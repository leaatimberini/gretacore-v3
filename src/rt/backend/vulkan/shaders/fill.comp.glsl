#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer OutBuf {
    uint data[];
} outb;

layout(push_constant) uniform Push {
    uint value;
    uint n;     // number of uint elements
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.n) {
        outb.data[idx] = pc.value;
    }
}
