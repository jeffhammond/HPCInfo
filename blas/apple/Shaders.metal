#include <metal_stdlib>
using namespace metal;

kernel void matrixMultiply(
    const device float* matrixA [[buffer(0)]],
    const device float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint* dimensions [[buffer(3)]],
    uint2 position [[thread_position_in_grid]])
{
    const uint M = dimensions[0];
    const uint N = dimensions[1];
    const uint K = dimensions[2];
    
    // Check if the thread is within bounds
    if (position.x >= M || position.y >= N) {
        return;
    }
    
    // Compute the dot product for this position
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += matrixA[position.x * K + k] * matrixB[k * N + position.y];
    }
    
    // Store the result
    result[position.x * N + position.y] = sum;
} 