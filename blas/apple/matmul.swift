import Metal
import MetalKit
import Foundation

class MatrixMultiplication {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private let matrixMultiplyPipelineState: MTLComputePipelineState
    
    init?() {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return nil
        }
        self.device = device
        
        // Create command queue
        guard let commandQueue = device.makeCommandQueue() else {
            print("Could not create command queue")
            return nil
        }
        self.commandQueue = commandQueue
        
        // Get the path to our compiled metal library
        let libraryPath = "build/metal/default.metallib"
        let libraryURL = URL(fileURLWithPath: libraryPath)
        
        let tempLibrary: MTLLibrary
        do {
            // Load the library from our compiled metallib file
            tempLibrary = try device.makeLibrary(URL: libraryURL)
        } catch {
            print("Could not load Metal library: \(error)")
            return nil
        }
        self.library = tempLibrary
        
        // Create compute pipeline state
        guard let matrixMultiplyFunction = library.makeFunction(name: "matrixMultiply"),
              let pipelineState = try? device.makeComputePipelineState(function: matrixMultiplyFunction) else {
            print("Could not create pipeline state")
            return nil
        }
        self.matrixMultiplyPipelineState = pipelineState
    }
    
    func multiply(matrixA: [Float], matrixB: [Float], M: Int, N: Int, K: Int) -> [Float]? {
        let resultSize = M * N
        var result = [Float](repeating: 0, count: resultSize)
        
        // Create buffers
        guard let bufferA = device.makeBuffer(bytes: matrixA, length: matrixA.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferB = device.makeBuffer(bytes: matrixB, length: matrixB.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let bufferResult = device.makeBuffer(length: result.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            print("Could not create buffers")
            return nil
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Could not create command buffer or encoder")
            return nil
        }
        
        computeEncoder.setComputePipelineState(matrixMultiplyPipelineState)
        computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferResult, offset: 0, index: 2)
        
        // Set dimensions as constants
        var dimensions = [UInt32(M), UInt32(N), UInt32(K)]
        computeEncoder.setBytes(&dimensions, length: dimensions.count * MemoryLayout<UInt32>.size, index: 3)
        
        // Calculate grid and threadgroup sizes
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let gridSize = MTLSize(
            width: (M + threadgroupSize.width - 1) / threadgroupSize.width * threadgroupSize.width,
            height: (N + threadgroupSize.height - 1) / threadgroupSize.height * threadgroupSize.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        // Execute command buffer
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Copy result back to CPU
        let resultPtr = bufferResult.contents().bindMemory(to: Float.self, capacity: resultSize)
        result = Array(UnsafeBufferPointer(start: resultPtr, count: resultSize))
        
        return result
    }
}
