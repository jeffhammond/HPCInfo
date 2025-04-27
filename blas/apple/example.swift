// Example usage of GPU Matrix Multiplication

@main
struct MatrixMultiplicationExample {
    static func main() {
        // Create test matrices
        let M = 1024  // Number of rows in A
        let N = 1024  // Number of columns in B
        let K = 1024  // Number of columns in A / rows in B

        // Initialize test matrices with random values
        let matrixA = (0..<M*K).map { _ in Float.random(in: 0...1) }
        let matrixB = (0..<K*N).map { _ in Float.random(in: 0...1) }

        // Create matrix multiplication instance
        if let matMul = MatrixMultiplication() {
            // Perform multiplication
            if let result = matMul.multiply(matrixA: matrixA, matrixB: matrixB, M: M, N: N, K: K) {
                print("Matrix multiplication completed successfully!")
                print("Result matrix dimensions: \(M)x\(N)")
                
                // Print a small subset of the result for verification
                print("\nFirst few elements of the result matrix:")
                for i in 0..<min(3, M) {
                    for j in 0..<min(3, N) {
                        print(String(format: "%.4f ", result[i * N + j]), terminator: "")
                    }
                    print()
                }
            } else {
                print("Failed to perform matrix multiplication")
            }
        } else {
            print("Failed to initialize MatrixMultiplication")
        }
    }
} 