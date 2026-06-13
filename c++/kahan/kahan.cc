#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>

// Structure to hold both simple and Kahan sums for comparison
struct SumResults {
    float simple_sum;
    float kahan_sum;
    float pairwise_sum;
    double reference_sum;  // Using double as reference
};

// Kahan summation
float kahan_sum(const float* data, size_t n)
{
    float sum = 0.0f;
    float compensation = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float y = data[i] - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}

// Simple summation
float simple_sum(const float* data, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// Pairwise summation for comparison
float pairwise_sum(const float* data, size_t n)
{
    if (n == 0) return 0.0f;
    if (n == 1) return data[0];
    
    size_t half = n / 2;
    return pairwise_sum(data, half) + pairwise_sum(data + half, n - half);
}

// Reference sum using double precision
double reference_sum(const float* data, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += (double)data[i];
    }
    return sum;
}

// Generate test data with wide range of exponents
void generate_test_data(float* data, size_t n, unsigned int seed)
{
    std::mt19937 gen(seed);
    
    // Generate random exponents in range [-126, 127] (float range)
    std::uniform_int_distribution<int> exp_dist(-50, 50);
    
    // Generate random mantissas
    std::uniform_real_distribution<float> mantissa_dist(1.0f, 2.0f);
    
    for (size_t i = 0; i < n; i++) {
        float mantissa = mantissa_dist(gen);
        int exponent   = exp_dist(gen);
        
        // Construct number: mantissa * 2^exponent
        data[i] = mantissa * powf(2.0f, (float)exponent);
        
        // Randomly make some numbers negative
        if (gen() & 1) {
            data[i] = -data[i];
        }
    }
}

// Print detailed analysis of a number
void analyze_float(float x)
{
    int exp;
    float mantissa = frexpf(fabsf(x), &exp);
    printf("Value: %g\n", x);
    printf("  Sign: %s\n", x < 0 ? "negative" : "positive");
    printf("  Mantissa: %g\n", mantissa);
    printf("  Exponent: %d\n", exp);
}

// Run summation test
SumResults run_test(size_t n, unsigned int seed)
{
    float* data = new float[n];
    generate_test_data(data, n, seed);
    
    SumResults results;
    
    // Perform all summations
    results.simple_sum    = simple_sum(data, n);
    results.kahan_sum     = kahan_sum(data, n);
    results.pairwise_sum  = pairwise_sum(data, n);
    results.reference_sum = reference_sum(data, n);
    
    // Print some sample values from the dataset
    printf("\nSample values from dataset:\n");
    for (size_t i = 0; i < 5 && i < n; i++) {
        printf("\nValue %zu:\n", i);
        analyze_float(data[i]);
    }
    
    delete[] data;
    return results;
}

// Analyze and print results
void analyze_results(const SumResults& results)
{
    printf("\nSummation Results:\n");
    printf("Simple sum:     %g\n", results.simple_sum);
    printf("Kahan sum:      %g\n", results.kahan_sum);
    printf("Pairwise sum:   %g\n", results.pairwise_sum);
    printf("Reference sum:  %.17g\n", results.reference_sum);
    
    printf("\nRelative Errors (vs double precision):\n");
    double ref = results.reference_sum;
    printf("Simple sum:   %.3e\n", fabs((results.simple_sum - ref) / ref));
    printf("Kahan sum:    %.3e\n", fabs((results.kahan_sum - ref) / ref));
    printf("Pairwise sum: %.3e\n", fabs((results.pairwise_sum - ref) / ref));
    
    // Analyze the sums
    printf("\nDetailed analysis of sums:\n");
    printf("\nSimple sum:\n");
    analyze_float(results.simple_sum);
    printf("\nKahan sum:\n");
    analyze_float(results.kahan_sum);
    printf("\nPairwise sum:\n");
    analyze_float(results.pairwise_sum);
}

int main(void)
{
    const size_t test_sizes[] = {1000, 10000, 100000, 1000000};
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    for (size_t n : test_sizes) {
        printf("\n=== Test with %zu elements ===\n", n);
        SumResults results = run_test(n, seed);
        analyze_results(results);
        
        // Add some spacing between tests
        printf("\n----------------------------------------\n");
    }
    
    return 0;
}
