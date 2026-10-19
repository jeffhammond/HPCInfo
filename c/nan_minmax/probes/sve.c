#include <arm_sve.h>

svfloat64_t f(svbool_t p, svfloat64_t a, svfloat64_t b)
{
    return svmaxnm_f64_x(p, svadd_f64_x(p, a, b), b);
}
