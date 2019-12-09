#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "util.h"
#include "setup.h"

#ifdef STATIC_ALLOCATION
const size_t nelem = 16777216;
static double a[nelem];
static double b[nelem];
#endif

int main(int argc, char* argv[])
{
    if ((argc > 1) && ( 0==strncmp("-h",argv[1],2) ||
                        0==strncmp("--h",argv[1],3) ) ) {
        printf("./driver.x <nelem> [<niter> [<nwarm> [<max stride>]]]\n");
        exit(0);
    }

#ifndef STATIC_ALLOCATION
    size_t nelem = (argc > 1) ? atol(argv[1]) : 1000;
#endif
    size_t niter = (argc > 2) ? atol(argv[2]) : 10;
    size_t nwarm = (argc > 3) ? atol(argv[3]) : niter/5;
    size_t max_s = (argc > 4) ? atol(argv[4]) : nelem;

    printf("SIMD memtest\n");
    printf("elements   = %zu\n", nelem);
    printf("iterations = %zu\n", niter);
    printf("warmups    = %zu\n", nwarm);
    printf("max_stride = %zu\n", max_s);

    size_t bytes = nelem * sizeof(double);
    // the number of bytes actually allocated are padded out by 7 * max_stride
    // to provide a buffer for non-power of true stides which read slightly beyond
    // the end of the array
    size_t alloc_bytes = (nelem + 7 * 64) * sizeof(double);

    if (nelem >= 1024*1024) {
        printf("bytes      = %zu MiB\n", bytes/(1024*1024));
    } else if (nelem >= 1024) {
        printf("bytes      = %zu KiB\n", bytes/1024);
    } else {
        printf("bytes      = %zu B\n", bytes);
    }

    printf("num_threads = %d\n", omp_get_max_threads() );

    int debug = ( getenv("JEFFDEBUG") != NULL );

#ifndef STATIC_ALLOCATION
    double * a = (double*)mymalloc(alloc_bytes);
    double * b = (double*)mymalloc(alloc_bytes);
    if (debug) {
        printf("allocation finished\n");
    }
#endif

    //set_doubles(nelem, 7777.3333, a);
    init_doubles(nelem, a);

    int numtest = setup();

    for (int i=0; i<numtest; i++) {
        if (testfns[i] != NULL) {

            set_doubles(nelem, 1111.9999, b);

            double t0=0., t1=0.;
            for (size_t j = 0; j<niter; j++) {
                if (j == nwarm) t0 = omp_get_wtime();
                testfns[i](nelem, a, b);
            }
            t1 = omp_get_wtime();
            testtime[i] = t1-t0;
            testtime[i] /= (niter-nwarm);

            size_t testerrs = compare_doubles(nelem, a, b);

            if (testerrs != 0 || debug ) {
                printf("====== %s ======\n", testname[i]);
                printf("There were %zu errors!\n", testerrs);
                print_doubles_2(nelem, a, b);
            } else {
                //printf("There were no errors.\n");
                printf("%20s Time = %lf seconds Bandwidth = %lf GB/s\n",
                       testname[i], testtime[i], (2.e-9*bytes)/testtime[i]);
            }
        }
    }

    int numtest2 = setup_stride();

    int strides[13] = {1,2,3,4,5,6,7,8,12,16,24,32,64};
    for (int j=0; j<(int)(sizeof(strides)/sizeof(strides[0])); j++) {
        int s = strides[j];
        if (s > max_s) break;
        for (int i=0; i<numtest2; i++) {
            if (testfns2[i] != NULL) {

                double v = 1111.9999;
                set_doubles(nelem, v, b);

                double t0=0., t1=0.;
                for (size_t j = 0; j<niter; j++) {
                    if (j == nwarm) t0 = omp_get_wtime();
                    testfns2[i](nelem, a, b, s);
                }
                t1 = omp_get_wtime();
                testtime2[i] = t1-t0;
                testtime2[i] /= (niter-nwarm);

                size_t testerrs = compare_doubles_stride_holes(nelem, a, b, s, v);

                if (testerrs != 0 || debug ) {
                    printf("====== %s (stride=%d) ======\n", testname2[i], s);
                    printf("There were %zu errors!\n", testerrs);
                    //print_doubles_2(nelem, a, b);
                    print_compare_doubles_stride_holes(nelem, a, b, s, v);
                    if (testerrs != 0) { exit(1); }
                } else {
                    //printf("There were no errors.\n");
                    printf("%20s (stride=%2d) Time = %lf seconds Bandwidth = %lf GB/s\n",
                           testname2[i], s, testtime2[i], (2.e-9*bytes)/s/testtime2[i]);
                }
            }
        }
    }

#ifndef STATIC_ALLOCATION
    free(a);
    free(b);
#endif

    return 0;
}
