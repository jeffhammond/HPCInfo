#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double getusec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double t = 1.e6 * tv.tv_sec + 1. * tv.tv_usec;
    return t;
}

int main(int argc, char * argv[])
{
    int n = (argc>1) ? atoi(argv[1]) : 100000;

    double t0 = getusec();
    double junk = 0.0;
    for (int i=0; i<n; i++) {
        junk += getusec();
    }
    double t1 = getusec();
    printf("dt = %lf s junk = %4.1e\n", 1.e-6 * (t1-t0), junk);
    return 0;
}
