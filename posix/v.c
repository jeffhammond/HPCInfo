/* this has to be set before headers are included */
#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <assert.h>

int get_num_cpus_from_mask(void)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);

    /* 0 = current process */
    int rc = sched_getaffinity(0, sizeof(cpu_set_t), &mask);
    if (rc != 0) {
        fprintf(stderr, "sched_getaffinity has failed (rc=%d)\n", rc);
        return -1;
    }

    return CPU_COUNT(&mask);
}
