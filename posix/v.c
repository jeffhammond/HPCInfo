/* this has to be set before headers are included */
#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

static inline
int count_commas(const char* buffer)
{
    const char* pos = buffer;
    int count = 0;
    while ((pos = strstr(pos, ",")) != NULL) {
        count++;
        pos++;
    }
    return count;
}

static inline
int get_threads_per_core(int cpu)
{
    char path[256] = {0};
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);

    const int fd = open(path, O_RDONLY);
    if (fd < 0) return 1;

    char line[256] = {0};
    const int br = read(fd, line, sizeof(line)-1);
    if (br <= 0) return 1;

    const int cc = 1 + count_commas(line);

    close(fd);

    return cc;
}

static inline
int get_my_cpu(const cpu_set_t * mask)
{
    for (size_t i=0; i<CPU_SETSIZE; i++) {
        const int on = CPU_ISSET(i, mask);
        if (on) return i;
    }
    return 0;
}

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

    const int hwthreads = CPU_COUNT(&mask);
    const int mycpu     = get_my_cpu(&mask);
    const int hwthpcore = get_threads_per_core(mycpu);

#ifdef DEBUG
    printf("hwthreads=%d mycpu=%d hwthpcore=%d\n", hwthreads, mycpu, hwthpcore);
#endif

    return hwthreads / hwthpcore;
}
