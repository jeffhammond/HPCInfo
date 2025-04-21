#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#define MAX_CPUS 1024
#define MAX_LINE 256

// Structure to hold sibling relationships
typedef struct {
    int cpu;
    int* siblings;
    int num_siblings;
} CoreSiblings;

// Function to read and parse thread siblings list for a CPU
CoreSiblings* read_thread_siblings(int cpu) {
    char path[256];
    char line[MAX_LINE];
    CoreSiblings* result = malloc(sizeof(CoreSiblings));
    result->cpu = cpu;
    result->siblings = malloc(MAX_CPUS * sizeof(int));
    result->num_siblings = 0;

    snprintf(path, sizeof(path), 
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", 
             cpu);

    FILE* f = fopen(path, "r");
    if (!f) {
        // If we can't open the file, this CPU might not exist
        free(result->siblings);
        free(result);
        return NULL;
    }

    if (fgets(line, sizeof(line), f)) {
        char* token = strtok(line, ",");
        while (token) {
            // Handle ranges like "0-3"
            if (strchr(token, '-')) {
                int start, end;
                sscanf(token, "%d-%d", &start, &end);
                for (int i = start; i <= end; i++) {
                    result->siblings[result->num_siblings++] = i;
                }
            } else {
                result->siblings[result->num_siblings++] = atoi(token);
            }
            token = strtok(NULL, ",");
        }
    }

    fclose(f);
    return result;
}

// Function to check if two CPUs are siblings
int are_siblings(CoreSiblings** siblings_list, int num_cores, int cpu1, int cpu2) {
    for (int i = 0; i < num_cores; i++) {
        if (siblings_list[i] && siblings_list[i]->cpu == cpu1) {
            for (int j = 0; j < siblings_list[i]->num_siblings; j++) {
                if (siblings_list[i]->siblings[j] == cpu2) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

int main() {
    cpu_set_t mask;
    CoreSiblings* siblings_list[MAX_CPUS] = {NULL};
    int num_cores = 0;
    int active_cpus[MAX_CPUS];
    int num_active = 0;

    // Get the current CPU affinity mask
    if (sched_getaffinity(0, sizeof(mask), &mask) == -1) {
        perror("sched_getaffinity failed");
        return 1;
    }

    // Get the number of available CPUs
    num_cores = sysconf(_SC_NPROCESSORS_CONF);
    if (num_cores > MAX_CPUS) {
        fprintf(stderr, "System has too many CPUs\n");
        return 1;
    }

    // Read thread siblings information for all CPUs
    for (int i = 0; i < num_cores; i++) {
        siblings_list[i] = read_thread_siblings(i);
    }

    // Count and store active CPUs from affinity mask
    for (int i = 0; i < num_cores; i++) {
        if (CPU_ISSET(i, &mask)) {
            active_cpus[num_active++] = i;
        }
    }

    printf("Active CPUs in affinity mask: ");
    for (int i = 0; i < num_active; i++) {
        printf("%d ", active_cpus[i]);
    }
    printf("\n");

    // If exactly two CPUs are in the mask, check if they're siblings
    if (num_active == 2) {
        if (are_siblings(siblings_list, num_cores, active_cpus[0], active_cpus[1])) {
            printf("CPUs %d and %d are hyperthreads on the same core\n", 
                   active_cpus[0], active_cpus[1]);
        } else {
            printf("CPUs %d and %d are on different physical cores\n", 
                   active_cpus[0], active_cpus[1]);
        }
    } else {
        printf("Affinity mask contains %d CPUs (not exactly 2)\n", num_active);
    }

    // Print detailed sibling information for each active CPU
    printf("\nDetailed thread sibling information:\n");
    for (int i = 0; i < num_active; i++) {
        int cpu = active_cpus[i];
        if (siblings_list[cpu]) {
            printf("CPU %d siblings: ", cpu);
            for (int j = 0; j < siblings_list[cpu]->num_siblings; j++) {
                printf("%d ", siblings_list[cpu]->siblings[j]);
            }
            printf("\n");
        }
    }

    // Cleanup
    for (int i = 0; i < num_cores; i++) {
        if (siblings_list[i]) {
            free(siblings_list[i]->siblings);
            free(siblings_list[i]);
        }
    }

    return 0;
}
