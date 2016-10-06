#ifndef HUGE_ALLOC_H
#define HUGE_ALLOC_H

/* This is required to get MAP_HUGETLB, MAP_ANONYMOUS, MAP_POPULATE in C99 mode. */
/* It must be before all the heads, and not just before sys/mman.h. */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <sys/mman.h>
#include <errno.h>

void * huge_alloc(size_t bytes, size_t pagesize);
void huge_free(void* ptr, size_t bytes);

#endif /* HUGE_ALLOC_H */
