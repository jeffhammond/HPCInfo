# Specification

See the [OpenMP home page](http://openmp.org/wp/openmp-specifications/) for the latest specification.

On Blue Gene/Q, the IBM XL compilers provide support for OpenMP v3.1. The GNU compilers provide support for OpenMP v3.0, although .  LLVM does not yet support OpenMP on any platform.

More information about OpenMP support in GCC can be found at http://gcc.gnu.org/wiki/openmp and http://gcc.gnu.org/projects/gomp/.

# Documentation

The [LLNL OpenMP page](https://computing.llnl.gov/tutorials/openMP/) is the best description I've found anywhere.  Blaise Barney is awesome.

NERSC has an [OpenMP tutorial)http://www.nersc.gov/assets/Uploads/IntroToOpenMP.pdf) as well.

# Dummy Runtime

## omp.h
```
/* 
 * Copyright(c) 2005-2013 PathScale Inc.
 * Copyright(c) 2013-     Argonne National Laboratory. 
 *
 * All rights reserved.  
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files(the "Software"), to deal 
 * with the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 * of the Software, and to permit persons to whom the Software is furnished to do so, 
 * subject to the following conditions: 
 *
 * - Redistributions of source code must retain the above copyright notice, this list of 
 *   conditions and the following disclaimers.  
 * - Redistributions in binary form must reproduce the above copyright notice, this list of 
 *   conditions and the following disclaimers in the documentation and/or other materials 
 *   provided with the distribution. 
 * - Neither the names of PathScale Inc. nor Argonne National Laboratory, nor the names of 
 *   its contributors may be used to endorse or promote products derived from this Software 
 *   without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE 
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS WITH THE SOFTWARE.
 */

#ifndef include_omp_h
#define include_omp_h

#ifdef __cplusplus
extern "C" {
#endif

typedef int omp_lock_t;
typedef int omp_nest_lock_t;

extern double omp_get_wtime(void);
extern double omp_get_wtick(void);

extern int  omp_get_num_threads(void);
extern void omp_set_num_threads(int nthreads);
extern int  omp_get_max_threads(void);
extern int  omp_get_thread_num(void);
extern int  omp_get_num_procs(void);
extern int  omp_in_parallel(void);
extern int  omp_get_dynamic(void);
extern void omp_set_dynamic(int nthreads);
extern int  omp_get_nested(void);
extern void omp_set_nested(int nested);

extern void omp_init_lock(omp_lock_t *lock);
extern void omp_destroy_lock(omp_lock_t *lock);
extern void omp_set_lock(omp_lock_t *lock);
extern void omp_unset_lock(omp_lock_t *lock);
extern int  omp_test_lock(omp_lock_t *lock);

extern void omp_init_nest_lock(omp_nest_lock_t *lock);
extern void omp_destroy_nest_lock(omp_nest_lock_t *lock);
extern void omp_set_nest_lock(omp_nest_lock_t *lock);
extern void omp_unset_nest_lock(omp_nest_lock_t *lock);
extern int  omp_test_nest_lock(omp_nest_lock_t *lock);

#ifdef __cplusplus
}
#endif

#endif /* include_omp_h */
```

## omp.c
```
/* 
 * Copyright(c) 2013-     Argonne National Laboratory. 
 *
 * All rights reserved.  
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files(the "Software"), to deal 
 * with the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 * of the Software, and to permit persons to whom the Software is furnished to do so, 
 * subject to the following conditions: 
 *
 * - Redistributions of source code must retain the above copyright notice, this list of 
 *   conditions and the following disclaimers.  
 * - Redistributions in binary form must reproduce the above copyright notice, this list of 
 *   conditions and the following disclaimers in the documentation and/or other materials 
 *   provided with the distribution. 
 * - Neither the names of nor Argonne National Laboratory, nor the names of its contributors 
 *   may be used to endorse or promote products derived from this Software without specific 
 *   prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE 
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS WITH THE SOFTWARE.
 */

#include <stdio.h>
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

omp_lock_t _unset_lock     =  0;
omp_lock_t _set_lock       =  1;
omp_lock_t _destroyed_lock = -1;

omp_nest_lock_t _unset_nest_lock     =  0;
omp_nest_lock_t _set_nest_lock       =  1;
omp_nest_lock_t _destroyed_nest_lock = -1;

double omp_get_wtime(void) 
{ 
    return 0.0; 
}

double omp_get_wtick(void) 
{ 
    return 0.0; 
}

int  omp_get_num_threads(void) 
{ 
    return 1; 
}

void omp_set_num_threads(int nthreads) 
{ 
    if (nthreads>1)
        fprintf(stderr, "You are using a dummy OpenMP runtime and cannot set anything! \n");
    return;
}

int  omp_get_max_threads(void)
{ 
    return 1; 
}

int  omp_get_thread_num(void)
{ 
    return 0; 
}

int  omp_get_num_procs(void)
{ 
    return 1; 
}

int  omp_in_parallel(void)
{ 
    return 0; 
}

int  omp_get_dynamic(void)
{ 
    return 0; 
}

void omp_set_dynamic(int nthreads)
{ 
    if (nthreads>1)
        fprintf(stderr, "You are using a dummy OpenMP runtime and cannot set anything! \n");
    return;
}

int  omp_get_nested(void)
{ 
    return 0; 
}

void omp_set_nested(int nested)
{ 
    fprintf(stderr, "You are using a dummy OpenMP runtime and cannot set anything! \n");
    return;
}

void omp_init_lock(omp_lock_t *lock)
{ 
    *lock = _unset_lock;
    return; 
}

void omp_destroy_lock(omp_lock_t *lock)
{ 
    if ((*lock)==_unset_lock)
        *lock = _destroyed_lock;
    else 
        fprintf(stderr, "You cannot destroy a set lock! (%d) \n", (int)(*lock));
    return; 
}

void omp_set_lock(omp_lock_t *lock)
{ 
    if ((*lock)==_unset_lock)
        *lock = _set_lock;
    else 
        fprintf(stderr, "You cannot set an set lock! (%d) \n", (int)(*lock));
    return; 
}

void omp_unset_lock(omp_lock_t *lock)
{ 
    if ((*lock)==_set_lock)
        *lock = _unset_lock;
    else 
        fprintf(stderr, "You cannot unset an unset lock! (%d) \n", (int)(*lock));
    return; 
}

int  omp_test_lock(omp_lock_t *lock)
{ 
    if ((*lock)==_unset_lock) {
        *lock = _set_lock;
        return 1;
    } else {
        return 0; 
    }
}

void omp_init_nest_lock(omp_nest_lock_t *lock)
{ 
    *lock = _unset_nest_lock;
    return; 
}

void omp_destroy_nest_lock(omp_nest_lock_t *lock)
{ 
    if ((*lock)==0)
        *lock = _destroyed_nest_lock;
    else 
        fprintf(stderr, "You cannot destroy a set nest lock! (%d) \n", (int)(*lock));
    return; 
}

void omp_set_nest_lock(omp_nest_lock_t *lock)
{ 
    if ((*lock)>=0)
        (*lock)++;
    else 
        fprintf(stderr, "You cannot set a destroyed nest lock! (%d) \n", (int)(*lock));
    return; 
}

void omp_unset_nest_lock(omp_nest_lock_t *lock)
{ 
    if ((*lock)>0)
        (*lock)--;
    else 
        fprintf(stderr, "You cannot unset a set nest lock! (%d) \n", (int)(*lock));
    return; 
}

int  omp_test_nest_lock(omp_nest_lock_t *lock)
{ 
    if ((*lock)>=0) {
        (*lock)++;
        return 1;
    } else {
        fprintf(stderr, "You cannot test a destroyed nest lock! (%d) \n", (int)(*lock));
        return 0; 
    }
}

#ifdef __cplusplus
}
#endif
```

# Open-Source Runtime Implementations

## Background

* http://openmp.llvm.org/

## Intel OSS OpenMP runtime library

* http://www.openmprtl.org/
* https://github.com/pathscale/openmprtl/
