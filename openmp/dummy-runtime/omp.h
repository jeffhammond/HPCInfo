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
