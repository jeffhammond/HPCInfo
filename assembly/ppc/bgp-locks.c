#include <bpcore/bgp_atomic_ops.h>
// This is the relevant excerpt from the header to show that 
// nothing about this code is really BGP-specific:
/**********************************
//@MG: Note alignment need not be this coarse (32B), but must be >= 8B.
typedef struct T_BGP_Atomic
                {
                volatile uint32_t atom;
                }
                ALIGN_L1D_CACHE _BGP_Atomic;
    
// eg: _BGP_Atomic my_atom = _BGP_ATOMIC_INIT( 0 );
//
#define _BGP_ATOMIC_INIT(val) { (val) }
**********************************/

_BGP_Atomic global_atomic = _BGP_ATOMIC_INIT(0);
volatile uint32_t global_lock __attribute__((__aligned__(16)));

static __inline__ uint32_t testandset(volatile uint32_t *global_lock)
{
  uint32_t ret;
  uint32_t val = 1;

  __asm__ volatile(
                   "loop:   lwarx   %0,0,%1   \n"
                   "            stwcx.  %2,0,%1   \n"
                   "            bne-    loop      "
                   : "=r" (ret)
                   : "r" (global_lock)
                   : "r" (val)
                  );
   return ret;
}

static __inline__ void reset(volatile uint32_t *global_lock)
{
  uint32_t val = 0;

  __asm__ volatile(
                   "            mr      %0,%1"
                   : "=r" (global_lock)
                   : "r" (val)
                  );
   return;
}

void global_lock_acquire()
{
   while(testandset(&global_lock));
   return;
}

void global_lock_release()
{
   reset(&global_lock);
   return;
}
