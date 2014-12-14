/*
 * Written by Jeff Hammond, July 2012
 * Copyright Argonne National Laboratory
 *
 * This implementation of BGQ atomics is based upon 
 * hwi/include/bqc/A2_inlines.h but uses signed integers
 * instead of unsigned integer and/or long types.
 *
 * In theory, this should work for any PowerPC system.  Need to verify on Blue Gene/P and POWER7. 
 */

#include <stdint.h>

#ifndef __INLINE__
#define __INLINE__ extern inline __attribute__((always_inline))
#endif

/* Not yet verified for use on non-BGQ PPC but not active there anyways */
#if defined(__bgq__) || defined(__bgp__) || defined(__powerpc__)
__INLINE__ int32_t LoadReservedSigned32( volatile int32_t *pVar )
{
   register int32_t Val;
   asm volatile ("lwarx   %[rc],0,%[pVar];"
                 : [rc] "=&b" (Val)
                 : [pVar] "b" (pVar));
   return(Val);
}

__INLINE__ int StoreConditionalSigned32( volatile int32_t *pVar, int32_t Val )
{
   register int rc = 1; // assume success
   asm volatile ("  stwcx.  %2,0,%1;"
                 "  beq     1f;"       // conditional store succeeded
                 "  li      %0,0;"
                 "1:;"
                 : "=b" (rc)
                 : "b"  (pVar),
                   "b"  (Val),
                   "0"  (rc)
                 : "cc", "memory" );
   return(rc);
}

__INLINE__ int32_t CompareAndSwapSigned32( volatile int32_t *var, int32_t  Compare, int32_t  NewValue )
{
    asm volatile ("msync" : : : "memory");

    int32_t OldValue = *var;

    do {
       int32_t TmpValue = LoadReservedSigned32( var );
       if ( Compare != TmpValue  ) return(OldValue);
       }
       while( !StoreConditionalSigned32( var, NewValue ) );

    return(OldValue);
}

__INLINE__ int32_t FetchAndAddSigned32( volatile int32_t *pVar, int32_t value )
{
    asm volatile ("msync" : : : "memory");

    register int32_t old_val, tmp_val;

    do
    {
        old_val = LoadReservedSigned32( pVar );
        tmp_val = old_val + value;
    }
    while ( !StoreConditionalSigned32( pVar, tmp_val ) );

    return( old_val );
}

__INLINE__ void MemoryBarrier(void)
{
    asm volatile ("msync" : : : "memory");

    return;
}

#else
#error You cannot use PowerPC assembly on a non-PowerPC architecture!
#endif
