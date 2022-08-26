#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "pgif90.h"

void foo(int * buffer, int * m, int * n, int * o)
{
    printf("FOO buffer = %p\n", buffer);
    printf("FOO m,n,o = %d,%d,%d\n", *m, *n, *o);
}

void print_kind(long long kind)
{
    char * name;
    switch (kind) {
        case 14: name = "CHARACTER(1)"; break; // no other character kinds supported
        case  9: name = "COMPLEX(4)"  ; break;
        case 10: name = "COMPLEX(8)"  ; break;
        case 32: name = "INTEGER(1)"  ; break; // byte appears as this
        case 24: name = "INTEGER(2)"  ; break;
        case 25: name = "INTEGER(4)"  ; break;
        case 26: name = "INTEGER(8)"  ; break;
        case 17: name = "LOGICAL(1)"  ; break;
        case 18: name = "LOGICAL(2)"  ; break;
        case 19: name = "LOGICAL(4)"  ; break;
        case 20: name = "LOGICAL(8)"  ; break;
        case 45: name = "REAL(2)"     ; break;
        case 27: name = "REAL(4)"     ; break;
        case 28: name = "REAL(8)"     ; break; // double precision appears as this
        default: name = "UNKNOWN"     ; break;
    }
    printf("kind = %s\n", name);
}

void print_flags(long long flags)
{
    bool TEMPLATE           = flags & 0x00010000;
    bool OFF_TEMPLATE       = flags & 0x00080000;
    bool SECTZBASE          = flags & 0x00400000;
    bool BOGUSBOUNDS        = flags & 0x00800000;
    bool NOT_COPIED         = flags & 0x01000000;
    bool NOREINDEX          = flags & 0x02000000;
    bool SEQUENTIAL_SECTION = flags & 0x20000000;

    printf("TEMPLATE           = %s\n", TEMPLATE           ? "true" : "false");
    printf("OFF_TEMPLATE       = %s\n", OFF_TEMPLATE       ? "true" : "false");
    printf("SECTZBASE          = %s\n", SECTZBASE          ? "true" : "false");
    printf("BOGUSBOUNDS        = %s\n", BOGUSBOUNDS        ? "true" : "false");
    printf("NOT_COPIED         = %s\n", NOT_COPIED         ? "true" : "false");
    printf("NOREINDEX          = %s\n", NOREINDEX          ? "true" : "false");
    printf("SEQUENTIAL_SECTION = %s\n", SEQUENTIAL_SECTION ? "true" : "false");
}

void bar(int * buffer, int * m, int * n, int * o, F90_Desc_la * d)
{
    printf("BAR buffer = %p\n", buffer);
    printf("BAR m,n,o = %d,%d,%d\n", *m, *n, *o);
    printf("BAR F90_Desc = %p\n", d);
    printf("BAR F90_Desc->tag   = %lld\n", d->tag  );
    printf("BAR F90_Desc->rank  = %lld\n", d->rank );
    printf("BAR F90_Desc->kind  = %lld\n", d->kind );
    print_kind(d->kind);
    printf("BAR F90_Desc->len   = %lld\n", d->len  );
    printf("BAR F90_Desc->flags = %lld\n", d->flags);
    print_flags(d->flags);
    printf("BAR F90_Desc->lsize = %lld\n", d->lsize);
    printf("BAR F90_Desc->gsize = %lld\n", d->gsize);
    printf("BAR F90_Desc->lbase = %lld\n", d->lbase);
    printf("BAR F90_Desc->gbase = %p\n",   d->gbase);
#if 1
    for (int i=0; i<d->rank; i++) {
        printf("BAR F90_Desc->dim.lbound  = %lld\n", d->dim[i].lbound );
        printf("BAR F90_Desc->dim.extent  = %lld\n", d->dim[i].extent );
        printf("BAR F90_Desc->dim.sstride = %lld\n", d->dim[i].sstride);
        printf("BAR F90_Desc->dim.soffset = %lld\n", d->dim[i].soffset);
        printf("BAR F90_Desc->dim.lstride = %lld\n", d->dim[i].lstride);
        printf("BAR F90_Desc->dim.ubound  = %lld\n", d->dim[i].ubound );
    }
#endif
}
