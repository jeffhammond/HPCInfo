#include "safemalloc.h"

void * safemalloc(int n) 
{
    //void * ptr = malloc( n );
    int rc;
    void * ptr;
    rc = posix_memalign( &ptr , ALIGNMENT , n );

    if ( ptr == NULL )
    {
        fprintf( stderr , "%d bytes could not be allocated \n" , n );
        exit(1);
    }

    return ptr;
}

