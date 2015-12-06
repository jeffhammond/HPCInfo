#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_MIN_SIZE 100
#endif

#ifdef MPI
#   include <mpi.h>
#endif

#include "getticks.c"
#include "safemalloc.h"

#define ALIGNMENT 128
#define REPEAT 16

int main(int argc, char* argv[])
{
    int rank = 0;

#ifdef MPI
    int rc;
    int provided;
    MPI_Init_thread( &argc , &argv , MPI_THREAD_FUNNELED , &provided );
    MPI_Comm_rank( MPI_COMM_WORLD , &rank );
#endif

    int min = ( argc>1 ? atoi(argv[1]) : 2 );
    int max = ( argc>2 ? atoi(argv[2]) : 400 );
    int inc = ( argc>3 ? atoi(argv[3]) : 1 );

    if ( rank == 0 )
    {
        unsigned long long * d0 = safemalloc( max * sizeof(unsigned long long) );
        unsigned long long * d1 = safemalloc( max * sizeof(unsigned long long) );
        unsigned long long * d2 = safemalloc( max * sizeof(unsigned long long) );
        unsigned long long * d3 = safemalloc( max * sizeof(unsigned long long) );
        unsigned long long * d4 = safemalloc( max * sizeof(unsigned long long) );
        unsigned long long * d5 = safemalloc( max * sizeof(unsigned long long) );
        unsigned long long * d6 = safemalloc( max * sizeof(unsigned long long) );

        /*              12345678901234567890123456789012 */
        char * name0 = "memcpy";
        char * name1 = "loop copy";
        char * name2 = "basic + s1S";
        char * name3 = "basic + s1L";
        char * name4 = "pur 4x4 + s1L";
        char * name5 = "mur 4x4 + s1L";
        char * name6 = "murv 4x4 + s1L";

#ifdef OMP
        printf( "# starting test: OpenMP ON with %d threads... \n", omp_get_max_threads() );
        fprintf( stderr , "starting test: OpenMP ON with %d threads... \n", omp_get_max_threads() );
#else
        printf( "# starting test: OpenMP OFF... \n" );
        fprintf( stderr , "starting test: OpenMP OFF... \n" );
#endif
        printf( "# sN[S/L] = stride-N [store/load] \n" );
        printf( "# pur  = pragma unroll \n" );
        printf( "# mur  = manual unroll \n" );
        printf( "# murv = manual unroll and vectorize \n" );

#ifdef DEBUG
        fprintf( stderr , "sN[S/L] = stride-N [store/load] \n" );
        fprintf( stderr , "pur  = pragma unroll \n" );
        fprintf( stderr , "mur  = manual unroll \n" );
        fprintf( stderr , "murv = manual unroll and vectorize \n" );
        fprintf( stderr , "%4s %8s %16s %16s %16s %16s %16s %16s \n" , "n" , name0 , name1 , name2 , name3 , name4 , name5 , name6);
        fflush( stderr );
#endif

        for ( int n=min ; n<max ; n+=inc ) {
            const int N = (n*n);

            double * A;
            double * B;

            unsigned long long t0, t1;

            A = safemalloc( N * sizeof(double) );
            B = safemalloc( N * sizeof(double) );

            int k = 0;
#ifdef OMP
#pragma omp parallel for if(n>OMP_MIN_SIZE)
#endif
            for ( int i=0 ; i<n ; i++ )
                for ( int j=0 ; j<n ; j++ )
                    A[i*n+j] = (double)(k++);

#ifdef OMP
#pragma omp parallel for if(n>OMP_MIN_SIZE)
#endif
            for ( int i=0 ; i<n ; i++ )
                for ( int j=0 ; j<n ; j++ )
                    B[i*n+j] = 0.0;

            /* reference - memcpy */
            t0 = getticks();
            for ( int t=0 ; t<REPEAT ; t++ )
                memcpy( B , A , N );

            t1 = getticks();
            d0[n] = (t1-t0)/REPEAT;

            /* reference - direct copy */
            t0 = getticks();
            for ( int t=0 ; t<REPEAT ; t++ )
            {
#ifdef OMP
#pragma omp parallel for if(n>OMP_MIN_SIZE)
#endif
                for ( int i=0 ; i<n ; i++ )
                    for ( int j=0 ; j<n ; j++ )
                        B[i*n+j] = A[i*n+j];
            }
            t1 = getticks();
            d1[n] = (t1-t0)/REPEAT;

            /* basic w/ stride-1 stores */
            t0 = getticks();
            for ( int t=0 ; t<REPEAT ; t++ )
            {
#ifdef OMP
#pragma omp parallel for if(n>OMP_MIN_SIZE)
#endif
                for ( int i=0 ; i<n ; i++ )
                    for ( int j=0 ; j<n ; j++ )
                        B[i*n+j] = A[j*n+i];
            }
            t1 = getticks();
            d2[n] = (t1-t0)/REPEAT;

            /* verify */
            for ( int j=0 ; j<n ; j++ )
                for ( int i=0 ; i<n ; i++ )
                    assert( B[i*n+j] == A[j*n+i] );

            /* basic w/ stride-1 loads */
            t0 = getticks();
            for ( int t=0 ; t<REPEAT ; t++ )
            {
#ifdef OMP
#pragma omp parallel for if(n>OMP_MIN_SIZE)
#endif
                for ( int j=0 ; j<n ; j++ )
                    for ( int i=0 ; i<n ; i++ )
                        B[i*n+j] = A[j*n+i];
            }
            t1 = getticks();
            d3[n] = (t1-t0)/REPEAT;

            /* verify */
            for ( int j=0 ; j<n ; j++ )
                for ( int i=0 ; i<n ; i++ )
                    assert( B[i*n+j] == A[j*n+i] );

            /* pragma unroll 4x4 + s1 loads */
            t0 = getticks();
            for ( int t=0 ; t<REPEAT ; t++ ) {
#ifdef OMP
#pragma omp parallel for if(n>OMP_MIN_SIZE)
#endif
                //#pragma unroll(4)
                #pragma unroll_and_jam
                for ( int j=0 ; j<n ; j++ )
                    //#pragma unroll(4)
                    #pragma unroll_and_jam
                    for ( int i=0 ; i<n ; i++ )
                        B[i*n+j] = A[j*n+i];
            }
            t1 = getticks();
            d4[n] = (t1-t0)/REPEAT;

            /* verify */
            for ( int j=0 ; j<n ; j++ )
                for ( int i=0 ; i<n ; i++ )
                    assert( B[i*n+j] == A[j*n+i] );

            /* manual unroll 4x4 + s1 loads */
            t0 = getticks();
            for ( int t=0 ; t<REPEAT ; t++ ) {
#ifdef OMP
#pragma omp parallel if(n>OMP_MIN_SIZE)
#endif
              {
                int n4 = n-(n%4);    /* divisible-by-4 part */
#ifdef OMP
#pragma omp for private(i,j,n4)
#endif
                for ( int j=0 ; j<n4 ; j+=4 ) {
                    for ( int i=0 ; i<n4 ; i+=4 ) {
                        B[(i  )*n+j  ] = A[(j  )*n+i  ];
                        B[(i  )*n+j+1] = A[(j+1)*n+i  ];
                        B[(i  )*n+j+2] = A[(j+2)*n+i  ];
                        B[(i  )*n+j+3] = A[(j+3)*n+i  ];
                        B[(i+1)*n+j  ] = A[(j  )*n+i+1];
                        B[(i+1)*n+j+1] = A[(j+1)*n+i+1];
                        B[(i+1)*n+j+2] = A[(j+2)*n+i+1];
                        B[(i+1)*n+j+3] = A[(j+3)*n+i+1];
                        B[(i+2)*n+j  ] = A[(j  )*n+i+2];
                        B[(i+2)*n+j+1] = A[(j+1)*n+i+2];
                        B[(i+2)*n+j+2] = A[(j+2)*n+i+2];
                        B[(i+2)*n+j+3] = A[(j+3)*n+i+2];
                        B[(i+3)*n+j  ] = A[(j  )*n+i+3];
                        B[(i+3)*n+j+1] = A[(j+1)*n+i+3];
                        B[(i+3)*n+j+2] = A[(j+2)*n+i+3];
                        B[(i+3)*n+j+3] = A[(j+3)*n+i+3];
                    }
                    for ( int i=n4 ; i<n ; i++ )
                        B[i*n+j] = A[j*n+i];
                }
                for ( int j=n4 ; j<n ; j++ )
                {
                    for ( int i=0 ; i<n4 ; i+=4 )
                    {
                        B[(i  )*n+j] = A[j*n+i  ];
                        B[(i+1)*n+j] = A[j*n+i+1];
                        B[(i+2)*n+j] = A[j*n+i+2];
                        B[(i+3)*n+j] = A[j*n+i+3];
                    }
                    for ( int i=n4 ; i<n ; i++ )
                        B[i*n+j] = A[j*n+i];
                }
              }
            }
            t1 = getticks();
            d5[n] = (t1-t0)/REPEAT;

            /* verify */
            for ( int j=0 ; j<n ; j++ )
                for ( int i=0 ; i<n ; i++ )
                    assert( B[i*n+j] == A[j*n+i] );

            /* manual unroll 4x4 and vectorize + s1 loads */
            t0 = getticks();
            for ( int t=0 ; t<REPEAT ; t++ ) {
#ifdef OMP
#pragma omp parallel if(n>OMP_MIN_SIZE)
#endif
              {
                int n4 = n-(n%4);    /* divisible-by-4 part */
#ifdef OMP
#pragma omp for private(i,j,n4)
#endif
                for ( int j=0 ; j<n4 ; j+=4 ) {
                    for ( int i=0 ; i<n4 ; i+=4 ) {
                        double a00, a01, a02, a03;
                        double a10, a11, a12, a13;
                        double a20, a21, a22, a23;
                        double a30, a31, a32, a33;

                        double b00, b01, b02, b03;
                        double b10, b11, b12, b13;
                        double b20, b21, b22, b23;
                        double b30, b31, b32, b33;

                        a00 = A[(j  )*n+i  ];
                        a01 = A[(j  )*n+i+1];
                        a02 = A[(j  )*n+i+2];
                        a03 = A[(j  )*n+i+3];
                        a10 = A[(j+1)*n+i  ];
                        a11 = A[(j+1)*n+i+1];
                        a12 = A[(j+1)*n+i+2];
                        a13 = A[(j+1)*n+i+3];
                        a20 = A[(j+2)*n+i  ];
                        a21 = A[(j+2)*n+i+1];
                        a22 = A[(j+2)*n+i+2];
                        a23 = A[(j+2)*n+i+3];
                        a30 = A[(j+3)*n+i  ];
                        a31 = A[(j+3)*n+i+1];
                        a32 = A[(j+3)*n+i+2];
                        a33 = A[(j+3)*n+i+3];

                        b00=a00; b01=a10; b02=a20; b03=a30;
                        b10=a01; b11=a11; b12=a21; b13=a31;
                        b20=a02; b21=a12; b22=a22; b23=a32;
                        b30=a03; b31=a13; b32=a23; b33=a33;

                        B[(i  )*n+j  ] = b00;
                        B[(i  )*n+j+1] = b01;
                        B[(i  )*n+j+2] = b02;
                        B[(i  )*n+j+3] = b03;
                        B[(i+1)*n+j  ] = b10;
                        B[(i+1)*n+j+1] = b11;
                        B[(i+1)*n+j+2] = b12;
                        B[(i+1)*n+j+3] = b13;
                        B[(i+2)*n+j  ] = b20;
                        B[(i+2)*n+j+1] = b21;
                        B[(i+2)*n+j+2] = b22;
                        B[(i+2)*n+j+3] = b23;
                        B[(i+3)*n+j  ] = b30;
                        B[(i+3)*n+j+1] = b31;
                        B[(i+3)*n+j+2] = b32;
                        B[(i+3)*n+j+3] = b33;
                    }
                    for ( int i=n4 ; i<n ; i++ )
                        B[i*n+j] = A[j*n+i];
                }
                for ( int j=n4 ; j<n ; j++ ) {
                    for ( int i=0 ; i<n4 ; i+=4 )
                    {
                        int a00, a01, a02, a03;
                        int b00, b10, b20, b30;

                        a00 = A[j*n+i  ];
                        a01 = A[j*n+i+1];
                        a02 = A[j*n+i+2];
                        a03 = A[j*n+i+3];

                        b00=a00;
                        b10=a01;
                        b20=a02;
                        b30=a03;

                        B[(i  )*n+j] = b00;
                        B[(i+1)*n+j] = b10;
                        B[(i+2)*n+j] = b20;
                        B[(i+3)*n+j] = b30;
                    }
                    for ( int i=n4 ; i<n ; i++ )
                        B[i*n+j] = A[j*n+i];
                }
              }
            }
            t1 = getticks();
            d6[n] = (t1-t0)/REPEAT;

            /* verify */
            for ( int j=0 ; j<n ; j++ )
                for ( int i=0 ; i<n ; i++ )
                    assert( B[i*n+j] == A[j*n+i] );

#ifdef DEBUG
            if ( n<11 ) {
                printf( "A: \n" );
                for ( int i=0 ; i<n ; i++ ) {
                    for ( int j=0 ; j<n ; j++ ) printf( "%12.1lf" , A[i*n+j]);
                    printf( "\n" );
                }

                printf( "B: \n" );
                for ( int i=0 ; i<n ; i++ ) {
                    for ( int j=0 ; j<n ; j++ ) printf( "%12.1lf" , B[i*n+j]);
                    printf( "\n" );
                }
            }
            /* this is just for the neurotic person who cannot wait until the end for data */
            double c = 1.0 / d0[n];
            fprintf( stderr , "%4d %8llu %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) \n" ,
                                n , d0[n] , d1[n] , c*d1[n] , d2[n] , c*d2[n] , d3[n] , c*d3[n] , d4[n] , c*d4[n] , d5[n] , c*d5[n] , d6[n] , c*d6[n] );

            free(B);
            free(A);
            fflush( stderr );
#endif
        }

        /* print analysis */
        printf( "# timing in cycles (ratio relative to memcpy) \n" );
        printf( "%4s %8s %16s %16s %16s %16s %16s %16s \n" , "# n" , name0 , name1 , name2 , name3 , name4 , name5 , name6);
        for ( int n=min ; n<max ; n+=inc)
        {
            double c = 1.0 / d0[n];
            printf( "%4d %8llu %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) %8llu (%5.1lf) \n" ,
                      n , d0[n] , d1[n] , c*d1[n] , d2[n] , c*d2[n] , d3[n] , c*d3[n] , d4[n] , c*d4[n] , d5[n] , c*d5[n] , d6[n] , c*d6[n] );
        }

        printf( "# ...the end \n" );
        fflush( stdout );

        free(d6);
        free(d5);
        free(d4);
        free(d3);
        free(d2);
        free(d1);
        free(d0);
    }

#ifdef MPI
    MPI_Finalize();
#endif

    return 0;
}
