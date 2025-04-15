// icx -fiopenmp -fopenmp-targets=spir64 slm.c && ./a.out

#include <stdio.h>
#include <omp.h>

int main(void)
{
    int mxtm = 0;//omp_get_max_teams();
    int mxtd = 0;//omp_get_teams_thread_limit();
    printf("omp_get_max_teams = %d\n",mxtm);
    printf("omp_get_teams_thread_limit = %d\n",mxtd);
    #pragma omp target teams reduction(max:mxtm,mxtd)
    {
        mxtm = omp_get_num_teams();
        mxtd = omp_get_max_threads();
    }
    printf("omp_get_max_teams = %d\n",mxtm);
    printf("omp_get_teams_thread_limit = %d\n",mxtd);

    //int * a = omp_target_alloc(mxtm*mxtd*sizeof(int),omp_get_default_device());
    int * restrict a = calloc(mxtm*mxtd,sizeof(int));
    printf("a=%p\n",a);
    #pragma omp target teams map(tofrom:a[0:mxtm*mxtd]) //num_teams(4)
    {
        const int team = omp_get_team_num();
        #pragma omp parallel shared(a) //num_threads(4)
        {
            int tls[1024] = {0};
            mxtd = omp_get_max_threads();
            const int thrd = omp_get_thread_num();
            //printf("team %6d thread %6d total %9d\n", team, thrd, mxtd*team+thrd);
            #pragma omp atomic write
            a[mxtd*team+thrd] = mxtd*team+thrd;
            if (thrd < sizeof(tls)/sizeof(*tls)) {
                #pragma omp atomic write
                tls[thrd] = team;
            }
            if (thrd==0) {
                printf("team=%d tls=%p tls[1]=%d\n", team, tls, tls[1]);
            }
        }
    }
    for (int i=0; i<mxtd*mxtm; i++) {
        //if (a[i] != 0) printf("%d,",a[i]);
    }
    return 0;
}
