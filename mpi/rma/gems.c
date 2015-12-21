#include <stdio.h>
#include <stdlib.h>

/***************************************************************/
#include <mpi.h>
int main(void) {
    MPI_Init(NULL,NULL);
    int me,np;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    if (np<2) MPI_Abort(MPI_COMM_WORLD,1);
    /* allocate from the shared heap */
    int * Abuf;
    MPI_Win Awin;
    MPI_Win_allocate(sizeof(int),sizeof(int),MPI_INFO_NULL,MPI_COMM_WORLD,&Abuf,&Awin);
    MPI_Win_lock_all(MPI_MODE_NOCHECK,Awin);
    int B = 134;
    /* store local B at PE 0 into A at PE 1 */
    if (me==0) MPI_Put(&B,1,MPI_INT,1,0,1,MPI_INT,Awin);
    /* global synchronization of execution and data */
    MPI_Win_flush_all(Awin);
    MPI_Barrier(MPI_COMM_WORLD);
    /* observe the result of the store */
    if (me==1) printf("A@1=%d\n",*Abuf);
    MPI_Win_unlock_all(Awin);
    MPI_Win_free(&Awin);
    MPI_Finalize();
    return 0;
}
/***************************************************************/
