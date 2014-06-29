#include "bench.h"

void put_static_nocallback()
{

    if (myrank == 0)
    {

        DCMF_Request_t put_req[ITERATIONS + SKIP];
        DCMF_Request_t put_flush;
        DCMF_Callback_t put_done, put_ack;
        DCMF_Callback_t put_flush_done, put_flush_ack;
        int flush_done_count, flush_ack_count;
        int msgsize, i;

        put_done.function = NULL;
        put_done.clientdata = NULL;
        put_ack.function = NULL;
        put_ack.clientdata = NULL;

        put_flush_done.function = done;
        put_flush_done.clientdata = (void *) &flush_done_count;
        put_flush_ack.function = done;
        put_flush_ack.clientdata = (void *) &flush_ack_count;

        char buffer[50];
        sprintf(buffer, "%20s  %20s", "Msg Size", "Latency (usec)");
        printf("%s \n", buffer);
        fflush(stdout);

        barrier();

        for (msgsize = 1; msgsize < MAX_MSG_SIZE; msgsize *= 2)
        {

            /***********************
             * warmup               *
             ***********************/
            for (i = 0; i < SKIP; i++)
            {
                DCMF_Put(&put_reg,
                         &put_req[i],
                         put_done,
                         DCMF_SEQUENTIAL_CONSISTENCY,
                         1,
                         msgsize,
                         memregion[myrank],
                         memregion[myrank + 1],
                         i * msgsize,
                         i * msgsize,
                         put_ack);
            }

            flush_ack_count = 1;
            DCMF_Put(&put_reg,
                     &put_flush,
                     put_flush_done,
                     DCMF_SEQUENTIAL_CONSISTENCY,
                     1,
                     1,
                     memregion[myrank],
                     memregion[myrank + 1],
                     0,
                     0,
                     put_flush_ack);
            while (flush_ack_count)
                DCMF_Messager_advance();

            /***********************
             * start timer          *
             ***********************/

            t_start = DCMF_Timebase();

            for (i = SKIP; i < ITERATIONS + SKIP; i++)
            {
                DCMF_Put(&put_reg,
                         &put_req[i],
                         put_done,
                         DCMF_SEQUENTIAL_CONSISTENCY,
                         1,
                         msgsize,
                         memregion[myrank],
                         memregion[myrank + 1],
                         i * msgsize,
                         i * msgsize,
                         put_ack);
            }

            flush_ack_count = 1;
            DCMF_Put(&put_reg,
                     &put_flush,
                     put_flush_done,
                     DCMF_SEQUENTIAL_CONSISTENCY,
                     1,
                     1,
                     memregion[myrank],
                     memregion[myrank + 1],
                     0,
                     0,
                     put_flush_ack);
            while (flush_ack_count)
                DCMF_Messager_advance();
            t_stop = DCMF_Timebase();

            /***********************
             * stop timer          *
             ***********************/

            t_usec = ((t_stop - t_start) / clockMHz);
            printf("%20d %20.0f \n", msgsize, t_usec / (ITERATIONS));

        }

        barrier();

    }
    else
    {

        barrier();

        barrier();

    }
}

void put_static_callback()
{

    if (myrank == 0)
    {

        DCMF_Request_t put_req[ITERATIONS + SKIP];
        DCMF_Callback_t put_done, put_ack;
        int done_count, ack_count;
        int msgsize, i;

        put_done.function = done;
        put_done.clientdata = (void *) &done_count;
        put_ack.function = done;
        put_ack.clientdata = (void *) &ack_count;

        char buffer[50];
        sprintf(buffer, "%20s  %20s", "Msg Size", "Latency (us)");
        printf("%s \n", buffer);
        fflush(stdout);

        barrier();

        for (msgsize = 1; msgsize < MAX_MSG_SIZE; msgsize *= 2)
        {

            /***********************
             * warmup               *
             ***********************/
            ack_count = SKIP;
            for (i = 0; i < SKIP; i++)
            {
                DCMF_Put(&put_reg,
                         &put_req[i],
                         put_done,
                         DCMF_SEQUENTIAL_CONSISTENCY,
                         1,
                         msgsize,
                         memregion[myrank],
                         memregion[myrank + 1],
                         i * msgsize,
                         i * msgsize,
                         put_ack);
            }
            while (ack_count)
                DCMF_Messager_advance();

            /***********************
             * start timer          *
             ***********************/

            t_start = DCMF_Timebase();
            ack_count = ITERATIONS;

            for (i = SKIP; i < ITERATIONS + SKIP; i++)
            {
                DCMF_Put(&put_reg,
                         &put_req[i],
                         put_done,
                         DCMF_SEQUENTIAL_CONSISTENCY,
                         1,
                         msgsize,
                         memregion[myrank],
                         memregion[myrank + 1],
                         i * msgsize,
                         i * msgsize,
                         put_ack);
            }

            while (ack_count)
                DCMF_Messager_advance();
            t_stop = DCMF_Timebase();

            /***********************
             * stop timer          *
             ***********************/

            t_usec = ((t_stop - t_start) / clockMHz);
            printf("%20d %20.0f \n", msgsize, t_usec / (ITERATIONS));

        }

        barrier();

    }
    else
    {

        barrier();

        barrier();

    }
}

void put_dynamic_callback()
{

    if (myrank == 0)
    {

        DCMF_Request_t put_req[ITERATIONS + SKIP];
        DCMF_Callback_t put_done, put_ack;
        int done_count, ack_count;
        int msgsize, i;

        put_done.function = done;
        put_done.clientdata = (void *) &done_count;
        put_ack.function = done;
        put_ack.clientdata = (void *) &ack_count;

        char buffer[50];
        sprintf(buffer, "%20s  %20s", "Msg Size", "Latency (usec)");
        printf("%s \n", buffer);
        fflush(stdout);

        barrier();

        for (msgsize = 1; msgsize < MAX_MSG_SIZE; msgsize *= 2)
        {

            /***********************
             * warmup               *
             ***********************/
            ack_count = SKIP;
            for (i = 0; i < SKIP; i++)
            {
                DCMF_Put(&put_reg,
                         &put_req[i],
                         put_done,
                         DCMF_RELAXED_CONSISTENCY,
                         1,
                         msgsize,
                         memregion[myrank],
                         memregion[myrank + 1],
                         i * msgsize,
                         i * msgsize,
                         put_ack);
            }
            while (ack_count)
                DCMF_Messager_advance();

            /***********************
             * start timer          *
             ***********************/

            t_start = DCMF_Timebase();
            ack_count = ITERATIONS;

            for (i = SKIP; i < ITERATIONS + SKIP; i++)
            {
                DCMF_Put(&put_reg,
                         &put_req[i],
                         put_done,
                         DCMF_RELAXED_CONSISTENCY,
                         1,
                         msgsize,
                         memregion[myrank],
                         memregion[myrank + 1],
                         i * msgsize,
                         i * msgsize,
                         put_ack);
            }

            while (ack_count)
                DCMF_Messager_advance();
            t_stop = DCMF_Timebase();

            /***********************
             * stop timer          *
             ***********************/

            t_usec = ((t_stop - t_start) / clockMHz);
            printf("%20d %20.0f \n", msgsize, t_usec / (ITERATIONS));

        }

        barrier();

    }
    else
    {

        barrier();

        barrier();

    }
}

int main()
{
    DCMF_Messager_initialize();

    init();

    barrier_init(DCMF_DEFAULT_GLOBALBARRIER_PROTOCOL);

    control_init(DCMF_DEFAULT_CONTROL_PROTOCOL, DCMF_DEFAULT_NETWORK);

    memregion_init(MAX_BUF_SIZE * nranks);

    put_init(DCMF_DEFAULT_PUT_PROTOCOL, DCMF_TORUS_NETWORK);

    barrier();

    if (myrank == 0)
    {
        printf("Put Latency (usec) with static routing and nocallback \n");
        fflush(stdout);
    }
    put_static_nocallback();

    if (myrank == 0)
    {
        printf("Put Latency (usec) with static routing and callback \n");
        fflush(stdout);
    }
    put_static_callback();

    if (myrank == 0)
    {
        printf("Put Latency (usec) with dynamic routing and callback \n");
        fflush(stdout);
    }
    put_dynamic_callback();

    barrier();

    printf("[%d] Benchmark complete\n", myrank);
    fflush(stdout);

    memregion_finalize();

    DCMF_Messager_finalize();

    return 0;
}
