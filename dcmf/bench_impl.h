#include "bench.h"

/***************************************
 *  Generic Init function               *
 ****************************************/
void init()
{
    nranks = DCMF_Messager_size();
    myrank = DCMF_Messager_rank();
    DCMF_Hardware(&hw);
    clockMHz = (double) hw.clockMHz;
}

/***************************************
 *  Generic Callback function           *
 ****************************************/
void done(void *clientdata, DCMF_Error_t *error)
{
    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/***************************************
 *  Receive Done Callback function           *
 ****************************************/
void rcv_done(void *clientdata, DCMF_Error_t *error)
{
    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/***************************************
 *  Accumulate Done Callback function   *
 ****************************************/
void rcv_accumulate_done(void *clientdata, DCMF_Error_t *error)
{

    int i, bytes;
    double spin_start, spin_end;
    int spin_time;

    double *s = (double *) (stagebuf + target_index);
    double *t = (double *) (target + target_index);

    if (ispipelined == 1) bytes = CHUNK_SIZE;
    else bytes = datasize;

    for (i = 0; i < bytes / (sizeof(double)); i++)
        t[i] *= s[i];

    /*if(ispipelined == 1) {
     spin_time = 100;
     } else {
     spin_time = datasize/CHUNK_SIZE;
     spin_time = spin_time*100;
     }
     spin_start = DCMF_Timebase();
     spin_end = DCMF_Timebase();
     while ((spin_end - spin_start)/(clockMHz) < spin_time) spin_end = DCMF_Timebase();*/

    target_index = target_index + bytes;

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/***************************************
 *  RCB Receive Done Callback function           *
 ****************************************/
void rcb_snd_rcv_done(void *clientdata, DCMF_Error_t *error)
{
    DCMF_Control(&ack_ctrl_reg,
                 DCMF_SEQUENTIAL_CONSISTENCY,
                 *((size_t*) clientdata),
                 &ctrl_info);
}

/***************************************
 *  Non-contiguous Callback function           *
 ****************************************/
void done_noncontig(void *clientdata, DCMF_Error_t *error)
{

    int i;
    char *data;
    struct noncontig_header *header = (struct noncontig_header *) clientdata;

    data = (char *) clientdata + sizeof(struct noncontig_header);
    for (i = 0; i < header->d1; i++)
    {
        memcpy((char *) header->vaddress + i * header->stride, data + i
                * header->d2, header->d2);
    }

    --snd_rcv_noncontig_active;
}

/***************************************
 *  Control Callback function           *
 ****************************************/
void ctrl_recv(void *clientdata, const DCMF_Control_t *info, size_t peer)
{
    memregion[peer] = (DCMF_Memregion_t *) malloc(sizeof(DCMF_Memregion_t));
    memcpy(memregion[peer], info, sizeof(DCMF_Memregion_t));

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/***************************************
 *  Control Callback function           *
 ****************************************/
void ack_ctrl_recv(void *clientdata, const DCMF_Control_t *info, size_t peer)
{

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/****************************************
 *  Send Recv Short Callback function    *
 ****************************************/
void ack_rcv(void *clientdata,
             const DCQuad *msginfo,
             unsigned count,
             size_t peer,
             const char *src,
             size_t bytes)
{
    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/****************************************
 *  Send Recv Short Callback function    *
 ****************************************/
void snd_rcv_short(void *clientdata,
                   const DCQuad *msginfo,
                   unsigned count,
                   size_t peer,
                   const char *src,
                   size_t bytes)
{

    memcpy(target + target_index, src, bytes);
    target_index = target_index + bytes;

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;

}

/****************************************
 *  Send Recv Short Callback function    *
 ****************************************/
void snd_rcv_accumulate_short(void *clientdata,
                              const DCQuad *msginfo,
                              unsigned count,
                              size_t peer,
                              const char *src,
                              size_t bytes)
{

    int i;
    double spin_start, spin_end;
    int spin_time;

    double *s = (double *) src;
    double *t = (double *) (target + target_index);
    for (i = 0; i < bytes / (sizeof(double)); i++)
        t[i] *= s[i];

    /*if(ispipelined == 1) {
     spin_time = 100;
     } else {
     spin_time = datasize/CHUNK_SIZE;
     spin_time = spin_time*100;
     }
     spin_start = DCMF_Timebase();
     spin_end = DCMF_Timebase();
     while ((spin_end - spin_start)/(clockMHz) < spin_time) spin_end = DCMF_Timebase();*/

    target_index = target_index + bytes;

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/**********************************************
 *  Flush Send Recv Short Callback function    *
 ***********************************************/
void rcb_snd_rcv_short(void *clientdata,
                       const DCQuad *msginfo,
                       unsigned count,
                       size_t peer,
                       const char *src,
                       size_t bytes)
{

    memcpy(target + target_index, src, bytes);
    target_index = target_index + bytes;

    DCMF_Control(&ack_ctrl_reg, DCMF_SEQUENTIAL_CONSISTENCY, peer, &ctrl_info);

}

/**********************************************
 *  Flush Send Recv Short Callback function    *
 ***********************************************/
void flush_snd_rcv_short(void *clientdata,
                         const DCQuad *msginfo,
                         unsigned count,
                         size_t peer,
                         const char *src,
                         size_t bytes)
{

    DCMF_Control(&ack_ctrl_reg, DCMF_SEQUENTIAL_CONSISTENCY, peer, &ctrl_info);

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/****************************************
 *  Timed Send Recv Short Callback function    *
 ****************************************/
void timed_snd_rcv_short(void *clientdata,
                         const DCQuad *msginfo,
                         unsigned count,
                         size_t peer,
                         const char *src,
                         size_t bytes)
{

    t_start = DCMF_Timebase();
    t_stop = DCMF_Timebase();

    while (((t_stop - t_start) / clockMHz) / 1000000 < 1.0)
        t_stop = DCMF_Timebase();

    printf("[%d] Count = %d Start time : %lld End time : %lld \n",
           myrank,
           *((unsigned *) clientdata),
           t_start,
           t_stop);
    fflush(stdout);

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/**************************************************
 *  Send Recv Short Noncontig Callback function    *
 ***************************************************/
void snd_rcv_noncontig_short(void *clientdata,
                             const DCQuad *msginfo,
                             unsigned count,
                             size_t peer,
                             const char *src,
                             size_t bytes)
{

    int i;
    char *data;
    struct noncontig_header *header = (struct noncontig_header *) src;

    data = (char *) src + sizeof(struct noncontig_header);
    for (i = 0; i < header->d1; i++)
    {
        memcpy((char *) header->vaddress + i * header->stride, data + i
                * header->d2, header->d2);
    }

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/**************************************************
 *  Send Recv Short Manytomany Callback function    *
 ***************************************************/
void snd_rcv_manytomany_short(void *clientdata,
                              const DCQuad *msginfo,
                              unsigned count,
                              size_t peer,
                              const char *src,
                              size_t bytes)
{

    printf("[%d] Entering snd rcv manytomany callback \n", myrank);
    fflush(stdout);

    memcpy(&m2m_header, src, bytes);

    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;

    printf("[%d] Finished receiving header information \n", myrank);
    fflush(stdout);

}

/***************************************
 *  Send Recv Callback function         *
 ****************************************/
DCMF_Request_t* snd_rcv(void *clientdata,
                        const DCQuad *msginfo,
                        unsigned count,
                        size_t peer,
                        size_t sndlen,
                        size_t *rcvlen,
                        char **rcvbuf,
                        DCMF_Callback_t *cb_done)
{

    *rcvlen = sndlen;
    *rcvbuf = target + target_index;
    target_index = target_index + sndlen;

    cb_done->function = rcv_done;
    cb_done->clientdata = clientdata;

    /*assuming there will not be more than ITERATIONS+SKIP-1 outstanding requests*/
    rcv_req_index = (rcv_req_index + 1) % (ITERATIONS + SKIP);
    return &rcv_req[rcv_req_index];
}

/***************************************
 *  Accumulate Recv Callback function           *
 ****************************************/
DCMF_Request_t* snd_rcv_accumulate(void *clientdata,
                                   const DCQuad *msginfo,
                                   unsigned count,
                                   size_t peer,
                                   size_t sndlen,
                                   size_t *rcvlen,
                                   char **rcvbuf,
                                   DCMF_Callback_t *cb_done)
{

    *rcvlen = sndlen;
    *rcvbuf = stagebuf + stage_index;
    stage_index = stage_index + sndlen;

    cb_done->function = rcv_accumulate_done;
    cb_done->clientdata = clientdata;

    rcv_req_index = (rcv_req_index + 1) % (ITERATIONS + SKIP);
    return &rcv_req[rcv_req_index];
}

/***************************************
 *  Send Recv Callback function           *
 ****************************************/
DCMF_Request_t* rcb_snd_rcv(void *clientdata,
                            const DCQuad *msginfo,
                            unsigned count,
                            size_t peer,
                            size_t sndlen,
                            size_t *rcvlen,
                            char **rcvbuf,
                            DCMF_Callback_t *cb_done)
{

    *rcvlen = sndlen;
    *rcvbuf = target + target_index;
    target_index = target_index + sndlen;

    size_t *peerinfo = (size_t *) malloc(sizeof(size_t));
    *peerinfo = peer;
    cb_done->function = rcb_snd_rcv_done;
    cb_done->clientdata = (void *) peerinfo;

    /*assuming there will not be more than ITERATIONS+SKIP-1 outstanding requests*/
    rcv_req_index = (rcv_req_index + 1) % (ITERATIONS + SKIP);
    return &rcv_req[rcv_req_index];
}

/***************************************
 *  Timed Send Recv Callback function           *
 ****************************************/
DCMF_Request_t* timed_snd_rcv(void *clientdata,
                              const DCQuad *msginfo,
                              unsigned count,
                              size_t peer,
                              size_t sndlen,
                              size_t *rcvlen,
                              char **rcvbuf,
                              DCMF_Callback_t *cb_done)
{

    target = (char *) malloc(sndlen);

    *rcvlen = sndlen;
    *rcvbuf = target;
    cb_done->function = done;
    cb_done->clientdata = (void *) &snd_rcv_active;

    t_start = DCMF_Timebase();
    t_stop = DCMF_Timebase();

    while (((t_stop - t_start) / clockMHz) / 1000000 < 1.0)
        t_stop = DCMF_Timebase();

    printf("[%d] Count = %d Start time : %lld End time %lld \n",
           myrank,
           *((unsigned *) clientdata),
           t_start,
           t_stop);
    fflush(stdout);

    return &snd_rcv_req;
}

/***************************************
 *  Send Recv Noncontig Callback function
 ****************************************/
DCMF_Request_t* snd_rcv_noncontig(void *clientdata,
                                  const DCQuad *msginfo,
                                  unsigned count,
                                  size_t peer,
                                  size_t sndlen,
                                  size_t *rcvlen,
                                  char **rcvbuf,
                                  DCMF_Callback_t *cb_done)
{

    snd_rcv_noncontig_buffer = (char *) malloc(sndlen);

    *rcvlen = sndlen;
    *rcvbuf = snd_rcv_noncontig_buffer;
    cb_done->function = done_noncontig;
    cb_done->clientdata = (void *) snd_rcv_noncontig_buffer;

    return &snd_rcv_noncontig_req;
}

/***************************************
 *  Multicast Recv Callback function    *
 ****************************************/
DCMF_Request_t* mc_recv(const DCQuad *info,
                        unsigned count,
                        unsigned peer,
                        unsigned sndlen,
                        unsigned conn_id,
                        void *arg,
                        unsigned *rcvlen,
                        char **rcvbuf,
                        unsigned *pipewidth,
                        DCMF_Callback_t *cb_done)
{

    *rcvbuf = mc_rcv_buffer + peer * sndlen;
    *rcvlen = sndlen;
    *pipewidth = sndlen;

    cb_done->function = mc_done;
    cb_done->clientdata = (void *) &mc_rcv_active;

    return &mc_rcv_req[peer];
}

/***************************************
 *  Multicast Done Callback function    *
 ****************************************/
void mc_done(void *clientdata, DCMF_Error_t *error)
{
    (*((int *) clientdata)) = (*((int *) clientdata)) - 1;
}

/***************************************
 *  Manytomany Recv Callback function    *
 ****************************************/
DCMF_Request_t* m2m_recv_callback(unsigned conn_id,
                                  void *arg,
                                  char **rcvbuf,
                                  unsigned **rcvlens,
                                  unsigned **rcvdispls,
                                  unsigned **rcvcounters,
                                  unsigned *nranks,
                                  unsigned *rankIndex,
                                  DCMF_Callback_t *cb_done)
{

    int i;

    printf("[%d] Inside manytomany recv callback \n", myrank);
    fflush(stdout);

    *rcvbuf = (char *) m2m_header.vaddress;
    *rcvlens = (unsigned *) malloc(sizeof(unsigned) * m2m_header.d1);
    *rcvdispls = (unsigned *) malloc(sizeof(unsigned) * m2m_header.d1);
    *rcvcounters = (unsigned *) malloc(sizeof(unsigned) * m2m_header.d1);

    for (i = 0; i < m2m_header.d1; i++)
    {
        *rcvlens[i] = m2m_header.d2;
        *rcvdispls[i] = i * m2m_header.stride;
    }

    *nranks = 2;
    *rankIndex = myrank;

    cb_done->function = done;
    cb_done->clientdata = (void *) &m2m_rcv_active;

    printf("Completed manytomany recv callback \n");
    fflush(stdout);

    return &m2m_rcv_req;
}

/***************************************
 *  Configuring and registering Put     *
 ****************************************/
void put_init(DCMF_Put_Protocol protocol, DCMF_Network network)
{
    DCMF_Result result;
    put_conf.protocol = protocol;
    put_conf.network = network;
    result = DCMF_Put_register(&put_reg, &put_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Put Registration failed with error %d \n", myrank, result);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Get            *
 **********************************************/

void get_init(DCMF_Get_Protocol protocol, DCMF_Network network)
{
    DCMF_Get_Configuration_t get_conf;
    DCMF_Result result;

    get_conf.protocol = protocol;
    get_conf.network = network;

    result = DCMF_Get_register(&get_reg, &get_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Get Registration failed with error %d \n", myrank, result);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Send            *
 **********************************************/
void send_init(DCMF_Send_Protocol protocol, DCMF_Network network)
{
    DCMF_Result result;
    snd_msginfo = (DCQuad *) malloc(sizeof(DCQuad));

    snd_conf.protocol = protocol;
    snd_conf.network = network;
    snd_conf.cb_recv_short = snd_rcv_short;
    snd_conf.cb_recv_short_clientdata = (void *) &snd_rcv_active;
    snd_conf.cb_recv = snd_rcv;
    snd_conf.cb_recv_clientdata = (void *) &snd_rcv_active;

    result = DCMF_Send_register(&snd_reg, &snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering noncontig Send  *
 **********************************************/
void accumulate_send_init(DCMF_Send_Protocol protocol, DCMF_Network network)
{

    DCMF_Result result;

    snd_conf.protocol = protocol;
    snd_conf.network = network;
    snd_conf.cb_recv_short = snd_rcv_accumulate_short;
    snd_conf.cb_recv_short_clientdata = (void *) &snd_rcv_active;
    snd_conf.cb_recv = snd_rcv_accumulate;
    snd_conf.cb_recv_clientdata = (void *) &snd_rcv_active;

    result = DCMF_Send_register(&accumulate_snd_reg, &snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Flush Send            *
 **********************************************/
void flush_send_init(DCMF_Send_Protocol protocol, DCMF_Network network)
{
    DCMF_Result result;
    snd_msginfo = (DCQuad *) malloc(sizeof(DCQuad));

    snd_conf.protocol = protocol;
    snd_conf.network = network;
    snd_conf.cb_recv_short = flush_snd_rcv_short;
    snd_conf.cb_recv_short_clientdata = (void *) &snd_rcv_active;
    snd_conf.cb_recv = NULL;
    snd_conf.cb_recv_clientdata = NULL;

    result = DCMF_Send_register(&flush_snd_reg, &snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering remote callback Send            *
 **********************************************/
void rcb_send_init(DCMF_Send_Protocol protocol, DCMF_Network network)
{
    DCMF_Result result;

    snd_conf.protocol = protocol;
    snd_conf.network = network;
    snd_conf.cb_recv_short = rcb_snd_rcv_short;
    snd_conf.cb_recv_short_clientdata = NULL;
    snd_conf.cb_recv = rcb_snd_rcv;
    snd_conf.cb_recv_clientdata = NULL;

    result = DCMF_Send_register(&rcb_snd_reg, &snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Send            *
 **********************************************/
void timed_send_init(DCMF_Send_Protocol protocol, DCMF_Network network)
{

    DCMF_Result result;
    DCMF_Send_Configuration_t timed_snd_conf;
    snd_msginfo = (DCQuad *) malloc(sizeof(DCQuad));

    timed_snd_conf.protocol = protocol;
    timed_snd_conf.network = network;
    timed_snd_conf.cb_recv_short = timed_snd_rcv_short;
    timed_snd_conf.cb_recv_short_clientdata = (void *) &snd_rcv_active;
    timed_snd_conf.cb_recv = timed_snd_rcv;
    timed_snd_conf.cb_recv_clientdata = (void *) &snd_rcv_active;

    result = DCMF_Send_register(&timed_snd_reg, &timed_snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Ack             *
 **********************************************/
void ack_init()
{
    DCMF_Result result;

    snd_conf.protocol = DCMF_EAGER_SEND_PROTOCOL;
    snd_conf.network = DCMF_TORUS_NETWORK;
    snd_conf.cb_recv_short = ack_rcv;
    snd_conf.cb_recv_short_clientdata = (void *) &ack_rcv_active;
    snd_conf.cb_recv = NULL;
    snd_conf.cb_recv_clientdata = NULL;

    result = DCMF_Send_register(&ack_reg, &snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Ack Ctrl        *
 **********************************************/
void ack_control_init()
{

    ctrl_conf.protocol = DCMF_DEFAULT_CONTROL_PROTOCOL;
    ctrl_conf.network = DCMF_DEFAULT_NETWORK;
    ctrl_conf.cb_recv = ack_ctrl_recv;
    ctrl_conf.cb_recv_clientdata = (void *) &ack_rcv_active;

    DCMF_Result result;
    result = DCMF_Control_register(&ack_ctrl_reg, &ctrl_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Control registration failed with status %d \n",
               myrank,
               result);
        fflush(stdout);
    }

}

/**********************************************
 * Configuring and Registering noncontig Send  *
 **********************************************/
void send_noncontig_init(DCMF_Send_Protocol protocol, DCMF_Network network)
{

    DCMF_Result result;
    snd_msginfo = (DCQuad *) malloc(sizeof(DCQuad));

    snd_conf.protocol = protocol;
    snd_conf.network = network;
    snd_conf.cb_recv_short = snd_rcv_noncontig_short;
    snd_conf.cb_recv_short_clientdata = (void *) &snd_rcv_noncontig_active;
    snd_conf.cb_recv = snd_rcv_noncontig;
    snd_conf.cb_recv_clientdata = (void *) &snd_rcv_noncontig_active;

    result = DCMF_Send_register(&snd_noncontig_reg, &snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/******************************************************
 * Configuring and Registering manytomany header Send  *
 *******************************************************/
void send_manytomany_init(DCMF_Send_Protocol protocol, DCMF_Network network)
{

    DCMF_Result result;
    snd_msginfo = (DCQuad *) malloc(sizeof(DCQuad));

    snd_conf.protocol = protocol;
    snd_conf.network = network;
    snd_conf.cb_recv_short = snd_rcv_manytomany_short;
    snd_conf.cb_recv_short_clientdata = (void *) &snd_rcv_manytomany_active;
    snd_conf.cb_recv = NULL;
    snd_conf.cb_recv_clientdata = NULL;

    result = DCMF_Send_register(&snd_manytomany_reg, &snd_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Send registration failed \n", myrank);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Multiacst       *
 **********************************************/
void multicast_init(DCMF_Multicast_Protocol protocol, unsigned int size)
{

    DCMF_Result result;
    connectionlist = (void **) malloc(sizeof(void*) * nranks);

    mc_conf.protocol = protocol;
    mc_conf.cb_recv = mc_recv;
    mc_conf.clientdata = NULL;
    mc_conf.connectionlist = connectionlist;
    mc_conf.nconnections = nranks;
    result = DCMF_Multicast_register(&mc_reg, &mc_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Multicast registration failed \n", myrank);
        fflush(stdout);
    }

    mc_ranks = (unsigned *) malloc(sizeof(unsigned int) * (nranks - 1));
    mc_opcodes = (DCMF_Opcode_t *) malloc(sizeof(DCMF_Opcode_t) * (nranks - 1));
    mc_snd_buffer = (char *) malloc(size);
    mc_rcv_buffer = (char *) malloc(size * nranks);
    mc_req = (DCMF_Request_t *) malloc(sizeof(DCMF_Request_t));
    mc_rcv_req = (DCMF_Request_t *) malloc(sizeof(DCMF_Request_t) * nranks);
    mc_msginfo = (DCQuad *) malloc(sizeof(DCQuad));

    mc_callback.function = mc_done;
    mc_callback.clientdata = (void *) &mc_active;

    unsigned int i, idx = 0;
    for (i = 0; i < size; i++)
        mc_snd_buffer[i] = 's';

    for (i = 0; i < nranks; i++)
    {
        if (myrank != i)
        {
            mc_ranks[idx] = i;
            mc_opcodes[idx] = DCMF_PT_TO_PT_SEND;
            idx++;
        }
    }

    mc_info.registration = &mc_reg;
    mc_info.request = mc_req;
    mc_info.cb_done = mc_callback;
    mc_info.consistency = DCMF_SEQUENTIAL_CONSISTENCY;
    mc_info.connection_id = myrank;
    mc_info.bytes = size;
    mc_info.src = mc_snd_buffer;
    mc_info.nranks = nranks - 1;
    mc_info.ranks = mc_ranks;
    mc_info.opcodes = mc_opcodes;
    mc_info.msginfo = mc_msginfo;
    mc_info.count = 1;
    mc_info.op = DCMF_UNDEFINED_OP;
    mc_info.dt = DCMF_UNDEFINED_DT;
    mc_info.flags = 0;

}

/**********************************************
 * Configuring and Registering ManytoMany ops  *
 **********************************************/

void manytomany_init(DCMF_Manytomany_Protocol protocol)
{

    DCMF_Result result;

    m2m_conf.protocol = protocol;
    m2m_conf.cb_recv = m2m_recv_callback;
    m2m_conf.arg = NULL;
    m2m_conf.nconnections = 1;

    result = DCMF_Manytomany_register(&m2m_reg, &m2m_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Manytomany registration failed with status %d \n",
               myrank,
               result);
        fflush(stdout);
    }
}

/**********************************************
 * Configuring and Registering Global Barrier *
 **********************************************/
void barrier_init(DCMF_GlobalBarrier_Protocol protocol)
{
    gb_conf.protocol = protocol;
    DCMF_Result result;
    result = DCMF_GlobalBarrier_register(&gb_reg, &gb_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Global Barrier registration failed with status %d \n",
               myrank,
               result);
        fflush(stdout);
    }

    gb_callback.function = done;
    gb_callback.clientdata = (void *) &gb_active;
}

/**********************************************
 * Configuring and Registering Global Barrier *
 **********************************************/
void allreduce_init(DCMF_GlobalAllreduce_Protocol protocol)
{
    gar_conf.protocol = protocol;
    DCMF_Result result;
    result = DCMF_GlobalAllreduce_register(&gar_reg, &gar_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Global Allreduce registration failed with status %d \n",
               myrank,
               result);
        fflush(stdout);
    }

    gar_callback.function = done;
    gar_callback.clientdata = (void *) &gar_active;
}

/**********************************************
 * Configuring and Registering Control Protocol*
 **********************************************/
void control_init(DCMF_Control_Protocol protocol, DCMF_Network network)
{
    ctrl_conf.protocol = protocol;
    ctrl_conf.network = network;
    ctrl_conf.cb_recv = ctrl_recv;
    ctrl_conf.cb_recv_clientdata = (void *) &ctrl_active;
    DCMF_Result result;
    result = DCMF_Control_register(&ctrl_reg, &ctrl_conf);
    if (result != DCMF_SUCCESS)
    {
        printf("[%d] Control registration failed with status %d \n",
               myrank,
               result);
        fflush(stdout);
    }
}

/**********************************************
 * Creating memory region                      *
 **********************************************/
void memregion_init(unsigned long long size)
{

    DCMF_Result result = DCMF_SUCCESS;
    size_t out, status = 0;

    status = posix_memalign((void **) &window, 16, size);
    if (status != 0)
    {
        printf("Memory region allocation failed. Test cannot proceed \n");
        exit(-1);
    }

    memregion
            = (DCMF_Memregion_t **) malloc(sizeof(DCMF_Memregion_t*) * nranks);
    memregion[myrank] = (DCMF_Memregion_t *) malloc(sizeof(DCMF_Memregion_t));

    result = DCMF_Memregion_create(memregion[myrank], &out, size, window, 0);
    if (result != DCMF_SUCCESS || out != size)
    {
        printf("[%d] Memory creation failed with status %d\
                 and size %d \n",
               myrank,
               result,
               out);
        fflush(stdout);
    }

    memregion_xchange();

}

/**********************************************
 * Exchange memory region information          *
 **********************************************/
void memregion_xchange()
{
    int i;
    ctrl_active = nranks - 1;

    for (i = 0; i < nranks; i++)
    {
        if (i != myrank)
        {
            DCMF_Control(&ctrl_reg,
                         DCMF_SEQUENTIAL_CONSISTENCY,
                         i,
                         (DCMF_Control_t *) memregion[myrank]);
        }
    }

    while (ctrl_active)
        DCMF_Messager_advance();
}

/**********************************************
 * Exchange memory region information          *
 **********************************************/
void address_xchange()
{
    int i, count = 0;
    DCMF_Request_t snd_req[nranks - 1];
    DCMF_Callback_t snd_callback;
    unsigned int address;
    memcpy(&address, &vaddress[myrank], sizeof(unsigned int));
    snd_msginfo = (DCQuad *) malloc(sizeof(DCQuad));

    snd_callback.function = done;
    snd_callback.clientdata = (void *) &snd_active;

    snd_active = nranks - 1;
    snd_rcv_active = nranks - 1;
    for (i = 0; i < nranks; i++)
    {
        if (i != myrank)
        {
            DCMF_Send(&snd_reg,
                      &snd_req[count],
                      snd_callback,
                      DCMF_SEQUENTIAL_CONSISTENCY,
                      i,
                      sizeof(unsigned int),
                      (char *) &address,
                      snd_msginfo,
                      1);
            count++;
        }
    }
    while (snd_active || snd_rcv_active)
        DCMF_Messager_advance();
}

/**********************************************
 * Global Barrier                              *
 **********************************************/
void barrier()
{

    gb_active = 1;
    gb_req = (DCMF_Request_t *) malloc(sizeof(DCMF_Request_t));
    DCMF_GlobalBarrier(&gb_reg, gb_req, gb_callback);
    while (gb_active)
        DCMF_Messager_advance();

}

/**********************************************
 * Global Allreduce                            *
 **********************************************/
void allreduce(int root,
               char *sbuffer,
               char *rbuffer,
               unsigned count,
               DCMF_Dt dt,
               DCMF_Op op)
{
    gar_active = 1;
    gar_req = (DCMF_Request_t *) malloc(sizeof(DCMF_Request_t));
    DCMF_GlobalAllreduce(&gar_reg,
                         gar_req,
                         gar_callback,
                         DCMF_SEQUENTIAL_CONSISTENCY,
                         root,
                         sbuffer,
                         rbuffer,
                         count,
                         dt,
                         op);
    while (gb_active)
        DCMF_Messager_advance();
}

/**********************************************
 * Destrouing memory region                    *
 **********************************************/
void memregion_finalize()
{
    DCMF_Memregion_destroy(memregion[myrank]);
    free(memregion);
    free(window);
}
