#include "dcmf.h"
#include "dcmf_globalcollectives.h"
#include "dcmf_multisend.h"

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#define ITERATIONS 100 
#define SKIP 10
#define CHUNK_SIZE 8*1024 
#define MAX_MSG_SIZE 4*1024*1024
#define VECTOR 16 
#define MAX_BUF_SIZE MAX_MSG_SIZE*(ITERATIONS+SKIP)

#define MAX_DIM 1024
#define ITERS 1

/***************************************
*  Header for noncontig transfers      *
***************************************/

struct noncontig_header {
   void *vaddress;
   int dist;
   int stride;
   int d1;
   int d2;
   char flag;
};

struct rbuf_info {
   void *rbuf;
   void *active;
};

/***************************************
*  Global Helper variables             *
***************************************/
int nranks, myrank;
volatile unsigned long long t_start, t_stop;
volatile double t_sec, t_usec, t_msec;
volatile double t_sec1, t_usec1, t_msec1;
volatile double t_sec2, t_usec2, t_msec2;
volatile double t_max, t_avg, t_min;
volatile double t_max1, t_avg1, t_min1;
volatile double t_max2, t_avg2, t_min2;
volatile unsigned long long t_cycles, t_cycles1, t_avg_cycles, t_avg_cycles1;
volatile unsigned long long t_cycles2, t_avg_cycles2;
volatile double clockMHz, bw, bw_avg; 
DCMF_Hardware_t hw;

/***************************************
*  Global DCMF structures for Window   *
***************************************/
DCMF_Memregion_t **memregion;
void **vaddress;
char *window;

/**********************************************
* Global DCMF structures for Put              *
**********************************************/
DCMF_Put_Configuration_t put_conf;
DCMF_Protocol_t put_reg;
DCMF_Callback_t put_done, put_ack;
int put_count, ack_count;

/**********************************************
* Global DCMF structures for Get              *
**********************************************/
DCMF_Put_Configuration_t get_conf;
DCMF_Protocol_t get_reg;
DCMF_Callback_t get_done, get_ack;
int get_count;

/**********************************************
* Global DCMF structures for Send      *
**********************************************/
DCMF_Send_Configuration_t snd_conf;
DCMF_Callback_t snd_callback;
DCMF_Protocol_t snd_reg;
DCMF_Request_t snd_rcv_req, rcv_req[ITERATIONS+SKIP];
volatile int snd_rcv_active, snd_active;
volatile int target_index, stage_index, rcv_req_index;
volatile int ispipelined;
DCQuad *snd_msginfo;
char *source, *target, *stagebuf;

/**********************************************************
* Global DCMF structures for Send with remote callback     *
***********************************************************/
DCMF_Protocol_t accumulate_snd_reg;
int datasize;

/**********************************************************
* Global DCMF structures for Send with remote callback     *
***********************************************************/
DCMF_Protocol_t rcb_snd_reg;

/**********************************************
* Global DCMF structures for Flush Send      *
**********************************************/
DCMF_Protocol_t flush_snd_reg;

/**********************************************
* Global DCMF structures for Timed Send      *
**********************************************/
DCMF_Protocol_t timed_snd_reg;

/**********************************************
* Global DCMF structures for Noncontig Send      *
**********************************************/
DCMF_Protocol_t snd_noncontig_reg;
DCMF_Request_t snd_rcv_noncontig_req;
volatile int snd_rcv_noncontig_active;
char *snd_rcv_noncontig_buffer;

/**********************************************
* Global DCMF structures for Manytomany Send      *
**********************************************/
DCMF_Protocol_t snd_manytomany_reg;
DCMF_Request_t snd_rcv_manytomany_req;
volatile int snd_rcv_manytomany_active;
char *snd_rcv_manytomany_buffer;

/**********************************************
* Global DCMF structures for Acked Send      *
**********************************************/
DCMF_Protocol_t acked_snd_reg;
DCMF_Request_t acked_snd_rcv_req;
volatile int acked_snd_rcv_active;
char *acked_snd_rcv_buffer;
char ack;

/**********************************************
* Global DCMF structures for Ack              *
**********************************************/
DCMF_Protocol_t ack_reg, ack_ctrl_reg;
DCMF_Request_t ack_rcv_req;
volatile int ack_rcv_active;

/**********************************************
* Global DCMF structures for Multisend      *
**********************************************/

DCMF_Multicast_Configuration_t mc_conf;
DCMF_Multicast_t mc_info;
DCMF_MulticastRecv_t mc_rcv_info;
DCMF_Protocol_t mc_reg;
DCMF_Request_t *mc_req, *mc_rcv_req;
DCMF_Callback_t mc_callback, mc_rcv_callback;
DCMF_Opcode_t *mc_opcodes;
unsigned int *mc_ranks;
volatile int mc_active, mc_rcv_active;
void **connectionlist;
char *mc_rcv_buffer, *mc_snd_buffer;
DCQuad *mc_msginfo;

/**********************************************
* Global DCMF structures for Many2Many        *
**********************************************/

DCMF_Manytomany_Configuration_t m2m_conf;
DCMF_Request_t m2m_req, m2m_rcv_req;
DCMF_Protocol_t m2m_reg;
DCMF_Callback_t m2m_callback, m2m_rcv_callback;
struct noncontig_header m2m_header;
volatile int m2m_rcv_active, m2m_active;
unsigned rankindex;

/***************************************
*  Global DCMF structures for Barrier  *
****************************************/
DCMF_GlobalBarrier_Configuration_t gb_conf;
DCMF_Protocol_t gb_reg;
DCMF_Request_t *gb_req;
DCMF_Callback_t gb_callback;
volatile int gb_active;

/***************************************
*  Global DCMF structures for Allreduce  *
****************************************/
DCMF_GlobalAllreduce_Configuration_t gar_conf;
DCMF_Protocol_t gar_reg;
DCMF_Request_t *gar_req;
DCMF_Callback_t gar_callback;
volatile int gar_active;

/***************************************
*  Global DCMF structures for Control  *
****************************************/
DCMF_Control_Configuration_t ctrl_conf;
DCMF_Protocol_t ctrl_reg;
DCMF_Control_t ctrl_info;
DCMF_Request_t ctrl_req;
DCMF_Callback_t ctrl_callback;
volatile int ctrl_active;

/***************************************
*  Generic init                        *
****************************************/
void init();

/***************************************
*  Generic Callback function           *
****************************************/
void done(void *, DCMF_Error_t *); 

/***************************************
*  Control Callback function           *
****************************************/
void ctrl_recv(void *, const DCMF_Control_t *, size_t); 

/***************************************
*  Multicast Recv Callback function         *
****************************************/
DCMF_Request_t* mc_recv(const DCQuad *, unsigned, unsigned, unsigned,\
             unsigned , void *, unsigned *, char **,\
             unsigned *, DCMF_Callback_t *);

/***************************************
*  Multicast Done Callback function    *
****************************************/
void mc_done(void *, DCMF_Error_t *); 

/***************************************
*  Configuring and registering Put     *
****************************************/
void put_init (DCMF_Put_Protocol, DCMF_Network);

/***************************************
*  Configuring and registering Get     *
****************************************/
void get_init (DCMF_Get_Protocol, DCMF_Network);

/****************************************
* Configuring and Registering Send *
*****************************************/
void send_init(DCMF_Send_Protocol, DCMF_Network);

/****************************************
* Configuring and Registering Send *
*****************************************/
void accumulate_send_init(DCMF_Send_Protocol, DCMF_Network);

/****************************************
* Configuring and Registering Send with Remote Callback*
*****************************************/
void rcb_send_init(DCMF_Send_Protocol, DCMF_Network);

/**********************************************
* Configuring and Registering Flush Send      *
**********************************************/
void flush_send_init(DCMF_Send_Protocol, DCMF_Network); 

/****************************************
* Configuring and Registering Timed Send *
*****************************************/
void timed_send_init(DCMF_Send_Protocol, DCMF_Network);

/*****************************************
* Configuring and Registering Acked Send *
******************************************/
void acked_send_init();
void ack_init();

/**********************************************
* Configuring and Registering Ack Ctrl        *
**********************************************/
void ack_control_init();

/****************************************
* Configuring and Registering Send Noncontig *
*****************************************/
void send_noncontig_init(DCMF_Send_Protocol, DCMF_Network);

/******************************************************
* Configuring and Registering manytomany header Send  *
*******************************************************/
void send_manytomany_init(DCMF_Send_Protocol, DCMF_Network);

/**********************************************
* Configuring and Registering ManytoMany ops  *
**********************************************/
void manytomany_init(DCMF_Manytomany_Protocol); 

/****************************************
* Configuring and Registering Multicast *
*****************************************/
void multicast_init(DCMF_Multicast_Protocol, unsigned int size);

/*********************************************
* Configuring and Registering Global Barrier *
**********************************************/
void barrier_init(DCMF_GlobalBarrier_Protocol); 

/*********************************************
* Configuring and Registering Global Allreduce*
**********************************************/
void allreduce_init(DCMF_GlobalAllreduce_Protocol);

/**********************************************
* Configuring and Registering Control Protocol*
**********************************************/
void control_init(DCMF_Control_Protocol, DCMF_Network);

/**********************************************
* Creating memory region                      *
**********************************************/
void memregion_init(unsigned long long);

/**********************************************
* Exchange memory region information          *
**********************************************/
void memregion_xchange();

/**********************************************
* Exchange memory region information          *
**********************************************/
void address_xchange();

/**********************************************
* Global Barrier                              *
**********************************************/
void barrier();

/**********************************************
* Global Allreduce                            *
**********************************************/
void allreduce(int, char *, char *, unsigned, DCMF_Dt, DCMF_Op);

/**********************************************
* Destroy memory region                       *
**********************************************/
void memregion_finalize();
