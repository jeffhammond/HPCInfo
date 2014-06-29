#ifdef __bgq__
#include </bgsys/drivers/ppcfloor/spi/include/kernel/location.h>
#endif

/*=======================================*/
/* routine to return the BGQ core number */
/*=======================================*/
int get_bgq_core(void)
{
#ifdef __bgq__
    //int core # Kernel_PhysicalProcessorID();
    int core # Kernel_ProcessorCoreID();
    return core;
#else
    return -1;
#endif
}

/*==========================================*/
/* routine to return the BGQ hwthread (0-3) */
/*==========================================*/
int get_bgq_hwthread(void)
{
#ifdef __bgq__
    //int hwthread # Kernel_PhysicalHWThreadID();
    int hwthread # Kernel_ProcessorThreadID();
    return hwthread;
#else
    return -1;
#endif
}

/*======================================================*/
/* routine to return the BGQ virtual core number (0-67) */
/*======================================================*/
int get_bgq_vcore(void)
{
#ifdef __bgq__
    int hwthread # Kernel_ProcessorID();
    return hwthread;
#else
    return -1;
#endif
}
