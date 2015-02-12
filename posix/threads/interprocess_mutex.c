#include <pthread.h>

pthread_mutex_t shm_mutex;

int main(void)
{
    int err;
    pthread_mutexattr_t attr;
    err = pthread_mutexattr_init(&attr); if (err) return err;
    err = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED); if (err) return err;
    err = pthread_mutex_init(&shm_mutex, &attr); if (err) return err;
    err = pthread_mutexattr_destroy(&attr); if (err) return err;
    err = pthread_mutex_lock(&shm_mutex); if (err) return err;
    err = pthread_mutex_unlock(&shm_mutex); if (err) return err;
    err = pthread_mutex_destroy(&shm_mutex); if (err) return err;
    return 0;
}
