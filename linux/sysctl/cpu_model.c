#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

int main(void)
{
#ifdef __APPLE__
    int rc;
    size_t len = 1024;
    char buf[1024] = {0};
    rc = sysctlbyname("machdep.cpu.brand_string", &buf,  &len, NULL, 0);
    printf("len=%zu\n", len);
    printf("buf=%s\n", buf);
#endif
    return 0;
}
