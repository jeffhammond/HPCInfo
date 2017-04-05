#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/sysctl.h>

int main(void)
{
    int rc;
    size_t len = 1024;
    char buf[1024] = {0};
    rc = sysctlbyname("machdep.cpu.brand_string", &buf,  &len, NULL, 0);
    printf("len=%zu\n", len);
    printf("buf=%s\n", buf);
    return 0;
}
