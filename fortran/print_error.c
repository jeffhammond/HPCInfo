#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ISO_Fortran_binding.h>

void print_error(int code, CFI_cdesc_t * message)
{
    char * buffer = calloc(message -> elem_len + 1, 1);
    memcpy(buffer, message -> base_addr, message -> elem_len);
    printf("code=%d, message=%s\n", code, buffer);
}
