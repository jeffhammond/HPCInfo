#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MKL
#include "mkl.h"
#else
#include "cblas.h"
#endif

#if (__STDC_VERSION__ < 199901L)
#error You need to enable C99 or remove vararg debug printing by hand.
#endif

#ifdef DEBUG
#define PDEBUG(fmt, ...) do { printf("DEBUG: "); printf(fmt, __VA_ARGS__); } while (0)
#else
#define PDEBUG(fmt, ...)
#endif

int main(int argc, char* argv[])
{
    char type;
    int iterations;
    int m, n, k;

    if (argc<2) {
        goto input_error;
    }

    for (int a=1; a<argc; ++a) {
        PDEBUG("argv[%d]=\"%s\"\n",a,argv[a]);
        char copy[64] = {0};
        if (strlen(argv[a])>63) {
            printf("argument %s is too long\n", argv[a]);
            goto input_error;
        }
        strncpy(copy,argv[a],strlen(argv[a]));
        PDEBUG("copy=\"%s\"\n",copy);
        char * key = strtok(copy," -=");
        char * value;
        PDEBUG("key=\"%s\"\n",key);
        if (key[0]=='t') {
            PDEBUG("%s","type argument detected\n");
            value = strtok(NULL," -=");
            PDEBUG("value=\"%s\"\n",value);
            type = value[0];
        } else if (key[0]=='s') {
            PDEBUG("%s","size argument detected\n");
            value = strtok(NULL," -=,");
            PDEBUG("value=\"%s\"\n",value);
            if (value!=NULL) {
                m = n = k = atoi(value);
            } else {
                printf("size argument missing\n");
                goto input_error;
            }
        } else if (key[0]=='d') {
            PDEBUG("%s","size argument detected\n");
            value = strtok(NULL," -=,");
            PDEBUG("value=\"%s\"\n",value);
            if (value!=NULL) {
                m = atoi(value);
            } else {
                printf("first dims argument missing\n");
                goto input_error;
            }
            value = strtok(NULL," -=,");
            PDEBUG("value=\"%s\"\n",value);
            if (value!=NULL) {
                n = atoi(value);
            } else {
                printf("second dims argument missing\n");
                goto input_error;
            }
            value = strtok(NULL," -=,");
            PDEBUG("value=\"%s\"\n",value);
            if (value!=NULL) {
                k = atoi(value);
            } else {
                printf("third dims argument missing\n");
                goto input_error;
            }
        }
    }

    if (type=='s') {
        printf("single precision\n");
    } else if (type=='d') {
        printf("double precision\n");
    } else if (type=='c') {
        printf("complex single precision\n");
    } else if (type=='z') {
        printf("complex double precision\n");
    } else {
        printf("You have chosen an unsupported datatype...\n");
        goto input_error;
    }

    printf("SUCCESS\n");
    return 0;

    input_error:
        printf("Options:\n");
        printf(" --type=[sdcz] (trailing characters ignored)\n");
        printf(" --size=n ~or~ --dims=m,n,k (m,n,k are integers)\n");
        printf("Your input was:\n");
        for (int a=0; a<argc; ++a) {
            printf("%s ",argv[a]);
        }
        printf("\n");
        goto fail;

    fail:
        return 1;
}
