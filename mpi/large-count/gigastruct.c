#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char * argv[])
{
    const long n = (argc > 1) ? atol(argv[1]) : 100;
    FILE * f = fopen("temp.c","w+");
    if (!f) printf("oh shit\n");
    fprintf(f,"#include <stdio.h>\n"
              "#include <stdint.h>\n"
              "typedef struct {\n");
    for (long i=0; i<n; i++) {
        fprintf(f,"  int8_t  a%ld;\n", i);
    }
    fprintf(f,"} giga_s;\n\n"
              "int main(void) {\n"
              "  printf(\"sizeof=%%zu\\n\",sizeof(giga_s));\n"
              "  return 0;\n"
              "}\n");
    int rc = fclose(f);
    if (rc) printf("oh shit\n");
    return 0;
}
