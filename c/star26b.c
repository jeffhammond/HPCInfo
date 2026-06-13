#include <stdio.h>
#include <stdbool.h>

/***************************************

                    a
                   / \
              b---c---d---e
               \ /     \ /
                f       g
               / \     / \
              h---i---j---k
                   \ /
                    l

****************************************/

int main(void)
{
    int n = 0;
    for (int b=1; b<=12 ; b++)
     for (int c=1; c<=12; c++)
      if (c!=b)
       for (int d=1; d<=12; d++)
        if (d!=c && d!=b)
         for (int e=1; e<=12; e++)
          if (e!=d && e!=c && e!=b)
           if ((b+c+d+e)==26) // top row
            for (int a=1; a<=12; a++)
             if (a!=b && a!=c && a!=d && a!=e)
              for (int f=1; f<=12; f++)
               if (f!=e && f!=d && f!=c && f!=b && f!=a)
                for (int h=1; h<=12; h++)
                 if ((a+c+f+h)==26) // left left down
                  if (h!=f && h!=e && h!=d && h!=c && h!=b && h!=a)
                   for (int g=1; g<=12; g++)
                    if (g!=h && g!=f && g!=e && g!=d && g!=c && g!=b && g!=a)
                     for (int k=1; k<=12; k++)
                      if (k!=h && k!=g && k!=f && k!=e && k!=d && k!=c && k!=b && k!=a)
                       if ((a+d+g+k)==26) // right right down
                        for (int i=1; i<=12; i++)
                         if (i!=k && i!=h && i!=g && i!=f && i!=e && i!=d && i!=c && i!=b && i!=a)
                          for (int j=1; j<=12; j++)
                           if (j!=k && j!=i && j!=h && j!=g && j!=f && j!=e && j!=d && j!=c && j!=b && j!=a)
                            if ((h+i+j+k)==26) // bottom row
                             for (int l=1; l<=12; l++)
                              if (l!=k && l!=j && l!=i && l!=h && l!=g && l!=f && l!=e && l!=d && l!=c && l!=b && l!=a) 
                               if ((e+g+j+l)==26) // right left down
                                if ((b+f+i+l)==26) // left right down
                                 if ((a+e+k+l+h+b)==26) // outer vertices condition
                                  {
#if 1
                                   fprintf(stderr,
                                           "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d\n",
                                           a,b,c,d,e,f,g,h,i,j,k,l,"pass",n);
#endif
                                   n++;
                                  }
    printf("%d\n",n);
    return 0;
}

