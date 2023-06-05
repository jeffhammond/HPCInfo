#include <iostream>

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

bool pass(int a, int b, int c,
          int d, int e, int f,
          int g, int h, int i,
          int j, int k, int l)
{
    return ( ((b+c+d+e)==26) && // top row
             ((h+i+j+k)==26) && // bottom row
             ((a+c+f+h)==26) && // left left down
             ((e+g+j+l)==26) && // right left down
             ((b+f+i+l)==26) && // left right down
             ((a+d+g+k)==26));  // right right down
}

int main(void)
{
    int n = 0;
    for (int a=1; a<=12; a++)
     for (int b=1; b<=12 ; b++)
      if (b!=a)
       for (int c=1; c<=12; c++)
        if (c!=b && c!=a)
         for (int d=1; d<=12; d++)
          if (d!=c && d!=b && d!=a)
           for (int e=1; e<=12; e++)
            if (e!=d && e!=c && e!=b && e!=a)
             for (int f=1; f<=12; f++)
              if (f!=e && f!=d && f!=c && f!=b && f!=a)
               for (int g=1; g<=12; g++)
                if (g!=f && g!=e && g!=d && g!=c && g!=b && g!=a)
                 for (int h=1; h<=12; h++)
                  if (h!=g && h!=f && h!=e && h!=d && h!=c && h!=b && h!=a)
                   for (int i=1; i<=12; i++)
                    if (i!=h && i!=g && i!=f && i!=e && i!=d && i!=c && i!=b && i!=a)
                     for (int j=1; j<=12; j++)
                      if (j!=i && j!=h && j!=g && j!=f && j!=e && j!=d && j!=c && j!=b && j!=a)
                       for (int k=1; k<=12; k++)
                        if (k!=j && k!=i && k!=h && k!=g && k!=f && k!=e && k!=d && k!=c && k!=b && k!=a)
                         for (int l=1; l<=12; l++)
                          if (l!=k && l!=j && l!=i && l!=h && l!=g && l!=f && l!=e && l!=d && l!=c && l!=b && l!=a) 
                           if ( pass(a,b,c,d,e,f,g,h,i,j,k,l) ) {
                               std::cout << a << "," << b << "," << c << "," << d << "," 
                                         << e << "," << f << "," << g << "," << h << "," 
                                         << i << "," << j << "," << k << "," << l << ","
                                         << " pass " << n++ << "\n";
                           }
    return 0;
}

