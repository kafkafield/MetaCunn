#include <stdio.h>

main ()
{
	int a = 32;
	int i;
	for (i=1;i<50;i++)
	{
	//	printf ("%d ", i);
		a = a + 32;
		printf ("%d ", a);
		if (a>512)
			break;
	}
}
