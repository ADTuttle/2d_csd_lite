#include <stdlib.h>
#include <math.h>
#include "petscsys.h"

PetscReal array_max(PetscReal *array,size_t size)
{
	PetscReal max=0;
	for(size_t ind=0;ind<size;ind++) {
		if(fabs(array[ind])>max) {
			max=fabs(array[ind]);
		}
	}
	return max;
}
PetscReal array_diff_max(PetscReal *array1,PetscReal *array2,size_t size)
{
	PetscReal max=0;
	for(size_t ind=0;ind<size;ind++) {
		if(fabs(array1[ind]-array2[ind])>max) {
			max=fabs(array1[ind]-array2[ind]);
		}
	}
	return max;
}

PetscReal l2_norm(PetscReal *array1,size_t size)
{
	PetscReal max=0;
	for(size_t ind=0;ind<size;ind++) {
		max += array1[ind]*array1[ind];
	}
	return sqrt(max);
}
