#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <omp.h>
#if _SSGRIND_
#include "ss_valgrind.h"
#endif

// N and MIN must be powers of 2
long N;
long MIN_SORT_SIZE;
long MIN_MERGE_SIZE;

#if _SSGRIND_
char stringMessage[256];
#endif 

#define T int


void basicsort(long n, T data[n]);

void basicmerge(long n, T left[n], T right[n], T result[n*2], long start, long length);

void merge_rec(long n, T left[n], T right[n], T result[n*2], long start, long length) {
	if (length < MIN_MERGE_SIZE*2L) {
		// Base case
#if _SSGRIND_
		sprintf(stringMessage, "merge %ld", n/4);
		start_task_valgrind(NULL, stringMessage);
		basicmerge(n, left, right, result, start, length);
		end_task_valgrind();
#else
		basicmerge(n, left, right, result, start, length);
#endif
	} else {
		// Recursive decomposition
		
		#pragma omp task input (left[0;n], right[0;n]) output (result[start;length/2])
		merge_rec(n, left, right, result, start, length/2);
		#pragma omp task input (left[0;n], right[0;n]) output (result[start+length/2;length/2])
		merge_rec(n, left, right, result, start + length/2, length/2);
	}
}


void multisort(long n, T data[n], T tmp[n]) {
	if (n >= MIN_SORT_SIZE*4L) {
		// Recursive decomposition
		#pragma omp task inout (data[0;n/4L])
		multisort(n/4L, &data[0], &tmp[0]);
		#pragma omp task inout(data[n/4L;n/4L])
		multisort(n/4L, &data[n/4L], &tmp[n/4L]);
		#pragma omp task inout (data[n/2L;n/4L])
		multisort(n/4L, &data[n/2L], &tmp[n/2L]);
		#pragma omp task inout (data[3L*n/4L; n/4L])
		multisort(n/4L, &data[3L*n/4L], &tmp[3L*n/4L]);
		#pragma omp taskwait

		#pragma omp task input (data[0;n/4L], data[n/4L;n/4L]) output (tmp[0; n/2L])
		merge_rec(n/4L, &data[0], &data[n/4L], &tmp[0], 0, n/2L);
		#pragma omp task input (data[n/2L;n/4L], data[3L*n/4L; n/4L]) output (tmp[n/2L; n/2L])
		merge_rec(n/4L, &data[n/2L], &data[3L*n/4L], &tmp[n/2L], 0, n/2L);
		
		//#pragma omp taskwait
		#pragma omp task input (tmp[0; n/2L], tmp[n/2L; n/2L]) output (data[0; n])
		merge_rec(n/2L, &tmp[0], &tmp[n/2L], &data[0], 0, n);
		//#pragma omp taskwait
	} else {
		// Base case


#if _SSGRIND_
		sprintf(stringMessage, "sort %ld", n/4);
		start_task_valgrind(NULL, stringMessage);
		basicsort(n, data);
		end_task_valgrind();

#else
		basicsort(n, data);
#endif

	}
}

static void initialize(long length, T data[length]) {
	for (long i = 0; i < length; i++) {
		if (i==0) {
			data[i] = rand();
		} else {
			data[i] = (data[i-1] * i * 104723L) % N;
		}
		if (data[i] == 0) data[i] = rand();
	}
}

int check_solution(long length, T data[length]) {
	int failure=0;
	for (long i = 0; i < length-1; i++) {
		if (data[i]>data[i+1]){
		 failure=1; 
		 break;
		}
	}
	return (failure);
}
static void touch(long length, T data[length]) {
	for (long i = 0; i < length; i++) {
		data[i] = 0;
	}
}

#ifdef CHECKSUM
int get_checksum(int n, T array[n])
{
   int c = 0, i;
   for (i=0; i < n; i++) c+=array[i];
   return c;
}
#endif


int main(int argc, char **argv) {
#ifdef _CHECK_
 	int failure = 0;
#ifdef CHECKSUM
	int checksum0, checksum1;
#endif
#endif

	if (argc != 4) {
		fprintf(stderr, "Usage: %s <vector size in K> <sort size in K> <merge size in K>\n", argv[0]);
		return 1;
	}

	N = atol(argv[1]) * 1024L;
	MIN_SORT_SIZE = atol(argv[2]) * 1024L;
	MIN_MERGE_SIZE = atol(argv[3]) * 1024L;
	
	T *data = malloc(N*sizeof(T));
	T *tmp = malloc(N*sizeof(T));
	
	FILE *fp;
	if((fp=fopen("multisort-serial.out", "wb"))==NULL) {
		fprintf(stderr, "Unable to open file\n");
		return EXIT_FAILURE;
	}

#if _SSGRIND_
	start_css_valgrind();
#endif

	double init_time = omp_get_wtime();
	initialize(N, data);
	touch(N, tmp);
	init_time = omp_get_wtime() - init_time;
	double sort_time = omp_get_wtime();
#ifdef CHECKSUM
	checksum0 = get_checksum(N, data);
#endif
	multisort(N, data, tmp);
#pragma omp taskwait
#ifdef CHECKSUM
	checksum1 = get_checksum(N, data);
#endif

	sort_time = omp_get_wtime() - sort_time;
#if _SSGRIND_
	end_css_valgrind();
#endif

	if(fwrite(data, sizeof(T), N, fp) != N) {
		fprintf(stderr, "Output file not written correctly\n");
	}

#if _CHECK_
#ifdef CHECKSUM
	printf ("Checksum0 %x checksum1 %x\n", checksum0, checksum1);
	if (checksum0!=checksum1) failure = 1000;
#endif

	failure += check_solution(N, data);
	if (failure) printf ("SORTING FAILURE (%d)\n", failure); 
	else printf ("SORTING SUCCESS\n"); 

#endif

    	fprintf(stdout, "Multisort program\n");
    	fprintf(stdout, "   Initialization time in seconds = %g\n", init_time);
    	fprintf(stdout, "   Multisort time in seconds = %g\n", sort_time);
    	fprintf(stdout, "\n");
	return 0;
}
