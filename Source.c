#include <stdio.h>
#include <malloc.h>
#include <omp.h>
#include <time.h>


int f;
double t1;

void blas_dgemm(int M, int N, int K, double *A, double *B, double *C)
{
	double start, stop;
	double  t;
	int i;
	start = omp_get_wtime();
#pragma omp parallel for  num_threads(f) collapse(2)
	for (i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			*(C + i + j * M) = 0;
			for (int z = 0;z < K;z = z + 4) {
				*(C + i + j * M) += (*(A + i + z * M)) * (*(B + z + j * N));
				*(C + i + j * M) += (*(A + i + (z+1) * M)) * (*(B + (z + 1) + j * N));
				*(C + i + j * M) += (*(A + i + (z + 2) * M)) * (*(B + (z + 2) + j * N));
				*(C + i + j * M) += (*(A + i + (z + 3) * M)) * (*(B + (z + 3) + j * N));
			}
		}
	}
	stop = omp_get_wtime();
	t = (double)(stop - start);
	if (f == 1)
		t1 = t;
	printf("boost=%f, threads=%d time=%f\n", t1/t,f,t);
}

int main()
{
	double *A, *B, *C;
	int M, N, K;
	M = 32;
	N = M;
	K = M;
	A = (double*)malloc(M * K * sizeof(double));
	B = (double*)malloc(K * N * sizeof(double));
	C = (double*)malloc(M * N * sizeof(double));
	for (int i = 0;i <= (N * N - 1);i++)
	{
		A[i] = i;
		B[i] = (i-1);
		C[i] = 0;
		//A[i] = 0;
		//B[i] = 0;
	}	
	printf("N=%d\n",N);
	for (f=1;f < 17; f=f*2) 
		blas_dgemm(M, N, K, A, B, C);
	printf("\nYour matrix C=");
		for (int i = 0;i < N;i++) {
			printf("\n");
			for (int j = 0;j < N;j++) {
				printf("%0.2lf ", C[i + j * N]);
			}
		}

	free(A);
	free(B);
	free(C);
	return(0);
}