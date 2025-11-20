void striped_gemm(double *A, double *B, double *C,
                    int M, int N, int K,
                    int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i += 2) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                double a_val = A[i * lda + k];
                double b_val = B[k * ldb + j];
                C[i * ldc + j] += a_val * b_val;
            }
        }
    }
}
