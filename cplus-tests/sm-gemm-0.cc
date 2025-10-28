// GEMM case with basic submatrix offsets - KernelFaRer misses this 
// because it doesn't recognize 'base + offset + i*ld + j' patterns
void submatrix_gemm(int m, int n, int k, float alpha, const float *A, int lda, int offsetA, const float *B, int ldb, int offsetB, float beta, float *C, int ldc, int offsetC) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
      float a = A[offsetA + i * lda + p];    // Pattern: base + constant_offset + i*ld + p
      float b = B[offsetB + p * ldb + j];    // Pattern: base + constant_offset + p*ld + j
        sum += a * b;
      }
      C[offsetC + i * ldc + j] = alpha * sum + beta * C[offsetC + i * ldc + j];
    }
  }
}