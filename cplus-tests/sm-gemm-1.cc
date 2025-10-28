// GEMM case with separate row/column offsets - KernelFaRer misses this because 
// it doesn't recognize 'base + (row_offset + i)*ld + (col_offset + p)' patterns
void submatrix_rowcol_offset(int m, int n, int k, float alpha, 
                            const float *A, int lda, int start_rowA, int start_colA,
                            const float *B, int ldb, int start_rowB, int start_colB,
                            float beta, float *C, int ldc, int start_rowC, int start_colC) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        float a = A[(start_rowA + i) * lda + (start_colA + p)];    // Pattern: base + (constant + i)*ld + (constant + p)
        float b = B[(start_rowB + p) * ldb + (start_colB + j)];    // Pattern: base + (constant + p)*ld + (constant + j)
        sum += a * b;
      }
      C[(start_rowC + i) * ldc + (start_colC + j)] = alpha * sum + beta * C[(start_rowC + i) * ldc + (start_colC + j)];  // Pattern: base + (constant + i)*ld + (constant + j)
    }
  }
}