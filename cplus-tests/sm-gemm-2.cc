// GEMM case with submatrix using size-based offsets - KernelFaRer misses this 
// because it doesn't recognize 'base + (block_start + i)*full_ld + (block_start + p)' patterns  
void block_submatrix_gemm(int block_size, const float *full_A, int full_lda, int block_startA,
                         const float *full_B, int full_ldb, int block_startB,
                         float *full_C, int full_ldc, int block_startC) {
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < block_size; ++p) {
        float a = full_A[(block_startA + i) * full_lda + (block_startA + p)];    // Pattern: base + (block_start + i)*full_ld + (block_start + p)
        float b = full_B[(block_startB + p) * full_ldb + (block_startB + j)];    // Pattern: base + (block_start + p)*full_ld + (block_start + j)
        sum += a * b;
      }
      full_C[(block_startC + i) * full_ldc + (block_startC + j)] = sum;          // Pattern: base + (block_start + i)*full_ld + (block_start + j)
    }
  }
}