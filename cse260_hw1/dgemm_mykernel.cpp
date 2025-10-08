#include "dgemm_mykernel.h"
#include "parameters.h"

#include <stdexcept>

void DGEMM_mykernel::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();

    my_dgemm(m, n, k, A.data(), k, B.data(), n, C.data(), n);
}

string DGEMM_mykernel::name() {
    return "my_kernel";
}

void DGEMM_mykernel::my_dgemm(
        int    m,
        int    n,
        int    k,
        const double *XA,
        int    lda,
        const double *XB,
        int    ldb,
        double *C,       
        int    ldc       
        )
{
    int    ic, ib, jc, jb, pc, pb;
    const double *packA, *packB;

    // Using NOPACK option for simplicity
    #define NOPACK

    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            #ifdef NOPACK
            packA = &XA[pc + ic * lda ];
            #else
            // Implement pack_A if you want to use PACK option
            #endif

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );

                #ifdef NOPACK
                packB = &XB[ldb * pc + jc ];
                #else
                // Implement pack_B if you want to use PACK option
                #endif

                // Implement your macro-kernel here
                my_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA,
                        packB,
                        &C[ ic * ldc + jc ], 
                        ldc
                        );
            }                                               // End 3.rd loop around micro-kernel
        }                                                 // End 4.th loop around micro-kernel
    }                                                     // End 5.th loop around micro-kernel
}

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based microkernel (NOPACK version)
//
// Implement your micro-kernel here
void DGEMM_mykernel::my_dgemm_ukr( int    kc,
                                  int    mr,
                                  int    nr,
                                  const double *a,
                                  const double *b,
                                  double *c,
                                  int ldc)
{
    int l, j, i;
    double cloc[param_mr][param_nr] = {{0}};
    
    // Load C into local array
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            cloc[i][j] = c(i, j, ldc);
        }
    }
    
    // Perform matrix multiplication
    for ( l = 0; l < kc; ++l ) {                 
        for ( i = 0; i < mr; ++i ) { 
            double as = a(i, l, ldc);
            for ( j = 0; j < nr; ++j ) { 
                cloc[i][j] +=  as * b(l, j, ldc);
            }
        }
    }
    
    // Store local array back to C
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            c(i, j, ldc) = cloc[i][j];
        }
    }
}

// Implement your macro-kernel here
void DGEMM_mykernel::my_macro_kernel(
        int    ib,
        int    jb,
        int    pb,
        const double * packA,
        const double * packB,
        double * C,
        int    ldc
        )
{
    int    i, j;

    for ( i = 0; i < ib; i += param_mr ) {                      // 2-th loop around micro-kernel
        for ( j = 0; j < jb; j += param_nr ) {                  // 1-th loop around micro-kernel
            my_dgemm_ukr (
                        pb,
                        min(ib-i, param_mr),
                        min(jb-j, param_nr),
                        &packA[i * ldc],          // assumes sq matrix, otherwise use lda
                        &packB[j],                
                        &C[ i * ldc + j ],
                        ldc
                        );
        }                                                       // 1-th loop around micro-kernel
    }
}

void DGEMM_mykernel::pack_A(
        int m,
        int k,
        const double * A,
        int lda, // row or column dimension?
        double * packed_A // not an array, just pointer to first element?
    )
{
    for (int i = 0; i < m; i += param_mr) { // iterating through the Mr subpanels of Ap (rows)
        // handle fringe case where n is not divisible by blocking size
        // so there is remainder where true number of rows is less than param_mr
        int true_row = min(param_mr, m - i);
        for (int j = 0; j < k; j++) { // iterating through each column in Kc of Ap subpanel
            for (int k = 0; k < true_row; k++) { // iterating through each row in the column of subpanel
                // i + l is current row, lda should be the column dimension (?) based on tutorial, j is the current column
                // need to get row and multiply by total columns to get to current row, then add the current column to get right address
                *packed_A = A[(i + k) * lda + j];
                *packed_A++; // move to the next position
            }
        }
    }
}

void DGEMM_mykernel::pack_B(
    int k,
    int n,
    const double * B,
    int ldb,
    double * packed_B
    )
{
    for (int i = 0; i < n; i += param_nr) { // iterating through the Nr subpanels of Bp (columns)
        for (int j = 0; j < k; j++) { // iterating through each row in Kc of Bp subpanel
            for (int k = 0; k < param_nr; k++) { // iterating through each column in the row of subpanel:
                ;
            }
        }
    }
}