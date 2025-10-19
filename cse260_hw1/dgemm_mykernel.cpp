#include "dgemm_mykernel.h"
#include "parameters.h"

#include <stdexcept>
#include <iostream>
#include <arm_sve.h>

void DGEMM_mykernel::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();

    // TODO: remove
    // A.print();
    // B.print();
    // C.print();

    my_dgemm(m, n, k, A.data(), k, B.data(), n, C.data(), n);

    // TODO: remove
    // C.print();
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
    // const double *packA, *packB;
    // allocate memory for packed_A and packed_B
    double* packed_A = new double[param_mc * param_kc];
    double* packed_B = new double[param_kc * param_nc];

    // Using NOPACK option for simplicity
    // #define NOPACK

    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc ); // the row number of Ap
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc ); // the column number of Ap and row number of Bp
            
            #ifdef NOPACK
            packA = &XA[pc + ic * lda ];
            #else
            // Implement pack_A if you want to use PACK option
            pack_A(ib, pb, &XA[pc + ic * lda ], lda, packed_A);
            // TODO: remove
            // for (double * p = packed_A; p < packed_A + ib * pb; ++p) {
            //         std::cout << *p << std::endl;
            //     }
            
            #endif

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc ); // the column number of Bp

                #ifdef NOPACK
                packB = &XB[ldb * pc + jc ];
                #else
                // Implement pack_B if you want to use PACK option
                pack_B(pb, jb, &XB[ldb * pc + jc ], ldb, packed_B);
                // TODO: remove
                // for (double * p = packed_B; p < packed_B + ib * pb; ++p) {
                //     std::cout << *p << std::endl;
                // }
                #endif

                // Implement your macro-kernel here
                my_macro_kernel(
                        ib,
                        jb,
                        pb,
                        // packA,
                        // packB,
                        packed_A,
                        packed_B,
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
// Now uses packed A and B
void DGEMM_mykernel::my_dgemm_ukr( int    kc,
                                  int    mr,
                                  int    nr,
                                  const double *a,
                                  const double *b,
                                  double *c,
                                  int ldc)
{
    // SIMD SVE VERSION
    // have 32 vector registers, exact size unknown, 128 to 2048, can't seem to find exact size...
    // assuming 128 bits, so 2 doubles per vector, total of 64 doubles in registers
    // need registers for C (most), A (some), B (some)
    // want to maximize data reuse and minimize loads from memory
    // logic flow: load C into registers for reading and writing, load column of A, load row of B
    // multiply each value in column of A with each value in row of B, accumulate into C
    // column of A is loaded once, values of B row are loaded one at a time, want maximum reuse of column of A
    // since one register holds 2, if column of A has 2 values, then that uses one register
    // need another register to hold value from row B
    // have 30 registers for C, which is 60 doubles
    // since A is 2, then B can be 30
    // try a 2x30 kernel, param_mr = 2, param_nr = 30

    const double *a_curr = a; // pointer to current A subpanel row
    const double *b_curr = b; // pointer to current B subpanel column

    // registers to hold 2x30 subblock of C
    svfloat64_t c0, c1, c2, c3, c4, c5, c6, c7, c8, c9;
    svfloat64_t c10, c11, c12, c13, c14, c15, c16, c17, c18, c19;
    svfloat64_t c20, c21, c22, c23, c24, c25, c26, c27, c28, c29;

    svfloat64_t* c_sub[30] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8, &c9,
                             &c10, &c11, &c12, &c13, &c14, &c15, &c16, &c17, &c18, &c19,
                             &c20, &c21, &c22, &c23, &c24, &c25, &c26, &c27, &c28, &c29};

    svbool_t pred_mr = svwhilelt_b64(0, mr); // predicate to activate only mr rows that are valid

    for (int i = 0; i < nr; i++) { // only initialize valid columns of C based on columns in B
        *c_sub[i] = svdup_f64(0.0);
    }

    for (int j = 0; j < kc; j++) { // iterate through for every kc set of A column and B row in subpanels
        svfloat64_t a_col = svld1_f64(pred_mr, a_curr); // load current column of A subpanel

        for (int k = 0; k < nr; k++) { // iterate through each column of B row
            svfloat64_t b_val = svdup_f64(b_curr[k]); // grab value from B and put in vector register
            *c_sub[k] = svmla_f64_m(pred_mr, *c_sub[k], a_col, b_val); // multiply accumulate into C subblock
        }

        a_curr += mr; // move to next column of A subpanel
        b_curr += nr; // move to next row of B subpanel
    }

    // store results back to C, remember C is row-major but c_sub is column-major
    for (int i = 0; i < nr; i++) { // iterate through each column of c subblock
        double temp_c[2]; // temp array to hold c
        svst1_f64(pred_mr, temp_c, *c_sub[i]); // store column i of c_sub into temp c

        for (int j = 0; j < mr; j++) { // iterate through each row of c subblock
            c[j * ldc + i] += temp_c[j]; // add to original C
        }
        // svfloat64_t c_orig = svld1_f64(pred_mr, c + i * ldc); // load one column of 2x30 C subblock
        // svfloat64_t c_new = svadd_f64_m(pred_mr, c_orig, *c_sub[i]); // add computed values
        // svst1_f64(pred_mr, c + i * ldc, c_new); // store back to C
    }


    // PLAIN PACKING VERSION
    // for (int i = 0; i < kc; i++) { // iterating through columns of Ap subpanel and rows of Bp subpanel
    //     for (int j = 0; j < mr; j++) { // iterating through rows of Ap subpanel
    //         // subpanel of A is packed column-major, so multiply column by mr and add row
    //         double a_ji = a[i * mr + j];
    //         for (int k = 0; k < nr; k++) { // iterating through columns of Bp subpanel
    //             // subpanel of B is packed row-major, so multiply row by nr and add column
    //             double b_ik = b[i * nr + k];
    //             // adding the product of a_ji and b_ik to loction c_jk 
    //             c[j * ldc + k] += a_ji * b_ik;
    //         }
    //     }
    // }

    // ORIGINAL VERSION
    // int l, j, i;
    // double cloc[param_mr][param_nr] = {{0}};
    
    // // Load C into local array
    // for (i = 0; i < mr; ++i) {
    //     for (j = 0; j < nr; ++j) {
    //         cloc[i][j] = c(i, j, ldc);
    //     }
    // }
    
    // // Perform matrix multiplication
    // for ( l = 0; l < kc; ++l ) {                 
    //     for ( i = 0; i < mr; ++i ) { 
    //         double as = a(i, l, ldc);
    //         for ( j = 0; j < nr; ++j ) { 
    //             cloc[i][j] +=  as * b(l, j, ldc);
    //         }
    //     }
    // }
    
    // // Store local array back to C
    // for (i = 0; i < mr; ++i) {
    //     for (j = 0; j < nr; ++j) {
    //         c(i, j, ldc) = cloc[i][j];
    //     }
    // }
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
                        // &packA[i * ldc],          // assumes sq matrix, otherwise use lda
                        // &packB[j],                
                        &packA[i * pb], // sets pointer to start of current subpanel of Ap in packed_A
                        &packB[j * pb], // sets pointer to start of current subpanel of Bp in packed_B
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
            for (int l = 0; l < true_row; l++) { // iterating through each row in the column of subpanel
                // i + k is current row, lda should be the column dimension (?) based on tutorial, j is the current column
                // need to get row and multiply by total columns to get to current row, then add the current column to get right address
                *packed_A++ = A[(i + l) * lda + j];
                // *packed_A++; // move to the next position
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
        int true_col = min(param_nr, n - i);
        for (int j = 0; j < k; j++) { // iterating through each row in Kc of Bp subpanel
            for (int l = 0; l < true_col; l++) { // iterating through each column in the row of subpanel
                // j is the current row, ldb should be column dimension, i + k is current column
                // get row and multiple by total columns for current row, add current column
                *packed_B++ = B[j * ldb + i + l];
                // *packed_B++; // move to next position
            }
        }
    }
}