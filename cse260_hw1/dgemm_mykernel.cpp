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
    // double* packed_A = new double[param_mc * param_kc];
    // double* packed_B = new double[param_kc * param_nc];

    // Using NOPACK option for simplicity
    // #define NOPACK

    size_t packed_A_size = (size_t)param_mc * (size_t)param_kc;
    double* packed_A = nullptr;
    if (posix_memalign((void**)&packed_A, 64, packed_A_size * sizeof(double)) != 0) {
        throw std::bad_alloc();
    }

    size_t packed_B_size = (size_t)param_kc * (size_t)param_nc;
    double* packed_B = nullptr;
    if (posix_memalign((void**)&packed_B, 64, packed_B_size * sizeof(double)) != 0) {
        throw std::bad_alloc();
    }

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
        } 
    }                                                     // End 5.th loop around micro-kernel
    free(packed_A);                                                // End 4.th loop around micro-kernel
    free(packed_B);
}

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]


void print_svfloat64(svfloat64_t vec, svbool_t pred) {
    double buffer[svcntd()];
    svst1(pred, buffer, vec);
    for (int i = 0; i < int(svcntd()); i++) {
        printf("Element %d: %f\n", i, buffer[i]);
    }
}

//
// C-based microkernel (NOPACK version)
//
// Implement your micro-kernel here
// Now uses packed A and B
void DGEMM_mykernel::my_dgemm_ukr( int    kc,
                                  int    mr,
                                  int    nr,
                                  const double *__restrict__ a,
                                  const double *__restrict__ b,
                                  double *__restrict__ c,
                                  int ldc)
{
    // cout << "mr: " << mr << endl;
    // cout << "nr: " << nr << endl;
    // SIMD SVE VERSION 2
    const int VL = svcntd(); // get number of doubles that can fit per vector
    const int num_vecs = (nr + VL - 1) / VL; // number of vectors to cover nr columns
    
    for (int i = 0; i < num_vecs; i++) {
        // cout << mr << endl;
        int col_offset = i * VL;
        svbool_t npred = svwhilelt_b64((uint64_t)col_offset, (uint64_t)nr);

        /*
        svfloat64_t c_vecs[param_mr];
        for (int j = 0; j < mr; j++) {
            svfloat64_t c_vec = svld1_f64(npred, &c[j * ldc + col_offset]);
            c_vecs[j] = &c_vec;
        }

        for (int j = 0; j < kc; j++) {
            const double* b_row = &b[j * nr]; // pointer to current row in B
            svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
            cout << "B" << endl;
            print_svfloat64(b_vec, npred);
            
            const double* a_col = &a[j * mr]; // pointer to current column in A
            svfloat64_t* a_vals[param_mr];
            for (int k = 0; k < mr; k++) {
                svfloat64_t a_val = svdup_f64(a_col[k]); // broadcast A values
                a_vals[k] = &a_val;
                cout << "A" << endl;
                print_svfloat64(a_val, npred);
                *c_vecs[k] = svmla_f64_m(npred, *c_vecs[k], b_vec, *a_vals[k]); // multiply-accumulate
                cout << "C" << endl;
                print_svfloat64(*c_vecs[k], npred);
            }

            // for (int k = 0; k < mr; k++) {
            // }
        }

        for (int j = 0; j < mr; j++) {
            svst1_f64(npred, &c[j * ldc + col_offset], *c_vecs[j]); // store results back to C
        }
        */

        switch (mr) {
            case 15: { // theoretical max, 20.7056
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                svfloat64_t c8x = svld1_f64(npred, &c[8 * ldc + col_offset]);
                svfloat64_t c9x = svld1_f64(npred, &c[9 * ldc + col_offset]);
                svfloat64_t c10x = svld1_f64(npred, &c[10 * ldc + col_offset]);
                svfloat64_t c11x = svld1_f64(npred, &c[11 * ldc + col_offset]);
                svfloat64_t c12x = svld1_f64(npred, &c[12 * ldc + col_offset]);
                svfloat64_t c13x = svld1_f64(npred, &c[13 * ldc + col_offset]);
                svfloat64_t c14x = svld1_f64(npred, &c[14 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                    svfloat64_t a8 = svdup_f64(a_col[8]);
                    svfloat64_t a9 = svdup_f64(a_col[9]);
                    svfloat64_t a10 = svdup_f64(a_col[10]);
                    svfloat64_t a11 = svdup_f64(a_col[11]);
                    svfloat64_t a12 = svdup_f64(a_col[12]);
                    svfloat64_t a13 = svdup_f64(a_col[13]);
                    svfloat64_t a14 = svdup_f64(a_col[14]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                    c8x = svmla_f64_m(npred, c8x, b_vec, a8);
                    c9x = svmla_f64_m(npred, c9x, b_vec, a9);
                    c10x = svmla_f64_m(npred, c10x, b_vec, a10);
                    c11x = svmla_f64_m(npred, c11x, b_vec, a11);
                    c12x = svmla_f64_m(npred, c12x, b_vec, a12);
                    c13x = svmla_f64_m(npred, c13x, b_vec, a13);
                    c14x = svmla_f64_m(npred, c14x, b_vec, a14);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
                svst1_f64(npred, &c[8 * ldc + col_offset], c8x);
                svst1_f64(npred, &c[9 * ldc + col_offset], c9x);
                svst1_f64(npred, &c[10 * ldc + col_offset], c10x);
                svst1_f64(npred, &c[11 * ldc + col_offset], c11x);
                svst1_f64(npred, &c[12 * ldc + col_offset], c12x);
                svst1_f64(npred, &c[13 * ldc + col_offset], c13x);
                svst1_f64(npred, &c[14 * ldc + col_offset], c14x);
            
                break;
            }
            case 14: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                svfloat64_t c8x = svld1_f64(npred, &c[8 * ldc + col_offset]);
                svfloat64_t c9x = svld1_f64(npred, &c[9 * ldc + col_offset]);
                svfloat64_t c10x = svld1_f64(npred, &c[10 * ldc + col_offset]);
                svfloat64_t c11x = svld1_f64(npred, &c[11 * ldc + col_offset]);
                svfloat64_t c12x = svld1_f64(npred, &c[12 * ldc + col_offset]);
                svfloat64_t c13x = svld1_f64(npred, &c[13 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                    svfloat64_t a8 = svdup_f64(a_col[8]);
                    svfloat64_t a9 = svdup_f64(a_col[9]);
                    svfloat64_t a10 = svdup_f64(a_col[10]);
                    svfloat64_t a11 = svdup_f64(a_col[11]);
                    svfloat64_t a12 = svdup_f64(a_col[12]);
                    svfloat64_t a13 = svdup_f64(a_col[13]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                    c8x = svmla_f64_m(npred, c8x, b_vec, a8);
                    c9x = svmla_f64_m(npred, c9x, b_vec, a9);
                    c10x = svmla_f64_m(npred, c10x, b_vec, a10);
                    c11x = svmla_f64_m(npred, c11x, b_vec, a11);
                    c12x = svmla_f64_m(npred, c12x, b_vec, a12);
                    c13x = svmla_f64_m(npred, c13x, b_vec, a13);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
                svst1_f64(npred, &c[8 * ldc + col_offset], c8x);
                svst1_f64(npred, &c[9 * ldc + col_offset], c9x);
                svst1_f64(npred, &c[10 * ldc + col_offset], c10x);
                svst1_f64(npred, &c[11 * ldc + col_offset], c11x);
                svst1_f64(npred, &c[12 * ldc + col_offset], c12x);
                svst1_f64(npred, &c[13 * ldc + col_offset], c13x);
            
                break;
            }
            case 13: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                svfloat64_t c8x = svld1_f64(npred, &c[8 * ldc + col_offset]);
                svfloat64_t c9x = svld1_f64(npred, &c[9 * ldc + col_offset]);
                svfloat64_t c10x = svld1_f64(npred, &c[10 * ldc + col_offset]);
                svfloat64_t c11x = svld1_f64(npred, &c[11 * ldc + col_offset]);
                svfloat64_t c12x = svld1_f64(npred, &c[12 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                    svfloat64_t a8 = svdup_f64(a_col[8]);
                    svfloat64_t a9 = svdup_f64(a_col[9]);
                    svfloat64_t a10 = svdup_f64(a_col[10]);
                    svfloat64_t a11 = svdup_f64(a_col[11]);
                    svfloat64_t a12 = svdup_f64(a_col[12]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                    c8x = svmla_f64_m(npred, c8x, b_vec, a8);
                    c9x = svmla_f64_m(npred, c9x, b_vec, a9);
                    c10x = svmla_f64_m(npred, c10x, b_vec, a10);
                    c11x = svmla_f64_m(npred, c11x, b_vec, a11);
                    c12x = svmla_f64_m(npred, c12x, b_vec, a12);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
                svst1_f64(npred, &c[8 * ldc + col_offset], c8x);
                svst1_f64(npred, &c[9 * ldc + col_offset], c9x);
                svst1_f64(npred, &c[10 * ldc + col_offset], c10x);
                svst1_f64(npred, &c[11 * ldc + col_offset], c11x);
                svst1_f64(npred, &c[12 * ldc + col_offset], c12x);
            
                break;
            }
            case 12: { // 20.7115
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                svfloat64_t c8x = svld1_f64(npred, &c[8 * ldc + col_offset]);
                svfloat64_t c9x = svld1_f64(npred, &c[9 * ldc + col_offset]);
                svfloat64_t c10x = svld1_f64(npred, &c[10 * ldc + col_offset]);
                svfloat64_t c11x = svld1_f64(npred, &c[11 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                    svfloat64_t a8 = svdup_f64(a_col[8]);
                    svfloat64_t a9 = svdup_f64(a_col[9]);
                    svfloat64_t a10 = svdup_f64(a_col[10]);
                    svfloat64_t a11 = svdup_f64(a_col[11]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                    c8x = svmla_f64_m(npred, c8x, b_vec, a8);
                    c9x = svmla_f64_m(npred, c9x, b_vec, a9);
                    c10x = svmla_f64_m(npred, c10x, b_vec, a10);
                    c11x = svmla_f64_m(npred, c11x, b_vec, a11);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
                svst1_f64(npred, &c[8 * ldc + col_offset], c8x);
                svst1_f64(npred, &c[9 * ldc + col_offset], c9x);
                svst1_f64(npred, &c[10 * ldc + col_offset], c10x);
                svst1_f64(npred, &c[11 * ldc + col_offset], c11x);
            
                break;
            }
            case 11: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                svfloat64_t c8x = svld1_f64(npred, &c[8 * ldc + col_offset]);
                svfloat64_t c9x = svld1_f64(npred, &c[9 * ldc + col_offset]);
                svfloat64_t c10x = svld1_f64(npred, &c[10 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                    svfloat64_t a8 = svdup_f64(a_col[8]);
                    svfloat64_t a9 = svdup_f64(a_col[9]);
                    svfloat64_t a10 = svdup_f64(a_col[10]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                    c8x = svmla_f64_m(npred, c8x, b_vec, a8);
                    c9x = svmla_f64_m(npred, c9x, b_vec, a9);
                    c10x = svmla_f64_m(npred, c10x, b_vec, a10);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
                svst1_f64(npred, &c[8 * ldc + col_offset], c8x);
                svst1_f64(npred, &c[9 * ldc + col_offset], c9x);
                svst1_f64(npred, &c[10 * ldc + col_offset], c10x);
            
                break;
            }
            case 10: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                svfloat64_t c8x = svld1_f64(npred, &c[8 * ldc + col_offset]);
                svfloat64_t c9x = svld1_f64(npred, &c[9 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                    svfloat64_t a8 = svdup_f64(a_col[8]);
                    svfloat64_t a9 = svdup_f64(a_col[9]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                    c8x = svmla_f64_m(npred, c8x, b_vec, a8);
                    c9x = svmla_f64_m(npred, c9x, b_vec, a9);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
                svst1_f64(npred, &c[8 * ldc + col_offset], c8x);
                svst1_f64(npred, &c[9 * ldc + col_offset], c9x);
            
                break;
            }
            case 9: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                svfloat64_t c8x = svld1_f64(npred, &c[8 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                    svfloat64_t a8 = svdup_f64(a_col[8]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                    c8x = svmla_f64_m(npred, c8x, b_vec, a8);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
                svst1_f64(npred, &c[8 * ldc + col_offset], c8x);
            
                break;
            }
            case 8: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                svfloat64_t c7x = svld1_f64(npred, &c[7 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                    svfloat64_t a7 = svdup_f64(a_col[7]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                    c7x = svmla_f64_m(npred, c7x, b_vec, a7);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
                svst1_f64(npred, &c[7 * ldc + col_offset], c7x);
            
                break;
            }
            case 7: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                svfloat64_t c6x = svld1_f64(npred, &c[6 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                    svfloat64_t a6 = svdup_f64(a_col[6]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                    c6x = svmla_f64_m(npred, c6x, b_vec, a6);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
                svst1_f64(npred, &c[6 * ldc + col_offset], c6x);
            
                break;
            }
            case 6: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                svfloat64_t c5x = svld1_f64(npred, &c[5 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                    svfloat64_t a5 = svdup_f64(a_col[5]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                    c5x = svmla_f64_m(npred, c5x, b_vec, a5);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
                svst1_f64(npred, &c[5 * ldc + col_offset], c5x);
            
                break;
            }
            case 5: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
                svfloat64_t c4x = svld1_f64(npred, &c[4 * ldc + col_offset]);
                
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
                    svfloat64_t a4 = svdup_f64(a_col[4]);
                
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                    c4x = svmla_f64_m(npred, c4x, b_vec, a4);
                }
            
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
                svst1_f64(npred, &c[4 * ldc + col_offset], c4x);
            
                break;
            }
            case 4: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
                svfloat64_t c3x = svld1_f64(npred, &c[3 * ldc + col_offset]);
 
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
 
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
                    svfloat64_t a3 = svdup_f64(a_col[3]);
 
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                    c3x = svmla_f64_m(npred, c3x, b_vec, a3);
                }
 
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
                svst1_f64(npred, &c[3 * ldc + col_offset], c3x);
 
                break;
            }
            case 3: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
                svfloat64_t c2x = svld1_f64(npred, &c[2 * ldc + col_offset]);
 
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
            
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    svfloat64_t a2 = svdup_f64(a_col[2]);
 
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                    c2x = svmla_f64_m(npred, c2x, b_vec, a2);
                }
 
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
                svst1_f64(npred, &c[2 * ldc + col_offset], c2x);
 
                break;
            }
            case 2: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
                svfloat64_t c1x = svld1_f64(npred, &c[1 * ldc + col_offset]);
 
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    // for (int idx = 0; idx < nr; idx++) {
                    //     cout << "B row " << j << " col " << idx << ": " << b_row[idx] << endl;
                    // }
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                    // print_svfloat64(b_vec, npred);
 
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
                    svfloat64_t a1 = svdup_f64(a_col[1]);
                    // print_svfloat64(a0, npred);
                    // print_svfloat64(a1, npred);
 
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                    c1x = svmla_f64_m(npred, c1x, b_vec, a1);
                }
 
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
                svst1_f64(npred, &c[1 * ldc + col_offset], c1x);
 
                break;
            }
            case 1: {
                svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
 
                for (int j = 0; j < kc; j++) {
                    const double* b_row = &b[j * nr]; // pointer to current row in B
                    svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
                
                    const double* a_col = &a[j * mr]; // pointer to current column in A
                    svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
 
                    c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
                }
 
                svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
 
                break;
            }
        }
    }

    // PACKING VERSION (6.8)
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
    //     const double* a_col = a + (size_t)l * (size_t)param_mr;
    //     const double* b_row = b + (size_t)l * (size_t)param_nr;

    //     for ( i = 0; i < mr; ++i ) { 
    //         double a_il = a_col[i];
    //         for ( j = 0; j < nr; ++j ) { 
    //             cloc[i][j] +=  a_il * b_row[j];
    //         }
    //     }
    // }
    
    // // Store local array back to C
    // for (i = 0; i < mr; ++i) {
    //     for (j = 0; j < nr; ++j) {
    //         c(i, j, ldc) = cloc[i][j];
    //     }
    // }

    // ORIGINAL VERSION (6.4)
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
    // for (int i = 0; i < jb; i++) {
    //     for (int j = 0; j < ib; j++) {
    //         std::cout << "packB[" << i << "]: " << packB[i * pb + j] << std::endl;
    //     }
    // }
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
        int m, // starting row of panel
        int k, // starting column of panel
        const double * A, // pointer to first element of panel
        int lda, // row or column dimension?
        double * packed_A // not an array, just pointer to first element?
    )
{
    // double* a_ptr = packed_A;
    for (int i = 0; i < m; i += param_mr) { // iterating through the Mr subpanels of Ap (rows)
        // handle fringe case where n is not divisible by blocking size
        // so there is remainder where true number of rows is less than param_mr
        int true_row = std::min(param_mr, m - i);
        for (int j = 0; j < k; j++) { // iterating through each column in Kc of Ap subpanel
            // const double* a_col_ptr = &A[(size_t)(i) * (size_t)(lda) + (size_t)(j)]; // pointer to current column in Ap subpanel
            for (int l = 0; l < true_row; l++) { // iterating through each row in the column of subpanel
                // i + k is current row, lda should be the column dimension (?) based on tutorial, j is the current column
                // need to get row and multiply by total columns to get to current row, then add the current column to get right address
                *packed_A++ = A[(i + l) * lda + j];
                // a_ptr[l] = a_col_ptr[l * lda]; // moves down the column
            }
            // for (int l = true_row; l < param_mr; l++) { // padding for fringe case
            //     a_ptr[l] = 6.5;
            // }
            // a_ptr += param_mr; // move to next column location in packed_A
            // for (int l = 0; l < true_row; l+=4) {
            //     *packed_A++ = A[(i + l + 0) * lda + j];
            //     *packed_A++ = A[(i + l + 1) * lda + j];
            //     *packed_A++ = A[(i + l + 2) * lda + j];
            //     *packed_A++ = A[(i + l + 3) * lda + j];
            // }
            // for (int l = true_row - (true_row % 4); l < true_row; l++) {
            //     *packed_A++ = A[(i + l) * lda + j];
            // }
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
    // double* b_ptr = packed_B;
    for (int i = 0; i < n; i += param_nr) { // iterating through the Nr subpanels of Bp (columns)
        int true_col = min(param_nr, n - i);
        for (int j = 0; j < k; j++) { // iterating through each row in Kc of Bp subpanel
            // const double* b_row_ptr = &B[(size_t)(j) * (size_t)(ldb) + (size_t)(i)];
            for (int l = 0; l < true_col; l++) { // iterating through each column in the row of subpanel
                // j is the current row, ldb should be column dimension, i + k is current column
                // get row and multiple by total columns for current row, add current column
                *packed_B++ = B[j * ldb + i + l];
                // b_ptr[l] = b_row_ptr[l]; // moves across the row
            }
            // for (int l = true_col; l < param_nr; l++) { // padding for fringe case
            //     b_ptr[l] = 7.5;
            // }
            // b_ptr += param_nr; // move to next row location in packed_B
            // for (int l = 0; l < true_col; l+=4) {
            //     *packed_B++ = B[j * ldb + i + l + 0];
            //     *packed_B++ = B[j * ldb + i + l + 1];
            //     *packed_B++ = B[j * ldb + i + l + 2];
            //     *packed_B++ = B[j * ldb + i + l + 3];
            // }
            // for (int l = true_col - (true_col % 4); l < true_col; l++) {
            //     *packed_B++ = B[j * ldb + i + l];
            // }
        }
    }
}