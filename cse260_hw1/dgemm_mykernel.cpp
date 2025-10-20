#include "dgemm_mykernel.h"
#include "parameters.h"

#include <stdexcept>
#include <iostream>
#include <arm_sve.h>
#include <omp.h>

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

        
        if (mr == 4) {
            // cout << "mr = 4" << endl;
            // load C values into vector registers
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
 
            continue;
        }
        else if (mr == 3) {
            // cout << "mr = 3" << endl;
            // load C values into vector registers
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
 
            continue;
        }
        else if (mr == 2) {
            // cout << "mr = 2" << endl;
            // load C values into vector registers
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
 
            continue;
        }
        else if (mr == 1) {
            // cout << "mr = 1" << endl;
            // load C values into vector registers
            svfloat64_t c0x = svld1_f64(npred, &c[0 * ldc + col_offset]);
 
            for (int j = 0; j < kc; j++) {
                const double* b_row = &b[j * nr]; // pointer to current row in B
                svfloat64_t b_vec = svld1_f64(npred, b_row + col_offset); // load B row vector
             
                const double* a_col = &a[j * mr]; // pointer to current column in A
                svfloat64_t a0 = svdup_f64(a_col[0]); // broadcast A values
 
                c0x = svmla_f64_m(npred, c0x, b_vec, a0); // multiply-accumulate
            }
 
            svst1_f64(npred, &c[0 * ldc + col_offset], c0x); // store results back to C
 
            continue;
        }
    }


    // SIMD SVE VERSION 1
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
    // try a 2x30 kernel, param_mr = 2, param_nr = 30, not working well (4.48)

    //try 4x12 kernel, param_mr = 4, param_nr = 12, slightly better (5.5)

    //try 8x6 kernel, param_mr = 8, param_nr = 6, same (5.6)

    // const double *a_curr = a; // pointer to current A subpanel row
    // const double *b_curr = b; // pointer to current B subpanel column

    // //registers to hold 8x6 subblock of C
    // svfloat64_t c0_0, c0_1, c0_2, c0_3, c0_4, c0_5;
    // svfloat64_t c1_0, c1_1, c1_2, c1_3, c1_4, c1_5;
    // svfloat64_t c2_0, c2_1, c2_2, c2_3, c2_4, c2_5;
    // svfloat64_t c3_0, c3_1, c3_2, c3_3, c3_4, c3_5;

    // svfloat64_t* c_sub[4][6] = {
    //     {&c0_0, &c0_1, &c0_2, &c0_3, &c0_4, &c0_5},
    //     {&c1_0, &c1_1, &c1_2, &c1_3, &c1_4, &c1_5},
    //     {&c2_0, &c2_1, &c2_2, &c2_3, &c2_4, &c2_5},
    //     {&c3_0, &c3_1, &c3_2, &c3_3, &c3_4, &c3_5}
    // };

    // // registers to hold 4x12 subblock of C
    // // svfloat64_t c0_0, c0_1, c0_2, c0_3, c0_4, c0_5, c0_6, c0_7, c0_8, c0_9, c0_10, c0_11;
    // // svfloat64_t c1_0, c1_1, c1_2, c1_3, c1_4, c1_5, c1_6, c1_7, c1_8, c1_9, c1_10, c1_11;
    
    // // svfloat64_t* c_sub[2][12] = {
    // //     {&c0_0, &c0_1, &c0_2, &c0_3, &c0_4, &c0_5, &c0_6, &c0_7, &c0_8, &c0_9, &c0_10, &c0_11},
    // //     {&c1_0, &c1_1, &c1_2, &c1_3, &c1_4, &c1_5, &c1_6, &c1_7, &c1_8, &c1_9, &c1_10, &c1_11}
    // // };

    // // registers to hold 2x30 subblock of C
    // // svfloat64_t c0, c1, c2, c3, c4, c5, c6, c7, c8, c9;
    // // svfloat64_t c10, c11, c12, c13, c14, c15, c16, c17, c18, c19;
    // // svfloat64_t c20, c21, c22, c23, c24, c25, c26, c27, c28, c29;
 
    // // svfloat64_t* c_sub[30] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8, &c9,
    // //                          &c10, &c11, &c12, &c13, &c14, &c15, &c16, &c17, &c18, &c19,
    // //                          &c20, &c21, &c22, &c23, &c24, &c25, &c26, &c27};

    // svbool_t pred_0 = svwhilelt_b64(0, mr); // predicate to activate only mr rows that are valid
    // svbool_t pred_2 = svwhilelt_b64(2, mr);
    // svbool_t pred_4 = svwhilelt_b64(4, mr);
    // svbool_t pred_6 = svwhilelt_b64(6, mr);

    // for (int i = 0; i < nr; i++) { // only initialize valid columns of C based on columns in B
    //     *c_sub[0][i] = svdup_f64(0.0);
    //     *c_sub[1][i] = svdup_f64(0.0);
    //     *c_sub[2][i] = svdup_f64(0.0);
    //     *c_sub[3][i] = svdup_f64(0.0);
    // }

    // for (int j = 0; j < kc; j++) { // iterate through for every kc set of A column and B row in subpanels
    //     svfloat64_t a_col0 = svld1_f64(pred_0, a_curr); // load current column of A subpanel
    //     svfloat64_t a_col1 = svld1_f64(pred_2, a_curr + 2);        
    //     svfloat64_t a_col2 = svld1_f64(pred_4, a_curr + 4);        
    //     svfloat64_t a_col3 = svld1_f64(pred_6, a_curr + 6);        

    //     #pragma GCC unroll 6
    //     for (int k = 0; k < nr; k++) { // iterate through each column of B row
    //         svfloat64_t b_val = svdup_f64(b_curr[k]); // grab value from B and put in vector register
    //         *c_sub[0][k] = svmla_f64_m(pred_0, *c_sub[0][k], a_col0, b_val); // multiply accumulate into C subblock
    //         *c_sub[1][k] = svmla_f64_m(pred_2, *c_sub[1][k], a_col1, b_val);
    //         *c_sub[2][k] = svmla_f64_m(pred_4, *c_sub[2][k], a_col2, b_val);
    //         *c_sub[3][k] = svmla_f64_m(pred_6, *c_sub[3][k], a_col3, b_val);
    //     }

    //     a_curr += mr; // move to next column of A subpanel
    //     b_curr += nr; // move to next row of B subpanel
    // }

    // // store results back to C, remember C is row-major but c_sub is column-major
    // for (int i = 0; i < nr; i++) { // iterate through each column of c subblock
    //     double temp_c[8]; // temp array to hold c
    //     svst1_f64(pred_0, &temp_c[0], *c_sub[0][i]); // store column i of c_sub into temp c
    //     svst1_f64(pred_2, &temp_c[2], *c_sub[1][i]);
    //     svst1_f64(pred_4, &temp_c[4], *c_sub[2][i]);
    //     svst1_f64(pred_6, &temp_c[6], *c_sub[3][i]);

    //     for (int j = 0; j < mr; j++) { // iterate through each row of c subblock
    //         c[j * ldc + i] += temp_c[j]; // add to original C
    //     }
    //     // svfloat64_t c_orig = svld1_f64(pred_mr, c + i * ldc); // load one column of 2x30 C subblock
    //     // svfloat64_t c_new = svadd_f64_m(pred_mr, c_orig, *c_sub[i]); // add computed values
    //     // svst1_f64(pred_mr, c + i * ldc, c_new); // store back to C
    // }

    // PLAIN PACKING VERSION 1 (4.7)
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

    // PACKING VERSION 2 (6.8)
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