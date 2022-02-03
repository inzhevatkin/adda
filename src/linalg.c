/* File: linalg.c
 * $Date::                            $
 * Descr: different linear algebra operations for use with iterative solvers; highly specialized
 *
 *        Common feature of many functions is accepting timing argument. If it is not NULL, it is incremented by the
 *        time used for communication.
 *
 * Copyright (C) 2006-2008,2010-2014 ADDA contributors
 * This file is part of ADDA.
 *
 * ADDA is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * ADDA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ADDA. If not, see
 * <http://www.gnu.org/licenses/>.
 */
#include "linalg.h" // corresponding header
// project headers
#include "cmplx.h"
#include "comm.h"
#include "function.h"
#include "types.h"
#include "vars.h"
// system headers
#include <string.h>

/* There are several optimization ideas used in this file:
 * - If usage of some function has coinciding arguments, than a special function for such case is created. In
 * particular, this allows consistent usage of 'restrict' keyword almost for all function arguments.
 * - Deeper optimizations, such as loop unrolling, are left to the compiler.
 *
 * !!! TODO: Further optimizations (pragmas, or gcc attributes, e.g. 'expect') should be done only together with
 * profiling to see the actual difference
 */
//======================================================================================================================

void nInit(doublecomplex * restrict a)
// initialize vector a with null values
{
	register size_t i;
	register const size_t n=local_nRows;
	LARGE_LOOP;
	for (i=0;i<n;i++) a[i]=0;
}

//======================================================================================================================

void nCopy(doublecomplex * restrict a,const doublecomplex * restrict b)
// copy vector b to a (a=b); !!! they must not alias !!!
{
	memcpy(a,b,local_nRows*sizeof(doublecomplex));
}

//======================================================================================================================

double nNorm2(const doublecomplex * restrict a,TIME_TYPE *comm_timing)
// squared norm of a large vector a
{
	register size_t i;
	register const size_t n=local_nRows;
	double sum=0;

	LARGE_LOOP;
	for (i=0;i<n;i++) sum+=cAbs2(a[i]);
	// this function is not called inside the main iteration loop
	MyInnerProduct(&sum,double_type,1,comm_timing);
	return sum;
}

//======================================================================================================================

doublecomplex nDotProd(const doublecomplex * restrict a,const doublecomplex * restrict b,TIME_TYPE *comm_timing)
/* dot product of two large vectors; a.b; here the dot implies conjugation
 * !!! a and b must not alias !!! (to enforce use of function nNorm2 for coinciding arguments)
 */
{
	register size_t i;
	register const size_t n=local_nRows;
	doublecomplex sum=0;

	LARGE_LOOP;
	for (i=0;i<n;i++) sum+=a[i]*conj(b[i]);
	MyInnerProduct(&sum,cmplx_type,1,comm_timing);
	return sum;
}

//======================================================================================================================

doublecomplex nDotProd_conj(const doublecomplex * restrict a,const doublecomplex * restrict b,TIME_TYPE *comm_timing)
/* conjugate dot product of two large vectors; c=a.b*=b.a*; here the dot implies conjugation
 * !!! a and b must not alias !!! (to enforce use of function nDotProdSelf_conj for coinciding arguments)
 */
{
	register size_t i;
	register const size_t n=local_nRows;
	doublecomplex sum=0;

	LARGE_LOOP;
	for (i=0;i<n;i++) sum+=a[i]*b[i];
	MyInnerProduct(&sum,cmplx_type,1,comm_timing);
	return sum;
}

//======================================================================================================================

doublecomplex nDotProdSelf_conj(const doublecomplex * restrict a,TIME_TYPE *comm_timing)
// conjugate dot product of vector on itself; c=a.a*; here the dot implies conjugation
{
	register size_t i;
	register const size_t n=local_nRows;
	doublecomplex sum=0;

	LARGE_LOOP;
	/* Explicit writing the following through real and imaginary types can lead to delaying the multiplication by two
	 * until the sum is complete. But that is not believed to be significant
	 */
	for (i=0;i<n;i++) sum+=a[i]*a[i];
	MyInnerProduct(&sum,cmplx_type,1,comm_timing);
	return sum;
}

//======================================================================================================================

void equate_matrices(doublecomplex ** a, doublecomplex ** b) {
	for(size_t i=0;i<BLOCK_SIZE;i++) {
		// number of rows = local_nRows.
		nCopy(a[i], b[i]);
	}
}

void inv(doublecomplex ** ro){
	size_t i, j, idx;
	size_t N=BLOCK_SIZE;
    for (i = 0; i != N; ++i) { //row
        for (j = 0; j != N; ++j){ //column
            idx = i*N + j;
            inv_auxiliary[idx] = ro[j][i]; //(MAT[i][j]).real() + _Complex_I*(MAT[i][j]).imag();
        }
    }
    // Condition number.
	// see:
	// https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga188b8d30443d14b1a3f7f8331d87ae60.html
	// lapack_int LAPACKE_zgecon( int matrix_layout, char norm, lapack_int n, const lapack_complex_double* a,
	//								lapack_int lda, double anorm, double* rcond );
	//double rcond;
	//LAPACKE_zgecon(LAPACK_ROW_MAJOR, '1', N, inv_auxiliary, N, '1', &rcond);
	//fprintf(log,"%27.25f \n", rcond);


	// Norm.
	// see:
	// https://www.netlib.org/lapack/explore-html/de/d39/group__double_g_eauxiliary_gaefa80dbd8cd1732740478618b8b622a1.html
	// double LAPACKE_zlange( int matrix_layout, char norm, lapack_int m,
	//                           lapack_int n, const lapack_complex_double* a,
	//                           lapack_int lda );
	//double norm;
	//norm = LAPACKE_zlange(LAPACK_ROW_MAJOR, '1', N, N, inv_auxiliary, N);
	//fprintf(logfile,"Norm=%f \n", norm);

    int* IPIV = malloc(N*sizeof(int));
    // lapack_int LAPACKE_zgetrf( int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a,
    // lapack_int lda, lapack_int* ipiv )
    // matrix_layout - LAPACK_ROW_MAJOR/LAPACK_COL_MAJOR
    // lda - the size of the part of the matrix that we want to invert
    // IPIV - list of instructions of what rows to swap in order to invert the matrix
    LAPACKE_zgetrf(LAPACK_ROW_MAJOR, N, N, inv_auxiliary, N, IPIV); // LU factorization
    LAPACKE_zgetri(LAPACK_ROW_MAJOR, N, inv_auxiliary, N, IPIV);
    for (i = 0; i != N; ++i){ //row
        for (j = 0; j != N; ++j){ //column
            idx = i*N + j;
            ro[j][i] = inv_auxiliary[idx];
        }
    }
    free(IPIV);
}

void QR(doublecomplex ** b, doublecomplex ** R_, size_t rows, size_t columns){
	// The right side of the linear equation is fed to the input - b. When outputting, this matrix stores Q.
	// The matrix R stores R part from QR decomposition.
	//Copy data from 2D array (b) to auxiliary 1D array.
	//The use of an auxiliary array is not optimal, but I have not come up with another way.
	//Perhaps if you use LAPACK_COL_MAJOR, you can avoid using the auxiliary matrix.
	//To do this, you need to move away from 2D arrays in favor of 1D ones.
	size_t i, j;
	lapack_int info = 0;
	int lda = columns;
	lapack_complex_double *R_auxiliary, *QR_auxiliary, *tau;
	R_auxiliary = calloc(columns*columns, sizeof(lapack_complex_double));
	QR_auxiliary = calloc(rows*columns, sizeof(lapack_complex_double));
	for (i = 0; i != rows; ++i)
		for (j = 0; j != columns; ++j)
	        QR_auxiliary[i*columns + j] = b[j][i];
	tau = calloc(columns, sizeof(lapack_complex_double));
	info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, (int) rows, (int) columns, QR_auxiliary, lda, tau); // returns the Q, R in a packed format
	if(info != 0)
		fprintf(logfile,"LAPACKE_zgeqrf error. Output info != 0. \n");
	else
		fprintf(logfile,"LAPACKE_zgeqrf worked successfully. \n");
	// Copy the upper triangular Matrix R (columns x columns).
	for(i = 0; i < columns; ++i)
		memcpy(R_auxiliary+i*columns+i, QR_auxiliary+i*columns+i, (columns-i)*sizeof(doublecomplex));
	info = LAPACKE_zungqr(LAPACK_ROW_MAJOR, (int) rows, (int) columns, (int) columns, QR_auxiliary, lda, tau); // returns the Q in qr_auxiliary
	if(info != 0)
		fprintf(logfile,"LAPACKE_zungqr error. Output info != 0. \n");
	else
		fprintf(logfile,"LAPACKE_zungqr worked successfully. \n");
	// inverse copy data from auxiliary 1D array to 2D array (b).
	for (i = 0; i != rows; ++i) { //row
		for (j = 0; j != columns; ++j) { //column
			b[j][i] = QR_auxiliary[i*columns + j];
		}
	}
	for (i = 0; i != columns; ++i) { //row
		for (j = 0; j != columns; ++j) { //column
			R_[j][i] = R_auxiliary[i*columns + j];
		}
	}
	free(tau);
	free(R_auxiliary);
	free(QR_auxiliary);
}


void QR2(doublecomplex ** b, doublecomplex ** R_, size_t rows, size_t columns){
	// LAPACK_COL_MAJOR
	// The right side of the linear equation is fed to the input - b. When outputting, this matrix stores Q.
	// The matrix R stores R part from QR decomposition.
	// Copy data from 2D array (b) to auxiliary 1D array (QR_auxiliary).
	// The use of an auxiliary array is not optimal, but I have not come up with another way.
	// Perhaps if you use LAPACK_COL_MAJOR, you can avoid using the auxiliary matrix.
	// To do this, you need to move away from 2D arrays in favor of 1D ones.
	size_t i, j;
	int lda = rows;
	int ldc = rows;
	char side = 'R';
	char trans = 'N';
	lapack_complex_double *R_auxiliary, *QR_auxiliary, *tau, *C;
	R_auxiliary = calloc(columns*columns, sizeof(lapack_complex_double));
	QR_auxiliary = calloc(rows*columns, sizeof(lapack_complex_double));
	C = calloc(rows*columns, sizeof(lapack_complex_double));
	for (j = 0; j != columns; ++j) {
		for (i = 0; i != rows; ++i) {
			QR_auxiliary[i+j*rows] = b[j][i];
		}
	}
	for (j = 0; j != columns; ++j) {
		for (i = 0; i != rows; ++i) {
			if(i==j)
				C[i+j*rows] = 1;
			else
				C[i+j*rows] = 0;
		}
	}
	tau = calloc(columns, sizeof(lapack_complex_double));
	LAPACKE_zgeqrf(LAPACK_COL_MAJOR, (int) columns, (int) rows, QR_auxiliary, lda, tau); // returns the Q, R in a packed format
	//LAPACKE_zungqr(LAPACK_COL_MAJOR, (int) columns, (int) rows, (int) columns, QR_auxiliary, lda, tau); // returns the Q in qr_auxiliary
	LAPACKE_zunmqr(LAPACK_COL_MAJOR, side, trans,  (int) columns, (int) rows, (int) columns, QR_auxiliary, lda, tau, C, ldc);

	// inverse copy data from auxiliary 1D array to 2D array (b).
	for (j = 0; j != columns; ++j) {
		for (i = 0; i != rows; ++i) {
			b[j][i] = C[i+j*rows];
		}
	}
	/*for (i = 0; i != columns; ++i) { //row
		for (j = 0; j != columns; ++j) { //column
			R_[j][i] = R_auxiliary[i*columns + j];
		}
	}*/
	free(tau);
	free(R_auxiliary);
	free(QR_auxiliary);
	free(C);
}


// check: ||A-QR||/||A|| < threshold
bool QR_first_check(doublecomplex ** Q_, doublecomplex ** R_, doublecomplex ** A_, size_t rows, size_t columns, double thresh){
	size_t i ,j;
	lapack_complex_double *Q_auxiliary, *R_auxiliary, *QR_auxiliary, *A_auxiliary;
	Q_auxiliary = calloc(rows*columns, sizeof(lapack_complex_double));
	R_auxiliary = calloc(columns*columns, sizeof(lapack_complex_double));
	QR_auxiliary = calloc(rows*columns, sizeof(lapack_complex_double));
	A_auxiliary = calloc(rows*columns, sizeof(lapack_complex_double));
	for (i = 0; i != rows; ++i)
		for (j = 0; j != columns; ++j)
	        Q_auxiliary[i*columns + j] = Q_[j][i];
	for (i = 0; i != columns; ++i)
		for (j = 0; j != columns; ++j)
			R_auxiliary[i*columns + j] = R_[j][i];
	for (i = 0; i != rows; ++i)
		for (j = 0; j != columns; ++j)
			A_auxiliary[i*columns + j] = A_[j][i];

	// QR
	// C := alpha*op( A )*op( B ) + beta*C
	const doublecomplex alpha_zgemm = 1;
	const doublecomplex beta_zgemm = 0;
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, columns, &alpha_zgemm, Q_auxiliary,
				columns, R_auxiliary, columns, &beta_zgemm, QR_auxiliary, columns);
	// A - QR
	// We implement through the standard loop, not through the LAPACK/BLAS functionality.
	// QR_auxiliary stores the difference.
	for (i = 0; i != rows; ++i)
		for (j = 0; j != columns; ++j)
			QR_auxiliary[i*columns + j] = A_auxiliary[i*columns + j] - QR_auxiliary[i*columns + j];

	// ||A - QR||
	// Calculate Frobenius norm.
	double norm_num = LAPACKE_zlange(LAPACK_ROW_MAJOR, 'F', rows, columns, QR_auxiliary, columns);
	fprintf(logfile,"||A-QR|| = %.30f,\n", norm_num);
	// ||A||
	double norm_den = LAPACKE_zlange(LAPACK_ROW_MAJOR, 'F', rows, columns, A_auxiliary, columns);
	fprintf(logfile,"||A|| = %.30f,\n", norm_den);

	double ratio = norm_num / norm_den;
	fprintf(logfile,"||A-QR||/||A|| = %.30f,\n", ratio);

	free(Q_auxiliary);
	free(R_auxiliary);
	free(QR_auxiliary);
	free(A_auxiliary);

	if(ratio < thresh)
		return true;
	else
		return false;
}

// check:  ||I-Q^HQ|| < threshold
bool QR_second_check(doublecomplex ** Q_, size_t rows, size_t columns, double thresh) {
	size_t i, j;
	lapack_complex_double *Q_auxiliary;
	Q_auxiliary = calloc(rows*columns, sizeof(lapack_complex_double));
	for (i = 0; i != rows; ++i)
		for (j = 0; j != columns; ++j)
			Q_auxiliary[i*columns + j] = Q_[j][i];
	lapack_complex_double * QHQ = calloc(columns*columns, sizeof(lapack_complex_double));
	// Q^HQ
	// C := alpha*A*A**H + beta*C
	cblas_zherk(CblasRowMajor, CblasUpper, CblasConjTrans, columns, rows, 1, Q_auxiliary, columns, 0, QHQ, columns);
	// I-Q^HQ
	// QHQ stores the difference.
	for (i = 0; i != columns; ++i) {
		for (j = 0; j != columns; ++j) {
			if (i==j)
				QHQ[i*columns + j] = 1 - QHQ[i*columns + j];
			else
				QHQ[i*columns + j] = - QHQ[i*columns + j];
		}
	}
	// ||I-Q^HQ||
	double norm = LAPACKE_zlange(LAPACK_ROW_MAJOR, 'F', columns, columns, QHQ, columns);
	fprintf(logfile,"||I-Q^HQ|| = %.30f,\n", norm);

	free(QHQ);
	free(Q_auxiliary);

	if(norm < thresh)
		return true;
	else
		return false;
}

void sq_matrix_mult(doublecomplex ** res, doublecomplex ** a, doublecomplex ** b){
	size_t i, j, idx;
	size_t size = BLOCK_SIZE;
	doublecomplex * A = inv_auxiliary;
	doublecomplex * B = mutrix_mult_B_auxiliary;
	doublecomplex * C = mutrix_mult_C_auxiliary;
	for (i = 0; i != size; ++i) { //row
		for (j = 0; j != size; ++j){ //column
			idx = i*size + j;
			A[idx] = a[j][i];
			B[idx] = b[j][i];
		}
	}
	const CBLAS_LAYOUT layout=CblasRowMajor;//CblasRowMajor;
	const CBLAS_TRANSPOSE TRANS=CblasNoTrans; // no operation
	const int M = (int)size; // the number  of rows  of the  matrix op( A ),
	const int N = (int)size; // the number  of columns of the matrix op( B ),
	const int K = (int)size; // K  specifies  the number of columns of the matrix
	    		             // op( A ) and the number of rows of the matrix op( B ),
	double complex ALPHA = 1.0;
	const int LDA = (int)size;
	const int LDB = (int)size;
	doublecomplex BETA = 0.0;
	const int LDC = (int)size;
	cblas_zgemm(layout, TRANS, TRANS, M, N, K, &ALPHA, A, LDA, B, LDB, &BETA, C, LDC);
	for (i = 0; i != size; ++i){ //row
		for (j = 0; j != size; ++j){ //column
			idx = i*size + j;
			res[j][i] = C[idx];
		}
	}
}

void matrix_mult(doublecomplex ** res, doublecomplex ** a, doublecomplex ** b, size_t rows, size_t columns){
	// It is impossible to use here already implemented functions of vector products,
	// since these functions imply multiplication of vectors of size local_nRows.
	doublecomplex sum;
	size_t j, k, i;
	for (j=0;j<rows;j++) { //row of first matrix
		for (k=0;k<columns;k++) { //column of second matrix
			sum=0;
			for (i=0;i<columns;i++) sum+=a[i][j]*b[k][i];
			res[k][j]=sum;
		}
	}
}

void matrix_mult_sq(doublecomplex ** res, doublecomplex ** a, doublecomplex ** b, size_t size){
	doublecomplex sum;
	for (size_t j=0;j<size;j++) { //row of first matrix
		for (size_t k=0;k<size;k++) { //column of second matrix
			sum=0;
			for (size_t i=0;i<size;i++) sum+=a[i][j]*b[k][i];
			res[k][j]=sum;
		}
	}
}

void mTm(doublecomplex ** res, doublecomplex ** a) {
	size_t j, k, i;
	doublecomplex sum;

	for (j=0;j<BLOCK_SIZE;j++) { //columns
		for (k=0;k<BLOCK_SIZE;k++) { //columns
			sum=0;
			for (i=0;i<local_nRows;i++) sum+=a[j][i]*a[k][i];
			res[k][j]=sum;
		}
	}
}

void mTAm(doublecomplex ** res, doublecomplex ** a, doublecomplex ** b)
// res - BLOCK_SIZE*BLOCK_SIZE matrix
{
	doublecomplex sum;
	for (size_t j=0;j<BLOCK_SIZE;j++) { // column of a
		for (size_t k=0;k<BLOCK_SIZE;k++) { // column of b
			sum=0;
			for (size_t i=0;i<local_nRows;i++) sum+=a[j][i]*b[k][i];
			res[k][j]=sum;
		}
	}
}

void aTb(doublecomplex ** res, doublecomplex ** a, doublecomplex ** b, TIME_TYPE *comm_timing)
// res - BLOCK_SIZE*BLOCK_SIZE matrix
{
	for (size_t j=0;j<BLOCK_SIZE;j++) { // column of a
		for (size_t k=0;k<BLOCK_SIZE;k++) { // column of b
			res[k][j]=nDotProd_conj(a[j],b[k],comm_timing);
		}
	}
}


void X_new(doublecomplex ** res, doublecomplex ** p_old, doublecomplex ** alfa)
// x_new=x_old+p_old*alfa
{
	matrix_mult(pvec_koeff, p_old, alfa, local_nRows, BLOCK_SIZE);
	for (size_t j=0;j<BLOCK_SIZE;j++) { //column
		for (size_t k=0;k<local_nRows;k++) { //row
			res[j][k]+=pvec_koeff[j][k];
		}
	}
}

void R_new(doublecomplex ** res, doublecomplex ** r_old, doublecomplex ** Ap, doublecomplex ** alfa)
//r_new=r_old - A*p_old*alfa
{
	matrix_mult(res, Ap, alfa, local_nRows, BLOCK_SIZE);
	for (size_t j=0;j<BLOCK_SIZE;j++) {
		for (size_t k=0;k<local_nRows;k++) {
			res[j][k]=-res[j][k]+r_old[j][k];
		}
	}
}

void vector_new(doublecomplex ** res, doublecomplex ** a_old, doublecomplex ** b_old, doublecomplex ** koeff, int sign)
// a_new=a_old+sign*b_old*koeff
{
	matrix_mult(pvec_koeff, b_old, koeff, local_nRows, BLOCK_SIZE);
	for (size_t j=0;j<BLOCK_SIZE;j++) { //column
		for (size_t k=0;k<local_nRows;k++) { //row
			res[j][k]=a_old[j][k]+sign*pvec_koeff[j][k];
		}
	}
}

void P_new(doublecomplex ** res, doublecomplex ** r_new, doublecomplex ** p_old, doublecomplex ** beta)
// p_new=r_new+p_old*beta
{
	size_t j, k;
	matrix_mult(pvec_koeff, p_old, beta, local_nRows, BLOCK_SIZE);
	for (j=0;j<BLOCK_SIZE;j++) {
		for (k=0;k<local_nRows;k++) {
			res[j][k]=r_new[j][k]+pvec_koeff[j][k];
		}
	}
}

double find_max(void) {
	double sum_cur;
	double sum_max=0;
	for(size_t i=0;i<BLOCK_SIZE;i++){
		sum_cur=0;
		for(size_t j=0;j<local_nRows;j++){
			sum_cur += cAbs2(rvecArray[i][j]);
		}
		if(sum_max<sum_cur) sum_max=sum_cur;
	}
	return sum_max;
}

//======================================================================================================================

doublecomplex nDotProdSelf_conj_Norm2(const doublecomplex * restrict a,double * restrict norm,TIME_TYPE *comm_timing)
// Computes both conjugate dot product of vector on itself (c=a.a*) and its Hermitian squared norm=||a||^2
{
	register size_t i;
	register const size_t n=local_nRows;
	double buf[3]={0,0,0};

	LARGE_LOOP;
	// Here the optimization for explicit treatment seems significant, so we keep the old code
	for (i=0;i<n;i++) {
		buf[0]+=creal(a[i])*creal(a[i]);
		buf[1]+=cimag(a[i])*cimag(a[i]);
		buf[2]+=creal(a[i])*cimag(a[i]);
	}
	MyInnerProduct(buf,double_type,3,comm_timing);
	*norm=buf[0]+buf[1];
	return buf[0] - buf[1] + I*2*buf[2];
}

//======================================================================================================================

void nIncrem110_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	const doublecomplex c1,const doublecomplex c2)
// a=c1*a+c2*b+c; !!! a,b,c must not alias !!!
{
	register size_t i;
	register const size_t n=local_nRows;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] = c1*a[i] + c2*b[i] + c[i];
}

//======================================================================================================================

void nIncrem011_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	const doublecomplex c1,const doublecomplex c2)
// a+=c1*b+c2*c; !!! a,b,c must not alias !!!
{
	register size_t i;
	register const size_t n=local_nRows;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] += c1*b[i] + c2*c[i];
}

//======================================================================================================================

void nIncrem110_d_c_conj(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	const double c1,const doublecomplex c2,double * restrict inprod,TIME_TYPE *comm_timing)
/* a=c1*a(*)+c2*b(*)+c; one constant is real, another - complex, vectors a and b are conjugated during the evaluation;
 * !!! a,b,c must not alias !!!
 */
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = c1*conj(a[i]) + c2*conj(b[i]) + c[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = c1*conj(a[i]) + c2*conj(b[i]) + c[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nIncrem111_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	const doublecomplex c1,const doublecomplex c2,const doublecomplex c3)
// a=c1*a+c2*b+c3*c; !!! a,b,c must not alias !!!
{
	register size_t i;
	register const size_t n=local_nRows;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] = c1*a[i] + c2*b[i] + c3*c[i];
}

//======================================================================================================================

void nIncrem(doublecomplex * restrict a,const doublecomplex * restrict b,double * restrict inprod,
	TIME_TYPE *comm_timing)
// a+=b, inprod=|a|^2; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] += b[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] += b[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nDecrem(doublecomplex * restrict a,const doublecomplex * restrict b,double * restrict inprod,
	TIME_TYPE *comm_timing)
// a-=b, inprod=|a|^2; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] -= b[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] -= b[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nIncrem01(doublecomplex * restrict a,const doublecomplex * restrict b,const double c,double * restrict inprod,
	TIME_TYPE *comm_timing)
// a=a+c*b, inprod=|a|^2; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] += c*b[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] += c*b[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nIncrem10(doublecomplex * restrict a,const doublecomplex * restrict b,const double c,double * restrict inprod,
	TIME_TYPE *comm_timing)
// a=c*a+b, inprod=|a|^2; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = c*a[i] + b[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = c*a[i] + b[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nIncrem11_d_c(doublecomplex * restrict a,const doublecomplex * restrict b,const double c1,const doublecomplex c2,
	double * restrict inprod,TIME_TYPE *comm_timing)
// a=c1*a+c2*b, inprod=|a|^2 , one constant is double, another - complex; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = c1*a[i] + c2*b[i];
	}
	else {
		*inprod=0.0;
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = c1*a[i] + c2*b[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nIncrem01_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex c,
	double * restrict inprod,TIME_TYPE *comm_timing)
// a=a+c*b, inprod=|a|^2; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] += c*b[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] += c*b[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nIncrem10_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex c,
	double * restrict inprod,TIME_TYPE *comm_timing)
// a=c*a+b, inprod=|a|^2; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = c*a[i] + b[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = c*a[i] + b[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nLinComb_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	const doublecomplex c1,const doublecomplex c2,double * restrict inprod,TIME_TYPE *comm_timing)
// a=c1*b+c2*c, inprod=|a|^2; !!! a,b,c must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = c1*b[i] + c2*c[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = c1*b[i] + c2*c[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nLinComb1_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	const doublecomplex c1,double * restrict inprod,TIME_TYPE *comm_timing)
// a=c1*b+c, inprod=|a|^2; !!! a,b,c must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = c1*b[i] + c[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = c1*b[i] + c[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nLinComb1_cmplx_conj(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	const doublecomplex c1,double * restrict inprod,TIME_TYPE *comm_timing)
// a=c1*b(*)+c, inprod=|a|^2; !!! a,b,c must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = c1*conj(b[i]) + c[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = c1*conj(b[i]) + c[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}


//======================================================================================================================

void nSubtr(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex * restrict c,
	double * restrict inprod,TIME_TYPE *comm_timing)
// a=b-c, inprod=|a|^2; !!! a,b,c must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;
	double sum=0;

	if (inprod==NULL) {
		LARGE_LOOP;
		for (i=0;i<n;i++) a[i] = b[i] - c[i];
	}
	else {
		LARGE_LOOP;
		for (i=0;i<n;i++) {
			a[i] = b[i] - c[i];
			sum += cAbs2(a[i]);
		}
		(*inprod)=sum;
		MyInnerProduct(inprod,double_type,1,comm_timing);
	}
}

//======================================================================================================================

void nMult(doublecomplex * restrict a,const doublecomplex * restrict b,const double c)
// multiply vector by a real constant; a=c*b; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] = c*b[i];
}

//======================================================================================================================

void nMult_cmplx(doublecomplex * restrict a,const doublecomplex * restrict b,const doublecomplex c)
// multiply vector by a complex constant; a=c*b; !!! a and b must not alias !!!
{
	register const size_t n=local_nRows;
	register size_t i;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] = c*b[i];
}
//======================================================================================================================

void nMultSelf(doublecomplex * restrict a,const double c)
// multiply vector by a real constant; a*=c
{
	register const size_t n=local_nRows;
	register size_t i;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] *= c;
}
//======================================================================================================================

void nMultSelf_conj(doublecomplex * restrict a,const double c)
// conjugate vector and multiply it by a real constant; a=c*a(*)
{
	register const size_t n=local_nRows;
	register size_t i;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] = c*conj(a[i]);
}

//======================================================================================================================

void nMultSelf_cmplx(doublecomplex * restrict a,const doublecomplex c)
// multiply vector by a complex constant; a*=c
{
	register const size_t n=local_nRows;
	register size_t i;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i] *= c;
}

//======================================================================================================================

void nMult_mat(doublecomplex * restrict a,const doublecomplex * restrict b,/*const*/ doublecomplex (* restrict c)[3])
/* multiply by a function of material of a dipole and component; a[3*i+j]=c[mat[i]][j]*b[3*i+j]
 * !!! a,b,c must not alias !!!
 * It seems impossible to declare c as constant (due to two pointers)
 */
{
	register const size_t nd=local_nvoid_Ndip; // name 'nd' to distinguish with 'n' used elsewhere
	register size_t i,k;
	/* Hopefully, the following declaration is enough to allow efficient loop unrolling. So the compiler should
	 * understand that none of the used vectors alias. Otherwise, deeper optimization should be used.
	 */
	const doublecomplex * restrict val;

	LARGE_LOOP;
	for (i=0,k=0;i<nd;i++,k+=3) {
		val=c[material[i]];
		a[k] = val[0]*b[k];
		a[k+1] = val[1]*b[k+1];
		a[k+2] = val[2]*b[k+2];
	}
}

//======================================================================================================================

void nMultSelf_mat(doublecomplex * restrict a,/*const*/ doublecomplex (* restrict c)[3])
/* multiply by a function of material of a dipole and component; a[3*i+j]*=c[mat[i]][j]
 * !!! a and c must not alias !!!
 * It seems impossible to declare c as constant (due to two pointers)
 */
{
	register const size_t nd=local_nvoid_Ndip; // name 'nd' to distinguish with 'n' used elsewhere
	register size_t i,k;
	/* Hopefully, the following declaration is enough to allow efficient loop unrolling. So the compiler should
	 * understand that none of the used vectors alias. Otherwise, deeper optimization should be used.
	 */
	const doublecomplex * restrict val;

	LARGE_LOOP;
	for (i=0,k=0;i<nd;i++,k+=3) {
		val=c[material[i]];
		a[k] *= val[0];
		a[k+1] *= val[1];
		a[k+2] *= val[2];
	}
}

//======================================================================================================================

void nConj(doublecomplex * restrict a)
// complex conjugate of the vector
{
	register const size_t n=local_nRows;
	register size_t i;

	LARGE_LOOP;
	for (i=0;i<n;i++) a[i]=conj(a[i]);
}
