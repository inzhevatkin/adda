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
    int* IPIV = malloc(N*sizeof(int));
    //LU factorization
    LAPACKE_zgetrf(LAPACK_ROW_MAJOR, N, N, inv_auxiliary, N, IPIV);
    LAPACKE_zgetri(LAPACK_ROW_MAJOR, N, inv_auxiliary, N, IPIV);
    for (i = 0; i != N; ++i){ //row
        for (j = 0; j != N; ++j){ //column
            idx = i*N + j;
            ro[j][i] = inv_auxiliary[idx];
        }
    }
    free(IPIV);
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
	doublecomplex sum;
	for (size_t j=0;j<rows;j++) { //row of first matrix
		for (size_t k=0;k<columns;k++) { //column of second matrix
			sum=0;
			for (size_t i=0;i<columns;i++) sum+=a[i][j]*b[k][i];
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
			res[j][k]=sum;
		}
	}
}

void mTAm(doublecomplex ** res, doublecomplex ** a, doublecomplex ** b){
	doublecomplex sum;

	for (size_t j=0;j<BLOCK_SIZE;j++) { // column of a
		for (size_t k=0;k<BLOCK_SIZE;k++) { // column of b
			sum=0;
			for (size_t i=0;i<local_nRows;i++) sum+=a[j][i]*b[k][i];
			res[j][k]=sum;
		}
	}
}


void X_new(doublecomplex ** res, doublecomplex ** p_old, doublecomplex ** alfa)
// x_new=x_old+p_old*alfa
{
	matrix_mult(pvec_koeff, p_old, alfa, local_nRows, BLOCK_SIZE);
	for (size_t j=0;j<BLOCK_SIZE;j++) {
		for (size_t k=0;k<local_nRows;k++) {
			res[j][k]+=pvec_koeff[j][k];
		}
	}
}

void R_new(doublecomplex ** res, doublecomplex ** r_old, doublecomplex ** Ap, doublecomplex ** alfa)
//r_new=r_old - A*p_old*alfa
{
	size_t j, k;
	matrix_mult(res, Ap, alfa, local_nRows, BLOCK_SIZE);
	for (j=0;j<BLOCK_SIZE;j++) {
		for (k=0;k<local_nRows;k++) {
			res[j][k]=-res[j][k]+r_old[j][k];
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

double find_max() {
	double sum_cur;
	double sum_max=0;
	for(size_t i=0;i<BLOCK_SIZE;i++){
		sum_cur=0;
		for(size_t j=0;j<local_nRows;j++){
			sum_cur += cAbs2(rvecArray_new[i][j]);
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
