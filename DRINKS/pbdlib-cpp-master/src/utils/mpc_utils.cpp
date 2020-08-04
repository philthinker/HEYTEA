// Interactive MPC helper functions

#include "mpc_utils.h"
//#include <LinAlg/LapackWrapperExtra.h>
//#include <SymEigsSolver.h>

#define PI 3.14159265359
// We are going to calculate the eigenvalues of M

using namespace arma;
/*
void symEigs( arma::vec* Dv, arma::mat* V, arma::mat A, int numEigs, int convFactor)
{
	// Construct matrix operation object using the wrapper class DenseGenMatProd
		DenseGenMatProd<double> op(A);
		// Construct eigen solver object, requesting the largest three eigenvalues
		SymEigsSolver< double, LARGEST_ALGE, DenseGenMatProd<double> > eigs(&op, numEigs, convFactor);

		eigs.init();
		int nconv = eigs.compute();

		if(nconv > 0)
		{
			*Dv = eigs.eigenvalues();
			*V = eigs.eigenvectors(nconv);
		}
		else
		{
			*Dv = arma::zeros(2);
			*V = arma::eye(2,2);
		}
}
*/

double factorials[15] =
{
   1,
	1,
	2,
	6,
	24,
	120,
	720,
	5040,
	40320,
	362880,
	3628800,
	39916800,
	479001600,
	6227020800,
	87178291200
};


static mat transformSigma( const mat& m, const mat& Sigma)
{
	mat V;
	vec d;
	eig_sym(d, V, Sigma);
	mat D = diagmat(d);
	V = m*V;
	return V*D*inv(V);
}

arma::mat rot2d( double theta, bool affine )
{
	int d = affine?3:2;
	arma::mat m = arma::eye(d,d);

	double ct = cos(theta);
	double st = sin(theta);

	m(0,0) = ct; m(0,1) = -st;
	m(1,0) = st; m(1,1) = ct;

	return m;
}

static mat makeSigma( float rot, vec scale )
{
	mat Q = rot2d(rot, false);
	mat Lambda = diagmat(scale%scale);
	return Q * Lambda * inv(Q);
}

/// Inits the system matrices with a chain of integrators
void makeIntegratorChain(arma::mat* A, arma::mat* B, int order)
{
	*A = zeros(order, order);
	A->submat(0, 1, order-2, order-1)
	= eye(order-1, order-1);
	*B = join_vert(zeros(order-1,1),
					  ones(1,1));
}

void discretizeSystem(arma::mat* _A, arma::mat* _B, arma::mat A, arma::mat B, double dt, bool zoh  )
{
	if(zoh)
	{
		// matrix eponential trick
		// From Linear Systems: A State Variable Approach with Numerical Implementation pg 215
		// adapted from scipy impl

		mat EM = join_horiz(A, B);
		mat zero = join_horiz(zeros(B.n_cols, B.n_rows),
							  zeros(B.n_cols, B.n_cols));
		EM = join_vert(EM, zero);
		mat M = expmat(EM * dt);
		*_A = M.submat(span(0, A.n_rows-1), span(0, A.n_cols-1));
		*_B = M.submat(span(0, B.n_rows-1), span(A.n_cols, M.n_cols-1));
	}
	else  // euler
	{
		*_A = eye(A.n_rows, A.n_cols) + A*dt;
		*_B = B * dt;
	}
}

/// Creates random gaussians as targets for LQR constraints
void randomCovariances(arma::cube* Sigma, const arma::mat& Mu, const arma::vec& covscale, bool semiTied, double theta, double minRnd )
{
	int m = Mu.n_cols;

	*Sigma = zeros(2,2,m);
	vec s;

	for( int i = 0; i < m; i++ )
	{
		vec s = (minRnd+(randu(2)*(1.-minRnd))) % covscale;
		if(!semiTied)
			theta = -PI + arma::randu()*2.*PI;
		Sigma->slice(i) = makeSigma(theta, s);
	}
}

double SHM_r(double d, double period, int order)
{
	double omega = (2. * PI) / period;
	return 1. / pow(d * pow(omega,order), 2);
}

arma::ivec stepwiseIndices(int m, int n)
{
	return arma::conv_to<arma::ivec>::from(
			arma::linspace(0, (float)m-0.1, n) );
}

arma::ivec viaPointIndices(int m, int n)
{
	float a = 0.;
	float b = n-1;

	ivec inds = zeros<ivec>(m);
	for( int i = 0; i < m; i++ )
		inds(i) = (int)(a + i*(b-a)/(m-1));
	return inds;
}

void stepwiseReference( arma::mat* MuQ, arma::mat* Q, const mat& Mu, const cube& Sigma, int n, int order, int dim, double endWeight )
{
	int m = Mu.n_cols;
	int cDim = order*dim;
	int muDim = Mu.n_rows;

	// equally distributed optimal states
	// Could try something along the lines of Fitts here based on covariance.
	arma::ivec qList =	stepwiseIndices(m, n);

	// precision matrices
	mat Lambda = zeros(muDim, m*muDim);
	for( int i = 0; i < m; i++ )
		Lambda.cols(i*muDim, muDim*(i+1)-1) = inv(Sigma.slice(i));

	*Q = zeros(cDim*n, cDim*n); // Precision matrix
	*MuQ = zeros(cDim, n); // Mean vector

	for( int i = 0; i < qList.n_rows; i++ )
	{
		// Precision matrix based on state sequence
		Q->submat(i*cDim,
				 i*cDim,
				 i*cDim+muDim-1,
				 i*cDim+muDim-1) =
		Lambda.cols(qList[i]*muDim, (qList[i]+1)*muDim-1);

		// Mean vector
		MuQ->submat(span(0,muDim-1), span(i,i)) = Mu.col(qList[i]);
	}

	if(endWeight > 0.0)
	{
		// Set last value for Q to enforce 0 end condition
		int ind = qList.n_rows-1;
		for( int i = 2; i < cDim; i++ )
			Q->operator()(ind*cDim+i, ind*cDim+i) = endWeight;
	}
}

void viaPointReference( arma::mat* MuQ, arma::mat* Q, const mat& Mu, const cube& Sigma, int n, int order, int dim, double endWeight )
{
	int m = Mu.n_cols;
	int cDim = order*dim;
	int muDim = Mu.n_rows;

	// equally distributed optimal states
	// Could try something along the lines of Fitts here based on covariance.
	arma::ivec qList =	stepwiseIndices(m, n);

	// precision matrices
	mat Lambda = zeros(muDim, m*muDim);
	for( int i = 0; i < m; i++ )
		Lambda.cols(i*muDim, muDim*(i+1)-1) = inv(Sigma.slice(i));

	*Q = zeros(cDim*n, cDim*n); // Precision matrix
	*MuQ = zeros(cDim, n); // Mean vector

	ivec vI = viaPointIndices(m, n);
	for( int i = 0; i < vI.n_rows; i++ )
	{
		int t = vI[i];
		Q->submat(t*cDim,
				 t*cDim,
				 t*cDim+muDim-1,
				 t*cDim+muDim-1)
		=
		Lambda.cols(i*muDim, (i+1)*muDim-1);

		MuQ->submat(span(0,muDim-1), span(t,t)) = Mu.col(i);
	}


	if(endWeight > 0.0)
	{
		// Set last value for Q to enforce 0 end condition
		int ind = n-1;
		for( int i = 2; i < cDim; i++ )
			Q->operator()(ind*cDim+i, ind*cDim+i) = endWeight;

	}
}
