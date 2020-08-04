/*
 * lqr.h
 *
 * Linear quadratic regulation related functions
 *
 * Original implementation in pbdlib, by Danilo Bruno and Sylvain Calinon
 *
 * Authors: Danilo Bruno, Sylvain Calinon, Philip Abbet
 */

#pragma once

#include <armadillo>


namespace lqr {

	arma::mat solve_algebraic_Riccati(const arma::mat& A, const arma::mat& B,
									  const arma::mat& Q, const arma::mat& R)
	{
		const int n = A.n_rows;

		arma::mat G = B * R.i() * B.t();

		arma::mat Z(2 * n, 2 * n);
		Z(arma::span(0, n - 1), arma::span(0, n - 1)) = A;
		Z(arma::span(n, 2 * n - 1), arma::span(0, n - 1)) = -Q;
		Z(arma::span(0, n - 1), arma::span(n, 2 * n - 1)) = -G;
		Z(arma::span(n, 2 * n - 1), arma::span(n, 2 * n - 1)) = -A.t();

		//Using diagonalization
		arma::cx_mat U(2 * n, n);
		arma::cx_vec dd(2 * n);
		arma::cx_mat V(2 * n, 2 * n);

		arma::eig_gen(dd, V, Z);
		int i = 0;
		for (int j = 0; j < 2 * n; ++j) {
			if (real(dd(j)) < 0) {
				U.col(i) = V.col(j);
				++i;
			}
		}

		arma::mat S1 = arma::zeros(n, n);
		arma::cx_mat Sc = U(arma::span(0, n - 1), arma::span(0, n - 1)).t().i() *
						  U(arma::span(n, 2 * n - 1), arma::span(0, n - 1)).t();

		return arma::real(Sc);
	};


	//-----------------------------------------------------


	arma::mat solve_algebraic_Riccati_discrete(const arma::mat& A, const arma::mat& B,
											   const arma::mat& Qx, const arma::mat& R)
	{
		// Ajay Tanwani, 2016

		int n = A.n_rows;

		arma::mat Q = (Qx + Qx.t()) / 2;  // mat Q here corresponds to (Q+Q')/2 specified
										  // in set_problem
		arma::mat G = B * R.i() * B.t();
		arma::mat Z(2 * n , 2 * n);
		Z(arma::span(0, n - 1), arma::span(0, n - 1)) = A + G * arma::inv(A.t()) * Q;
		Z(arma::span(0, n - 1), arma::span(n, 2 * n - 1)) = -G * arma::inv(A.t());
		Z(arma::span(n, 2 * n - 1), arma::span(0, n - 1)) = -arma::inv(A.t()) * Q;
		Z(arma::span(n, 2 * n - 1), arma::span(n, 2 * n - 1)) = arma::inv(A.t());

		// Using diagonalization
		arma::cx_mat V(2 * n, 2 * n), U(2 * n, n);
		arma::cx_vec dd(2 * n);

		arma::eig_gen(dd, V, Z);

		int i = 0;
		for (int j = 0; j < 2 * n; j++) {
			if (norm(dd(j)) < 1) {
				U.col(i) = V.col(j);
				i++;
			}
		}

		return arma::real(U(arma::span(n, 2 * n - 1), arma::span(0, n - 1)) *
						  U(arma::span(0, n - 1), arma::span(0, n - 1)).i());
	};


	//-----------------------------------------------------


	std::vector<arma::mat> evaluate_gains_infinite_horizon(const arma::mat& A,
														   const arma::mat& B,
														   const arma::mat& R,
														   const std::vector<arma::mat>& Q,
														   const arma::mat& Target)
	{
		const int nb_data = Q.size();
		const int nb_var = A.n_rows;
		const int nb_ctrl = B.n_cols;

		std::vector<arma::mat> S; // Riccati solution
		std::vector<arma::mat> L; // LQR Gains

		for (int i = 0; i < nb_data; ++i) {
			S.push_back(arma::zeros(nb_var, nb_var));
			L.push_back(arma::zeros(nb_ctrl, nb_var));
		}

		arma::mat invR = R.i();
		for (int t = 0; t < nb_data; ++t) {
			S[t] = solve_algebraic_Riccati(A, B, Q[t], R);
			L[t] = invR * B.t() * S[t];
		}

		return L;
	};
}
