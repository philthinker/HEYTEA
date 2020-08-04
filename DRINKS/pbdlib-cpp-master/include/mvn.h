/*
 * mvn.h
 *
 * Multivariate gaussian distribution related functions
 *
 * Original implementation in pbdlib, by Davide De Tommaso and Milad Malekzadeh
 *
 * Authors: Davide De Tommaso, Milad Malekzadeh, Philip Abbet
 */

#pragma once

#include <armadillo>


namespace mvn {

	arma::colvec getPDFValue(const arma::colvec& mu, const arma::mat& sigma,
							 const arma::mat& samples)
	{
		arma::colvec probs(samples.n_cols);

		arma::mat lambda = sigma.i();

		// Calculate difference (x - mu)
		arma::mat diff = arma::trans(samples) - arma::repmat(arma::trans(mu), samples.n_cols, 1);

		// Calculate exponential (x - mu)^T * inv(sigma) * (x - mu)
		probs = arma::sum((diff * lambda) % diff, 1);

		// Calculate exponential
		probs = sqrt(fabs(arma::det(lambda)) / pow(2 * arma::datum::pi, lambda.n_cols)) *
				exp(-0.5 * arma::abs(probs));

		return probs;
	}

}
