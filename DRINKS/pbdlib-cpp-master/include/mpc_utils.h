// Interactive MPC helper functions

#pragma once

#include "armadillo"

/// Create (continous) chain of @order integrators 
void makeIntegratorChain(arma::mat* A, arma::mat* B, int order);
/// Discretise system matrices A, B
void discretizeSystem(arma::mat* _A, arma::mat* _B, arma::mat A, arma::mat B, double dt, bool zoh=true  );
/// Creates random (bivariate) covariances for a sequence of means @Mu
void randomCovariances(arma::cube* Sigma, const arma::mat& Mu, const arma::vec& covscale, bool semiTied, double theta, double minRnd=0.8 );
/// Computes r factor based on a maximum displacement d for an oscillatory movement with period @period
double SHM_r(double d, double period, int order);
/// Creates stepwise reference matrices given a sequence of Gaussians
void stepwiseReference( arma::mat* MuQ, arma::mat* Q, const arma::mat& Mu, const arma::cube& Sigma, int n, int order, int dim, double endWeight=1e-10 );
/// Creates via point reference matrices given a sequence of Gaussians
void viaPointReference( arma::mat* MuQ, arma::mat* Q, const arma::mat& Mu, const arma::cube& Sigma, int n, int order, int dim, double endWeight=1e-10 );
