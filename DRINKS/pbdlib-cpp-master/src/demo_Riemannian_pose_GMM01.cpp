/*
 * demo_Riemannian_pose_GMM01.cpp
 *
 * GMM on R3 x S3.
 *
 * If this code is useful for your research, please cite the related publication:
 * @article{Calinon19,
 * 	author="Calinon, S. and Jaquier, N.",
 * 	title="Gaussians on {R}iemannian Manifolds for Robot Learning and Adaptive Control",
 * 	journal="arXiv:1909.05946",
 * 	year="2019",
 * 	pages="1--10"
 * }
 *
 * Author: Andras Kupcsik
 */

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <stdio.h>
#include <armadillo>
#include <cfloat>

using namespace arma;


static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//-----------------------------------------------

arma::mat QuatMatrix(arma::mat q) {
	arma::mat Q;
	Q = { {	q(0),-q(1),-q(2),-q(3)},
		{	q(1), q(0),-q(3), q(2)},
		{	q(2), q(3), q(0),-q(1)},
		{	q(3),-q(2), q(1), q(0)}};

	return Q;
}

//-----------------------------------------------

arma::colvec QuatVector(arma::mat Q) {
	arma::colvec q;
	q = Q.col(0);
	return q;
}

//-----------------------------------------------

mat QuatToRotMat(vec4 q) {
	float w = q(0);
	float x = q(1);
	float y = q(2);
	float z = q(3);
	mat RotMat(3, 3);
	RotMat << 1 - 2 * y * y - 2 * z * z << 2 * x * y - 2 * z * w << 2 * x * z + 2 * y * w << endr << 2 * x * y + 2 * z * w << 1 - 2 * x * x - 2 * z * z
			<< 2 * y * z - 2 * x * w << endr << 2 * x * z - 2 * y * w << 2 * y * z + 2 * x * w << 1 - 2 * x * x - 2 * y * y << endr;
	return RotMat;
}

//-----------------------------------------------

arma::mat acoslog(arma::mat x) {
	arma::mat acosx(1, x.size());

	for (int n = 0; n <= x.size() - 1; n++) {
		if (x(0, n) >= 1.0)
			x(0, n) = 1.0;
		if (x(0, n) <= -1.0)
			x(0, n) = -1.0;
		if (x(0, n) < 0) {
			acosx(0, n) = acos(x(0, n)) - M_PI;
		} else
			acosx(0, n) = acos(x(0, n));
	}

	return acosx;
}

//-----------------------------------------------

arma::mat Qexpfct(arma::mat u) {
	arma::mat normv = sqrt(pow(u.row(0), 2) + pow(u.row(1), 2) + pow(u.row(2), 2));
	arma::mat Exp(4, u.n_cols);

	Exp.row(0) = cos(normv);
	Exp.row(1) = u.row(0) % sin(normv) / normv;
	Exp.row(2) = u.row(1) % sin(normv) / normv;
	Exp.row(3) = u.row(2) % sin(normv) / normv;

	return Exp;
}

//-----------------------------------------------

arma::mat Qlogfct(arma::mat x) {
	arma::mat fullone;
	fullone.ones(size(x.row(0)));
	arma::mat scale(1, x.size() / 4);
	scale = acoslog(x.row(0)) / sqrt(fullone - pow(x.row(0), 2));

	if (scale.has_nan()) {
		for (int i = 0; i < x.n_cols; i++) {
			if (scale.row(0).has_nan()) {
				scale(0, i) = 1.0;
			}
		}
	}

	arma::mat Log(3, x.size() / 4);
	Log.row(0) = x.row(1) % scale;
	Log.row(1) = x.row(2) % scale;
	Log.row(2) = x.row(3) % scale;

	return Log;
}

//-----------------------------------------------

arma::mat Qexpmap(arma::mat u, arma::vec mu) {
	arma::mat x = QuatMatrix(mu) * Qexpfct(u);
	return x;
}

//-----------------------------------------------

arma::mat Qlogmap(arma::mat x, arma::vec mu) {
	arma::mat pole;
	arma::mat Q(4, 4, fill::ones);

	pole = {1,0,0,0};

	if (norm(mu - trans(pole)) < 1E-6)
		Q = { {	1,0,0,0},
			{	0,1,0,0},
			{	0,0,1,0},
			{	0,0,0,1}};
		else
		Q = QuatMatrix(mu);

	arma::mat u;
	u = Qlogfct(trans(Q) * x);

	return u;
}

//-----------------------------------------------

arma::mat Qtransp(vec g, vec h) {

	mat E;
	E << 0.0 << 0.0 << 0.0 << endr << 1.0 << 0.0 << 0.0 << endr << 0.0 << 1.0 << 0.0 << endr << 0.0 << 0.0 << 1.0;
	colvec tmpVec = zeros(4, 1);
	tmpVec.subvec(0, 2) = Qlogmap(h, g);

	vec vm = QuatMatrix(g) * tmpVec;
	double mn = norm(vm, 2);
	mat Ac;
	if (mn < 1E-10) {
		Ac = eye(3, 3);
	}

	colvec uv = vm / mn;
	mat Rpar = eye(4, 4) - sin(mn) * (g * uv.t()) - (1 - cos(mn)) * (uv * uv.t());
	Ac = E.t() * QuatMatrix(h).t() * Rpar * QuatMatrix(g) * E;

	return Ac;
}

//-----------------------------------------------

arma::mat transp(vec g, vec h) {
	mat Ac = eye(6, 6);

	if (norm(g - h, 2) == 0) {
		Ac = eye(6, 6);
	} else {
		Ac.submat(3, 3, 5, 5) = Qtransp(g.subvec(3, 6), h.subvec(3, 6));
	}

	return Ac;
}

//-----------------------------------------------

arma::mat logmap(mat x, vec mu) {
	mat u = join_vert(x.rows(0, 2), Qlogmap(x.rows(3, 6), mu.subvec(3, 6)));
	return u;
}

//-----------------------------------------------

arma::mat expmap(mat u, vec mu) {
	mat x = join_vert(u.rows(0, 2), Qexpmap(u.rows(3, 5), mu.subvec(3, 6)));
	return x;
}

//-----------------------------------------------

arma::vec gaussPDF(mat Data, colvec Mu, mat Sigma) {

	int nbVar = Data.n_rows;
	int nbData = Data.n_cols;
	Data = Data.t() - repmat(Mu.t(), nbData, 1);

	vec prob = sum((Data * inv(Sigma)) % Data, 1);

	prob = exp(-0.5 * prob) / sqrt(pow((2 * datum::pi), nbVar) * det(Sigma) + DBL_MIN);

	return prob;
}

//-----------------------------------------------

int main(int argc, char **argv) {

	arma_rng::set_seed_random();

	bool init = true;
	int nbStates = 5;
	int nbStates_prev = 5;
	int nbVar = 6;
	int nbSamples = 250;
	int nbVarMan = 7;
	int nbIterEM = 30;
	int nbIter = 10;		//Number of iteration for the Gauss Newton algorithm
	double params_diagRegFact = 1E-4;
	int nbDemo = 5;

	vec e0 = { 0, 0, 0, 1.0, 0.0, 0, 0.0 }; // center on manifold

	mat demo(nbVarMan, nbSamples);
	demo.load("./data/data_pose_gmm_XU.txt");

	mat demoX = demo; // data = [X, [U; 0]]
	mat demoU = demoX.submat(0, nbSamples, nbVar - 1, 2 * nbSamples - 1);
	demoX = demoX.submat(0, 0, nbVarMan - 1, nbSamples - 1);

	cout << "demoX ncols/nrows: " << demoX.n_cols << ", " << demoX.n_rows << endl;
	cout << "demoU ncols/nrows: " << demoU.n_cols << ", " << demoU.n_rows << endl;

	// ================== Recomputing tpgmm model with the latest task parameters ======================
	vec tmpVec;
	mat tmpCov;
	vec tmpEigenValues;
	mat tmpEigenVectors;
	vec Priors(nbStates);
	cube Sigma(nbVar, nbVar, nbStates);
	mat Mu(nbVar, nbStates);

	// ++++++++++++++++++++++ RANDOM INIT ++++++++++++++++++
	/*			arma_rng::set_seed_random();  // set the seed to a random value

	 Priors = ones(nbStates);
	 Priors /= nbStates;

	 // Random init the means and covariances
	 //cout << "init menas and covs" << endl;
	 // 5 x nbStates

	 for (int i = 0; i < nbStates; i++) { //for some reason the matrix randn doesn't work => doing it with vectors
	 for (int j = 0; j < (1 + 2 * (nbVar - 1)); j++) {
	 Mu(j, i) = randn();
	 }
	 }
	 for (int i = 0; i < nbStates; i++) {
	 Sigma.slice(i) = eye(1 + 2 * (nbVar - 1), 1 + 2 * (nbVar - 1)) * 0.5; //%Covariance in the tangent plane at point MuMan
	 }
	 */
	// +++++++++++++++++++++++ KBINS INIT +++++++++++++++++++++++
	vec nbSamplesCumSum = {1, 50, 100, 150, 200, 250 };

	if (init || (nbStates != nbStates_prev)) {
		cout << "Recomputing the model... " << endl;
		for (int i = 0; i < nbStates; i++) {
			rowvec id;

			for (int n = 0; n < nbDemo; n++) {
				vec indices = round(nbSamplesCumSum(n)-1 + linspace(0, nbSamples/nbDemo - 1, nbStates + 1));
				vec vecDummy = round(linspace(indices(i), indices(i + 1), indices(i + 1) - indices(i)));
				id = join_horiz(id, vecDummy.t());
			}

			Priors(i) = id.n_elem;

			mat demoUColID = zeros(nbVar, id.n_cols);

			for (int j = 0; j < id.n_elem; j++) {
				demoUColID.col(j) = demoU.col((unsigned int) id(j));
			}

			Mu.col(i) = mean(demoUColID, 1);
			Sigma.slice(i) = cov(demoUColID.t()) + eye(nbVar, nbVar) * 1E-4;
		}
		Priors = Priors / sum(Priors);

		int totalSamples = nbSamplesCumSum(nbSamplesCumSum.n_elem - 1);

		// +++++++++++++++++++++++++ INIT END +++++++++++++++++++++++++++++++++
		cout << " starting EM " << endl;

		mat MuMan = expmap(Mu, e0);
		Mu.rows(3, 5) = zeros(3, nbStates);

		mat utmp2;

		cube u(nbVar , nbSamples, nbStates);			//uTmp in matlab code

		for (int nb = 0; nb < nbIterEM; nb++) {

			//E-step
			mat L = zeros(nbStates, nbSamples);
			mat xcTmp;
			float multiplier = 1E-5;

			for (int i = 0; i < nbStates; i++) {

				xcTmp = logmap(demoX, MuMan.col(i));

				while (true) { // sometimes the starting Mu is far from our data => add bigger regularization
					L.row(i) = Priors(i) * (gaussPDF(xcTmp, Mu.col(i), Sigma.slice(i) + eye(nbVar, nbVar) * multiplier).t());
					if (norm(L.row(i), 2) == 0) {
						multiplier *= 10.0;
						//		cout << "EM step #" << i << ", multiplier: " << multiplier << endl;
					} else {
						break;
					}
				}
			}

			rowvec Lsum = sum(L, 0) + 1E-308;

			mat GAMMA = L / repmat(Lsum, nbStates, 1);

			colvec GammaSum = sum(GAMMA, 1);

			mat GAMMA2 = GAMMA / repmat(GammaSum, 1, nbSamples);

			//M-step
			for (int i = 0; i < nbStates; i++) {
				//Update Priors
				Priors(i) = sum(GAMMA.row(i)) / (nbSamples);

				//Update MuMan
				for (int n = 0; n < nbIter; n++) {
					u.slice(i) = logmap(demoX, MuMan.col(i));
					MuMan.col(i) = expmap(u.slice(i) * GAMMA2.row(i).t(), MuMan.col(i));
				}

				Mu.submat(0, i, 2, i) = MuMan.col(i).subvec(0, 2);

				utmp2 = u.slice(i);
				utmp2.each_col() -= Mu.col(i);

				//Update Sigma
				Sigma.slice(i) = utmp2 * diagmat(GAMMA2.row(i)) * utmp2.t() + eye( nbVar, nbVar) * params_diagRegFact;
			}
		}

		for (int i = 0; i < nbStates; i++) {
			cout << "============= Component #" << i << " ===========" << endl;
			cout << "Prior: " << Priors(i) << endl;
			cout << "MuMan:" << endl;
			cout << MuMan.col(i).t() << endl;
			cout << "Sigma: " << endl;
			cout << Sigma.slice(i) << endl << endl;
		}
	}

	return 0;
}
