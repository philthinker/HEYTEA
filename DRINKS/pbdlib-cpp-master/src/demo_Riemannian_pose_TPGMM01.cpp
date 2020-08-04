/*
 * demo_Riemannian_pose_TPGMM01.cpp
 *
 * TP-GMM on R3 x S3 with two frames.
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

	prob = exp(-0.5 * prob) / sqrt(pow((2 * datum::pi), nbVar) * det(Sigma));	// + DBL_MIN);

	return prob;
}

//-----------------------------------------------

int main(int argc, char **argv) {

	arma_rng::set_seed_random();

	int nbStates = 3;
	int nbStates_prev = 3;
	int nbVar = 6;
	int nbSamples = 531;
	int nbVarMan = 7;
	int nbIterEM = 30;
	int nbIter = 10;		//Number of iteration for the Gauss Newton algorithm
	double params_diagRegFact = 1E-4;
	int nbFrames = 2;
	vec e0 = { 0, 0, 0, 1.0, 0.0, 0, 0.0 }; // center on manifold
	int nbDemo = 8;

	// Load manifold data, first row is time, 2:4 is data in fame 1, 5:7 is data in frame 2
	mat demo(2 * nbVarMan, 2 * nbSamples);
	demo.load("./data/data_pose_tpgmm_XU2.txt");

	mat demoX = demo; // data = [X, [U; 0]]
	mat demoU = demoX.submat(0, nbSamples, 11, 2 * nbSamples - 1);
	demoX = demoX.submat(0, 0, 13, nbSamples - 1);

	cout << "demoX ncols/nrows: " << demoX.n_cols << ", " << demoX.n_rows << endl;
	cout << "demoU ncols/nrows: " << demoU.n_cols << ", " << demoU.n_rows << endl;

	mat A1;
	A1 = eye(6, 6);
	mat A2 = eye(6, 6);
	vec tmpq = { -0.034, 0.0684, 0.0097, -0.997 };
	A2.submat(0, 0, 2, 2) = QuatToRotMat(tmpq);
	A2.submat(3, 3, 5, 5) = QuatToRotMat(tmpq);
	colvec b1;
	b1 << 0.0 << endr << 0.0 << endr << 0.0 << endr << 0.0 << endr << 0.0 << endr << 0.0 << endr << 0.0;
	colvec b2;
	b2 << 0.7 << endr << -0.15 << endr << 1.0 << endr << 0.0 << endr << 0.0 << endr << 0.0 << endr << 0.0;

	mat MuManProduct;
	cube MuMan2;

	bool init = true;

	bool recompute = true;

	// ================== Recomputing tpgmm model with the latest task parameters ======================
	vec tmpVec;
	mat tmpCov;
	vec tmpEigenValues;
	mat tmpEigenVectors;
	vec Priors(nbStates);
	cube Sigma(nbVar * 2, nbVar * 2, nbStates);
	mat Mu(nbVar * 2, nbStates);

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
	if (init || (nbStates != nbStates_prev)) {
		cout << "Recomputing the model... " << endl;

		vec nbSamplesCumSum = { 0, 67, 143, 213, 284, 347, 417, 478, 531 };
		field<mat> tmpField(nbStates);


		for (int i = 0; i < nbSamplesCumSum.n_elem-1; i++) {
			vec id =  linspace(nbSamplesCumSum(i), nbSamplesCumSum(i + 1), nbSamplesCumSum(i + 1) - nbSamplesCumSum(i));
			vec idloc = round(linspace(0, id.n_elem, nbStates + 1));

			for (int j = 0; j < nbStates; j++) {
				vec idLoc = nbSamplesCumSum(i) + round(linspace(idloc(j), idloc(j+1)-1, idloc(j+1)-1-idloc(j)));
				for (int k = 0; k < idLoc.n_elem; k++) {
					tmpField(j) = join_horiz(tmpField(j), demoU.col((unsigned int) idLoc(k)));
				}
			}
		}

		for (int i = 0; i < nbStates; i++ ) {
			Mu.col(i) = mean(tmpField(i), 1);
			Sigma.slice(i) = cov(tmpField(i).t()) + eye(2*nbVar, 2*nbVar) * 1E-4;
			Priors(i) = tmpField(i).n_cols;
		}
		Priors = Priors / sum(Priors);

		nbSamples = nbSamplesCumSum(nbSamplesCumSum.n_elem - 1);

		mat MuMan = join_vert(expmap(Mu.rows(0, nbVar - 1), e0), expmap(Mu.rows(nbVar, 2 * nbVar - 1), e0));
		Mu.rows(3, 5) = zeros(3, nbStates);
		Mu.rows(9, 11) = zeros(3, nbStates); // center on manifold
		mat utmp2;

		cube u(nbVar * 2, nbSamples, nbStates);			//uTmp in matlab code

		for (int nb = 0; nb < nbIterEM; nb++) {

			//E-step
			mat L = zeros(nbStates, nbSamples);
			mat xcTmp;
			float multiplier = 1E-5;

			for (int i = 0; i < nbStates; i++) {

				xcTmp = join_vert(logmap(demoX.rows(0, 6), MuMan.submat(0, i, 6, i)), logmap(demoX.rows(7, 13), MuMan.submat(7, i, 13, i)));

				while (true) { // sometimes the starting Mu is far from our data => add bigger regularization
					L.row(i) = Priors(i) * (gaussPDF(xcTmp, Mu.col(i), Sigma.slice(i) + eye(12, 12) * multiplier).t());

					if (norm(L.row(i), 2) == 0) {
						multiplier *= 10.0;
					} else {
						break;
					}
				}
			}

			rowvec Lsum = sum(L, 0); // + 1E-308;

			mat GAMMA = L / repmat(Lsum, nbStates, 1);

			colvec GammaSum = sum(GAMMA, 1);
			cout << GammaSum << endl;

			mat GAMMA2 = GAMMA / repmat(GammaSum, 1, nbSamples);

			//M-step
			for (int i = 0; i < nbStates; i++) {
				//Update Priors
				Priors(i) = sum(GAMMA.row(i)) / (nbSamples);

				//Update MuMan
				for (int n = 0; n < nbIter; n++) {

					u.slice(i) = join_vert(logmap(demoX.rows(0, 6), MuMan.submat(0, i, 6, i)), logmap(demoX.rows(7, 13), MuMan.submat(7, i, 13, i)));

					MuMan.col(i) = join_vert(expmap(u.slice(i).rows(0, 5) * GAMMA2.row(i).t(), MuMan.submat(0, i, 6, i)),
							expmap(u.slice(i).rows(6, 11) * GAMMA2.row(i).t(), MuMan.submat(7, i, 13, i)));
				}

				Mu.submat(0, i, 2, i) = MuMan.col(i).subvec(0, 2);
				Mu.submat(6, i, 8, i) = MuMan.col(i).subvec(7, 9);
				utmp2 = u.slice(i);
				utmp2.each_col() -= Mu.col(i);

				//Update Sigma
				Sigma.slice(i) = utmp2 * diagmat(GAMMA2.row(i)) * utmp2.t() + eye(2 * nbVar, 2 * nbVar) * params_diagRegFact;
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

		//Reformatting as a tensor GMM
		mat MuManOld = MuMan;
		cube SigmaOld = Sigma;
		MuMan2 = zeros(nbVarMan, nbFrames, nbStates);
		field<cube> Sigma2(nbStates);
		for (int i = 0; i < nbStates; i++) {
			cube dummyCube(nbVar, nbVar, nbFrames);
			Sigma2.at(i) = dummyCube;
		}

		for (int i = 0; i < nbStates; i++) {
			for (int m = 0; m < nbFrames; m++) {

				// dummy way to construct id and idMan, works for 2 frames
				uvec id;
				uvec idMan;
				if (m == 0) {
					id = {0, 1, 2, 3, 4, 5};
				} else {
					id = {6, 7, 8, 9, 10, 11};
				}

				if (m == 0) {
					idMan = {0, 1, 2, 3, 4, 5, 6};
				} else {
					idMan = {7, 8, 9, 10, 11, 12, 13};
				}

				for (int ii = 0; ii < idMan.size(); ii++) {

					MuMan2.slice(i).col(m)(ii) = MuManOld.col(i)(idMan(ii));

				}

				Sigma2.at(i).slice(m) = SigmaOld.slice(i).submat(id, id);

			}
		}

		// Transforming gaussians with respective frames
		for (int i = 0; i < nbStates; i++) {

			// ============ Important ================
			// If you have real task parameters add the following #Frame lines:
			// b{#frame}.subvec(3, 6) = QuatVector(A.submat(3, 3, 6, 6));
			// this is important for proper expmap/logmap
			// =======================================

			vec uTmp = A1 * logmap(MuMan2.slice(i).col(0).rows(0, 6), e0);
			colvec norotq = { 1, 0, 0, 0 };
			b1.subvec(3, 6) = norotq; // manual
			MuMan2.slice(i).col(0).rows(0, 6) = expmap(uTmp, b1);
			MuMan2.slice(i).col(0).rows(0, 2) = MuMan2.slice(i).col(0).rows(0, 2) + b1.subvec(0, 2);

			mat Ac1 = transp(b1, MuMan2.slice(i).col(0).rows(0, 6));

			Sigma2.at(i).slice(0).submat(0, 0, 5, 5) = Ac1 * A1 * Sigma2.at(i).slice(0).submat(0, 0, 5, 5) * A1.t() * Ac1.t();

			uTmp = A2 * logmap(MuMan2.slice(i).col(1).rows(0, 6), e0);
			b2.subvec(3, 6) = tmpq; //manual
			MuMan2.slice(i).col(1).rows(0, 6) = expmap(uTmp, b2);
			MuMan2.slice(i).col(1).rows(0, 2) = MuMan2.slice(i).col(1).rows(0, 2) + b2.subvec(0, 2);
			mat Ac2 = transp(b2, MuMan2.slice(i).col(1).rows(0, 6));
			Sigma2.at(i).slice(1).submat(0, 0, 5, 5) = Ac2 * A2 * Sigma2.at(i).slice(1).submat(0, 0, 5, 5) * A2.t() * Ac2.t();

		}

		cout << MuMan2 << endl;
		cout << Sigma2 << endl;

		//Gaussian product
		MuManProduct = zeros(nbVarMan, nbStates);
		cube SigmaProduct(nbVar, nbVar, nbStates);

		for (int i = 0; i < nbStates; i++) {
			mat componentsMu = zeros(nbVarMan, nbVar); // current means of the components
			cube U0 = zeros(nbVar, nbVar, nbStates); //current covariances of the components

			vec tmpVec;
			mat tmpCov;
			vec tmpEigenValues;
			mat tmpEigenVectors;

			componentsMu.col(0) = MuMan2.slice(i).col(0).rows(0, 6);
			tmpCov = Sigma2.at(i).slice(0).submat(0, 0, 5, 5);

			eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);

			U0.slice(0) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

			componentsMu.col(1) = MuMan2.slice(i).col(1).rows(0, 6);

			tmpCov = Sigma2.at(i).slice(1).submat(0, 0, 5, 5);

			eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);

			U0.slice(1) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

			vec MuMan; // starting point on the manifold
			MuMan << 0.0 << endr << 0.0 << endr << 0.0 << endr << 1.0 << endr << 0.0 << endr << 0.0 << endr << 0.0;
			MuMan /= norm(MuMan, 2);
			mat Sigma = zeros(nbVar, nbVar);

			mat MuTmp = zeros(nbVar, nbVar); //nbVar (on tangent space) x components
			cube SigmaTmp = zeros(nbVar, nbVar, nbStates); // nbVar (on tangent space) x nbVar (on tangent space) x components

			for (int n = 0; n < 10; n++) { // compute the Gaussian product
				// nbVar = 3 for S3 sphere tangent space
				colvec Mu = zeros(nbVar, 1);
				mat SigmaSum = zeros(nbVar, nbVar);

				for (int j = 0; j < 2; j++) { // we have two frames
					mat Ac = transp(componentsMu.col(j), MuMan);
					mat U1 = Ac * U0.slice(j);
					SigmaTmp.slice(j) = U1 * U1.t();

					//Tracking component for Gaussian i
					SigmaSum += inv(SigmaTmp.slice(j));
					MuTmp.col(j) = logmap(componentsMu.col(j), MuMan);
					Mu += inv(SigmaTmp.slice(j)) * MuTmp.col(j);
				}

				Sigma = inv(SigmaSum);

				//Gradient computation
				Mu = Sigma * Mu;

				MuMan = expmap(Mu, MuMan);
			}

			MuManProduct.col(i) = MuMan;
			SigmaProduct.slice(i) = Sigma;

			cout << "================ Gaussian Product component #" << i << " ========================" << endl;
			cout << "MuMan: " << MuMan.t() << endl;
			cout << "Sigma: " << endl << Sigma << endl;
		}
	}

	return 0;
}
