/*
 * demo_Riemannian_S2_TPGMM01.cpp
 *
 * TPGMM on sphere with two frames.
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
 * Author: kupcsik
 */

#include <stdio.h>
#include <armadillo>

#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>

using namespace arma;


/****************************** HELPER FUNCTIONS *****************************/

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//-----------------------------------------------

arma::mat trans2cov(const ui::Trans2d& trans) {
	arma::mat RG = {
		{ trans.x.x / 200.0, trans.y.x / 200.0 },
		{ -trans.x.y / 200.0, -trans.y.y / 200.0 }
	};

	return RG * RG.t();
}

//-----------------------------------------------

arma::mat rotM(arma::vec v) {
	arma::vec st = v / norm(v);
	float acangle = st(2);
	float cosa = acangle;
	float sina = sqrt(1 - pow(acangle, 2));

	if ((1 - acangle) > 1E-16) {
		v << st(1) << endr << -st(0) << endr << 0 << endr;
		v = v / sina;
	} else {
		v = zeros(3);
	}

	float vera = 1 - cosa;
	float x = v(0);
	float y = v(1);
	float z = v(2);

	arma::mat rot;
	rot = {
		{ cosa + pow(x, 2) * vera, x * y * vera - z * sina, x * z * vera + y * sina },
		{ x * y * vera + z * sina, cosa + pow(y, 2) * vera, y * z * vera - x * sina },
		{ x * z * vera - y * sina, y * z * vera + x * sina, cosa + pow(z, 2) * vera }
	};

	return rot;
}

//-----------------------------------------------

arma::mat expfct(arma::mat u) {
	arma::mat normv = sqrt(pow(u.row(0), 2) + pow(u.row(1), 2));
	arma::mat Exp(3, u.n_cols);

	Exp.row(0) = u.row(0) % sin(normv) / normv;
	Exp.row(1) = u.row(1) % sin(normv) / normv;
	Exp.row(2) = cos(normv);

	return Exp;
}

//-----------------------------------------------

arma::mat logfct(arma::mat x) {

	arma::mat fullone;
	fullone.ones(size(x.row(2)));
	arma::mat scale(1, x.n_cols);
	scale = acos(x.row(2)) / sqrt(fullone - pow(x.row(2), 2));
	mat ac = acos(x.row(2));
	mat sq = sqrt(fullone - pow(x.row(2), 2));

	arma::mat Log(2, x.n_cols);
	Log.row(0) = x.row(0) % scale;
	Log.row(1) = x.row(1) % scale;

	return Log;
}

//-----------------------------------------------

arma::mat expmap(arma::mat u, arma::vec mu) {
	arma::mat x = trans(rotM(mu)) * expfct(u);

	return x;
}

//-----------------------------------------------

arma::mat logmap(arma::mat x, arma::vec mu) {
	arma::mat pole;
	pole = {0, 0, 1};
	arma::mat R(3, 3, fill::ones);

	if (norm(mu - trans(pole)) < 1E-6) {
		R = { {	1, 0, 0},
			{	0, -1, 0},
			{	0, 0, -1}};
	}
	else {
		R = rotM(mu);
	}

	arma::mat u;
	u = logfct(R * x);

	return u;
}

//-----------------------------------------------

arma::mat transp(vec g, vec h) {
	mat E;
	E << 1.0 << 0.0 << endr << 0.0 << 1.0 << endr << 0.0 << 0.0;
	vec logmapTmp = logmap(h, g);
	vec logmapTmp_bottom = zeros(1, 1);

	vec vm = rotM(g).t() * arma::join_vert(logmapTmp, logmapTmp_bottom);
	double mn = norm(vm, 2);
	vec uv = vm / (mn);
	mat Rpar = arma::eye(3, 3) - sin(mn) * (g * uv.t()) - (1 - cos(mn)) * (uv * uv.t());
	mat Ac = E.t() * rotM(h) * Rpar * rotM(g).t() * E; //Transportation operator from g to h
	return Ac;

}

//-----------------------------------------------

arma::mat sample_gaussian_points(const ui::Trans2d& gaussian, const arma::vec& target) {

	arma::mat uCov = trans2cov(gaussian);

	arma::mat V;
	arma::vec d;
	arma::eig_sym(d, V, uCov);

	arma::mat sample_points({
		{ 0.0f, 0.0f },
		{ 1.0f, 0.0f },
		{ 0.0f, 1.0f },
		{ -1.0f, 0.0f },
		{ 0.0f, -1.0f },
	});

	arma::mat points = expmap(V * diagmat(sqrt(d)) * sample_points.t(), target);

	points.col(0) = target;

	return points;
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


/********************************* RENDERING *********************************/

//-----------------------------------------------------------------------------
// Create a gaussian mesh, colored (no lightning)
//-----------------------------------------------------------------------------
gfx2::model_t create_gaussian(const arma::fvec& color, const ui::Trans2d& gaussian,
							  const arma::vec& target) {

	arma::mat uCov = trans2cov(gaussian);

	arma::rowvec tl = arma::linspace<arma::rowvec>(-arma::datum::pi, arma::datum::pi, 100);

	arma::mat V;
	arma::vec d;
	arma::eig_sym(d, V, uCov);

	arma::mat points = expmap(V * diagmat(sqrt(d)) * join_cols(cos(tl), sin(tl)), target);

	return gfx2::create_line(color, points);
}


/********************************* FUNCTIONS *********************************/

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	arma::vec target( { 0.0, 0.2, 0.6 });
	target = target / norm(target);

	arma::vec target2( { 0.2, 0.2, 0.2 });
	target2 = target2 / norm(target2);

	arma::vec target_product( { -1.0, -1.0, 0.0 });
	target_product = target_product / norm(target_product);

	ui::Trans2d gaussian(ImVec2(50, 0), ImVec2(0, 50), ImVec2(50, 680));
	ui::Trans2d gaussian_gmm(ImVec2(50, 0), ImVec2(0, 100), ImVec2(50, 500));

	// Initialise GLFW
	gfx2::window_size_t window_size;
	window_size.win_width = 800;
	window_size.win_height = 800;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;

	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	// Open a window and create its OpenGL context
	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Demo Riemannian sphere tpgmm", window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	// Setup OpenGL
	gfx2::init();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);
	glDepthFunc(GL_LESS);

	// Setup ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);

	ui::config.color = 0x770000ff;
	ui::config.lineColor = 0x770000ff;


	// Projection matrix
	arma::fmat projection;

	// Camera matrix
	arma::fmat view = gfx2::lookAt(
		arma::fvec( { 0, 0, 3 }),  // Position of the camera
		arma::fvec( { 0, 0, 0 }),  // Look at the origin
		arma::fvec( { 0, 1, 0 })   // Head is up
	);


	// Creation of the models
	//
	// The following hierarchy is used:
	//   - transforms node ("node")
	//    |__ sphere ("sphere")
	//
	// This allows to rotate everything by just changing the rotation of the
	// root transforms node.

	//-- The root transforms node
	gfx2::transforms_t node;
	node.position.zeros(3);
	node.rotation.eye(4, 4);

	//-- The sphere
	gfx2::model_t sphere = gfx2::create_sphere(0.99f);
	sphere.transforms.parent = &node;

	arma::mat points1 = sample_gaussian_points(gaussian, arma::vec( { 1.0f, 0.0f, 0.0f }));
	arma::mat points2 = sample_gaussian_points(gaussian, target);

	arma::fmat rotation;
	arma::fvec translation;
	gfx2::rigid_transform_3D(conv_to<fmat>::from(points1), conv_to<fmat>::from(points2),
							 rotation, translation);

	sphere.transforms.rotation.submat(0, 0, 2, 2) = rotation;

	// Creation of the light
	gfx2::point_light_t light;
	light.transforms.position = arma::fvec({ 0.0f, 0.0f, 5.0f, 1.0f });
	light.diffuse_color = arma::fvec({ 1.0f, 1.0f, 1.0f });
	light.ambient_color = arma::fvec({ 0.1f, 0.1f, 0.1f });

	gfx2::light_list_t lights;
	lights.push_back(light);

	// Mouse control
	double mouse_x, mouse_y, previous_mouse_x, previous_mouse_y;
	bool rotating = false;
	GLFWcursor* crosshair_cursor = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
	GLFWcursor* arrow_cursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

	glfwGetCursorPos(window, &previous_mouse_x, &previous_mouse_y);

	// Main loop
	bool must_recompute = true;
	int nbStates = 5;
	int previous_nbStates = nbStates;
	int nbVar = 3; //including time
	int nbSamples = 80;
	int nbVarMan = 4; //including time
	int nbIterEM = 30;
	int nbIter = 10; //Number of iteration for the Gauss Newton algorithm
	double params_diagRegFact = 1E-4;
	int nbFrames = 2;
	vec e0 = { 0, 0, 1 }; // center on manifold

	// Load manifold data:
	//  - first row is time
	//  - 2:4 is data in frame 1
	//  - 5:7 is data in frame 2
	mat demo(1 + 2 * nbVarMan, nbSamples);
	demo.load("./data/data_sphere_tpgmm_transformedXU.txt");

	mat demoX = demo; // data = [X, [U; 0]]
	mat demoU = demoX.submat(0, nbSamples, 4, 2 * nbSamples - 1);
	demoX = demoX.submat(0, 0, 6, nbSamples - 1);

	// cout << "demoX ncols/nrows: " << demoX.n_cols << ", " << demoX.n_rows << endl;
	// cout << "demoU ncols/nrows: " << demoU.n_cols << ", " << demoU.n_rows << endl;

	rowvec xIn = demoX.row(0);
	mat xOut = demoX.rows(1, 6);

	std::vector<gfx2::model_t> gaussian_models;
	std::vector<gfx2::model_t> sample_points;

	for (int i = 0; i < nbSamples; i++) {
		vec tmpEigenValues;
		mat tmpEigenVectors;
		mat tmpSigma;
		tmpSigma << 1.0 << 0.0 << endr << 0.0 << 1.0;
		eig_sym(tmpEigenValues, tmpEigenVectors, tmpSigma);
		mat tmpMat = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5) * 1.0);

		if (tmpMat(0, 1) >= 0) {
			gaussian_gmm.x.x = -tmpMat(0, 0);
			gaussian_gmm.x.y = tmpMat(1, 0);
			gaussian_gmm.y.x = -tmpMat(0, 1);
			gaussian_gmm.y.y = tmpMat(1, 1);
		} else {
			gaussian_gmm.x.x = tmpMat(0, 0);
			gaussian_gmm.x.y = -tmpMat(1, 0);
			gaussian_gmm.y.x = tmpMat(0, 1);
			gaussian_gmm.y.y = -tmpMat(1, 1);
		}

		gfx2::model_t gaussian_gmm_model = create_gaussian(
			arma::fvec( { 0.0f, 0.0f, 0.5f }), gaussian_gmm, demoX.col(i).rows(1, 3)
		);

		gaussian_gmm_model.transforms.parent = &node;

		sample_points.push_back(gaussian_gmm_model);

		gaussian_gmm_model = create_gaussian(
			arma::fvec( { 0.5f, 0.0f, 0.0f }), gaussian_gmm, demoX.col(i).rows(4, 6)
		);

		gaussian_gmm_model.transforms.parent = &node;

		sample_points.push_back(gaussian_gmm_model);
	}

	mat A1;
	mat A1_prev;
	mat A2;
	colvec b1;
	colvec b2;

	// Task parameters
	A1 << 0.4122 << -0.9111 << endr << -0.9111 << -0.4122;
	gaussian.x.x = -34.9728;
	gaussian.x.y = -30.6904;
	gaussian.y.x = 43.0625;
	gaussian.y.y = -49.4486;

	A2 << -0.9252 << -0.3796 << endr << -0.3796 << 0.9252;

	b1 << 0.7046 << endr << 0.1091 << endr << -0.7012;
	b2 << -0.4133 << endr << 0.5748 << endr << -0.7062;

	bool tasp_params_changed = true;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		previous_nbStates = nbStates;

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		if ((window_result == gfx2::WINDOW_READY) || (window_result == gfx2::WINDOW_RESIZED)) {

			// Update the projection matrix
			projection = gfx2::perspective(
				gfx2::deg2rad(45.0f),
				(float) window_size.fb_width / (float) window_size.fb_height,
				0.1f, 10.0f
			);

			// Ensure that the gaussian widgets are still visible
			gaussian.pos.x = 50;
			gaussian.pos.y = window_size.win_height - 120;
		}

		if (must_recompute) {

			for (int i = 0; i < gaussian_models.size(); i++) {
				gfx2::destroy(gaussian_models.at(i));
			}

			//======== Recomputing tpgmm model with the latest task parameters ========
			vec tmpVec;
			mat tmpCov;
			vec tmpEigenValues;
			mat tmpEigenVectors;
			vec Priors(nbStates);
			cube Sigma(1 + 2 * (nbVar - 1), 1 + 2 * (nbVar - 1), nbStates);
			mat Mu(1 + 2 * (nbVar - 1), nbStates);

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
			int nbData = 20;

			vec tSep = round(linspace<vec>(0, nbData, nbStates + 1));

			for (int i = 0; i < nbStates; i++) {
				rowvec id;

				for (int n = 0; n < 4; n++) {
					vec vecDummy = round(n * nbData + linspace<vec>(
						tSep(i), tSep(i + 1) - 1, tSep(i + 1) - tSep(i))
					);

					id = join_horiz(id, vecDummy.t());
				}

				Priors(i) = id.n_elem;

				mat demoUColID = zeros(5, id.n_cols);

				for (int j = 0; j < id.n_elem; j++) {

					demoUColID.col(j) = demoU.col((unsigned int) id(j));
				}

				Mu.col(i) = mean(demoUColID, 1);
				Sigma.slice(i) = cov(demoUColID.t()) + eye(5, 5) * 1E-4;
			}
			Priors = Priors / sum(Priors);

			// +++++++++++++++++++++++++ INIT END +++++++++++++++++++++++++++++++++

			mat MuMan = join_vert(Mu.row(0), expmap(Mu.rows(1, 2), e0)); //Center on the manifold
			MuMan = join_vert(MuMan, expmap(Mu.rows(3, 4), e0));
			Mu = zeros(1 + (nbVar - 1) * 2, nbStates);

			cube u(1 + 2 * (nbVar - 1), nbSamples, nbStates); //uTmp in matlab code

			for (int nb = 0; nb < nbIterEM; nb++) {

				//E-step
				mat L = zeros(nbStates, nbSamples);
				mat xcTmp;

				for (int i = 0; i < nbStates; i++) {
					xcTmp = join_vert(xIn - MuMan(0, i), logmap(xOut.rows(0, 2), MuMan.submat(1, i, 3, i)));
					xcTmp = join_vert(xcTmp, logmap(xOut.rows(3, 5), MuMan.submat(4, i, 6, i)));

					L.row(i) = Priors(i) * (gaussPDF(xcTmp, Mu.col(i), Sigma.slice(i)).t());
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
						u.slice(i) = join_vert(join_vert(xIn - MuMan(0, i), logmap(xOut.rows(0, 2), MuMan.submat(1, i, 3, i))),
								logmap(xOut.rows(3, 5), MuMan.submat(4, i, 6, i)));
						MuMan.col(i) = join_vert(
								join_vert((MuMan(0, i) + u.slice(i).row(0)) * GAMMA2.row(i).t(),
										expmap(u.slice(i).rows(1, 2) * GAMMA2.row(i).t(), MuMan.submat(1, i, 3, i))),
								expmap(u.slice(i).rows(3, 4) * GAMMA2.row(i).t(), MuMan.submat(4, i, 6, i)));
					}
					//Update Sigma
					Sigma.slice(i) = u.slice(i) * diagmat(GAMMA2.row(i)) * u.slice(i).t() +
									 					  eye(1 + 2 * (nbVar - 1),
														  1 + 2 * (nbVar - 1)
												  ) * params_diagRegFact;
				}

			}

			// for (int i = 0; i < nbStates; i++) {
			// 	cout << "============= Component #" << i << " ===========" << endl;
			// 	cout << "Prior: " << Priors(i) << endl;
			// 	cout << "MuMan:" << endl;
			// 	cout << MuMan.col(i).t() << endl;
			// 	cout << "Sigma: " << endl;
			// 	cout << Sigma.slice(i) << endl << endl;
			// }

			//Reformatting as a tensor GMM
			mat MuManOld = MuMan;
			cube SigmaOld = Sigma;
			cube MuMan2 = zeros(1 + nbVar, nbFrames, nbStates);
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
						id = {0, 1, 2};
					} else {
						id = {0, 3, 4};
					}

					if (m == 0) {
						idMan = {0, 1, 2, 3};
					} else {
						idMan = {0, 4, 5, 6};
					}

					for (int ii = 0; ii < idMan.size(); ii++) {
						MuMan2.slice(i).col(m)(ii) = MuManOld.col(i)(idMan(ii));
					}

					Sigma2.at(i).slice(m) = SigmaOld.slice(i).submat(id, id);
				}
			}

			// Transforming gaussians with respective frames
			for (int i = 0; i < nbStates; i++) {

				vec uTmp = A1 * logmap(MuMan2.slice(i).col(0).rows(1, 3), e0);
				MuMan2.slice(i).col(0).rows(1, 3) = expmap(uTmp, b1);
				mat Ac1 = transp(b1, MuMan2.slice(i).col(0).rows(1, 3));

				Sigma2.at(i).slice(0).submat(1, 1, 2, 2) =
					Ac1 * A1 * Sigma2.at(i).slice(0).submat(1, 1, 2, 2) * A1.t() * Ac1.t();

				uTmp = A2 * logmap(MuMan2.slice(i).col(1).rows(1, 3), e0);
				MuMan2.slice(i).col(1).rows(1, 3) = expmap(uTmp, b2);
				mat Ac2 = transp(b2, MuMan2.slice(i).col(1).rows(1, 3));

				Sigma2.at(i).slice(1).submat(1, 1, 2, 2) =
					Ac2 * A2 * Sigma2.at(i).slice(1).submat(1, 1, 2, 2) * A2.t() * Ac2.t();
			}

			// Transforming sample points with respective frames
			for (int i = 0; i < sample_points.size(); i++) {
				gfx2::destroy(sample_points.at(i));
			}
			sample_points.clear();
			for (int i = 0; i < nbSamples; i++) {
				vec tmpEigenValues;
				mat tmpEigenVectors;
				mat tmpSigma;
				vec uTmp;
				tmpSigma << 1.0 << 0.0 << endr << 0.0 << 1.0;
				eig_sym(tmpEigenValues, tmpEigenVectors, tmpSigma);
				mat tmpMat = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5) * 1.0);

				gaussian_gmm.x.x = -tmpMat(0, 0);
				gaussian_gmm.x.y = tmpMat(1, 0);
				gaussian_gmm.y.x = -tmpMat(0, 1);
				gaussian_gmm.y.y = tmpMat(1, 1);

				uTmp = A1 * logmap(demoX.col(i).rows(1, 3), e0);
				uTmp = expmap(uTmp, b1);

				gfx2::model_t gaussian_gmm_model = create_gaussian(
					arma::fvec( { 0.0f, 0.0f, 1.0f }), gaussian_gmm, uTmp
				);
				gaussian_gmm_model.transforms.parent = &node;

				sample_points.push_back(gaussian_gmm_model);

				uTmp = A2 * logmap(demoX.col(i).rows(4, 6), e0);
				uTmp = expmap(uTmp, b2);

				gaussian_gmm_model = create_gaussian(
					arma::fvec( { 1.0f, 0.0f, 0.0f }), gaussian_gmm, uTmp
				);
				gaussian_gmm_model.transforms.parent = &node;

				sample_points.push_back(gaussian_gmm_model);
			}

			//Gaussian product
			mat MuManProduct(3, nbStates);
			cube SigmaProduct(2, 2, nbStates);

			for (int i = 0; i < nbStates; i++) {
				mat componentsMu = zeros(3, 2); // current means of the components
				cube U0 = zeros(2, 2, 2);  //current covariances of the components

				vec tmpVec;
				mat tmpCov;
				vec tmpEigenValues;
				mat tmpEigenVectors;

				componentsMu.col(0) = MuMan2.slice(i).col(0).rows(1, 3);
				tmpCov = Sigma2.at(i).slice(0).submat(1, 1, 2, 2);
				eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);
				U0.slice(0) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

				componentsMu.col(1) = MuMan2.slice(i).col(1).rows(1, 3);
				tmpCov = Sigma2.at(i).slice(1).submat(1, 1, 2, 2);
				eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);
				U0.slice(1) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

				vec MuMan; // starting point on the manifold
				MuMan << 0.0 << endr << 0.0 << endr << 1.0;
				MuMan /= norm(MuMan, 2);
				mat Sigma = zeros(2, 2);

				mat MuTmp = zeros(2, 2); //nbVar (on tangent space) x components
				cube SigmaTmp = zeros(2, 2, 2); // nbVar (on tangent space) x nbVar (on tangent space) x components

				for (int n = 0; n < 10; n++) { // compute the Gaussian product
					colvec Mu = zeros(2, 1);
					mat SigmaSum = zeros(2, 2);

					for (int i = 0; i < 2; i++) { // we have two states
						mat Ac = transp(componentsMu.col(i), MuMan);
						mat U1 = Ac * U0.slice(i);
						SigmaTmp.slice(i) = U1 * U1.t();

						//Tracking component for Gaussian i
						SigmaSum += inv(SigmaTmp.slice(i));
						MuTmp.col(i) = logmap(componentsMu.col(i), MuMan);
						Mu += inv(SigmaTmp.slice(i)) * MuTmp.col(i);
					}

					Sigma = inv(SigmaSum);
					//Gradient computation
					Mu = Sigma * Mu;

					MuMan = expmap(Mu, MuMan);
				}

				MuManProduct.col(i) = MuMan;
				SigmaProduct.slice(i) = Sigma;

				// cout << "================ Gaussian Product component #" << i << " ========================" << endl;
				// cout << "MuMan: " << MuMan.t() << endl;
				// cout << "Sigma: " << endl << Sigma << endl;
			}

			// Data for plotting
			gaussian_models.clear();
			for (int f = 0; f < nbFrames; f++) {
				uvec id = { 1, 2 };
				uvec idMan = { 1, 3 };

				for (int i = 0; i < nbStates; i++) {
					eig_sym(tmpEigenValues, tmpEigenVectors, Sigma2.at(i).slice(f).submat(id, id));
					mat tmpMat = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5) * 200.0);

					if (tmpMat(0, 1) >= 0) {
						gaussian_gmm.x.x = -tmpMat(0, 0);
						gaussian_gmm.x.y = tmpMat(1, 0);
						gaussian_gmm.y.x = -tmpMat(0, 1);
						gaussian_gmm.y.y = tmpMat(1, 1);
					} else {
						gaussian_gmm.x.x = tmpMat(0, 0);
						gaussian_gmm.x.y = -tmpMat(1, 0);
						gaussian_gmm.y.x = tmpMat(0, 1);
						gaussian_gmm.y.y = -tmpMat(1, 1);
					}

					gfx2::model_t gaussian_gmm_model;
					if (f == 0) {
						gaussian_gmm_model = create_gaussian(
							arma::fvec( { 0.0f, 0.0f, 1.0f }),
							gaussian_gmm, MuMan2.slice(i).col(f).rows(idMan(0), idMan(1))
						);
					} else {
						gaussian_gmm_model = create_gaussian(
							arma::fvec( { 1.0f, 0.0f, 0.0f }),
							gaussian_gmm, MuMan2.slice(i).col(f).rows(idMan(0), idMan(1))
						);
					}
					gaussian_gmm_model.transforms.parent = &node;

					gaussian_models.push_back(gaussian_gmm_model);

				}

			}
			//product
			for (int i = 0; i < nbStates; i++) {
				vec tmpVec;
				mat tmpCov;
				vec tmpEigenValues;
				mat tmpEigenVectors;

				eig_sym(tmpEigenValues, tmpEigenVectors, SigmaProduct.slice(i));
				mat tmpMat = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5) * 200.0);

				if (tmpMat(0, 1) >= 0) {
					gaussian_gmm.x.x = -tmpMat(0, 0);
					gaussian_gmm.x.y = tmpMat(1, 0);
					gaussian_gmm.y.x = -tmpMat(0, 1);
					gaussian_gmm.y.y = tmpMat(1, 1);
				} else {
					gaussian_gmm.x.x = tmpMat(0, 0);
					gaussian_gmm.x.y = -tmpMat(1, 0);
					gaussian_gmm.y.x = tmpMat(0, 1);
					gaussian_gmm.y.y = -tmpMat(1, 1);
				}

				gfx2::model_t gaussian_gmm_model;

				gaussian_gmm_model = create_gaussian(
					arma::fvec( { 0.0f, 0.0f, 0.0f }), gaussian_gmm, MuManProduct.col(i)
				);

				gaussian_gmm_model.transforms.parent = &node;

				gaussian_models.push_back(gaussian_gmm_model);
			}
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();
		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glMultMatrixf(projection.memptr());

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glMultMatrixf(view.memptr());

		// Drawing
		gfx2::draw(sphere, lights);

		for (int i = 0; i < gaussian_models.size(); i++) {
			gfx2::draw(gaussian_models.at(i), lights);
		}

		for (int i = 0; i < sample_points.size(); i++) {
			gfx2::draw(sample_points.at(i), lights);
		}

		// Gaussian UI widget
		ui::begin("Gaussian");
		gaussian = ui::affineSimple(0, gaussian);
		ui::end();

		mat tmpEigenVectors;
		vec tmpEigenValues;
		mat tmpCov = trans2cov(gaussian);
		eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);

		A1_prev = A1;
		A1 = tmpEigenVectors;

		// Parameter window
		ImGui::SetNextWindowSize(ImVec2(400, 74));
		ImGui::Begin("Parameters", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove);
		ImGui::SliderInt("Nb states", &nbStates, 1, 10);
		ImGui::Text("Hold the right mouse button to rotate the sphere");
		ImGui::End();

		// GUI rendering
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		if (previous_nbStates != nbStates) {
			must_recompute = true;
			// cout << "Recomputing GMM with " << nbStates << " states..." << endl;
		} else {
			must_recompute = false;
		}

		if (norm(A1 - A1_prev, 2) != 0) {
			tasp_params_changed = true;
			must_recompute = true;
		}

		// Swap buffers
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;

		// Mouse input
		glfwGetCursorPos(window, &mouse_x, &mouse_y);

		//-- Left click
		if (ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1)) {

			// Determine if the click was on the sphere, and where exactly
			gfx2::ray_t ray = gfx2::create_ray(
				arma::fvec( { 0, 0, 3 }), (int) mouse_x, (int) mouse_y, view,
				projection, window_size.win_width, window_size.win_height
			);

			arma::fvec intersection;
			if (gfx2::intersects(ray, node.position, 1.0f, intersection)) {
				// Change the target position
				arma::fvec p(4);
				p.rows(0, 2) = intersection;
				p(3) = 1.0f;

				p = arma::inv(gfx2::worldRotation(&node)) * p;

				target = arma::conv_to<arma::vec>::from(p.rows(0, 2));
				b1 = -target;
				tasp_params_changed = true;

				must_recompute = true;
			}
		}

		//-- Right mouse button: rotation of the meshes while held down
		if (ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_2)) {
			if (!rotating) {
				rotating = true;
				glfwSetCursor(window, crosshair_cursor);
			}

			arma::fmat rotation = gfx2::rotate(
				arma::fvec( { 0.0f, 1.0f, 0.0f }),
				0.2f * gfx2::deg2rad(mouse_x - previous_mouse_x)) *
					gfx2::rotate(arma::fvec( { 1.0f, 0.0f, 0.0f }),
				0.2f * gfx2::deg2rad(mouse_y - previous_mouse_y)
			);

			node.rotation = rotation * node.rotation;
		} else if (rotating) {
			rotating = false;
			glfwSetCursor(window, arrow_cursor);
		}

		previous_mouse_x = mouse_x;
		previous_mouse_y = mouse_y;
	}

	// Cleanup
	glfwDestroyCursor(crosshair_cursor);
	glfwDestroyCursor(arrow_cursor);

	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}
