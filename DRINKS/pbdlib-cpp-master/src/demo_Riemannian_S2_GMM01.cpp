/*
 * demo_Riemannian_S2_GMM01.cpp
 *
 * GMM on sphere.
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

#include <stdio.h>
#include <armadillo>
#include <sstream>

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

	arma::mat Log(2, x.n_cols);
	Log.row(0) = x.row(0) % scale;
	Log.row(1) = x.row(1) % scale;

	return Log;
}

//-----------------------------------------------

arma::mat expmap(arma::mat u, arma::vec mu) {
	return trans(rotM(mu)) * expfct(u);
}

//-----------------------------------------------

arma::mat logmap(arma::mat x, arma::vec mu) {
	arma::mat pole;
	pole = {0, 0, 1};
	arma::mat R(3, 3, fill::ones);

	if (norm(mu - trans(pole)) < 1E-6) {
		R = {
			{ 1,  0,  0 },
			{ 0, -1,  0 },
			{ 0,  0, -1 }
		};
	}
	else {
		R = rotM(mu);
	}

	arma::mat u;
	u = logfct(R * x);

	return u;
}

//-----------------------------------------------

arma::mat sample_gaussian_points(const ui::Trans2d& gaussian, const arma::vec& target) {

	arma::mat uCov = trans2cov(gaussian);

	arma::mat V;
	arma::vec d;
	arma::eig_sym(d, V, uCov);

	arma::mat sample_points( { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 0.0f, 1.0f }, { -1.0f, 0.0f }, { 0.0f, -1.0f }, });

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

	prob = exp(-0.5 * prob) / sqrt(pow((2 * datum::pi), nbVar) * det(Sigma));

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

	ui::Trans2d gaussian(ImVec2(50, 0), ImVec2(0, 100), ImVec2(50, 680));


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
		"Demo Riemannian sphere gmm", window_size.win_width, window_size.win_height
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

	// Creation of the models
	//
	// The following hierarchy is used:
	//   - transforms node ("node")
	//   |__ sphere ("sphere")
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
	ui::Trans2d previous_gaussian = gaussian;
	bool must_recompute = true;
	int display_mode = 0;
	int nbStates = 5;
	int previous_nbStates = 5;
	int nbVar = 2;
	int nbSamples = 250;
	int nbVarMan = 3;
	int nbIterEM = 30;
	int nbIter = 10; //Number of iteration for the Gauss Newton algorithm
	double params_diagRegFact = 1E-4;

	// Load S shape on manifold
	mat demo(nbVarMan, nbSamples * 2);
	demo.load("./data/data_sphere_gmm_letter_S.txt");
	mat demoX = demo.submat(0, 0, 2, nbSamples - 1);
	mat demoU = demo.submat(0, nbSamples, 1, nbSamples * 2 - 1);

	// cout << "demoX ncols/nrows: " << demoX.n_cols << ", " << demoX.n_rows << endl;
	// cout << "demoU ncols/nrows: " << demoU.n_cols << ", " << demoU.n_rows << endl;

	std::vector<gfx2::model_t> gaussian_models;
	std::vector<gfx2::model_t> sample_points;

	for (int i = 0; i < nbSamples; i++) {
		vec tmpEigenValues;
		mat tmpEigenVectors;
		mat tmpSigma;
		tmpSigma << 1.0 << 0.0 << endr << 0.0 << 1.0;
		eig_sym(tmpEigenValues, tmpEigenVectors, tmpSigma);
		mat tmpMat = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5) * 1.0);

		ui::Trans2d gaussian_gmm;

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
			arma::fvec( { 0.0f, 0.0f, 1.0f }), gaussian_gmm, demoX.col(i)
		);

		gaussian_gmm_model.transforms.parent = &node;

		sample_points.push_back(gaussian_gmm_model);
	}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		previous_nbStates = nbStates;

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;


		if (must_recompute) {
			for (int i = 0; i < gaussian_models.size(); i++) {
				gfx2::destroy(gaussian_models.at(i));
			}

			mat componentsMu = zeros(3, 2); // current means of the components
			cube U0 = zeros(2, 2, 2);  //current covariances of the components

			vec tmpVec;
			mat tmpCov;
			vec tmpEigenValues;
			mat tmpEigenVectors;

			arma_rng::set_seed_random();  // set the seed to a random value

			vec Priors = ones(nbStates);
			Priors /= nbStates;

			arma::mat randTmp(nbVar, nbStates);

			for (int i = 0; i < nbStates; i++) { //for some reason the matrix randn doesn't work => doing it with vectors
				for (int j = 0; j < nbVar; j++) {
					randTmp(j, i) = randn();
				}
			}

			mat MuMan = expmap(randTmp, randn(nbVarMan)); //Center on the manifold

			mat Mu = zeros(nbVar, nbStates); //Center in the tangent plane at point MuMan of the manifold
			cube Sigma(nbVar, nbVar, nbStates);
			cube u(nbVar, nbSamples, nbStates);
			for (int i = 0; i < nbStates; i++) {
				Sigma.slice(i) = eye(nbVar, nbVar) * 0.5; //%Covariance in the tangent plane at point MuMan
			}

			// cout << "Starting EM iteration" << endl;
			for (int nb = 0; nb < nbIterEM; nb++) {

				//E-step
				mat L = zeros(nbStates, nbSamples);
				for (int i = 0; i < nbStates; i++) {
					L.row(i) = Priors(i) * gaussPDF(logmap(demoX, MuMan.col(i)),
													Mu.col(i), Sigma.slice(i)).t();
				}

				rowvec Lsum = sum(L, 0);
				mat GAMMA = L / repmat(Lsum, nbStates, 1);
				colvec GammaSum = sum(GAMMA, 1);
				mat H = GAMMA / repmat(GammaSum, 1, nbSamples);

				//M-step
				for (int i = 0; i < nbStates; i++) {
					//Update Priors
					Priors(i) = sum(GAMMA.row(i)) / (nbSamples);
					//Update MuMan
					for (int n = 0; n < nbIter; n++) {
						u.slice(i) = logmap(demoX, MuMan.col(i));
						MuMan.col(i) = expmap(u.slice(i) * H.row(i).t(), MuMan.col(i));
					}
					//Update Sigma
					Sigma.slice(i) = u.slice(i) * diagmat(H.row(i)) * u.slice(i).t() +
									 eye(nbVar, nbVar) * params_diagRegFact;
				}
			}

			gaussian_models.clear();
			for (int i = 0; i < nbStates; i++) {
				// cout << "========= Component " << i << " =============" << endl;
				// cout << "MuMan: " << endl << MuMan.col(i).t() << endl;
				// cout << "Sigma: " << endl << Sigma.slice(i) << endl;

				eig_sym(tmpEigenValues, tmpEigenVectors, Sigma.slice(i));
				mat tmpMat = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5) * 200.0);

				ui::Trans2d gaussian_gmm;

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
					arma::fvec( { 1.0f, 0.0f, 0.0f }), gaussian_gmm, MuMan.col(i)
				);
				gaussian_gmm_model.transforms.parent = &node;

				gaussian_models.push_back(gaussian_gmm_model);
			}

			// cout << "Priors: " << Priors.t() << endl;
		}

		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();
		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		arma::fmat projection = gfx2::perspective(
			gfx2::deg2rad(45.0f), (float) window_size.fb_width / (float) window_size.fb_height,
			0.1f, 10.0f
		);
		glMultMatrixf(projection.memptr());

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glTranslatef(0.0f, 0.0f, -3.0f);

		// Drawing
		gfx2::draw(sphere, lights);

		for (int i = 0; i < nbStates; i++) {
			gfx2::draw(gaussian_models.at(i), lights);
		}

		for (int i = 0; i < sample_points.size(); i++) {
			gfx2::draw(sample_points.at(i), lights);
		}

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

		// Swap buffers
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;

		// Mouse input
		glfwGetCursorPos(window, &mouse_x, &mouse_y);

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
