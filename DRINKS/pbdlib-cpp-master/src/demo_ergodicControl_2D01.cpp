/*
 * demo_ergodicControl_2D01.cpp
 *
 * 2D ergodic control, inspired by G. Mathew and I. Mezic, "Spectral Multiscale  
 * Coverage: A Uniform Coverage Algorithm for Mobile Sensor Networks", CDC'2009 
 *
 * If this code is useful for your research, please cite the related publication:
 * @incollection{Calinon19MM,
 * 	author="Calinon, S.",
 * 	title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
 * 	booktitle="Mixture Models and Applications",
 * 	publisher="Springer",
 * 	editor="Bouguila, N. and Fan, W.", 
 * 	year="2019",
 * 	pages="39--57",
 * 	doi="10.1007/978-3-030-23876-6_3"
 * }
 *
 * Authors: Sylvain Calinon, Philip Abbet
 */

#include <stdio.h>
#include <armadillo>
#include <mvn.h>
#include <mpc_utils.h>

#include <gfx2.h>
#include <gfx_ui.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>
#include <GLFW/glfw3.h>

using namespace arma;


/***************************** ALGORITHM SECTION *****************************/

typedef std::vector<vec> vector_list_t;
typedef std::vector<mat> matrix_list_t;


//-----------------------------------------------------------------------------
// Contains all the parameters used by the algorithm. Some of them are
// modifiable through the UI, others are hard-coded.
//-----------------------------------------------------------------------------
struct parameters_t {
	int   nb_gaussians; // Number of gaussians that control the trajectory
	int   nb_data;      // Number of datapoints
	int   nb_fct[2];    // Number of basis functions along x and y
	int   nb_res;       // Resolution of discretization for the computation of
                        // Fourier coefficients of coverage distribution
	float dt;           // Time step
	mat   xlim;         // Domain limits
};


//-----------------------------------------------------------------------------
// Implementation of the algorithm
//-----------------------------------------------------------------------------
std::tuple<mat, mat> compute(const parameters_t& parameters, const mat& mu,
							 const cube& sigma) {

	const int nb_var = 2; 							 // Dimension of datapoint
	const float sp = ((float) nb_var + 1.0f) / 2.0f; // Sobolev norm parameter

	vec xsiz({ parameters.xlim(0, 1) - parameters.xlim(0, 0),
			   parameters.xlim(1, 1) - parameters.xlim(1, 0)
		     }); // Domain size

	vec dx = xsiz / parameters.nb_res; // Spatial increments
	vec x({0.1f, 0.3f}); // Initial position


	// Basis functions (Fourier coefficients of coverage distribution)
	//----------------------------------------------------------------
	vector_list_t rg;
	rg.push_back(linspace<vec>(0, parameters.nb_fct[0] - 1, parameters.nb_fct[0]));
	rg.push_back(linspace<vec>(0, parameters.nb_fct[1] - 1, parameters.nb_fct[1]));

	mat KX1 = repmat(rg[0], 1, parameters.nb_fct[1]);
	mat KX2 = repmat(rg[1].t(), parameters.nb_fct[0], 1);

	mat LK = pow(pow(KX1, 2) + pow(KX2, 2) + 1, -sp);	// Weighting matrix

	mat HK = join_vert(mat({1.0}), sqrt(0.5) * ones(parameters.nb_fct[0] - 1)) *
			 join_vert(mat({1.0}), sqrt(0.5) * ones(parameters.nb_fct[1] - 1)).t() *
			 sqrt(xsiz(0) * xsiz(1));		// Normalizing matrix

	mat X = repmat(linspace<vec>(parameters.xlim(0, 0), parameters.xlim(0, 1) - dx(0), parameters.nb_res).t(),
				   parameters.nb_res, 1);

	mat Y = repmat(linspace<vec>(parameters.xlim(1, 0), parameters.xlim(1, 1) - dx(1), parameters.nb_res),
				   1, parameters.nb_res);

	// Desired spatial distribution as mixture of Gaussians
	cube G_(X.n_rows, X.n_cols, mu.n_cols);

	mat XY(X.n_elem, 2);
	XY(span(0, X.n_elem - 1), 0) = reshape(X, X.n_elem, 1);
	XY(span(0, X.n_elem - 1), 1) = reshape(Y, Y.n_elem, 1);

	for (int k = 0; k < mu.n_cols; ++k) {
		G_.slice(k) = reshape(
				mvn::getPDFValue(mu(span::all, k), sigma.slice(k), XY.t()),
				X.n_rows, X.n_cols
			).t() / mu.n_cols;
	}

	mat G = sum(G_, 2);
	G /= accu(G);	// Spatial distribution

	// Computation of phi_k by discretization
	// mat phi = zeros(parameters.nb_fct[0], parameters.nb_fct[1]);
	// for (int kx = 0; kx < parameters.nb_fct[0]; ++kx) {
	// 	for (int ky = 0; ky < parameters.nb_fct[1]; ++ky) {
	// 		phi(kx, ky) = accu(G % cos(X * kx * datum::pi / xsiz(0)) % cos(Y * ky * datum::pi / xsiz(1))) /
	// 					  HK(kx, ky); // Fourier coefficients of spatial distribution
	// 	}
	// }

	// Explicit description of phi_k by exploiting the Fourier transform properties
	// of Gaussians
	mat w1 = KX1 * datum::pi / xsiz(0);
	mat w2 = KX2 * datum::pi / xsiz(1);

	mat w(2, w1.n_elem);
	w(0, span::all) = reshape(w1, 1, w1.n_elem);
	w(1, span::all) = reshape(w2, 1, w1.n_elem);

	// Enumerate symmetry operations for 2D signal ([-1,-1],[-1,1],[1,-1] and [1,1])
	mat op({ {-1, 1, -1, 1}, {-1, -1, 1, 1} });

	// Compute phi_k
	mat phi = zeros(parameters.nb_fct[0], parameters.nb_fct[1]);
	for (int k = 0; k < mu.n_cols; ++k) {
		for (int n = 0; n < op.n_cols; ++n) {
			mat MuTmp = diagmat(op(span::all, n)) * mu(span::all, k);
			mat SigmaTmp = diagmat(op(span::all, n)) * sigma.slice(k) * diagmat(op(span::all, n)).t();
			phi = phi + reshape(cos(w.t() * MuTmp) % diagvec(exp(-0.5 * w.t() * SigmaTmp * w)), size(HK));
		}
	}
	phi = phi / HK / mu.n_cols / op.n_cols;


	// Ergodic control with spectral multiscale coverage (SMC) algorithm
	//------------------------------------------------------------------
	mat Ck = zeros(parameters.nb_fct[0], parameters.nb_fct[1]);

	mat result = zeros(nb_var, parameters.nb_data);

	for (int t = 0; t < parameters.nb_data; ++t) {

		// Log data
		result(span::all, t) = x;

		// Updating Fourier cosine coefficients of coverage distribution for each dimension
		vector_list_t cx;
		vector_list_t dcx;

		for (int i = 0; i < nb_var; ++i) {
			cx.push_back(cos(rg[i] * datum::pi * (x(i) - parameters.xlim(i, 0)) / xsiz(i)));
			dcx.push_back(sin(rg[i] * datum::pi * (x(i) - parameters.xlim(i, 0)) / xsiz(i)) % rg[i] * datum::pi / xsiz(i));
		} 

		// Fourier cosine coefficients along trajectory
		Ck = Ck + (repmat(cx[0], 1, parameters.nb_fct[1]) % repmat(cx[1].t(), parameters.nb_fct[0], 1)) / HK * parameters.dt;

		// SMC feedback control law
		dx(0) = accu((LK / HK) % (Ck - phi * (t + 1) * parameters.dt) %
					 (repmat(dcx[0], 1, parameters.nb_fct[1]) % repmat(cx[1].t(), parameters.nb_fct[0], 1)));

		dx(1) = accu((LK / HK) % (Ck - phi * (t + 1) * parameters.dt) %
					 (repmat(cx[0], 1, parameters.nb_fct[1]) % repmat(dcx[1].t(), parameters.nb_fct[0], 1)));

		x = x + dx * parameters.dt;
	}

	return std::make_tuple(result, G);
}


/****************************** HELPER FUNCTIONS *****************************/

static void error_callback(int error, const char* description){
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//-----------------------------------------------

std::tuple<vec, mat> trans2d_to_gauss(const ui::Trans2d& gaussian_transforms,
									  const gfx2::window_size_t& window_size,
									  int background_size) {

	vec mu = gfx2::ui2fb_centered(vec({ gaussian_transforms.pos.x, gaussian_transforms.pos.y }),
								  window_size);

	mu(0) *= (float) window_size.fb_width / background_size;
	mu(1) *= (float) window_size.fb_height / background_size;

	vec t_x({
		gaussian_transforms.x.x * window_size.scale_x(),
		gaussian_transforms.x.y * window_size.scale_y()
	});

	vec t_y({
		gaussian_transforms.y.x * window_size.scale_x(),
		gaussian_transforms.y.y * window_size.scale_y()
	});

	mat RG = {
		{ t_x(0), t_y(0) },
		{ -t_x(1), -t_y(1) }
	};

	mat sigma = RG * RG.t();

	return std::make_tuple(mu, sigma);
}

//-----------------------------------------------

void gauss_to_trans2d(const vec& mu, const mat& sigma,
					  const gfx2::window_size_t& window_size, ui::Trans2d &t2d) {

	vec ui_mu = gfx2::fb2ui_centered(mu, window_size);

	t2d.pos.x = ui_mu(0);
	t2d.pos.y = ui_mu(1);

	mat V;
	vec d;
	eig_sym(d, V, sigma);
	mat VD = V * diagmat(sqrt(d));

	t2d.x.x = VD.col(0)(0) / window_size.scale_x();
	t2d.x.y = VD.col(1)(0) / window_size.scale_y();
	t2d.y.x = VD.col(0)(1) / window_size.scale_x();
	t2d.y.y = VD.col(1)(1) / window_size.scale_y();
}

//-----------------------------------------------

std::tuple<mat, cube, std::vector<ui::Trans2d> > create_random_gaussians(
	const parameters_t& parameters, const gfx2::window_size_t& window_size,
	int background_size) {

	mat Mu = randu(2, parameters.nb_gaussians);
	Mu.row(0) = Mu.row(0) * (window_size.fb_width - 200) - (window_size.fb_width / 2 - 100);
	Mu.row(1) = Mu.row(1) * (window_size.fb_height - 200) - (window_size.fb_height / 2 - 100);

	cube Sigma;
	randomCovariances(
		&Sigma, Mu,
		vec({ 100 * (float) window_size.fb_width / window_size.win_width,
			  100 * (float) window_size.fb_height / window_size.win_height
		}),
		true, 0.0, 0.2
	);

	std::vector<ui::Trans2d> gaussians(parameters.nb_gaussians, ui::Trans2d());

	for (int i = 0; i < parameters.nb_gaussians; ++i) {
		gauss_to_trans2d(Mu.col(i), Sigma.slice(i), window_size, gaussians[i]);

		fmat rotation = gfx2::rotate(fvec({ 0.0f, 0.0f, 1.0f }), randu() * 2 * datum::pi);

		fvec x = rotation * fvec({ gaussians[i].x.x, gaussians[i].x.y, 0.0f, 0.0f });
		gaussians[i].x.x = x(0);
		gaussians[i].x.y = x(1);

		fvec y = rotation * fvec({ gaussians[i].y.x, gaussians[i].y.y, 0.0f, 0.0f });
		gaussians[i].y.x = y(0);
		gaussians[i].y.y = y(1);

		vec mu_;
		mat sigma_;

		std::tie(mu_, sigma_) = trans2d_to_gauss(
			gaussians[i], window_size, background_size
		);

		Sigma.slice(i) = sigma_;
	}

	return std::make_tuple(Mu, Sigma, gaussians);
}


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv){

	arma_rng::set_seed_random();

	// Parameters
	parameters_t parameters;
	parameters.nb_gaussians = 2;
	parameters.nb_data      = 2000;
	parameters.nb_fct[0]    = 20;
	parameters.nb_fct[1]    = 20;
	parameters.nb_res       = 100;
	parameters.dt           = 0.01f;
	parameters.xlim         = mat({ { 0.0f, 1.0f }, { 0.0f, 1.0f } } );


	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 800;
	window_size.win_height = 800;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;


	// Initialise GLFW
	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	// Open a window and create its OpenGL context
	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Demo 2D ergodic control",
		window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);


	// Setup OpenGL
	gfx2::init();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Setup ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);


	// Gaussian widgets
	std::vector<ui::Trans2d> gaussians;

	// Covariances
	mat Mu;
	cube Sigma;

	// Main loop
	mat result;
	mat G;
	const float speed = 1.0f / 20.0f;
	float t = 0.0f;
	bool must_recompute = false;
	int background_size;

	gfx2::texture_t background_texture = {0};

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_size_t previous_size;

		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size, &previous_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		if ((window_result == gfx2::WINDOW_READY) || (window_result == gfx2::WINDOW_RESIZED)) {

			background_size = std::max(window_size.fb_width, window_size.fb_height);

			// Move and rescale the various objects so they stay in the window
			if (window_result == gfx2::WINDOW_RESIZED) {
				float scale = std::min((float) window_size.fb_width / previous_size.fb_width,
									   (float) window_size.fb_height / previous_size.fb_height);

				for (int i = 0; i < parameters.nb_gaussians; ++i) {
					arma::vec target = gfx2::fb2ui_centered(
						gfx2::ui2fb_centered({gaussians[i].pos.x, gaussians[i].pos.y},
											 previous_size.win_width, previous_size.win_height,
											 previous_size.fb_width, previous_size.fb_height) * scale,
						window_size.win_width, window_size.win_height,
						window_size.fb_width, window_size.fb_height
					);

					gaussians[i].pos.x = target(0);
					gaussians[i].pos.y = target(1);

					gaussians[i].x.x *= scale;
					gaussians[i].x.y *= scale;
					gaussians[i].y.x *= scale;
					gaussians[i].y.y *= scale;
				}
			}

			// At the very first frame: random initialisation of the gaussians (taking 4K
			// screens into account)
			else if (window_result == gfx2::WINDOW_READY) {
				std::tie(Mu, Sigma, gaussians) = create_random_gaussians(
					parameters, window_size, background_size
				);
			}

			must_recompute = true;
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();

		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-window_size.fb_width / 2, window_size.fb_width / 2,
				-window_size.fb_height / 2, window_size.fb_height / 2,
				-1.0f, 1.0f
		);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		if (result.n_cols > 0) {

			double max = G.max();
			double min = G.min();

			if (background_texture.width == 0) {
				background_texture = gfx2::create_texture(
					parameters.nb_res, parameters.nb_res, GL_RGB, GL_FLOAT
				);

				for (int j = 0; j < parameters.nb_res; ++j) {
					for (int i = 0; i < parameters.nb_res; ++i) {
						float color = 1.0f - (G(i, j) - min) / (max - min);

						background_texture.pixels_f[(j * parameters.nb_res + i) * 3] = color;
						background_texture.pixels_f[(j * parameters.nb_res + i) * 3 + 1] = 1.0f;
						background_texture.pixels_f[(j * parameters.nb_res + i) * 3 + 2] = color;
					}
				}
			}

			gfx2::draw_rectangle(background_texture, background_size, background_size);

			glClear(GL_DEPTH_BUFFER_BIT);

			mat result2(result.n_rows, result.n_cols);
			result2(0, span::all) = (result(0, span::all) - 0.5) * background_size;
			result2(1, span::all) = (result(1, span::all) - 0.5) * background_size;

			gfx2::draw_line(fvec({ 0.0f, 0.0f, 1.0f }), result2);

			int current_index = t * result2.n_cols;

			if (current_index > 0) {
				glClear(GL_DEPTH_BUFFER_BIT);
				glLineWidth(4.0f);
				gfx2::draw_line(fvec({ 1.0f, 0.0f, 0.0f }), result2(span::all, span(0, current_index)));
				glLineWidth(1.0f);
			}

			fvec current_position = zeros<fvec>(3);
			current_position(0) = result2(0, current_index);
			current_position(1) = result2(1, current_index);

			gfx2::draw_rectangle(fvec({ 1.0f, 0.0f, 0.0f }), 10.0f, 10.0f, current_position);
		}


		// Control panel GUI
		ImGui::SetNextWindowSize(ImVec2(650, 130));
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("Control Panel", NULL,
					 ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
					 ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings);

		int previous_nb_gaussians = parameters.nb_gaussians;
		int previous_nb_data = parameters.nb_data;
		int previous_nb_fct_x = parameters.nb_fct[0];
		int previous_nb_fct_y = parameters.nb_fct[1];
		int previous_nb_res = parameters.nb_res;

		ImGui::SliderInt("Nb gaussians", &parameters.nb_gaussians, 1, 10);
		ImGui::SliderInt("Nb data", &parameters.nb_data, 500, 10000);
		ImGui::SliderInt("Nb basis functions along X", &parameters.nb_fct[0], 5, 100);
		ImGui::SliderInt("Nb basis functions along Y", &parameters.nb_fct[1], 5, 100);
		ImGui::SliderInt("Resolution of discretization", &parameters.nb_res, 20, 500);

		if ((parameters.nb_gaussians != previous_nb_gaussians) ||
			(parameters.nb_data != previous_nb_data) ||
			(parameters.nb_fct[0] != previous_nb_fct_x) ||
			(parameters.nb_fct[1] != previous_nb_fct_y) ||
			(parameters.nb_res != previous_nb_res)) {

			must_recompute = true;
		}

		ImGui::End();


		// Gaussian widgets
		if (parameters.nb_gaussians != gaussians.size()) {
			std::tie(Mu, Sigma, gaussians) = create_random_gaussians(
				parameters, window_size, background_size
			);
		}

		ui::begin("Gaussian");

		if (!gaussians.empty()) {
			for (int i = 0; i < parameters.nb_gaussians; ++i) {
				ui::Trans2d previous_gaussian = gaussians[i];

				gaussians[i] = ui::affineSimple(i, gaussians[i]);

				vec mu;
				mat sigma;

			    std::tie(mu, sigma) = trans2d_to_gauss(
					gaussians[i], window_size, background_size
			    );

				Mu.col(i) = mu;
				Sigma.slice(i) = sigma;

				must_recompute = must_recompute ||
								 !gfx2::is_close(gaussians[i].pos.x, previous_gaussian.pos.x) ||
								 !gfx2::is_close(gaussians[i].pos.y, previous_gaussian.pos.y) ||
								 !gfx2::is_close(gaussians[i].x.x, previous_gaussian.x.x) ||
								 !gfx2::is_close(gaussians[i].x.y, previous_gaussian.x.y) ||
								 !gfx2::is_close(gaussians[i].y.x, previous_gaussian.y.x) ||
								 !gfx2::is_close(gaussians[i].y.y, previous_gaussian.y.y);
			}
		}

		ui::end();


		// Redo the computation when necessary
		if (must_recompute && !ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)) {
			mat mu = Mu;
			mu(0, span::all) = mu(0, span::all) / window_size.fb_width + 0.5;
			mu(1, span::all) = mu(1, span::all) / window_size.fb_height + 0.5;

			mat scaling({
				{ 1.0 / background_size, 0.0 },
				{ 0.0, 1.0 / background_size }
			});

			cube sigma(Sigma.n_rows, Sigma.n_cols, Sigma.n_slices);
			for (int i = 0; i < Sigma.n_slices; ++i)
				sigma.slice(i) = scaling * Sigma.slice(i) * scaling.t();

			std::tie(result, G) = compute(parameters, mu, sigma);
			t = 0.0f;

			if (background_texture.width > 0)
				gfx2::destroy(background_texture);

			must_recompute = false;
		}


		// GUI rendering
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		// Swap buffers
		glPopMatrix();
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;


		t += speed * ImGui::GetIO().DeltaTime;
		if (t >= 1.0f)
			t = 0.0f;
	}


	// Cleanup
	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}
