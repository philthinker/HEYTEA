/*
 * demo_Riemannian_SPD_interp02.cpp
 *
 * Covariance interpolation on Riemannian manifold from a GMM with augmented
 * covariances (Implementation based on Pennec, Fillard and Ayache (2006)
 * "A Riemannian Framework For Tensor Computing").
 *
 * If this code is useful for your research, please cite the related publication:
 * @article{Jaquier17IROS,
 *	 author="Jaquier, N. and Calinon, S.",
 *	 title="Gaussian Mixture Regression on Symmetric Positive Definite Matrices Manifolds:
 *	 Application to Wrist Motion Estimation with s{EMG}",
 *	 year="2017, submitted for publication",
 *	 booktitle = "{IEEE/RSJ} Intl. Conf. on Intelligent Robots and Systems ({IROS})",
 *	 address = "Vancouver, Canada"
 * }
 *
 * Authors: Sylvain Calinon, Philip Abbet
 */


#include <stdio.h>
#include <armadillo>

#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>
#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>

using namespace arma;


/***************************** ALGORITHM SECTION *****************************/

void trans2d_to_gauss(const ui::Trans2d& gaussian_transforms,
					  const gfx2::window_size_t& window_size,
					  arma::vec &mu, arma::mat &sigma) {

	mu = gfx2::ui2fb_centered(vec({ gaussian_transforms.pos.x, gaussian_transforms.pos.y }),
							  window_size);

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

	sigma = RG * RG.t();
}

//---------------------------------------------------------

arma::mat expmap(const arma::mat& U, const arma::mat& S) {
	return real(sqrtmat(S) * expmat(inv(sqrtmat(S)) * U * inv(sqrtmat(S))) * sqrtmat(S));
}

//---------------------------------------------------------

arma::mat logmap(const arma::mat& X, const arma::mat& S) {
	return real(sqrtmat(S) * logmat(inv(sqrtmat(S)) * X * inv(sqrtmat(S))) * sqrtmat(S));
}

//---------------------------------------------------------

void interpolate(const std::vector<ui::Trans2d>& gaussian_transforms,
				 int nb_data,
				 std::vector<arma::vec> &interpolated_mu,
				 std::vector<arma::mat> &interpolated_sigma,
				 const gfx2::window_size_t& window_size) {

	const int nb_var = 2;	// Number of variables (fixed, since we use a
							// ui::Trans2d to define a gaussian)
	const int nb_var2 = nb_var + 1;

	// Transformation to Gaussians with augmented covariances centered on zero
	std::vector<arma::mat> augmented_sigma;

	for (size_t i = 0; i < gaussian_transforms.size(); ++i) {

		arma::vec current_mu;
		arma::mat sigma;

		trans2d_to_gauss(gaussian_transforms[i], window_size, current_mu, sigma);

		arma::mat current_sigma(nb_var2, nb_var2);

		current_sigma(0, 0, arma::size(nb_var, nb_var)) = sigma + current_mu * current_mu.t();
		current_sigma(0, nb_var, arma::size(nb_var, 1)) = current_mu;
		current_sigma(nb_var, 0, arma::size(1, nb_var)) = current_mu.t();
		current_sigma(nb_var, nb_var) = 1;

		augmented_sigma.push_back(current_sigma);
	}

	// Geodesic interpolation
	arma::vec w = arma::linspace(0, 1, nb_data);

	for (size_t i = 1; i < augmented_sigma.size(); ++i) {
		for (size_t t = 0; t < nb_data; ++t) {
			// Interpolation between two covariances can be computed in closed form
			arma::mat sigma = expmap(w(t) * logmap(augmented_sigma[i], augmented_sigma[i-1]),
									 augmented_sigma[i-1]);

			double beta = sigma(sigma.n_elem - 1);
			arma::vec mu = sigma(sigma.n_rows - 1, 0, arma::size(1, sigma.n_cols-1)).t() / beta;

			interpolated_mu.push_back(mu);
			interpolated_sigma.push_back(sigma(0, 0, arma::size(sigma.n_rows-1, sigma.n_cols-1)) - beta * mu * mu.t());
		}
	}
}


/*************************** DEMONSTRATION SECTION ***************************/

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error %d: %s\n", error, description);
}

// -----------------------------------------------------------------------------
// Render a 2d gaussian from its parameters (mu and sigma)
// -----------------------------------------------------------------------------
void render_gaussian(const arma::vec& mu, const arma::mat& sigma,
					 const arma::vec& color,
					 const gfx2::window_size_t& window_size) {

	// Rendering of the Gaussian
	gfx2::draw_gaussian(conv_to<fvec>::from(color), mu, sigma);

	glClear(GL_DEPTH_BUFFER_BIT);

	// Rendering of the Gaussian position
	fvec position({ (float) mu(0), (float) mu(1), 0.0f });

	fvec darker_color = conv_to<fvec>::from(color(span(0, 2))) * 0.5f;

	gfx2::draw_rectangle(darker_color, 4.0f * window_size.scale_x(),
						 4.0f * window_size.scale_y(), position);

	glClear(GL_DEPTH_BUFFER_BIT);
}


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	// Parameters
	int nb_states = 2; // Number of states in the GMM
	int nb_data = 20;  // Length of each trajectory


	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 600;
	window_size.win_height = 600;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;

	// Setup GUI
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		exit(1);

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	// Open a window and create its OpenGL context
	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Covariance interpolation",
		window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	// Setup OpenGL
	gfx2::init();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);
	// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Setup ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);
	ImVec4 clear_color = ImColor(255, 255, 255);


	// Main loop
	std::vector<ui::Trans2d> gaussian_transforms;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		// Start of rendering
		ImGui_ImplGlfwGL2_NewFrame();
		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-window_size.fb_width / 2, window_size.fb_width / 2,
				-window_size.fb_height / 2, window_size.fb_height / 2,
				-1.0f, 10.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glPushMatrix();

		// Ensure that the number of desired states hasn't changed
		if (nb_states > gaussian_transforms.size()) {
			for (int i = gaussian_transforms.size(); i < nb_states; ++i) {
				arma::vec mu = arma::randu(2);
				mu(0) = mu(0) * (window_size.win_width - 200) + 100;
				mu(1) = mu(1) * (window_size.win_height - 200) + 100;

				arma::vec xy = arma::randu(2);
				xy(0) = (xy(0) * window_size.win_width / 6 + 20);
				xy(1) = (xy(1) * window_size.win_height / 6 + 20);

				gaussian_transforms.push_back(ui::Trans2d(ImVec2((int) xy(0), 0),
														  ImVec2(0, (int) xy(1)),
														  ImVec2((int) mu(0), (int) mu(1))));
			}
		}
		else if (nb_states < gaussian_transforms.size()) {
			gaussian_transforms.resize(nb_states);
		}

		// Interpolation between the gaussians
		std::vector<arma::vec> interpolated_mu;
		std::vector<arma::mat> interpolated_sigma;

		interpolate(gaussian_transforms, nb_data, interpolated_mu, interpolated_sigma,
					window_size);

		// Rendering of the gaussians
		for (size_t i = 0; i < interpolated_sigma.size(); ++i) {
			render_gaussian(interpolated_mu[i], interpolated_sigma[i],
							arma::vec({ 0.8, 0.0, 0.0, 0.02 }), window_size);
		}

		for (size_t i = 0; i < gaussian_transforms.size(); ++i) {
			arma::vec mu;
			arma::mat sigma;

			trans2d_to_gauss(gaussian_transforms[i], window_size, mu, sigma);

			render_gaussian(mu, sigma, arma::vec({ 0.5, 0.5, 0.5, 0.5 }), window_size);
		}

		gfx2::draw_line(arma::fvec({ 0.8f, 0.0f, 0.0f }), interpolated_mu);

		// Gaussian UI widgets
		ui::begin("Gaussians");
		for (size_t i = 0; i < gaussian_transforms.size(); ++i)
			gaussian_transforms[i] = ui::affineSimple(i, gaussian_transforms[i]);
		ui::end();

		// Parameter window
		ImGui::Begin("Parameters", NULL, ImVec2(250, 80), 0.5f,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings);
		ImGui::SliderInt("Nb states", &nb_states, 2, 5);
		ImGui::SliderInt("Nb data", &nb_data, 5, 30);
		ImGui::End();

		// GUI rendering
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		// End of rendering
		glPopMatrix();
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;
	}

	// Cleanup
	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}
