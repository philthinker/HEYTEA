/*
 * demo_proMP01.cpp
 *
 * Trajectory distributions with ProMP.
 *
 * @incollection{Paraschos13,
 *  title = {Probabilistic Movement Primitives},
 *  author = {Paraschos, A. and Daniel, C. and Peters, J. and Neumann, G.},
 *  booktitle = NIPS,
 *  pages = {2616--2624},
 *  year = {2013},
 *  publisher = {Curran Associates, Inc.}
 * }
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

#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>

using namespace arma;


/***************************** ALGORITHM SECTION *****************************/

typedef std::vector<vec> vector_list_t;
typedef std::vector<mat> matrix_list_t;


//-----------------------------------------------------------------------------
// Contains all the parameters used by the algorithm. Some of them are
// modifiable through the UI, others are hard-coded.
//-----------------------------------------------------------------------------
struct parameters_t {
	int	  nb_states;		// Number of components in the GMM
	int	  nb_data;			// Number of datapoints in a trajectory
	int	  nb_reproductions; // Number of reproductions
	float dt;				// Time step (without rescaling, large values such
							// as 1 has the advantage of creating clusers based
							// on position information)
	float rbf_width;
	float regularized_pseudoinverse_parameter;
	float minimum_variance_parameter;
};


//-----------------------------------------------------------------------------
// Model trained using the algorithm
//-----------------------------------------------------------------------------
struct model_t {
	parameters_t parameters; // Parameters used to train the model

	int			 nb_var;	 // Dimension of position data
	vec			 mu_w;
	mat			 sigma_w;
	mat			 psi;
	mat			 H;			 // Functions activation
};


//-----------------------------------------------------------------------------
// Training of the model
//-----------------------------------------------------------------------------
void learn(const matrix_list_t& demos, model_t &model) {

	model.nb_var = 2;

	vec timesteps = linspace<vec>(
		0, model.parameters.nb_data - 1, model.parameters.nb_data
	) * model.parameters.dt;

	// Create basis functions (GMM with components equally split in time)
	vec mu = linspace<vec>(
		timesteps(0), timesteps(timesteps.n_rows - 1), model.parameters.nb_states
	);

	// Compute basis functions activation
	model.H = mat(model.parameters.nb_states, model.parameters.nb_data);

	for (unsigned int i = 0; i < model.parameters.nb_states; ++i) {
		model.H.row(i) = mvn::getPDFValue(colvec{ mu(i) }, mat({ model.parameters.rbf_width }),
										  timesteps.t()).t();
	}

	model.H = model.H / repmat(sum(model.H, 0), model.parameters.nb_states, 1);


	//_____ ProMP __________

	model.psi = mat(model.nb_var * model.parameters.nb_data,
					model.nb_var * model.parameters.nb_states);

	for (unsigned int i = 0; i < model.parameters.nb_data; ++i) {
		model.psi.rows(i * model.nb_var, (i + 1) * model.nb_var - 1) =
			kron(model.H.col(i).t(), eye(model.nb_var, model.nb_var));
	}

	mat w(model.nb_var * model.parameters.nb_states, demos.size());
	for (size_t i = 0; i < demos.size(); ++i) {
		w.col(i) = solve(model.psi.t() * model.psi +
							eye(
								model.nb_var * model.parameters.nb_states,
								model.nb_var * model.parameters.nb_states
							) * model.parameters.regularized_pseudoinverse_parameter,
						 model.psi.t()
				   ) * (mat) vectorise(demos[i]);
	}

	model.mu_w = mean(w, 1);

	model.sigma_w = eye(model.nb_var * model.parameters.nb_states,
						model.nb_var * model.parameters.nb_states
					) * model.parameters.minimum_variance_parameter;

	if (w.n_cols == 1)
		model.sigma_w += rowvec(cov(w.t()))[0];
	else
		model.sigma_w += cov(w.t());
}


//-----------------------------------------------------------------------------
// Compute the trajectory distribution
//-----------------------------------------------------------------------------
mat compute_trajectory_distribution(const model_t& model) {

	vec points = model.psi * model.mu_w;
	return reshape(points, model.nb_var, model.parameters.nb_data);
}


//-----------------------------------------------------------------------------
// Compute a reproduction (conditioning on trajectory distribution,
// reconstruction from partial data)
//-----------------------------------------------------------------------------
mat compute_reproduction(const model_t& model, const vec& from, const vec& to) {

	// The following code produces (with nb_data = 10 and nb_var = 2):
	//	   in = [ 0, 1, 18, 19 ]
	//	   out = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ]

	uvec in_out = linspace<uvec>(
		0, model.parameters.nb_data * model.nb_var - 1,
		model.parameters.nb_data * model.nb_var
	);

	uvec in(2 * model.nb_var);

	in.head(model.nb_var) = in_out.head(model.nb_var);
	in.tail(model.nb_var) = in_out.tail(model.nb_var);

	uvec out = in_out.subvec(model.nb_var, in_out.n_elem - model.nb_var - 1);

	// Input data
	mat start_end_coords(model.nb_var, 2);
	start_end_coords(span::all, 0) = from;
	start_end_coords(span::all, 1) = to;

	vec mu_in = reshape(
		start_end_coords + repmat((randu(model.nb_var, 1) - 0.5) * 10.0, 1, 2),
		model.nb_var * 2, 1
	);

	// Gaussian conditioning with trajectory distribution
	vec points(in_out.n_elem, fill::zeros);

	points.rows(in) = mu_in;

	mat A = model.sigma_w * model.psi.rows(in).t();
	mat B = model.psi.rows(in) * model.sigma_w * model.psi.rows(in).t();

	mat mu_w_tmp = model.mu_w +
				   (inv(B.t()) * A.t()).t() *	// == A / B
				   (points.rows(in) - model.psi.rows(in) * model.mu_w);

	points.rows(out) = model.psi.rows(out) * mu_w_tmp;

	return reshape(points, model.nb_var, model.parameters.nb_data);
}


/****************************** HELPER FUNCTIONS *****************************/

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error %d: %s\n", error, description);
}


//-----------------------------------------------------------------------------
// Contains all the informations about a viewport
//-----------------------------------------------------------------------------
struct viewport_t {
	int x;
	int y;
	int width;
	int height;

	// Projection matrix parameters
	arma::vec projection_top_left;
	arma::vec projection_bottom_right;
	double projection_near;
	double projection_far;
};


//-----------------------------------------------------------------------------
// Helper function to setup a viewport
//-----------------------------------------------------------------------------
void setup_viewport(viewport_t* viewport, int x, int y, int width, int height,
					double near_distance = -1.0, double far_distance = 1.0) {

	viewport->x = x;
	viewport->y = y;
	viewport->width = width;
	viewport->height = height;
	viewport->projection_top_left = vec({ (double) -width / 2,
										  (double) height / 2 });
	viewport->projection_bottom_right = vec({ (double) width / 2,
											  (double) -height / 2 });
	viewport->projection_near = near_distance;
	viewport->projection_far = far_distance;
}


//-----------------------------------------------------------------------------
// Converts some coordinates from UI-space to OpenGL-space, taking the
// coordinates of a viewport into account
//-----------------------------------------------------------------------------
arma::vec ui2fb(const arma::vec& coords, const gfx2::window_size_t& window_size,
				const viewport_t& viewport) {
	arma::vec result = coords;

	// ui -> viewport
	result(0) = coords(0) * (float) window_size.fb_width / (float) window_size.win_width - viewport.x;
	result(1) = (window_size.win_height - coords(1)) *
				(float) window_size.fb_height / (float) window_size.win_height - viewport.y;

	// viewport -> fb
	result(0) = result(0) - (float) viewport.width * 0.5f;
	result(1) = result(1) - (float) viewport.height * 0.5f;

	return result;
}


//-----------------------------------------------------------------------------
// Colors of the displayed lines and gaussians
//-----------------------------------------------------------------------------
const mat COLORS({
	{ 0.0,  0.0,  1.0  },
	{ 0.0,  0.5,  0.0  },
	{ 1.0,  0.0,  0.0  },
	{ 0.0,  0.75, 0.75 },
	{ 0.75, 0.0,  0.75 },
	{ 0.75, 0.75, 0.0  },
	{ 0.25, 0.25, 0.25 },
});


//-----------------------------------------------------------------------------
// Create a demonstration (with a length of 'nb_samples') from a trajectory
// (of any length)
//-----------------------------------------------------------------------------
mat sample_trajectory(const vector_list_t& trajectory, int nb_samples) {

	// Resampling of the trajectory
	vec x(trajectory.size());
	vec y(trajectory.size());
	vec x2(trajectory.size());
	vec y2(trajectory.size());

	for (size_t i = 0; i < trajectory.size(); ++i) {
		x(i) = trajectory[i](0);
		y(i) = trajectory[i](1);
	}

	vec from_indices = linspace<vec>(0, trajectory.size() - 1, trajectory.size());
	vec to_indices = linspace<vec>(0, trajectory.size() - 1, nb_samples);

	interp1(from_indices, x, to_indices, x2, "*linear");
	interp1(from_indices, y, to_indices, y2, "*linear");

	// Create the demonstration
	mat demo(2, nb_samples);
	for (int i = 0; i < nb_samples; ++i) {
		demo(0, i) = x2[i];
		demo(1, i) = y2[i];
	}

	return demo;
}


//-----------------------------------------------------------------------------
// Contains all the needed infos about the state of the application (values of
// the parameters modifiable via the UI, which action the user is currently
// doing, ...)
//-----------------------------------------------------------------------------
struct gui_state_t {
	// Indicates if the user is currently drawing a new demonstration
	bool is_drawing_demonstration;

	// Indicates if the parameters dialog is displayed
	bool is_parameters_dialog_displayed;

	// Indicates if the parameters were modified through the UI
	bool are_parameters_modified;

	// Indicates if the reproductions must be recomputed
	bool must_recompute_reproductions;

	// Parameters modifiable via the UI (they correspond to the ones declared
	// in parameters_t)
	int parameter_nb_states;
	int parameter_nb_data;
	int parameter_nb_reproductions;
	float parameter_rbf_width;
	float parameter_regularized_pseudoinverse_parameter;
	float parameter_minimum_variance_parameter;

	bool display_colored_points;
};


//-----------------------------------------------------------------------------
// Render the "demonstrations" viewport
//-----------------------------------------------------------------------------
void draw_demos_viewport(const viewport_t& viewport,
						 const vector_list_t& current_trajectory,
						 const matrix_list_t& demonstrations) {

	glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
	glScissor(viewport.x, viewport.y, viewport.width, viewport.height);
	glClearColor(0.7f, 0.7f, 0.7f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(viewport.projection_top_left(0), viewport.projection_bottom_right(0),
			viewport.projection_bottom_right(1), viewport.projection_top_left(1),
			viewport.projection_near, viewport.projection_far);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Draw the currently created demonstration (if any)
	if (current_trajectory.size() > 1)
		gfx2::draw_line(arma::fvec({0.33f, 0.97f, 0.33f}), current_trajectory);

	// Draw the demonstrations
	int color_index = 0;
	for (size_t i = 0; i < demonstrations.size(); ++i) {
		arma::mat datapoints = demonstrations[i];

		arma::fvec color = arma::conv_to<arma::fvec>::from(COLORS.row(color_index));

		gfx2::draw_line(color, datapoints);

		++color_index;
		if (color_index >= COLORS.n_rows)
			color_index = 0;
	}
}


//-----------------------------------------------------------------------------
// Render a "model" viewport
//-----------------------------------------------------------------------------
void draw_model_viewport(const viewport_t& viewport,
						 const model_t& model,
						 const mat& trajectory_distribution) {

	glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
	glScissor(viewport.x, viewport.y, viewport.width, viewport.height);
	glClearColor(0.9f, 0.9f, 0.9f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(viewport.projection_top_left(0), viewport.projection_bottom_right(0),
			viewport.projection_bottom_right(1), viewport.projection_top_left(1),
			viewport.projection_near, viewport.projection_far);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if (trajectory_distribution.n_cols > 0) {

		// Draw the GMM states
		for (int i = 0; i < model.parameters.nb_states; ++i) {
			glClear(GL_DEPTH_BUFFER_BIT);

			span id = span(i * model.nb_var, i * model.nb_var + 1);

			gfx2::draw_gaussian(conv_to<fvec>::from(COLORS.row(i % COLORS.n_rows).t()),
								model.mu_w(id), model.sigma_w(id, id));
		}

		glClear(GL_DEPTH_BUFFER_BIT);

		// Draw the trajectory distribution (if any)
		gfx2::draw_line(arma::fvec({0.0f, 0.0f, 0.0f}), trajectory_distribution);
	}
}


//-----------------------------------------------------------------------------
// Render a "reproduction" viewport
//-----------------------------------------------------------------------------
void draw_reproductions_viewport(const viewport_t& viewport,
								 const model_t& model,
								 const matrix_list_t& reproductions,
								 bool display_colored_points) {

	glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
	glScissor(viewport.x, viewport.y, viewport.width, viewport.height);
	glClearColor(0.9f, 0.9f, 0.9f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(viewport.projection_top_left(0), viewport.projection_bottom_right(0),
			viewport.projection_bottom_right(1), viewport.projection_top_left(1),
			viewport.projection_near, viewport.projection_far);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if (display_colored_points) {

		// Create a color map
		mat colors(model.parameters.nb_states, 3);
		for (int i = 0; i < model.parameters.nb_states; ++i)
			colors(i, span::all) = COLORS(i % COLORS.n_rows, span::all);

		// Draw the reproductions as points colored by the gmm states (if any)
		for (auto iter = reproductions.begin(); iter != reproductions.end(); ++iter) {
			const mat& datapoints = (*iter);

			for (unsigned int i = 0; i < datapoints.n_cols; ++i) {
				fvec color = conv_to<fvec>::from(model.H.col(i).t() * colors);

				fvec position = zeros<fvec>(3);
				position(0) = datapoints(0, i);
				position(1) = datapoints(1, i);

				gfx2::draw_rectangle(color, 8.0f, 8.0f, position);
			}
		}
	}

	// Draw the reproductions as lines (if any)
	glClear(GL_DEPTH_BUFFER_BIT);

	for (auto iter = reproductions.begin(); iter != reproductions.end(); ++iter) {
		gfx2::draw_line(arma::fvec({0.0f, 0.0f, 0.0f}), *iter);
	}
}


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	// Model
	model_t model;

	// Parameters
	model.parameters.nb_states                           = 10;
	model.parameters.nb_data                             = 200;
	model.parameters.nb_reproductions                    = 10;
	model.parameters.dt                                  = 0.01f;
	model.parameters.rbf_width                           = 1e-2f;
	model.parameters.regularized_pseudoinverse_parameter = 3e-2f;
	model.parameters.minimum_variance_parameter          = 1.0f;


	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 1200;
	window_size.win_height = 400;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;
	int viewport_width = 0;
	int viewport_height = 0;


	// Initialise GLFW
	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	// Open a window and create its OpenGL context
	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Demo Conditioning on trajectory distributions",
		window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);


	// Setup OpenGL
	gfx2::init();
	glEnable(GL_SCISSOR_TEST);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Setup ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);


	// Viewports
	viewport_t viewport_demos;
	viewport_t viewport_model;
	viewport_t viewport_repros;


	// GUI state
	gui_state_t gui_state;
	gui_state.is_drawing_demonstration = false;
	gui_state.is_parameters_dialog_displayed = false;
	gui_state.are_parameters_modified = false;
	gui_state.must_recompute_reproductions = false;
	gui_state.parameter_nb_states = model.parameters.nb_states;
	gui_state.parameter_nb_data = model.parameters.nb_data;
	gui_state.parameter_nb_reproductions = model.parameters.nb_reproductions;
	gui_state.parameter_rbf_width = model.parameters.rbf_width;
	gui_state.parameter_regularized_pseudoinverse_parameter = model.parameters.regularized_pseudoinverse_parameter;
	gui_state.parameter_minimum_variance_parameter = model.parameters.minimum_variance_parameter;
	gui_state.display_colored_points = false;


	// List of demonstrations and reproductions
	matrix_list_t demos;
	mat trajectory_distribution;
	matrix_list_t reproductions;


	// Main loop
	vector_list_t current_trajectory;
	std::vector<vector_list_t> original_trajectories;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		if ((window_result == gfx2::WINDOW_READY) || (window_result == gfx2::WINDOW_RESIZED)) {

			viewport_width = window_size.fb_width / 3 - 1;
			viewport_height = window_size.fb_height;

			// Update all the viewports
			setup_viewport(&viewport_demos, 0, 0, viewport_width, viewport_height);

			setup_viewport(&viewport_model, viewport_width + 2, 0, viewport_width,
						   viewport_height);

			setup_viewport(&viewport_repros, window_size.fb_width - viewport_width, 0,
						   viewport_width, viewport_height);
		}


		// If the parameters changed, learn the model again
		if (gui_state.are_parameters_modified) {

			if (model.parameters.nb_data != gui_state.parameter_nb_data) {
				demos.clear();

				for (size_t i = 0; i < original_trajectories.size(); ++i) {
					demos.push_back(sample_trajectory(original_trajectories[i],
													  gui_state.parameter_nb_data)
					);
				}
			}

			model.parameters.nb_data = gui_state.parameter_nb_data;
			model.parameters.nb_states = gui_state.parameter_nb_states;
			model.parameters.rbf_width = gui_state.parameter_rbf_width;
			model.parameters.regularized_pseudoinverse_parameter = gui_state.parameter_regularized_pseudoinverse_parameter;
			model.parameters.minimum_variance_parameter = gui_state.parameter_minimum_variance_parameter;

			if (!demos.empty()) {
				learn(demos, model);
				gui_state.must_recompute_reproductions = true;
			}

			gui_state.are_parameters_modified = false;
		}


		// Recompute the reproductions (if necessary)
		if (gui_state.must_recompute_reproductions) {

			model.parameters.nb_reproductions = gui_state.parameter_nb_reproductions;

			trajectory_distribution = compute_trajectory_distribution(model);

			reproductions.clear();
			for (int i = 0; i < model.parameters.nb_reproductions; ++i) {
				reproductions.push_back(compute_reproduction(
						model, demos[0].col(0), demos[0].col(model.parameters.nb_data - 1)
				));
			}

			gui_state.must_recompute_reproductions = false;
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();

		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glScissor(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		draw_demos_viewport(viewport_demos, current_trajectory, demos);

		draw_model_viewport(viewport_model, model, trajectory_distribution);

		draw_reproductions_viewport(viewport_repros, model, reproductions,
									gui_state.display_colored_points);


		// Window: Demonstrations
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("Demonstrations", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Demonstrations		");
		ImGui::SameLine();

		if (ImGui::Button("Clear")) {
			original_trajectories.clear();
			demos.clear();
			reproductions.clear();
			trajectory_distribution = mat();
			model.mu_w = vec();
			model.sigma_w = mat();
		}

		ImGui::SameLine();
		ImGui::Text("	  ");
		ImGui::SameLine();

		if (ImGui::Button("Parameters"))
			gui_state.is_parameters_dialog_displayed = true;

		ImGui::End();


		// Window: Model
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width / 3, 0));
		ImGui::Begin("Model", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Model");

		ImGui::End();


		// Window: Reproductions
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width - window_size.win_width / 3, 0));
		ImGui::Begin("Reproductions", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Reproductions");

		ImGui::End();


		// Window: Parameters
		ImGui::SetNextWindowSize(ImVec2(800, 216));
		ImGui::SetNextWindowPos(ImVec2((window_size.win_width - 800) / 2,
									   (window_size.win_height - 216) / 2));
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 255));

		if (gui_state.is_parameters_dialog_displayed)
			ImGui::OpenPopup("Parameters");

		if (ImGui::BeginPopupModal("Parameters", NULL,
								   ImGuiWindowFlags_NoResize |
								   ImGuiWindowFlags_NoSavedSettings)) {

			ImGui::SliderInt("Nb states", &gui_state.parameter_nb_states, 2, 20);
			ImGui::SliderInt("Nb data", &gui_state.parameter_nb_data, 100, 300);
			ImGui::SliderInt("Nb reproductions", &gui_state.parameter_nb_reproductions, 2, 10);
			ImGui::SliderFloat("RBF width", &gui_state.parameter_rbf_width, 1e-3f, 3e-2f);
			ImGui::SliderFloat("Regularized pseudoinverse parameter",
							   &gui_state.parameter_regularized_pseudoinverse_parameter,
							   3e-2f, 1e-1f);
			ImGui::SliderFloat("Minimum variance parameter",
							   &gui_state.parameter_minimum_variance_parameter, 1e-3f, 1e0f);
			ImGui::Checkbox("Show colored points in reproductions", &gui_state.display_colored_points);

			if (ImGui::Button("Close")) {
				ImGui::CloseCurrentPopup();
				gui_state.is_parameters_dialog_displayed = false;
				gui_state.are_parameters_modified = true;
			}

			ImGui::EndPopup();
		}

		ImGui::PopStyleColor();


		// GUI rendering
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		// Swap buffers
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;


		if (!gui_state.is_drawing_demonstration && !gui_state.is_parameters_dialog_displayed) {
			// Left click: start a new demonstration (only if not on the UI and in the
			// demonstrations viewport)
			if (ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1)) {
				double mouse_x, mouse_y;
				glfwGetCursorPos(window, &mouse_x, &mouse_y);

				if (!ImGui::GetIO().WantCaptureMouse && (mouse_x <= window_size.win_width / 3))
				{
					gui_state.is_drawing_demonstration = true;

					vec coords = ui2fb({ mouse_x, mouse_y }, window_size, viewport_demos);
					current_trajectory.push_back(coords);
				}
			}
		} else if (gui_state.is_drawing_demonstration) {
			double mouse_x, mouse_y;
			glfwGetCursorPos(window, &mouse_x, &mouse_y);

			vec coords = ui2fb({ mouse_x, mouse_y }, window_size, viewport_demos);

			vec last_point = current_trajectory[current_trajectory.size() - 1];
			vec diff = abs(coords - last_point);

			if ((diff(0) > 1e-6) && (diff(1) > 1e-6))
				current_trajectory.push_back(coords);

			// Left mouse button release: end the demonstration creation
			if (!ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)) {
				gui_state.is_drawing_demonstration = false;

				if (current_trajectory.size() > 1) {
					demos.push_back(sample_trajectory(current_trajectory, model.parameters.nb_data));

					original_trajectories.push_back(current_trajectory);

					learn(demos, model);

					gui_state.must_recompute_reproductions = true;
				}

				current_trajectory.clear();
			}
		}
	}


	// Cleanup
	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}
