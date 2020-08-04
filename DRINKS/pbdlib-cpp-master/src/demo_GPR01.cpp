/*
 * demo_GPR01.cpp
 *
 * Gaussian process regression (GPR) with RBF kernel
 *
 * If this code is useful for your research, please cite the related publication:
 * @incollection{Calinon19chapter,
 * 	author="Calinon, S. and Lee, D.",
 * 	title="Learning Control",
 * 	booktitle="Humanoid Robotics: a Reference",
 * 	publisher="Springer",
 * 	editor="Vadakkepat, P. and Goswami, A.", 
 * 	year="2019",
 * 	doi="10.1007/978-94-007-7194-9_68-1",
 * 	pages="1--52"
 * }
 *
 * Authors: Sylvain Calinon, Philip Abbet
 */


#include <stdio.h>
#include <armadillo>

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
	int nb_data;				// Number of datapoints in a trajectory
	int nb_data_reproduction;	// Number of datapoints for reproduction
	vec p;						// GPR parameters
};


//-----------------------------------------------------------------------------
// Create a demonstration (with a length of 'time_steps.size()') from a
// trajectory (of any length)
//-----------------------------------------------------------------------------
mat sample_trajectory(const mat& trajectory, const uvec& time_steps) {

	// Resampling of the trajectory
	vec x = trajectory.row(0).t();
	vec y = trajectory.row(1).t();
	vec x2;
	vec y2;

	vec from_indices = linspace<vec>(0, trajectory.n_cols - 1, trajectory.n_cols);
	vec to_indices = linspace<vec>(0, trajectory.n_cols - 1, time_steps[time_steps.size() - 1] + 1);

	interp1(from_indices, x, to_indices, x2, "*linear");
	interp1(from_indices, y, to_indices, y2, "*linear");

	// Create the demonstration
	mat demo(3, time_steps.size());
	for (int i = 0; i < time_steps.size(); ++i) {
		int j = time_steps[i];
		demo(0, i) = j;
		demo(1, i) = x2[j];
		demo(2, i) = y2[j];
	}

	return demo;
}


//-----------------------------------------------------------------------------
// Compute pairwise distance between two sets of vectors.
//-----------------------------------------------------------------------------
mat pdist2(const vec& x, const vec& y) {
	mat result(x.n_rows, y.n_rows);

	for (int i = 0; i < x.n_rows; ++i) {
		for (int j = 0; j < y.n_rows; ++j) {
			result(i, j) = fabsl(x(i) - y(j));
		}
	}

	return result;
}


//-----------------------------------------------------------------------------
// Gaussian mixture regression (GPR)
//-----------------------------------------------------------------------------
void compute_GPR(const parameters_t& parameters, const matrix_list_t& demos,
				 mat &points, matrix_list_t &sigma_out) {

	const int nb_var = demos[0].n_rows;

	sigma_out.clear();

	mat data(nb_var, demos[0].n_cols * demos.size());
	for (unsigned int m = 0; m < demos.size(); ++m) {
		data(span::all, span(m * demos[0].n_cols,
							 (m + 1) * demos[0].n_cols - 1)) =
			demos[m];
	}

	vec all_time_steps = linspace<vec>(0, parameters.nb_data - 1, parameters.nb_data_reproduction);

	mat K = parameters.p(0) * exp(-1.0 / parameters.p(1) * pow(pdist2(data.row(0).t(), data.row(0).t()), 2.0)) +
			parameters.p(2) * eye(data.n_cols, data.n_cols);

	mat Kd = parameters.p(0) * exp(-1.0 / parameters.p(1) * pow(pdist2(all_time_steps, data.row(0).t()), 2.0));

	points = mat(nb_var, all_time_steps.n_rows);
	points(0, span::all) = all_time_steps.t();

	points(span(1, nb_var - 1), span::all) = (Kd * solve(K, data(span(1, nb_var - 1), span::all).t())).t();

	mat Kdd = parameters.p(0) * exp(-1.0 / parameters.p(1) * pow(pdist2(all_time_steps, all_time_steps), 2));

	mat S = Kdd - Kd * solve(K, Kd.t());

	for (unsigned int t = 0; t < parameters.nb_data_reproduction; ++t) {
		mat sigma = eye(nb_var - 1, nb_var - 1) * S(t, t);
		sigma_out.push_back(sigma);
	}
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
// Converts some coordinates from OpenGL-space to UI-space, taking the
// coordinates of a viewport into account
//-----------------------------------------------------------------------------
arma::vec fb2ui(const arma::vec& coords, const gfx2::window_size_t& window_size,
				const viewport_t& viewport) {
	arma::vec result = coords;

	// fb -> viewport
	result(0) = coords(0) + (float) viewport.width * 0.5f;
	result(1) = coords(1) + (float) viewport.height * 0.5f;

	// viewport -> ui
	result(0) = (result(0) + viewport.x) * (float) window_size.win_width / (float) window_size.fb_width;

	result(1) = window_size.win_height - (result(1) + viewport.y) * (float) window_size.win_height / (float) window_size.fb_height;

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
	bool must_recompute_GPR;

	// Offset of the first missing point in the demonstrations
	int parameter_missing_data_offset;

	// Number of missing points in the demonstrations
	int parameter_missing_data_length;

	// Parameters modifiable via the UI (they correspond to the ones declared
	// in parameters_t)
	int parameter_nb_data;
	int parameter_nb_data_reproduction;
	fvec parameter_p;
};


//-----------------------------------------------------------------------------
// Render the "demonstrations & model" viewport
//-----------------------------------------------------------------------------
void draw_demos_viewport(const viewport_t& viewport,
						 const vector_list_t& current_trajectory,
						 const matrix_list_t& original_trajectories,
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
		arma::fvec color = conv_to<fvec>::from(COLORS.row(color_index));

		gfx2::draw_line(color, original_trajectories[i]);

		for (size_t j = 0; j < demonstrations[i].n_cols; ++j) {
			fvec position = zeros<fvec>(3);
			position(span(0, 1)) = conv_to<fvec>::from(demonstrations[i](span(1, 2), j));

			gfx2::draw_rectangle(color, 10.0f, 10.0f, position,
								 gfx2::rotate(fvec({ 0.0f, 0.0f, 1.0f }), datum::pi / 4));
		}

		++color_index;
		if (color_index >= COLORS.n_rows)
			color_index = 0;
	}
}


//-----------------------------------------------------------------------------
// Sample the GPR result points at the given time steps
//-----------------------------------------------------------------------------
mat sample_GPR_points(const mat& points, const vec& time_steps) {

	mat result(3, time_steps.n_rows);

	for (size_t i = 0; i < time_steps.n_rows; ++i) {
		for (size_t j = 0; j < points.n_cols - 1; ++j) {
			if ((points(0, j) <= time_steps(i)) && (time_steps(i) <= points(0, j + 1))) {
				result(0, i) = time_steps(i);
				result(1, i) = points(1, j) + (time_steps(i) - points(0, j)) *
							   (points(1, j + 1) - points(1, j)) / (points(0, j + 1) - points(0, j));
				result(2, i) = points(2, j) + (time_steps(i) - points(0, j)) *
							   (points(2, j + 1) - points(2, j)) / (points(0, j + 1) - points(0, j));
				break;
			}
		}
	}

	return result;
}


//-----------------------------------------------------------------------------
// Render a "reproduction" viewport
//-----------------------------------------------------------------------------
void draw_GPR_viewport(const viewport_t& viewport, const mat& points,
					   const std::vector<gfx2::model_t>& models,
					   const matrix_list_t& demonstrations) {

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

	if (!models.empty()) {
		for (int i = 0; i < models.size(); ++i) {
			glClear(GL_DEPTH_BUFFER_BIT);
			gfx2::draw(models[i]);
		}

		glClear(GL_DEPTH_BUFFER_BIT);

		glLineWidth(4.0f);
		gfx2::draw_line(arma::fvec({0.0f, 0.4f, 0.0f}), points(span(1, 2), span::all));
		glLineWidth(1.0f);

		glClear(GL_DEPTH_BUFFER_BIT);

		mat sampled_points = sample_GPR_points(points, vec(demonstrations[0].row(0).t()));

		for (size_t j = 0; j < sampled_points.n_cols; ++j) {
			fvec position = zeros<fvec>(3);
			position(span(0, 1)) = conv_to<fvec>::from(sampled_points(span(1, 2), j));

			gfx2::draw_rectangle(fvec({ 0.0f, 0.0f, 0.0f }), 10.0f, 10.0f, position,
								 gfx2::rotate(fvec({ 0.0f, 0.0f, 1.0f }), datum::pi / 4));
		}
	}
}


//-----------------------------------------------------------------------------
// Returns the dimensions that a plot should have inside the provided viewport
//-----------------------------------------------------------------------------
ivec get_plot_dimensions(const viewport_t& viewport) {

	const int MARGIN = 50;

	ivec result(2);
	result(0) = viewport.width - 2 * MARGIN;
	result(1) = viewport.height - 2 * MARGIN;

	return result;
}


//-----------------------------------------------------------------------------
// Render a "timeline" viewport
//-----------------------------------------------------------------------------
void draw_timeline_viewport(const gfx2::window_size_t& window_size,
							const viewport_t& viewport,
							const matrix_list_t& original_trajectories,
							const matrix_list_t& demonstrations,
							const mat& GPR_points,
							matrix_list_t GPR_sigma,
							unsigned int dimension) {

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

	ivec plot_dimensions = get_plot_dimensions(viewport);

	ivec plot_top_left({ -plot_dimensions(0) / 2, plot_dimensions(1) / 2 });
	ivec plot_bottom_right({ plot_dimensions(0) / 2, -plot_dimensions(1) / 2 });

	// Axis labels
	ui::begin("Text");

	vec coords = fb2ui(vec({ -20.0, double(-viewport.height / 2 + 45) }),
					   window_size, viewport);
	ui::text(ImVec2(coords(0), coords(1)), "t", ImVec4(0,0,0,1));

	std::stringstream label;
	label << "x" << dimension;

	coords = fb2ui(vec({ double(-viewport.width / 2) + 10, -20.0 }),
				   window_size, viewport);
	ui::text(ImVec2(coords(0), coords(1)), label.str(), ImVec4(0,0,0,1));

	ui::end();

	// Draw the axes
	gfx2::draw_line(fvec({0.0f, 0.0f, 0.0f}),
					mat({ { double(plot_top_left(0)), double(plot_bottom_right(0)) },
						  { double(plot_bottom_right(1)), double(plot_bottom_right(1)) }
						})
	);

	gfx2::draw_line(fvec({0.0f, 0.0f, 0.0f}),
					mat({ { double(plot_top_left(0)), double(plot_top_left(0)) },
						  { double(plot_bottom_right(1)), double(plot_top_left(1)) }
						})
	);

	// Check if there is something to display
	if (demonstrations.empty())
		return;

	// Draw the GPR
	double scale_x = (double) plot_dimensions(0) / GPR_points(0, GPR_points.n_cols - 1);
	double scale_y = (double) plot_dimensions(1) / viewport.height;

	mat top_vertices(2, GPR_points.n_cols);
	mat bottom_vertices(2, GPR_points.n_cols);

	for (int j = 0; j < GPR_points.n_cols; ++j) {
		top_vertices(0, j) = GPR_points(0, j) * scale_x - plot_dimensions(0) / 2;
		top_vertices(1, j) = (GPR_points(dimension, j) +
							  sqrt(GPR_sigma[j](dimension - 1, dimension - 1)) * 20.0) * scale_y;

		bottom_vertices(0, j) = top_vertices(0, j);
		bottom_vertices(1, j) = (GPR_points(dimension, j) -
								 sqrt(GPR_sigma[j](dimension - 1, dimension - 1)) * 20.0) * scale_y;
	}

	mat gmr_points(2, (GPR_points.n_cols - 1) * 6);

	for (int j = 0; j < GPR_points.n_cols - 1; ++j) {
		gmr_points(span::all, j * 6 + 0) = top_vertices(span::all, j);
		gmr_points(span::all, j * 6 + 1) = bottom_vertices(span::all, j);
		gmr_points(span::all, j * 6 + 2) = top_vertices(span::all, j + 1);

		gmr_points(span::all, j * 6 + 3) = top_vertices(span::all, j + 1);
		gmr_points(span::all, j * 6 + 4) = bottom_vertices(span::all, j);
		gmr_points(span::all, j * 6 + 5) = bottom_vertices(span::all, j + 1);
	}

	gfx2::model_t gmr_model = gfx2::create_mesh(fvec({ 0.0f, 0.8f, 0.0f, 0.05f }), gmr_points);
	gmr_model.use_one_minus_src_alpha_blending = true;
	gfx2::draw(gmr_model);

	glClear(GL_DEPTH_BUFFER_BIT);

	// Draw the demonstrations
	int color_index = 0;

	for (size_t i = 0; i < original_trajectories.size(); ++i) {
		uvec time_steps = linspace<uvec>(0, GPR_points(0, GPR_points.n_cols - 1),
										 original_trajectories[i].n_cols);

		mat datapoints = sample_trajectory(original_trajectories[i], time_steps).rows(uvec({ 0, dimension }));

		datapoints(0, span::all) = datapoints(0, span::all) * scale_x - plot_dimensions(0) / 2;
		datapoints(1, span::all) *= scale_y;

		arma::fvec color = arma::conv_to<arma::fvec>::from(COLORS.row(color_index));

		gfx2::draw_line(color, datapoints);

		++color_index;
		if (color_index >= COLORS.n_rows)
			color_index = 0;
	}

	// Draw the GPR result
	mat points(2, GPR_points.n_cols);
	points(0, span::all) = GPR_points(0, span::all);
	points(1, span::all) = GPR_points(dimension, span::all);

	points(0, span::all) = points(0, span::all) * scale_x - plot_dimensions(0) / 2;
	points(1, span::all) *= scale_y;

	glLineWidth(4.0f);
	gfx2::draw_line(arma::fvec({0.0f, 0.4f, 0.0f}), points);
	glLineWidth(1.0f);

	glClear(GL_DEPTH_BUFFER_BIT);

	mat sampled_points = sample_GPR_points(GPR_points, vec(demonstrations[0].row(0).t()));

	for (size_t j = 0; j < sampled_points.n_cols; ++j) {
		fvec position = zeros<fvec>(3);
		position(span(0, 1)) = conv_to<fmat>::from(sampled_points.rows(uvec({ 0, dimension }))).col(j);

		position(0, span::all) = position(0, span::all) * scale_x - plot_dimensions(0) / 2;
		position(1, span::all) *= scale_y;

		gfx2::draw_rectangle(fvec({ 0.0f, 0.0f, 0.0f }), 10.0f, 10.0f, position,
							 gfx2::rotate(fvec({ 0.0f, 0.0f, 1.0f }), datum::pi / 4));
	}
}


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	// Parameters
	parameters_t parameters;
	parameters.nb_data              = 20;
	parameters.nb_data_reproduction = 100;
	parameters.p                    = vec({ 100.0, 10.0, 1.0 });


	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 800;
	window_size.win_height = 800;
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
		"Demo Gaussian process regression (GPR) with RBF kernel",
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
	viewport_t viewport_GPR;
	viewport_t viewport_x1;
	viewport_t viewport_x2;


	// GUI state
	gui_state_t gui_state;
	gui_state.is_drawing_demonstration = false;
	gui_state.is_parameters_dialog_displayed = false;
	gui_state.are_parameters_modified = true;
	gui_state.must_recompute_GPR = false;
	gui_state.parameter_nb_data = parameters.nb_data;
	gui_state.parameter_nb_data_reproduction = parameters.nb_data_reproduction;
	gui_state.parameter_missing_data_offset = parameters.nb_data / 2 - 1;
	gui_state.parameter_missing_data_length = parameters.nb_data / 4;
	gui_state.parameter_p = conv_to<fvec>::from(parameters.p);


	// List of demonstrations and GMr results
	uvec time_steps;
	matrix_list_t demos;
	mat GPR_points;
	matrix_list_t GPR_sigma;
	std::vector<gfx2::model_t> GPR_models;


	// Main loop
	vector_list_t current_trajectory;
	matrix_list_t original_trajectories;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		if ((window_result == gfx2::WINDOW_READY) || (window_result == gfx2::WINDOW_RESIZED)) {

			viewport_width = window_size.fb_width / 2 - 1;
			viewport_height = window_size.fb_height / 2 - 1;

			// Update all the viewports
			setup_viewport(&viewport_demos, 0, window_size.fb_height - viewport_height,
						   viewport_width, viewport_height);

			setup_viewport(&viewport_GPR, window_size.fb_width - viewport_width,
						   window_size.fb_height - viewport_height,
						   viewport_width, viewport_height);

			setup_viewport(&viewport_x1, 0, 0, viewport_width, viewport_height);

			setup_viewport(&viewport_x2, window_size.fb_width - viewport_width, 0,
						   viewport_width, viewport_height);
		}


		// If the parameters changed, resample the trajectories and trigger a
		// recomputation
		if (gui_state.are_parameters_modified) {

			if (gui_state.parameter_missing_data_offset >= gui_state.parameter_nb_data - 1) {
				if (gui_state.parameter_missing_data_length > gui_state.parameter_nb_data / 2)
					gui_state.parameter_missing_data_length = gui_state.parameter_nb_data / 2;

				gui_state.parameter_missing_data_offset = gui_state.parameter_nb_data - (gui_state.parameter_missing_data_length + 1);

			} else if (gui_state.parameter_missing_data_length > gui_state.parameter_nb_data - (gui_state.parameter_missing_data_offset + 1)) {
				gui_state.parameter_missing_data_length = gui_state.parameter_nb_data - (gui_state.parameter_missing_data_offset + 1);
			}

			demos.clear();

			uvec all_time_steps = linspace<uvec>(
				0, gui_state.parameter_nb_data - 1, gui_state.parameter_nb_data
			);

			int offset1 = gui_state.parameter_missing_data_offset - 1;
			int offset2 = offset1 + gui_state.parameter_missing_data_length + 1;

			time_steps = uvec(gui_state.parameter_nb_data - (offset2 - offset1) + 1);

			time_steps(span(0, offset1)) = all_time_steps(span(0, offset1));
			time_steps(span(offset1 + 1, time_steps.n_rows - 1)) =
				all_time_steps(span(offset2, gui_state.parameter_nb_data - 1));

			for (size_t i = 0; i < original_trajectories.size(); ++i) {
				demos.push_back(sample_trajectory(original_trajectories[i],
												  time_steps)
				);
			}

			parameters.nb_data = gui_state.parameter_nb_data;
			parameters.nb_data_reproduction = gui_state.parameter_nb_data_reproduction;
			parameters.p = conv_to<vec>::from(gui_state.parameter_p);

			gui_state.must_recompute_GPR = !demos.empty();
			gui_state.are_parameters_modified = false;
		}


		// Recompute the GPR (if necessary)
		if (gui_state.must_recompute_GPR) {

			compute_GPR(parameters, demos, GPR_points, GPR_sigma);

			// Create one big mesh for the GPR viewport (for performance reasons)
			for (int i = 0; i < GPR_models.size(); ++i)
				gfx2::destroy(GPR_models[i]);

			GPR_models.clear();

			const int NB_POINTS = 60;

			mat vertices(2, NB_POINTS * 3 * GPR_sigma.size());
			mat lines(2, NB_POINTS * 2 * GPR_sigma.size());

			for (int j = 0; j < GPR_sigma.size(); ++j) {

				mat v = gfx2::get_gaussian_background_vertices(GPR_points(span(1, 2), j),
															   GPR_sigma[j] * 20.0, NB_POINTS);

				vertices(span::all, span(j * NB_POINTS * 3, (j + 1) * NB_POINTS * 3 - 1)) = v;

				mat p = gfx2::get_gaussian_border_vertices(GPR_points(span(1, 2), j),
														   GPR_sigma[j] * 20.0, NB_POINTS, false);

				lines(span::all, span(j * NB_POINTS * 2, (j + 1) * NB_POINTS * 2 - 1)) = p;
			}

			GPR_models.push_back(
				gfx2::create_mesh(fvec({ 0.0f, 0.8f, 0.0f, 0.1f }), vertices)
			);

			GPR_models[0].use_one_minus_src_alpha_blending = true;

			GPR_models.push_back(
				gfx2::create_line(fvec({ 0.0f, 0.4f, 0.0f, 0.1f }), lines,
								  arma::zeros<arma::fvec>(3),
								  arma::eye<arma::fmat>(4, 4), 0, false)
			);

			gui_state.must_recompute_GPR = false;
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();

		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glScissor(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		draw_demos_viewport(viewport_demos, current_trajectory, original_trajectories,
							demos);

		draw_GPR_viewport(viewport_GPR, GPR_points, GPR_models, demos);

		draw_timeline_viewport(window_size, viewport_x1, original_trajectories, demos,
							   GPR_points, GPR_sigma, 1);

		draw_timeline_viewport(window_size, viewport_x2, original_trajectories, demos,
							   GPR_points, GPR_sigma, 2);


		// Window: Demonstrations
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 2, 84));
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("Demonstrations", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Demonstrations       ");
		ImGui::SameLine();

		if (ImGui::Button("Clear")) {
			original_trajectories.clear();
			demos.clear();
			GPR_points = mat();
			GPR_sigma.clear();
			GPR_models.clear();
		}

		ImGui::SameLine();
		ImGui::Text("    ");
		ImGui::SameLine();

		if (ImGui::Button("Parameters"))
			gui_state.is_parameters_dialog_displayed = true;

		int previous_offset = gui_state.parameter_missing_data_offset;
		ImGui::SliderInt("Missing data offset", &gui_state.parameter_missing_data_offset,
						 std::max(parameters.nb_data / 8 - 1, 1),
						 std::min(parameters.nb_data - (gui_state.parameter_missing_data_length + 1), parameters.nb_data - 2));

		int previous_length = gui_state.parameter_missing_data_length;
		ImGui::SliderInt("Missing data length", &gui_state.parameter_missing_data_length,
						 1,
						 std::min(parameters.nb_data / 2, parameters.nb_data - (gui_state.parameter_missing_data_offset + 1)));

		if ((gui_state.parameter_missing_data_offset != previous_offset) ||
			(gui_state.parameter_missing_data_length != previous_length)) {
			gui_state.are_parameters_modified = true;
		}

		ImGui::End();


		// Window: GPR
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 2, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width - window_size.win_width / 2, 0));
		ImGui::Begin("GPR", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("GPR");

		ImGui::End();


		// Window: Timeline x1
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 2, 36));
		ImGui::SetNextWindowPos(ImVec2(0, window_size.win_height / 2));
		ImGui::Begin("Timeline: x1", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Timeline: x1");

		ImGui::End();


		// Window: Timeline x2
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 2, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width - window_size.win_width / 2,
									   window_size.win_height / 2));
		ImGui::Begin("Timeline: x2", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Timeline: x2");

		ImGui::End();


		// Window: Parameters
		ImGui::SetNextWindowSize(ImVec2(440, 170));
		ImGui::SetNextWindowPos(ImVec2((window_size.win_width - 440) / 2, (window_size.win_height - 170) / 2));
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 255));

		if (gui_state.is_parameters_dialog_displayed)
			ImGui::OpenPopup("Parameters");

		if (ImGui::BeginPopupModal("Parameters", NULL,
								   ImGuiWindowFlags_NoResize |
								   ImGuiWindowFlags_NoSavedSettings)) {

			ImGui::SliderInt("Nb data", &gui_state.parameter_nb_data, 10, 30);
			ImGui::SliderInt("Nb data for reproduction", &gui_state.parameter_nb_data_reproduction, 100, 300);
			ImGui::SliderFloat("RBF parameter 1", &gui_state.parameter_p[0], 1, 1000);
			ImGui::SliderFloat("RBF parameter 2", &gui_state.parameter_p[1], 1, 100);
			ImGui::SliderFloat("RBF parameter 3", &gui_state.parameter_p[2], .01, 10);

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

				if ((mouse_x <= window_size.win_width / 2) &&
					(mouse_y > 84) && (mouse_y <= window_size.win_height / 2))
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

					mat trajectory(2, current_trajectory.size());
					for (size_t i = 0; i < current_trajectory.size(); ++i) {
						trajectory(0, i) = current_trajectory[i](0);
						trajectory(1, i) = current_trajectory[i](1);
					}

					demos.push_back(sample_trajectory(trajectory, time_steps));

					original_trajectories.push_back(trajectory);

					gui_state.must_recompute_GPR = true;
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
