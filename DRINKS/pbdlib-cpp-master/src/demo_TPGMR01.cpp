/*
 * demo_TPGMR01.cpp
 *
 * TP-GMM with GMR (conditioning on the phase of the demonstration).
 *
 * If this code is useful for your research, please cite the related publication:
 * @article{Calinon16JIST,
 *   author="Calinon, S.",
 *   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
 *   journal="Intelligent Service Robotics",
 *   publisher="Springer Berlin Heidelberg",
 *   doi="10.1007/s11370-015-0187-9",
 *   year="2016",
 *   volume="9",
 *   number="1",
 *   pages="1--29"
 * }
 *
 * Authors: Sylvain Calinon, Philip Abbet, Andras Kupcsik
 */


#include <stdio.h>
#include <float.h>
#include <armadillo>
#include <chrono>

#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>

using namespace arma;


/***************************** ALGORITHM SECTION *****************************/

typedef std::vector<vec> vector_list_t;
typedef std::vector<mat> matrix_list_t;


//-----------------------------------------------------------------------------
// Contains all the parameters used by the algorithm. Some of them are
// modifiable through the UI, others are hard-coded.
//-----------------------------------------------------------------------------
struct parameters_t {
	int	   nb_states;		// Number of components in the GMM
	int	   nb_frames;		// Number of candidate frames of reference
	int	   nb_deriv;		// Number of static and dynamic features
	int	   nb_data;			// Number of datapoints in a trajectory <==== not used for product of Gaussians!!!
	float  dt;				// Time step duration					<==== not used for product of Gaussians!!!
};

//-----------------------------------------------------------------------------
// Model trained using the algorithm
//-----------------------------------------------------------------------------
struct model_t {
	parameters_t			   parameters;	// Parameters used to train the model

	// These lists contain one element per GMM state and per frame (access them
	// by doing: mu[state][frame])
	std::vector<vector_list_t> mu;
	std::vector<matrix_list_t> sigma;

	int						   nb_var;
	mat						   pix;
	vec						   priors;
};


//-----------------------------------------------------------------------------
// Represents a coordinate system, aka a reference frame
//-----------------------------------------------------------------------------
struct coordinate_system_t {

	coordinate_system_t(const arma::vec& position, const arma::mat& orientation,
						const parameters_t& parameters) {

		this->position = zeros(2 + (parameters.nb_deriv - 1) * 2);
		this->position(span(0, 1)) = position(span(0, 1));

		this->orientation = kron(eye(parameters.nb_deriv, parameters.nb_deriv),
								 orientation(span(0, 1), span(0, 1)));
	}

	vec position;
	mat orientation;
};


//-----------------------------------------------------------------------------
// Represents a list of coordinate systems
//-----------------------------------------------------------------------------
typedef std::vector<coordinate_system_t> coordinate_system_list_t;


//-----------------------------------------------------------------------------
// Contains all the needed informations about a demonstration, like:
//	 - its reference frames
//	 - the trajectory originally drawn by the user
//	 - the resampled trajectory
//	 - the trajectory expressed in each reference frame
//-----------------------------------------------------------------------------
class Demonstration
{
public:
	Demonstration(coordinate_system_list_t coordinate_systems,
				  const std::vector<arma::vec>& points,
				  const parameters_t& parameters)
	: coordinate_systems(coordinate_systems)
	{
		points_original = mat(3, points.size()); // added one for time (3rd dime)

		for (size_t i = 0; i < points.size(); ++i) {
			points_original(0, i) = points[i](0);
			points_original(1, i) = points[i](1);
			points_original(2, i) = points[i](2);
		}

		update(parameters);
	}


	//-------------------------------------------------------------------------
	// Resample the trajectory and recompute it in each reference frame
	// according to the provided parameters
	//-------------------------------------------------------------------------
	void update(const parameters_t& parameters)
	{
		// Resampling of the trajectory
		arma::vec x = points_original.row(0).t();
		arma::vec y = points_original.row(1).t();
		arma::vec x2(parameters.nb_data);
		arma::vec y2(parameters.nb_data);

		arma::vec from_indices = arma::linspace<arma::vec>(0, points_original.n_cols - 1, points_original.n_cols);
		arma::vec to_indices = arma::linspace<arma::vec>(0, points_original.n_cols - 1, parameters.nb_data);

		interp1(from_indices, x, to_indices, x2, "*linear");
		interp1(from_indices, y, to_indices, y2, "*linear");

		points = mat(2 * parameters.nb_deriv, parameters.nb_data);
		points(span(0), span::all) = x2.t();
		points(span(1), span::all) = y2.t();


		// Compute the derivatives
		mat D = (diagmat(ones(1, parameters.nb_data - 1), -1) -
				 eye(parameters.nb_data, parameters.nb_data)) / parameters.dt;

		D(parameters.nb_data - 1, parameters.nb_data - 1) = 0.0;

		points(span(2, 3), span::all) = points(span(0, 1), span::all) * pow(D, 1);

		// Compute the points in each coordinate system
		points_in_coordinate_systems.clear();


		points = join_vert(points, linspace(0, points.n_cols-1, points.n_cols).t());

		for (int m = 0; m < coordinate_systems.size(); ++m) {
			points_in_coordinate_systems.push_back(
					join_vert( pinv(coordinate_systems[m].orientation) *
					(points.rows(0, points.n_rows-2) - repmat(coordinate_systems[m].position, 1, parameters.nb_data)), points.row(points.n_rows-1)));
		}
	}


	//-------------------------------------------------------------------------
	// Returns the coordinates of a point in a specific reference frame of
	// the demonstration
	//-------------------------------------------------------------------------
	arma::vec convert_to_coordinate_system(const arma::vec& point, int frame) const {
		vec original_point = zeros(points.n_rows);
		original_point(span(0, 1)) = point(span(0, 1));

		vec result = pinv(coordinate_systems[frame].orientation) *
					 (original_point - coordinate_systems[frame].position);

		return result(span(0, 1));
	}


public:
	coordinate_system_list_t coordinate_systems;
	arma::mat				 points_original;
	arma::mat				 points;
	matrix_list_t			 points_in_coordinate_systems;
};


//-----------------------------------------------------------------------------
// Represents a list of demonstrations
//-----------------------------------------------------------------------------
typedef std::vector<Demonstration> demonstration_list_t;


//-----------------------------------------------------------------------------
// Computes the factorial of a number
//-----------------------------------------------------------------------------
int factorial(int n) {
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}


//-----------------------------------------------------------------------------
// Return the likelihood of datapoint(s) to be generated by a Gaussian
// parameterized by center and covariance
//-----------------------------------------------------------------------------
arma::vec gaussPDF(const mat& data, colvec mu, mat sigma) {

	int nb_var = data.n_rows;
	int nb_data = data.n_cols;

	mat data2 = data.t() - repmat(mu.t(), nb_data, 1);

	vec prob = sum((data2 * inv(sigma)) % data2, 1);

	prob = exp(-0.5 * prob) / sqrt(pow((2 * datum::pi), nb_var) * det(sigma) + DBL_MIN);

	return prob;
}


//-----------------------------------------------------------------------------
// Initialization of Gaussian Mixture Model (GMM) parameters by clustering an
// ordered dataset into equal bins
//-----------------------------------------------------------------------------
void init_tensorGMM_kbins(const demonstration_list_t& demos,
						  model_t &model) {

	model.priors.resize(model.parameters.nb_states);
	model.mu.clear();
	model.sigma.clear();

	model.nb_var = demos[0].points_in_coordinate_systems[0].n_rows;

	// Initialize bins
	uvec t_sep = linspace<uvec>(0, model.parameters.nb_data - 1,
								model.parameters.nb_states + 1);

	struct bin_t {
		mat data;
		vec mu;
		mat sigma;
	};

	std::vector<bin_t> bins;
	for (int i = 0; i < model.parameters.nb_states; ++i) {
		bin_t bin;
		bin.data = zeros(model.nb_var * model.parameters.nb_frames + 1, //adding time as last dimension
						 demos.size() * (t_sep(i + 1) - t_sep(i) + 1));

		bins.push_back(bin);
	}


	// Split each demonstration in K equal bins
	for (int n = 0; n < demos.size(); ++n) {

		for (int i = 0; i < model.parameters.nb_states; ++i) {
			int bin_size = t_sep(i + 1) - t_sep(i) + 1;

			for (int m = 0; m < model.parameters.nb_frames; ++m) {
				bins[i].data(span(m * model.nb_var, (m + 1) * model.nb_var - 1),
							 span(n * bin_size, (n + 1) * bin_size - 1)) =
						demos[n].points_in_coordinate_systems[m](span::all, span(t_sep(i), t_sep(i + 1)));
			}
		}
	}


	// Calculate statistics on bin data
	for (int i = 0; i < model.parameters.nb_states; ++i) {
		bins[i].mu = mean(bins[i].data, 1);
		bins[i].sigma = cov(bins[i].data.t());
		model.priors(i) = bins[i].data.n_elem;
	}


	// Reshape GMM into a tensor
	for (int i = 0; i < model.parameters.nb_states; ++i) {
		model.mu.push_back(vector_list_t());
		model.sigma.push_back(matrix_list_t());
	}

	for (int m = 0; m < model.parameters.nb_frames; ++m) {
		for (int i = 0; i < model.parameters.nb_states; ++i) {
			model.mu[i].push_back(bins[i].mu(span(m * model.nb_var, (m + 1) * model.nb_var - 1)));

			model.sigma[i].push_back(bins[i].sigma(span(m * model.nb_var, (m + 1) * model.nb_var - 1),
												   span(m * model.nb_var, (m + 1) * model.nb_var - 1)));

		}

	}

	model.priors /= sum(model.priors);
}


//-----------------------------------------------------------------------------
// Training of a task-parameterized Gaussian mixture model (GMM) with an
// expectation-maximization (EM) algorithm.
//
// The approach allows the modulation of the centers and covariance matrices of
// the Gaussians with respect to external parameters represented in the form of
// candidate coordinate systems.
//-----------------------------------------------------------------------------
void train_EM_tensorGMM(const demonstration_list_t& demos,
						model_t &model) {

	const int nb_max_steps = 100;			// Maximum number of iterations allowed
	const int nb_min_steps = 5;				// Minimum number of iterations allowed
	const double max_diff_log_likelihood = 1e-5;	// Likelihood increase threshold
	// to stop the algorithm
	const double diag_reg_fact = 1e-2;		// Regularization term is optional


	cube data(model.nb_var, model.parameters.nb_frames,
			  demos.size() * model.parameters.nb_data);


	for (int n = 0; n < demos.size(); ++n) {
		for (int m = 0; m < model.parameters.nb_frames; ++m) {
			data(span::all, span(m), span(n * model.parameters.nb_data,
										  (n + 1) * model.parameters.nb_data - 1)) =
					demos[n].points_in_coordinate_systems[m];
		}
	}


	std::vector<double> log_likelihoods;

	for (int iter = 0; iter < nb_max_steps; ++iter) {

		// E-step
		mat L = ones(model.parameters.nb_states, data.n_slices);

		for (int i = 0; i < model.parameters.nb_states; ++i) {
			for (int m = 0; m < model.parameters.nb_frames; ++m) {

				// Matricization/flattening of tensor
				mat data_mat(data(span::all, span(m), span::all));

				vec gamma0 = gaussPDF(data_mat, model.mu[i][m], model.sigma[i][m]);

				L(i, span::all) = L(i, span::all) % gamma0.t();
			}

			L(i, span::all) =  L(i, span::all) * model.priors(i);
		}

		mat gamma = L / repmat(sum(L, 0) + DBL_MIN, L.n_rows, 1);
		mat gamma2 = gamma / repmat(sum(gamma, 1), 1, data.n_slices);

		model.pix = gamma2;


		// M-step
		for (int i = 0; i < model.parameters.nb_states; ++i) {

			// Update priors
			model.priors(i) = sum(gamma(i, span::all)) / data.n_slices;

			for (int m = 0; m < model.parameters.nb_frames; ++m) {

				// Matricization/flattening of tensor
				mat data_mat(data(span::all, span(m), span::all));

				// Update mu
				model.mu[i][m] = data_mat * gamma2(i, span::all).t();

				// Update sigma
				mat data_tmp = data_mat - repmat(model.mu[i][m], 1, data.n_slices);
				model.sigma[i][m] = data_tmp * diagmat(gamma2(i, span::all)) * data_tmp.t() +
									eye(data_tmp.n_rows, data_tmp.n_rows) * diag_reg_fact;
			}
		}

		// Compute average log-likelihood
		log_likelihoods.push_back(vec(sum(log(sum(L, 0)), 1))[0] / data.n_slices);

		// Stop the algorithm if EM converged (small change of log-likelihood)
		if (iter >= nb_min_steps) {
			if (log_likelihoods[iter] - log_likelihoods[iter - 1] < max_diff_log_likelihood)
				break;
		}
	}
}


//-----------------------------------------------------------------------------
// Training of the model
//-----------------------------------------------------------------------------
void learn(const demonstration_list_t& demos, model_t &model) {

	init_tensorGMM_kbins(demos, model);
	train_EM_tensorGMM(demos, model);

}


/*************************** DEMONSTRATION SECTION ***************************/

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

	// View matrix parameters
	arma::fvec view;
};


//-----------------------------------------------------------------------------
// Helper function to setup a viewport
//-----------------------------------------------------------------------------
void setup_viewport(viewport_t* viewport, int x, int y, int width, int height,
					double near_distance = -1.0, double far_distance = 1.0,
					const fvec& view_transforms = zeros<fvec>(3)) {

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
	viewport->view = view_transforms;
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
	result(0) = result(0) + (float) viewport.width * 0.5f;
	result(1) = result(1) + (float) viewport.height * 0.5f;

	// viewport -> ui
	result(0) = (result(0) + viewport.x) * (float) window_size.win_width /
				(float) window_size.fb_width;

	result(1) = window_size.win_height - (result(1) + viewport.y) *
										 (float) window_size.win_height / (float) window_size.fb_height;

	return result;
}


//-----------------------------------------------------------------------------
// Colors used by the movable handlers
//-----------------------------------------------------------------------------
const arma::fmat HANDLER_COLORS({
	{ 0.0, 0.0,  0.0  },
	{ 0.0, 0.0,  1.0  },
	{ 0.0, 0.5,  0.0  },
	{ 1.0, 0.0,  0.0  },
	{ 0.0, 0.75, 0.75 },
});


//-----------------------------------------------------------------------------
// Colors used by the fixed handlers
//-----------------------------------------------------------------------------
const arma::fmat HANDLER_FIXED_COLORS({
	{ 0.4, 0.4,  0.4  },
	{ 0.4, 0.4,  1.0  },
	{ 0.2, 0.5,  0.2  },
	{ 1.0, 0.4,  0.4  },
	{ 0.3, 0.75, 0.75 },
});


//-----------------------------------------------------------------------------
// Colors of the displayed lines and gaussians
//-----------------------------------------------------------------------------
const arma::mat COLORS({
	{ 0.0,  0.0,  1.0  },
	{ 0.0,  0.5,  0.0  },
	{ 1.0,  0.0,  0.0  },
	{ 0.0,  0.75, 0.75 },
	{ 0.75, 0.0,  0.75 },
	{ 0.75, 0.75, 0.0  },
	{ 0.25, 0.25, 0.25 },
});


//-----------------------------------------------------------------------------
// An user-movable UI widget representing a coordinate system
//
// Can be "fixed", so the user can't move it anymore
//-----------------------------------------------------------------------------
class Handler
{
public:
	Handler(const viewport_t* viewport, const ImVec2& position, const ImVec2& y,
			int index, bool is_small)
	: viewport(viewport), hovered(false), fixed(false), moved(false), index(index)
	{
		ui_position = position;
		ui_y = y;

		fvec color = HANDLER_COLORS.row(index).t();

		if (is_small) {
			models[0] = gfx2::create_rectangle(color, 12.0f, 2.0f);

			models[1] = gfx2::create_rectangle(color, 30.0f, 2.0f);
			models[1].transforms.position(0) = 15.0f;
			models[1].transforms.position(1) = -5.0f;

			models[2] = gfx2::create_rectangle(color, 30.0f, 2.0f);
			models[2].transforms.position(0) = 15.0f;
			models[2].transforms.position(1) = 5.0f;
		} else {
			models[0] = gfx2::create_rectangle(color, 25.0f, 5.0f);

			models[1] = gfx2::create_rectangle(color, 60.0f, 5.0f);
			models[1].transforms.position(0) = 30.0f;
			models[1].transforms.position(1) = -10.0f;

			models[2] = gfx2::create_rectangle(color, 60.0f, 5.0f);
			models[2].transforms.position(0) = 30.0f;
			models[2].transforms.position(1) = 10.0f;
		}

		models[0].transforms.parent = &transforms;
		models[1].transforms.parent = &transforms;
		models[2].transforms.parent = &transforms;

		models[0].transforms.rotation = gfx2::rotate(arma::fvec({0.0f, 0.0f, 1.0f}),
													 gfx2::deg2rad(90.0f));
	}


	void update(const gfx2::window_size_t& window_size)
	{
		transforms.position.rows(0, 1) = arma::conv_to<arma::fvec>::from(
				ui2fb(ui_position, window_size, *viewport)
		);

		arma::vec delta = ui_y / arma::norm(arma::vec(ui_y));

		transforms.rotation(0, 0) = -delta(0);
		transforms.rotation(1, 0) = delta(1);
		transforms.rotation(0, 1) = -delta(1);
		transforms.rotation(1, 1) = -delta(0);
	}


	void update()
	{
		transforms.position(0) = ui_position.x;
		transforms.position(1) = ui_position.y;

		arma::vec delta = ui_y / arma::norm(arma::vec(ui_y));

		transforms.rotation(0, 0) = -delta(0);
		transforms.rotation(1, 0) = delta(1);
		transforms.rotation(0, 1) = -delta(1);
		transforms.rotation(1, 1) = -delta(0);
	}


	void viewport_resized(const gfx2::window_size_t& window_size)
	{
		ui_position = fb2ui(arma::conv_to<arma::vec>::from(transforms.position.rows(0, 1)),
							window_size, *viewport);
	}


	void fix()
	{
		fixed = true;

		for (int i = 0; i < 3; ++i)
			models[i].diffuse_color = HANDLER_FIXED_COLORS.row(index).t();
	}


	bool draw(const gfx2::window_size_t& window_size)
	{
		if (!fixed)
		{
			std::stringstream id;
			id << "handler" << (uint64_t) this;

			ImGuiWindow* window = ImGui::GetCurrentWindow();
			if (window->SkipItems)
				return false;

			ImGui::PushID(id.str().c_str());

			// Position
			ImVec2 current_position = ui_position;

			ui_position = ui::dragger(0, ui_position, false, ui::config.draggerSize );

			moved = moved || (ui_position.x != current_position.x) || (ui_position.y != current_position.y);

			hovered = ImGui::IsItemHovered();

			// y-axis
			ImVec2 py = ui_position + ui_y;
			current_position = ui_y;

			py = ui::dragger(1, py, false, ui::config.draggerSize * 0.7);
			ui_y = py - ui_position;

			moved = moved || ((ui_y.x != current_position.x) || (ui_y.y != current_position.y));

			hovered = ImGui::IsItemHovered() || hovered;

			window->DrawList->AddLine(ui_position, py, ui::config.lineColor);

			ImGui::PopID();

			update(window_size);
		}

		return hovered;
	}


	void draw_anchor() const
	{
		for (int i = 0; i < 3; ++i)
			gfx2::draw(models[i]);
	}


public:
	const viewport_t* viewport;
	int index;

	ImVec2 ui_position;
	ImVec2 ui_y;

	gfx2::transforms_t transforms;
	gfx2::model_t models[3];

	bool hovered;
	bool fixed;
	bool moved;
};


//-----------------------------------------------------------------------------
// Represents a list of handlers
//-----------------------------------------------------------------------------
typedef std::vector<Handler*> handler_list_t;


//-----------------------------------------------------------------------------
// Contains all the needed infos about the state of the application (values of
// the parameters modifiable via the UI, which action the user is currently
// doing, ...)
//-----------------------------------------------------------------------------
struct gui_state_t {
	// Indicates if the user can draw a new demonstration
	bool can_draw_demonstration;

	// Indicates if the user is currently drawing a new demonstration
	bool is_drawing_demonstration;

	// Indicates if the parameters dialog is displayed
	bool is_parameters_dialog_displayed;

	// Indicates if the parameters were modified through the UI
	bool are_parameters_modified;

	// Parameters modifiable via the UI (they correspond to the ones declared
	//in parameters_t)
	int parameter_nb_states;
	int parameter_nb_frames;
	int parameter_nb_data;

	int parameter_nb_gmr_components;
};


//-----------------------------------------------------------------------------
// Create a handler at a random position (within the given boundaries)
//-----------------------------------------------------------------------------
Handler* create_random_handler(const viewport_t* viewport, int index, int min_x,
							   int min_y, int max_x, int max_y, bool is_small) {
	return new Handler(viewport,
					   ImVec2((randu() * 0.8f + 0.1f) * (max_x - min_x) + min_x,
							  (randu() * 0.5f + 0.1f) * (max_y - min_y) + min_y),
					   ImVec2((randu() - 0.5) * 10, randu() * -10 - 10),
					   index,
					   is_small
	);
}


//-----------------------------------------------------------------------------
// Create handlers to be used for a new demonstration at random positions
//-----------------------------------------------------------------------------
void create_new_demonstration_handlers(const viewport_t& viewport,
									   const gfx2::window_size_t& window_size,
									   int nb_frames,
									   handler_list_t &handlers) {

	for (size_t i = 0; i < handlers.size(); ++i)
		delete handlers[i];

	handlers.clear();

	handlers.push_back(
		new Handler(&viewport, ImVec2(window_size.win_width / 6,
									  window_size.win_height / 2 - 50),
					ImVec2(0, 30), 0, (window_size.fb_width == window_size.win_width))
	);

	for (int n = 1; n < nb_frames; ++n) {
		handlers.push_back(
			create_random_handler(&viewport, n, 10, 20,
								  window_size.win_width / 3 - 10,
								  window_size.win_height / 2 - 20,
								  (window_size.fb_width == window_size.win_width))
		);
	};
}


//-----------------------------------------------------------------------------
// Create handlers to be used for a new reproduction at random positions
//-----------------------------------------------------------------------------
void create_reproduction_handlers(const viewport_t& viewport,
								  const gfx2::window_size_t& window_size,
								  int nb_frames,
								  handler_list_t &handlers) {

	for (size_t i = 0; i < handlers.size(); ++i)
		delete handlers[i];

	handlers.clear();

	handlers.push_back(
			new Handler(&viewport, ImVec2(window_size.win_width / 2,
										  window_size.win_height - 50),
						ImVec2(0, 30), 0, (window_size.fb_width == window_size.win_width))
	);

	for (int n = 1; n < nb_frames; ++n) {
		handlers.push_back(
				create_random_handler(&viewport, n,
									  window_size.win_width / 3 + 20,
									  20,
									  window_size.win_width * 2 / 3 - 20,
									  window_size.win_height / 2- 20,
									  (window_size.fb_width == window_size.win_width))
		);
	};
}

//-----------------------------------------------------------------------------
// Create handlers to be used for moving GMR window
//-----------------------------------------------------------------------------
void create_reproduction_handlers_gmrmoving(const viewport_t& viewport,
								  const gfx2::window_size_t& window_size,
								  int nb_frames,
								  handler_list_t &handlers) {

	for (size_t i = 0; i < handlers.size(); ++i)
		delete handlers[i];

	handlers.clear();

	handlers.push_back(
			new Handler(&viewport, ImVec2(window_size.win_width *5 / 6,
										  window_size.win_height - 50),
						ImVec2(0, 30), 0, (window_size.fb_width == window_size.win_width))
	);

	if (nb_frames > 1) {
		for (int n = 1; n < nb_frames; ++n) {
			handlers.push_back(
					create_random_handler(&viewport, n,
										  window_size.win_width * 2 / 3 + 20,
										  20,
										  window_size.win_width - 20,
										  window_size.win_height / 2 - 20,
										  (window_size.fb_width == window_size.win_width))
			);
		};
	};

	for (int i = 0; i < handlers.size(); i++) {
		handlers.at(i)->fix();
		handlers.at(i)->update(window_size);
	}


}


//-----------------------------------------------------------------------------
// Extract a list of coordinate systems from a list of handlers
//-----------------------------------------------------------------------------
void convert(const handler_list_t& from, coordinate_system_list_t &to,
			 const parameters_t& parameters) {

	to.clear();

	for (int n = 0; n < from.size(); ++n) {
		to.push_back(
				coordinate_system_t(arma::conv_to<arma::vec>::from(from[n]->transforms.position),
									arma::conv_to<arma::mat>::from(from[n]->transforms.rotation),
									parameters)
		);
	}
}


//-----------------------------------------------------------------------------
// Extract a list of coordinate systems from a list of demonstrations
//-----------------------------------------------------------------------------
void convert(const demonstration_list_t& from, std::vector<coordinate_system_list_t> &to) {

	to.clear();

	for (int n = 0; n < from.size(); ++n) {
		to.push_back(from[n].coordinate_systems);
	}
}


//-----------------------------------------------------------------------------
// Compute a view matrix centered on the given reference frame across all the
// demonstrations
//-----------------------------------------------------------------------------
arma::fvec compute_centered_view_matrix(const demonstration_list_t& demonstrations,
										int frame) {
	vec top_left({-30, 0});
	vec bottom_right({0, 0});

	for (auto iter = demonstrations.begin(); iter != demonstrations.end(); ++iter) {
		top_left(0) = fmin(top_left(0), iter->points_in_coordinate_systems[frame](0, span::all).min());
		top_left(1) = fmax(top_left(1), iter->points_in_coordinate_systems[frame](1, span::all).max());

		bottom_right(0) = fmax(bottom_right(0), iter->points_in_coordinate_systems[frame](0, span::all).max());
		bottom_right(1) = fmin(bottom_right(1), iter->points_in_coordinate_systems[frame](1, span::all).min());
	}

	vec center = (bottom_right - top_left) / 2 + top_left;

	return fvec({ (float) -center(0), (float) -center(1), 0.0f });
}


//-----------------------------------------------------------------------------
// Render the "demonstrations" viewport
//-----------------------------------------------------------------------------
void draw_demos_viewport(const viewport_t& viewport,
						 const std::vector<arma::vec>& current_trajectory,
						 const demonstration_list_t& demonstrations,
						 const std::vector<handler_list_t> fixed_demonstration_handlers,
						 const handler_list_t current_demonstration_handlers) {

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
	glTranslatef(viewport.view(0), viewport.view(1), viewport.view(2));

	// Draw the currently created demonstration (if any)
	if (current_trajectory.size() > 1) {
		std::vector<arma::vec> tmpvec;
		for (int i = 0; i < current_trajectory.size(); i++) {
			tmpvec.push_back(current_trajectory.at(i).subvec(0, 1));
		}
		gfx2::draw_line(arma::fvec({0.33f, 0.97f, 0.33f}), tmpvec);
		tmpvec.clear();
	}

	// Draw the demonstrations
	int color_index = 0;
	for (auto iter = demonstrations.begin(); iter != demonstrations.end(); ++iter) {
		arma::mat datapoints = iter->points(span(0, 1), span::all);

		arma::fvec color = arma::conv_to<arma::fvec>::from(COLORS.row(color_index));

		gfx2::draw_line(color, datapoints);

		++color_index;
		if (color_index >= COLORS.n_rows)
			color_index = 0;
	}

	// Draw the handlers
	for (size_t n = 0; n < fixed_demonstration_handlers.size(); ++n) {
		for (size_t i = 0; i < fixed_demonstration_handlers[n].size(); ++i)
			fixed_demonstration_handlers[n][i]->draw_anchor();
	}

	for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
		current_demonstration_handlers[i]->draw_anchor();
}


//-----------------------------------------------------------------------------
// Render a "reproduction" viewport
//-----------------------------------------------------------------------------
void draw_reproductions_viewport(const viewport_t& viewport,
								 const std::vector<handler_list_t>& handlers,
								 const matrix_list_t& reproductions) {

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
	glTranslatef(viewport.view(0), viewport.view(1), viewport.view(2));

	// Draw the handlers
	for (size_t n = 0; n < handlers.size(); ++n) {
		for (size_t i = 0; i < handlers[n].size(); ++i)
			handlers[n][i]->draw_anchor();
	}
}


//-----------------------------------------------------------------------------
// Render a "reproduction" viewport
//-----------------------------------------------------------------------------
void draw_reproductions_viewport(const viewport_t& viewport,
								 const handler_list_t& handlers,
								 const matrix_list_t& reproductions) {

	std::vector<handler_list_t> handler_list;
	handler_list.push_back(handlers);

	draw_reproductions_viewport(viewport, handler_list, reproductions);
}


//-----------------------------------------------------------------------------
// Render a "GMMs" viewport
//-----------------------------------------------------------------------------
void draw_gmrmoving_viewport(const viewport_t& viewport,
						const handler_list_t& handlers, model_t model, int nb_gmr_components, int drawIndex) {

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
	glTranslatef(viewport.view(0), viewport.view(1), viewport.view(2));


	std::vector<handler_list_t> handler_list;
	handler_list.push_back(handlers);

	// Draw the handlers
	for (size_t n = 0; n < handler_list.size(); ++n) {
		for (size_t i = 0; i < handler_list[n].size(); ++i)
			handler_list[n][i]->draw_anchor();
	}

	mat grey = {0.9, 0.9, 0.9};
	mat black = {0.0, 0.0, 0.0};

	if (model.mu.size() > 0) {
		mat inputs = linspace(0, 199, nb_gmr_components);
		inputs = inputs.t();
		mat H(model.parameters.nb_states, inputs.size());

		cube muGMR(2 * model.parameters.nb_deriv, inputs.size(), model.parameters.nb_frames);
		muGMR = zeros(2 * model.parameters.nb_deriv, inputs.size(), model.parameters.nb_frames);
		field<cube> sigmaGMR(model.parameters.nb_frames);

		for (int i = 0; i < model.parameters.nb_frames; i++) {
			sigmaGMR(i).set_size(2 * model.parameters.nb_deriv, 2 * model.parameters.nb_deriv, inputs.size());
			sigmaGMR(i) = zeros(2 * model.parameters.nb_deriv, 2 * model.parameters.nb_deriv, inputs.size());
		}

		for (int m = 0; m < model.parameters.nb_frames; m++) {
			for (int i = 0; i < model.parameters.nb_states; i++) {
				H.row(i) = model.priors(i) * gaussPDF(inputs, model.mu[i][m].row(model.nb_var - 1),
													  model.sigma[i][m].row(model.nb_var - 1).col(model.nb_var - 1)).t();
			}
			H = H / repmat(sum(H + 1e-300), model.parameters.nb_states, 1);

			mat muTmp(2 * model.parameters.nb_deriv, model.parameters.nb_states);
			mat sigmaTmp;

			for (int t = 0; t < inputs.size(); t++) {
				// Compute conditional means
				for (int i = 0; i < model.parameters.nb_states; i++) {
					muTmp.col(i) = model.mu[i][m].subvec(0, 2 * model.parameters.nb_deriv - 1) +
								   model.sigma[i][m].col(2 * model.parameters.nb_deriv).rows(0, 2 *
																								model.parameters.nb_deriv -
																								1) *
								   inv(model.sigma[i][m].row(model.nb_var - 1).col(model.nb_var - 1)) *
								   (inputs(t) - model.mu[i][m].row(model.nb_var - 1));
					muGMR.slice(m).col(t) += H(i, t) * muTmp.col(i);
				}

				// Compute conditional covariances
				for (int i = 0; i < model.parameters.nb_states; i++) {
					sigmaTmp = model.sigma[i][m].submat(0, 0, model.nb_var - 2, model.nb_var - 2) -
							   model.sigma[i][m].col(2 * model.parameters.nb_deriv).rows(0,
																						 2 * model.parameters.nb_deriv -
																						 1) *
							   inv(model.sigma[i][m].row(model.nb_var - 1).col(model.nb_var - 1)) *
							   model.sigma[i][m].row(2 * model.parameters.nb_deriv).cols(0,
																						 2 * model.parameters.nb_deriv -
																						 1);
					sigmaGMR(m).slice(t) += H(i, t) * (sigmaTmp + muTmp.col(i) * muTmp.col(i).t());
				}

				sigmaGMR(m).slice(t) += -muGMR.slice(m).col(t) * muGMR.slice(m).col(t).t() +
										eye(2 * model.parameters.nb_deriv, 2 * model.parameters.nb_deriv) * 1e-4;
			}
		}

		// transform mu/sigma GMR components into coordinate systems
		cube muGMRt = zeros(2, inputs.size(), model.parameters.nb_frames);
		field<cube> sigmaGMRt(model.parameters.nb_frames);
		for (int i = 0; i < model.parameters.nb_frames; i++) {
			sigmaGMRt(i).resize(2, 2, inputs.size());
			sigmaGMRt(i) = zeros(2, 2, inputs.size());
		}

		for (int f = 0; f < model.parameters.nb_frames; f++) {
			muGMRt.slice(f) = handler_list[0][f]->transforms.rotation.submat(0, 0, 1, 1) * muGMR.slice(f).rows(0, 1);
			vec b = {handler_list[0][f]->transforms.position(0), handler_list[0][f]->transforms.position(1)};
			muGMRt.slice(f).each_col() += b;
			for (int t = 0; t < inputs.size(); t++ ) {
				sigmaGMRt(f).slice(t) = handler_list[0][f]->transforms.rotation.submat(0, 0, 1, 1) *
													sigmaGMR(f).slice(t).submat(0, 0, 1, 1) *
							 handler_list[0][f]->transforms.rotation.submat(0, 0, 1, 1).t();
			}
		}

		// product
		vec maxMuP = zeros(2, 1);
		mat maxSigmaP = eye(2, 2);
		for (int t = 0; t < inputs.size(); t++) {
			vec muP = zeros(2, 1);
			mat sigmaP = zeros(2, 2);

			for (int m = 0; m < model.parameters.nb_frames; m++) {

				sigmaP += inv(sigmaGMRt(m).slice(t) + eye(2, 2) * 1e-4);
				muP += inv(sigmaGMRt(m).slice(t) + eye(2, 2) * 1e-4) * muGMRt.slice(m).col(t);
			}

			sigmaP = inv(sigmaP);
			muP = sigmaP * muP;


			if (t == drawIndex) {
			   maxMuP = muP;
			   maxSigmaP = sigmaP;
			}


			glClear(GL_DEPTH_BUFFER_BIT);
			gfx2::draw_gaussian(conv_to<fvec>::from(grey.row(0).t()), muP, sigmaP);
		}

		glClear(GL_DEPTH_BUFFER_BIT);
		gfx2::draw_gaussian(conv_to<fvec>::from(black.row(0).t()), maxMuP, maxSigmaP);

	}
}

//-----------------------------------------------------------------------------
// Render a "Product" viewport
//-----------------------------------------------------------------------------
void draw_gmr_viewport(const viewport_t& viewport,
						   const handler_list_t& handlers, model_t model, int nb_gmr_components, vec mouse, int &drawIndex) {

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
	glTranslatef(viewport.view(0), viewport.view(1), viewport.view(2));


	std::vector<handler_list_t> handler_list;
	handler_list.push_back(handlers);

	// Draw the handlers
	for (size_t n = 0; n < handler_list.size(); ++n) {
		for (size_t i = 0; i < handler_list[n].size(); ++i)
			handler_list[n][i]->draw_anchor();
	}

	mat grey = {0.9, 0.9, 0.9};
	mat black = {0.0, 0.0, 0.0};

	// Draw the Product GMM states

	// GMR
	//int numGauss = 20; // later load it from GUI
	if (model.mu.size() > 0) {
		mat inputs = linspace(0, 199, nb_gmr_components);
		inputs = inputs.t();
		mat H(model.parameters.nb_states, inputs.size());

		cube muGMR(2 * model.parameters.nb_deriv, inputs.size(), model.parameters.nb_frames);
		muGMR = zeros(2 * model.parameters.nb_deriv, inputs.size(), model.parameters.nb_frames);
		field<cube> sigmaGMR(model.parameters.nb_frames);

		for (int i = 0; i < model.parameters.nb_frames; i++) {
			sigmaGMR(i).set_size(2 * model.parameters.nb_deriv, 2 * model.parameters.nb_deriv, inputs.size());
			sigmaGMR(i) = zeros(2 * model.parameters.nb_deriv, 2 * model.parameters.nb_deriv, inputs.size());
		}

		for (int m = 0; m < model.parameters.nb_frames; m++) {
			for (int i = 0; i < model.parameters.nb_states; i++) {
				H.row(i) = model.priors(i) * gaussPDF(inputs, model.mu[i][m].row(model.nb_var - 1),
													  model.sigma[i][m].row(model.nb_var - 1).col(model.nb_var - 1)).t();
			}
			H = H / repmat(sum(H + 1e-300), model.parameters.nb_states, 1);

			mat muTmp(2 * model.parameters.nb_deriv, model.parameters.nb_states);
			mat sigmaTmp;

			for (int t = 0; t < inputs.size(); t++) {
				// Compute conditional means
				for (int i = 0; i < model.parameters.nb_states; i++) {
					muTmp.col(i) = model.mu[i][m].subvec(0, 2 * model.parameters.nb_deriv - 1) +
								   model.sigma[i][m].col(2 * model.parameters.nb_deriv).rows(0, 2 *
																								model.parameters.nb_deriv -
																								1) *
								   inv(model.sigma[i][m].row(model.nb_var - 1).col(model.nb_var - 1)) *
								   (inputs(t) - model.mu[i][m].row(model.nb_var - 1));
					muGMR.slice(m).col(t) += H(i, t) * muTmp.col(i);
				}

				// Compute conditional covariances
				for (int i = 0; i < model.parameters.nb_states; i++) {
					sigmaTmp = model.sigma[i][m].submat(0, 0, model.nb_var - 2, model.nb_var - 2) -
							   model.sigma[i][m].col(2 * model.parameters.nb_deriv).rows(0,
																						 2 * model.parameters.nb_deriv -
																						 1) *
							   inv(model.sigma[i][m].row(model.nb_var - 1).col(model.nb_var - 1)) *
							   model.sigma[i][m].row(2 * model.parameters.nb_deriv).cols(0,
																						 2 * model.parameters.nb_deriv -
																						 1);
					sigmaGMR(m).slice(t) += H(i, t) * (sigmaTmp + muTmp.col(i) * muTmp.col(i).t());
				}

				sigmaGMR(m).slice(t) += -muGMR.slice(m).col(t) * muGMR.slice(m).col(t).t() +
										eye(2 * model.parameters.nb_deriv, 2 * model.parameters.nb_deriv) * 1e-4;
			}
		}

		// transform mu/sigma GMR components into coordinate systems
		cube muGMRt = zeros(2, inputs.size(), model.parameters.nb_frames);
		field<cube> sigmaGMRt(model.parameters.nb_frames);
		for (int i = 0; i < model.parameters.nb_frames; i++) {
			sigmaGMRt(i).resize(2, 2, inputs.size());
			sigmaGMRt(i) = zeros(2, 2, inputs.size());
		}

		for (int f = 0; f < model.parameters.nb_frames; f++) {
			muGMRt.slice(f) = handler_list[0][f]->transforms.rotation.submat(0, 0, 1, 1) * muGMR.slice(f).rows(0, 1);
			vec b = {handler_list[0][f]->transforms.position(0), handler_list[0][f]->transforms.position(1)};
			muGMRt.slice(f).each_col() += b;
			for (int t = 0; t < inputs.size(); t++ ) {
				sigmaGMRt(f).slice(t) = handler_list[0][f]->transforms.rotation.submat(0, 0, 1, 1) *
													sigmaGMR(f).slice(t).submat(0, 0, 1, 1) *
							 handler_list[0][f]->transforms.rotation.submat(0, 0, 1, 1).t();
			}
		}

		// product
		double maxLL = 0.0;
		vec LLs = zeros(inputs.size(), 1);
		vec maxMuP = zeros(2, 1);
		mat maxSigmaP = eye(2, 2);
		for (int t = 0; t < inputs.size(); t++) {
			vec muP = zeros(2, 1);
			mat sigmaP = zeros(2, 2);

			for (int m = 0; m < model.parameters.nb_frames; m++) {

				sigmaP += inv(sigmaGMRt(m).slice(t) + eye(2, 2) * 1e-4);
				muP += inv(sigmaGMRt(m).slice(t) + eye(2, 2) * 1e-4) * muGMRt.slice(m).col(t);
			}

			sigmaP = inv(sigmaP);
			muP = sigmaP * muP;

			vec currLL = gaussPDF(mouse, muP, sigmaP);
			LLs.at(t) = currLL(0);

			if (LLs.at(t) >= maxLL) {
			   maxMuP = muP;
			   maxSigmaP = sigmaP;
			   maxLL =LLs.at(t);
			   drawIndex = t;
			}

			glClear(GL_DEPTH_BUFFER_BIT);
			gfx2::draw_gaussian(conv_to<fvec>::from(grey.row(0).t()), muP, sigmaP);
		}

		glClear(GL_DEPTH_BUFFER_BIT);
		gfx2::draw_gaussian(conv_to<fvec>::from(black.row(0).t()), maxMuP, maxSigmaP);
	}
}


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	// Model
	model_t model;

	// Parameters
	model.parameters.nb_states	= 4;
	model.parameters.nb_frames	= 2;
	model.parameters.nb_deriv	= 2;
	model.parameters.nb_data	= 200;
	model.parameters.dt			= 0.1f;

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
		"Demo - TPGMR",
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
	viewport_t viewport_gmr;
	viewport_t viewport_gmrmoving;

	// GUI state
	gui_state_t gui_state;
	gui_state.can_draw_demonstration = false;
	gui_state.is_drawing_demonstration = false;
	gui_state.is_parameters_dialog_displayed = false;
	gui_state.are_parameters_modified = false;
	gui_state.parameter_nb_data = model.parameters.nb_data;
	gui_state.parameter_nb_states = model.parameters.nb_states;
	gui_state.parameter_nb_frames = model.parameters.nb_frames;
	gui_state.parameter_nb_gmr_components = 20;
	std::chrono::microseconds ms2 = std::chrono::duration_cast< std::chrono::milliseconds >( std::chrono::system_clock::now().time_since_epoch());
	long startTimeMsec = ms2.count();
	ImVec2 initFramePos;
	int drawIndex = 0;
	bool reset_moving_handler = true;


	// Initial handlers
	std::vector<handler_list_t> fixed_demonstration_handlers;
	handler_list_t current_demonstration_handlers;
	handler_list_t reproduction_handlers;
	handler_list_t reproduction_handlers_moving;


	// List of demonstrations and reproductions
	demonstration_list_t demos;

	// Main loop
	std::vector<arma::vec> current_trajectory;

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
			setup_viewport(&viewport_demos, 0, window_size.fb_height - viewport_height,
						   viewport_width, viewport_height);

			setup_viewport(&viewport_gmr, viewport_width + 2,
						   window_size.fb_height - viewport_height,
						   viewport_width, viewport_height);

			setup_viewport(&viewport_gmrmoving, window_size.fb_width - viewport_width,
						   window_size.fb_height - viewport_height,
						   viewport_width, viewport_height);


			// Update all the handlers
			if (window_result == gfx2::WINDOW_RESIZED) {
				for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
					current_demonstration_handlers[i]->viewport_resized(window_size);

				for (size_t i = 0; i < reproduction_handlers.size(); ++i)
					reproduction_handlers[i]->viewport_resized(window_size);

				for (size_t i = 0; i < reproduction_handlers_moving.size(); ++i)
					reproduction_handlers_moving[i]->viewport_resized(window_size);
			}

			// At the very first frame: load initial data from files (can't be done
			// outside the loop because we need to know the size of the OpenGL front
			// buffer)
			//
			// Note: The loaded data was recorded in another demo with 3x2 viewports,
			// so appropriate scaling must be applied
			else if (window_result == gfx2::WINDOW_READY) {

				cube loaded_trajectories;
				mat loaded_frames;
				loaded_trajectories.load("data/data_4_trajectories.txt");
				loaded_frames.load("data/data_4_frames.txt");

				// adding time as 3rd dimension
				cube loaded_trajectories_tmp(loaded_trajectories.n_rows + 1, loaded_trajectories.n_cols, loaded_trajectories.n_slices);
				for (int i = 0; i < loaded_trajectories.n_slices; i++) {
					vec tmpvec = linspace(0, loaded_trajectories.n_cols-1, loaded_trajectories.n_cols);
					loaded_trajectories_tmp.slice(i) = join_vert(loaded_trajectories.slice(i), tmpvec.t());
				}
				loaded_trajectories.resize(loaded_trajectories.n_rows + 1, loaded_trajectories.n_cols, loaded_trajectories.n_slices);
				loaded_trajectories = loaded_trajectories_tmp;

				for (int n = 0; n < loaded_frames.n_cols / 2; ++n) {
					Handler* handler1 = new Handler(
							&viewport_demos,
							ImVec2(loaded_frames(0, n * 2) * window_size.win_width,
								   loaded_frames(1, n * 2) * window_size.win_height * 2),
							ImVec2(loaded_frames(2, n * 2) * window_size.win_width,
								   loaded_frames(3, n * 2) * window_size.win_height * 2),
							0,
							(window_size.fb_width == window_size.win_width)
					);
					handler1->update(window_size);
					handler1->fix();

					Handler* handler2 = new Handler(
							&viewport_demos,
							ImVec2(loaded_frames(0, n * 2 + 1) * window_size.win_width,
								   loaded_frames(1, n * 2 + 1) * window_size.win_height * 2),
							ImVec2(loaded_frames(2, n * 2 + 1) * window_size.win_width,
								   loaded_frames(3, n * 2 + 1) * window_size.win_height * 2),
							1,
							(window_size.fb_width == window_size.win_width)
					);
					handler2->update(window_size);
					handler2->fix();

					handler_list_t handlers;
					handlers.push_back(handler1);
					handlers.push_back(handler2);

					fixed_demonstration_handlers.push_back(handlers);
				}

				Handler* handler_reproduction_1 = new Handler(
						&viewport_gmr,
						fixed_demonstration_handlers[0][0]->ui_position +
						ImVec2(window_size.win_width / 3, 0),
						fixed_demonstration_handlers[0][0]->ui_y,
						0,
						(window_size.fb_width == window_size.win_width)
				);
				handler_reproduction_1->update(window_size);
				reproduction_handlers.push_back(handler_reproduction_1);

				Handler* handler_reproduction_2 = new Handler(
						&viewport_gmr,
						fixed_demonstration_handlers[2][1]->ui_position +
						ImVec2(window_size.win_width / 3, 0),
						fixed_demonstration_handlers[2][1]->ui_y,
						1,
						(window_size.fb_width == window_size.win_width)
				);
				handler_reproduction_2->update(window_size);
				reproduction_handlers.push_back(handler_reproduction_2);


				// moving frames
				Handler* handler_reproduction_moving1 = new Handler(
						&viewport_gmrmoving,
						fixed_demonstration_handlers[0][0]->ui_position +
						ImVec2(window_size.win_width * 2 / 3, 0),
						fixed_demonstration_handlers[0][0]->ui_y,
						0,
						(window_size.fb_width == window_size.win_width)
				);
				handler_reproduction_moving1->fix();
				handler_reproduction_moving1->update(window_size);
				reproduction_handlers_moving.push_back(handler_reproduction_moving1);

				Handler* handler_reproduction_moving2 = new Handler(
						&viewport_gmrmoving,
						fixed_demonstration_handlers[0][1]->ui_position +
						ImVec2(window_size.win_width * 2 / 3, 0),
						fixed_demonstration_handlers[0][1]->ui_y,
						1,
						(window_size.fb_width == window_size.win_width)
				);
				handler_reproduction_moving2->fix();
				handler_reproduction_moving2->update(window_size);
				reproduction_handlers_moving.push_back(handler_reproduction_moving2);


				for (int n = 0; n < loaded_trajectories.n_slices; ++n) {
					vector_list_t trajectory;
					for (int i = 0; i < loaded_trajectories.n_cols; ++i) {
						mat t = loaded_trajectories(span::all, span(i), span(n));
						t(0, span::all) *= window_size.fb_width;
						t(1, span::all) *= window_size.fb_height * 2;
						trajectory.push_back(t);
					}

					coordinate_system_list_t coordinate_systems;
					convert(fixed_demonstration_handlers[n], coordinate_systems, model.parameters);

					Demonstration demonstration(coordinate_systems, trajectory, model.parameters);
					demos.push_back(demonstration);
				}

				// Initial learning from the loaded data
				learn(demos, model);
			}
		}


		// If the parameters changed, recompute
		if (gui_state.are_parameters_modified) {

			// If the number of frames changed, clear everything
			if (model.parameters.nb_frames != gui_state.parameter_nb_frames) {
				demos.clear();
				model.mu.clear();
				model.sigma.clear();

				for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
					delete current_demonstration_handlers[i];

				for (size_t n = 0; n < fixed_demonstration_handlers.size(); ++n) {
					for (size_t i = 0; i < fixed_demonstration_handlers[n].size(); ++i)
						delete fixed_demonstration_handlers[n][i];
				}

				for (size_t i = 0; i < reproduction_handlers.size(); ++i)
					delete reproduction_handlers[i];

				for (size_t i = 0; i < reproduction_handlers_moving.size(); ++i)
					delete reproduction_handlers_moving[i];


				current_demonstration_handlers.clear();
				fixed_demonstration_handlers.clear();
				reproduction_handlers.clear();
				reproduction_handlers_moving.clear();


				create_new_demonstration_handlers(viewport_demos, window_size,
												  gui_state.parameter_nb_frames,
												  current_demonstration_handlers
				);

				create_reproduction_handlers(viewport_gmr, window_size,
											 gui_state.parameter_nb_frames,
											 reproduction_handlers);


				create_reproduction_handlers_gmrmoving(viewport_gmrmoving, window_size,
											 gui_state.parameter_nb_frames,
											 reproduction_handlers_moving);

				model.parameters.nb_frames = gui_state.parameter_nb_frames;

				gui_state.can_draw_demonstration = true;
				reset_moving_handler = true;
			}

			// If the number of states changed, recompute the model
			if (model.parameters.nb_states != gui_state.parameter_nb_states) {

				model.parameters.nb_states = gui_state.parameter_nb_states;

				for (auto iter = demos.begin(); iter != demos.end(); ++iter)
					iter->update(model.parameters);

				if (!demos.empty()) {
					learn(demos, model);
				}
			}

			gui_state.are_parameters_modified = false;
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();

		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glScissor(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		draw_demos_viewport(viewport_demos, current_trajectory, demos,
							fixed_demonstration_handlers,
							current_demonstration_handlers);

		double mouse_x, mouse_y;
		glfwGetCursorPos(window, &mouse_x, &mouse_y);
		vec mp = ui2fb({ mouse_x, mouse_y }, window_size, viewport_gmr);

		draw_gmr_viewport(viewport_gmr, reproduction_handlers, model, gui_state.parameter_nb_gmr_components, mp, drawIndex);



		if (reproduction_handlers.size() > 0) {
			int moving_handler_ix;
			if (model.parameters.nb_frames > 1)
				moving_handler_ix = 1;
			else
				moving_handler_ix = 0;

			if (reset_moving_handler) {
				initFramePos.x = reproduction_handlers_moving.at(moving_handler_ix)->ui_position.x;
				initFramePos.y = reproduction_handlers_moving.at(moving_handler_ix)->ui_position.y;
				reset_moving_handler = false;
			}

			ImVec2 diff;
			std::chrono::microseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >( std::chrono::system_clock::now().time_since_epoch());
			long currTimeMsec = ms.count();
			float phase =  (currTimeMsec-startTimeMsec)/1E6 ;
			diff.x = 50.0 * sin(phase);
			diff.y = 10.0 * sin(2 * phase);
			reproduction_handlers_moving.at(moving_handler_ix)->ui_position = initFramePos + diff;
			reproduction_handlers_moving.at(moving_handler_ix)->update(window_size);
		}

		draw_gmrmoving_viewport(viewport_gmrmoving, reproduction_handlers_moving, model, gui_state.parameter_nb_gmr_components, drawIndex);


		// Draw the UI controls of the handlers
		ui::begin("handlers");

		bool hovering_ui = false;

		for (size_t i = 0; i < reproduction_handlers.size(); ++i) {
			hovering_ui = reproduction_handlers[i]->draw(window_size) || hovering_ui;
		}


		for (size_t i = 0; i < reproduction_handlers_moving.size(); ++i) {
			hovering_ui = reproduction_handlers_moving[i]->draw(window_size) || hovering_ui;
		}

		for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
			hovering_ui = current_demonstration_handlers[i]->draw(window_size) || hovering_ui;

		ui::end();


		// Window: Demonstrations
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("Demonstrations", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Demonstrations		 ");
		ImGui::SameLine();

		if (!gui_state.can_draw_demonstration) {
			if (ImGui::Button("Add")) {
				create_new_demonstration_handlers(viewport_demos, window_size,
												  gui_state.parameter_nb_frames,
												  current_demonstration_handlers
				);

				gui_state.can_draw_demonstration = true;
			}
		}

		ImGui::SameLine();

		if (ImGui::Button("Clear")) {
			demos.clear();
			model.mu.clear();
			model.sigma.clear();

			for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
				delete current_demonstration_handlers[i];

			for (size_t n = 0; n < fixed_demonstration_handlers.size(); ++n) {
				for (size_t i = 0; i < fixed_demonstration_handlers[n].size(); ++i)
					delete fixed_demonstration_handlers[n][i];
			}

			current_demonstration_handlers.clear();
			fixed_demonstration_handlers.clear();

			create_new_demonstration_handlers(viewport_demos, window_size,
											  model.parameters.nb_frames,
											  current_demonstration_handlers
			);

			gui_state.can_draw_demonstration = true;
		}

		ImGui::SameLine();
		ImGui::Text("	 ");
		ImGui::SameLine();

		if (ImGui::Button("Parameters"))
			gui_state.is_parameters_dialog_displayed = true;

		ImGui::SameLine();
		if (ImGui::Button("Save")) {
			cube tmpTrajs(2, model.parameters.nb_data, fixed_demonstration_handlers.size());
			mat tmpFrames(4, 2 * fixed_demonstration_handlers.size());

			for (int i = 0; i < fixed_demonstration_handlers.size(); i++) {
				tmpFrames(0, i * 2) = fixed_demonstration_handlers.at(i).at(0)->ui_position.x / window_size.win_width ;
				tmpFrames(1, i * 2) =fixed_demonstration_handlers.at(i).at(0)->ui_position.y / window_size.win_height / 2;
				tmpFrames(2, i * 2) =fixed_demonstration_handlers.at(i).at(0)->ui_y.x / window_size.win_width ;
				tmpFrames(3, i * 2) =fixed_demonstration_handlers.at(i).at(0)->ui_y.y / window_size.win_height / 2;

				tmpFrames(0, i*2 + 1) = fixed_demonstration_handlers.at(i).at(1)->ui_position.x / window_size.win_width;
				tmpFrames(1, i*2 + 1) =fixed_demonstration_handlers.at(i).at(1)->ui_position.y / window_size.win_height / 2;
				tmpFrames(2, i*2 + 1) =fixed_demonstration_handlers.at(i).at(1)->ui_y.x / window_size.win_width;
				tmpFrames(3, i*2 + 1) =fixed_demonstration_handlers.at(i).at(1)->ui_y.y / window_size.win_height / 2;

			}
			tmpFrames.save("data/data_tpgmr_frames.txt", arma_ascii);

			for (int i = 0; i < demos.size(); i++ ) {
				tmpTrajs.slice(i).row(0) =	demos.at(i).points.row(0) / window_size.fb_width;
				tmpTrajs.slice(i).row(1)  = demos.at(i).points.row(1) / window_size.fb_height / 2;
			}

			tmpTrajs.save("data/data_tpgmr_trajectories.txt", arma_ascii);
		}

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;



		ImGui::End();

		// Window: GMR
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width / 3, 0));
		ImGui::Begin("TPGMR", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("TPGMR");
		ImGui::SameLine();
		ImGui::SliderInt("Nb components", &gui_state.parameter_nb_gmr_components, 10, 199);

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

		ImGui::End();

		// Window: GMMs in global coordinates
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width * 2 / 3, 0));
		ImGui::Begin("GMMs", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("TPGMR (changing TPs)");
		ImGui::SameLine();
		if (ImGui::Button("Randomize")) {
			for (size_t i = 0; i < reproduction_handlers_moving.size(); ++i)
				delete reproduction_handlers_moving[i];

			reproduction_handlers_moving.clear();

			create_reproduction_handlers_gmrmoving(viewport_gmrmoving, window_size,
										 gui_state.parameter_nb_frames,
										 reproduction_handlers_moving);

			reset_moving_handler = true;
		}

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

		ImGui::End();


		// Window: Parameters
		ImGui::SetNextWindowSize(ImVec2(600, 100));
		ImGui::SetNextWindowPos(ImVec2((window_size.win_width - 600) / 2, (window_size.win_height - 100) / 2));
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 255));

		if (gui_state.is_parameters_dialog_displayed)
			ImGui::OpenPopup("Parameters");

		if (ImGui::BeginPopupModal("Parameters", NULL,
								   ImGuiWindowFlags_NoResize |
								   ImGuiWindowFlags_NoSavedSettings)) {

			ImGui::SliderInt("Nb states", &gui_state.parameter_nb_states, 1, 10);
			ImGui::SliderInt("Nb frames", &gui_state.parameter_nb_frames, 1, 5);

			if (ImGui::Button("Close")) {
				ImGui::CloseCurrentPopup();
				gui_state.is_parameters_dialog_displayed = false;
				gui_state.are_parameters_modified = true;
			}

			ImGui::EndPopup();

			hovering_ui = true;
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


		// Left click: start a new demonstration (only if not on the UI and in the
		// demonstrations viewport)
		if (!gui_state.is_drawing_demonstration) {
			if (ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1) && gui_state.can_draw_demonstration) {
				double mouse_x, mouse_y;
				glfwGetCursorPos(window, &mouse_x, &mouse_y);

				if (!hovering_ui && (mouse_x <= window_size.win_width / 3))
				{
					gui_state.is_drawing_demonstration = true;

					vec coords = ui2fb({ mouse_x, mouse_y }, window_size, viewport_demos);
					vec tmpvec = {0.0};
					coords = join_vert(coords, tmpvec);

					current_trajectory.push_back(coords);
				}
			}
		} else {
			double mouse_x, mouse_y;
			glfwGetCursorPos(window, &mouse_x, &mouse_y);

			vec coords = ui2fb({ mouse_x, mouse_y }, window_size, viewport_demos);
			vec tmpvec = {(float)current_trajectory.size() - 1};
			coords = join_vert(coords, tmpvec);

			vec last_point = current_trajectory[current_trajectory.size() - 1];
			vec diff = abs(coords - last_point);

			if ((diff(0) > 1e-6) && (diff(1) > 1e-6))
				current_trajectory.push_back(coords);

			// Left mouse button release: end the demonstration creation
			if (!ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)) {
				gui_state.is_drawing_demonstration = false;

				if (current_trajectory.size() > 1) {

					coordinate_system_list_t coordinate_systems;
					convert(current_demonstration_handlers, coordinate_systems, model.parameters);

					Demonstration demonstration(coordinate_systems, current_trajectory,
												model.parameters);

					demos.push_back(demonstration);

					for (int i = 0; i < current_demonstration_handlers.size(); ++i)
						current_demonstration_handlers[i]->fix();

					fixed_demonstration_handlers.push_back(current_demonstration_handlers);
					current_demonstration_handlers.clear();

					learn(demos, model);

					gui_state.can_draw_demonstration = false;
				}

				current_trajectory.clear();
			}
		}

		ImGui::CaptureMouseFromApp();
	}


	// Cleanup
	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}

