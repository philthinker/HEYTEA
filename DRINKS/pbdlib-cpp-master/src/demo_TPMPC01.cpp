/*
 * demo_TPMPC01.cpp
 *
 * Linear quadratic control (unconstrained linear MPC) acting in multiple frames,
 * which is equivalent to a product of Gaussian controllers through a TP-GMM
 * representation.
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
 * Authors: Sylvain Calinon, Philip Abbet
 */


#include <stdio.h>
#include <float.h>
#include <armadillo>

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
	int	   nb_data;			// Number of datapoints in a trajectory
	double rfactor;			// Control cost in LQR
	float  dt;				// Time step duration
	int	   nb_stoch_repros; // Number of reproductions with stochastic sampling
};


//-----------------------------------------------------------------------------
// Contains values precomputed from the parameters and model, to speed up
// further computations
//-----------------------------------------------------------------------------
struct precomputed_t {
	// Control cost matrix
	mat R;

	// Transfer matrices in batch LQR, see Eq. (35)
	mat Su;
	mat Sx;

	// Eigen decompositions (indexing: V[state][frame])
	std::vector<matrix_list_t> V;
	std::vector<matrix_list_t> D;

	// Inverses (indexing: inv_sigma[state][frame])
	std::vector<matrix_list_t> inv_sigma;
};


//-----------------------------------------------------------------------------
// Model trained using the algorithm
//-----------------------------------------------------------------------------
struct model_t {
	parameters_t			   parameters;	// Parameters used to train the model
	precomputed_t			   precomputed; // Precomputed values

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
		points_original = mat(2, points.size());

		for (size_t i = 0; i < points.size(); ++i) {
			points_original(0, i) = points[i](0);
			points_original(1, i) = points[i](1);
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

		for (int m = 0; m < coordinate_systems.size(); ++m) {
			points_in_coordinate_systems.push_back(
				solve(coordinate_systems[m].orientation,
					  points - repmat(coordinate_systems[m].position, 1, parameters.nb_data))
			);
		}
	}


	//-------------------------------------------------------------------------
	// Returns the coordinates of a point in a specific reference frame of
	// the demonstration
	//-------------------------------------------------------------------------
	arma::vec convert_to_coordinate_system(const arma::vec& point, int frame) const {
		vec original_point = zeros(points.n_rows);
		original_point(span(0, 1)) = point(span(0, 1));

		vec result = solve(coordinate_systems[frame].orientation,
						   original_point - coordinate_systems[frame].position);

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
		bin.data = zeros(model.nb_var * model.parameters.nb_frames,
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
// Perform the precomputations needed to speed-up further processing
//-----------------------------------------------------------------------------
void precompute(model_t &model) {

	model.precomputed.inv_sigma.clear();
	model.precomputed.V.clear();
	model.precomputed.D.clear();

	// Precomputation of eigendecompositions and inverses
	for (int i = 0; i < model.parameters.nb_states; ++i) {
		model.precomputed.inv_sigma.push_back(matrix_list_t());
		model.precomputed.V.push_back(matrix_list_t());
		model.precomputed.D.push_back(matrix_list_t());

		for (int m = 0; m < model.parameters.nb_frames; ++m) {
			vec eigval;
			mat eigvec;

			eig_sym(eigval, eigvec, model.sigma[i][m](span(0, 1), span(0, 1)));

			uvec id = sort_index(eigval, "descend");

			model.precomputed.V[i].push_back(eigvec.cols(id));
			model.precomputed.D[i].push_back(diagmat(eigval(id)));
			model.precomputed.inv_sigma[i].push_back(inv(model.sigma[i][m]));
		}
	}

	// Control cost matrix
	model.precomputed.R = eye(2, 2) * model.parameters.rfactor;
	model.precomputed.R = kron(eye(model.parameters.nb_data - 1, model.parameters.nb_data - 1),
							   model.precomputed.R);

	// Integration with higher order Taylor series expansion
	mat A1d = zeros(model.parameters.nb_deriv, model.parameters.nb_deriv);
	mat B1d = zeros(model.parameters.nb_deriv, 1);

	for (int i = 0; i < model.parameters.nb_deriv; ++i) {
		// Discrete 1D
		A1d = A1d + diagmat(ones(model.parameters.nb_deriv - i, 1), i) *
			  pow(model.parameters.dt, i) * 1.0 / factorial(i);

		B1d(model.parameters.nb_deriv - i - 1) = pow(model.parameters.dt, i + 1) *
												 1.0 / factorial(i + 1);
	}

	mat A = kron(A1d, eye(2, 2)); // Discrete nD
	mat B = kron(B1d, eye(2, 2)); // Discrete nD

	// Build Sx and Su matrices (transfer matrices in batch LQR), see Eq. (35)
	model.precomputed.Su = zeros(model.nb_var * model.parameters.nb_data,
								 2 * (model.parameters.nb_data - 1));

	model.precomputed.Sx = kron(ones(model.parameters.nb_data, 1),
								eye(model.nb_var, model.nb_var));

	mat M = zeros(B.n_rows, 2 * model.parameters.nb_data);

	int n = 2 * model.parameters.nb_data - 2;
	M(span::all, span(n, n + 1)) = B;

	for (int n = 1; n < model.parameters.nb_data; ++n) {
		span id1(n * model.nb_var, model.parameters.nb_data * model.nb_var - 1);
		model.precomputed.Sx(id1, span::all) = model.precomputed.Sx(id1, span::all) * A;

		id1 = span(n * model.nb_var, (n + 1) * model.nb_var - 1);
		span id2(0, n * 2 - 1);

		int n2 = 2 * model.parameters.nb_data - n * 2;
		model.precomputed.Su(id1, id2) = M(span::all, span(n2, n2 + n * 2 - 1));

		M(span::all, span(n2 - 2, n2 - 1)) = A * M(span::all, span(n2, n2 + 1));
	}
}


//-----------------------------------------------------------------------------
// Training of the model
//-----------------------------------------------------------------------------
void learn(const demonstration_list_t& demos, model_t &model) {

	init_tensorGMM_kbins(demos, model);
	train_EM_tensorGMM(demos, model);
	precompute(model);
}


//-----------------------------------------------------------------------------
// Compute a reproduction
//-----------------------------------------------------------------------------
mat batch_lqr_reproduction(const model_t& model,
						   const coordinate_system_list_t& coordinate_systems,
						   const vec& start_coordinates,
						   int reference_demonstration_index = 0) {

	std::vector< vector_list_t > mu;		// Indexing: mu[state][frame]
	std::vector< matrix_list_t > inv_sigma; // Indexing: inv_sigma[state][frame]

	// GMM projection, see Eq. (5)
	for (int i = 0; i < model.parameters.nb_states; ++i) {
		mu.push_back(vector_list_t());
		inv_sigma.push_back(matrix_list_t());

		for (int m = 0; m < model.parameters.nb_frames; ++m) {
			mu[i].push_back(coordinate_systems[m].orientation * model.mu[i][m] +
							coordinate_systems[m].position);

			inv_sigma[i].push_back(coordinate_systems[m].orientation *
								   model.precomputed.inv_sigma[i][m] *
								   coordinate_systems[m].orientation.t());
		}
	}

	// Compute best path
	uvec q = index_max(model.pix(span::all,
								 span(reference_demonstration_index * model.parameters.nb_data,
									  (reference_demonstration_index + 1) * model.parameters.nb_data - 1)),
					   0).t();

	// Build a reference trajectory for each frame
	vector_list_t muQ;
	matrix_list_t frames_Q;

	mat Q = zeros(model.nb_var * model.parameters.nb_data,
				  model.nb_var * model.parameters.nb_data);

	for (int m = 0; m < model.parameters.nb_frames; ++m) {
		vec frame_muQ = zeros(model.nb_var * model.parameters.nb_data);
		mat frame_Q = zeros(model.nb_var, model.nb_var * model.parameters.nb_data);

		for (int i = 0; i < model.parameters.nb_data; ++i) {
			span id(i * model.nb_var, (i + 1) * model.nb_var - 1);
			frame_muQ(id) = mu[q(i)][m];
			frame_Q(span::all, id) = inv_sigma[q(i)][m];
		}

		frame_Q = kron(ones(model.parameters.nb_data, 1),
					   eye(model.nb_var, model.nb_var)) * frame_Q %
				  kron(eye(model.parameters.nb_data, model.parameters.nb_data),
					   ones(model.nb_var, model.nb_var));

		muQ.push_back(frame_muQ);
		frames_Q.push_back(frame_Q);

		Q = Q + frame_Q;
	}

	// Batch LQR (unconstrained linear MPC in multiple frames), corresponding to a
	// product of Gaussian controllers
	mat Rq = model.precomputed.Su.t() * Q * model.precomputed.Su + model.precomputed.R;

	vec X = zeros(model.nb_var);
	X(span(0, 1)) = start_coordinates(span(0, 1));

	mat rq = zeros(model.nb_var * model.parameters.nb_data);
	for (int m = 0; m < model.parameters.nb_frames; ++m) {
		rq = rq + frames_Q[m] * (muQ[m] - model.precomputed.Sx * X);
	}

	rq = model.precomputed.Su.t() * rq;

	mat u = solve(Rq, rq);

	return reshape(model.precomputed.Sx * X + model.precomputed.Su * u, model.nb_var,
				   model.parameters.nb_data);
}


//-----------------------------------------------------------------------------
// Compute stochastic reproductions
//-----------------------------------------------------------------------------
void batch_lqr_stochastic_reproduction(const model_t& model,
									   const coordinate_system_list_t& coordinate_systems,
									   const vec& start_coordinates,
									   matrix_list_t &reproductions,
									   std::vector< std::vector< vector_list_t > > &stochastic_mu) {

	const int nb_eigs = 2;	// Number of principal eigencomponents to keep

	reproductions.clear();
	stochastic_mu.clear();


	// Compute best path
	uvec q = index_max(model.pix(span::all, span(0, model.parameters.nb_data - 1)), 0).t();


	for (int n = 0; n < model.parameters.nb_stoch_repros; ++n) {
		std::vector< vector_list_t > mu;		// Indexing: mu[state][frame]
		std::vector< matrix_list_t > inv_sigma; // Indexing: inv_sigma[state][frame]

		// GMM projection by moving centers randomly
		std::vector<vector_list_t> stoch_mu = model.mu;

		cube N(nb_eigs, model.parameters.nb_frames, model.parameters.nb_states, fill::randn);
			// Noise on all components in all frames

		for (int i = 0; i < model.parameters.nb_states; ++i) {
			mu.push_back(vector_list_t());
			inv_sigma.push_back(matrix_list_t());

			for (int m = 0; m < model.parameters.nb_frames; ++m) {
				stoch_mu[i][m](span(0, nb_eigs - 1)) =
						model.mu[i][m](span(0, nb_eigs - 1)) +
						model.precomputed.V[i][m](span::all, span(0, nb_eigs - 1)) *
						pow(model.precomputed.D[i][m](span(0, nb_eigs - 1), span(0, nb_eigs - 1)), 0.5) *
						vec(N(span::all, span(m), span(i)));

				mu[i].push_back(coordinate_systems[m].orientation * stoch_mu[i][m] +
								coordinate_systems[m].position);

				inv_sigma[i].push_back(coordinate_systems[m].orientation *
									   model.precomputed.inv_sigma[i][m] *
									   coordinate_systems[m].orientation.t());
			}
		}

		// Build a reference trajectory for each frame
		vector_list_t muQ;
		matrix_list_t frames_Q;

		mat Q = zeros(model.nb_var * model.parameters.nb_data,
					  model.nb_var * model.parameters.nb_data);

		for (int m = 0; m < model.parameters.nb_frames; ++m) {
			vec frame_muQ = zeros(model.nb_var * model.parameters.nb_data);
			mat frame_Q = zeros(model.nb_var, model.nb_var * model.parameters.nb_data);

			for (int i = 0; i < model.parameters.nb_data; ++i) {
				span id(i * model.nb_var, (i + 1) * model.nb_var - 1);
				frame_muQ(id) = mu[q(i)][m];
				frame_Q(span::all, id) = inv_sigma[q(i)][m];
			}

			frame_Q = kron(ones(model.parameters.nb_data, 1),
						   eye(model.nb_var, model.nb_var)) * frame_Q %
					  kron(eye(model.parameters.nb_data, model.parameters.nb_data),
						   ones(model.nb_var, model.nb_var));

			muQ.push_back(frame_muQ);
			frames_Q.push_back(frame_Q);

			Q = Q + frame_Q;
		}

		// Batch LQR (unconstrained linear MPC in multiple frames), corresponding to a
		// product of Gaussian controllers
		mat Rq = model.precomputed.Su.t() * Q * model.precomputed.Su + model.precomputed.R;

		vec X = zeros(model.nb_var);
		X(span(0, 1)) = start_coordinates(span(0, 1));

		mat rq = zeros(model.nb_var * model.parameters.nb_data);
		for (int m = 0; m < model.parameters.nb_frames; ++m) {
			rq = rq + frames_Q[m] * (muQ[m] - model.precomputed.Sx * X);
		}

		rq = model.precomputed.Su.t() * rq;

		mat u = solve(Rq, rq);

		mat reproduction = reshape(model.precomputed.Sx * X + model.precomputed.Su * u,
								   model.nb_var, model.parameters.nb_data);

		reproductions.push_back(reproduction(span(0, 1), span::all));

		stochastic_mu.push_back(stoch_mu);
	}
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

	// Indicates if the reproductions must be recomputed
	bool must_recompute_reproductions;

	// Parameters modifiable via the UI (they correspond to the ones declared
	//in parameters_t)
	int parameter_nb_states;
	int parameter_nb_frames;
	int parameter_nb_data;
	int parameter_nb_stoch_repros;

	// The reference frame currently displayed for the models
	int displayed_frame;
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
								  window_size.win_height / 2 + 20,
								  window_size.win_width * 2 / 3 - 20,
								  window_size.win_height - 20,
								  (window_size.fb_width == window_size.win_width))
		);
	};
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
	if (current_trajectory.size() > 1)
		gfx2::draw_line(arma::fvec({0.33f, 0.97f, 0.33f}), current_trajectory);

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
// Render a "model" viewport
//-----------------------------------------------------------------------------
void draw_model_viewport(const viewport_t& viewport,
						 const gfx2::window_size_t window_size,
						 const demonstration_list_t& demonstrations,
						 int perspective,
						 const model_t& model,
						 std::vector< std::vector< vector_list_t > >* stochastic_mu = 0) {

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

	// Draw the GMM states
	if (!demonstrations.empty()) {
		if (!stochastic_mu) {
			for (int i = 0; i < model.parameters.nb_states; ++i) {
				glClear(GL_DEPTH_BUFFER_BIT);
				gfx2::draw_gaussian(conv_to<fvec>::from(COLORS.row(i).t()),
									model.mu[i][perspective],
									model.sigma[i][perspective]);
			}
		} else {
			for (int n = 0; n < stochastic_mu->size(); ++n) {
				for (int i = 0; i < model.parameters.nb_states; ++i) {
					glClear(GL_DEPTH_BUFFER_BIT);
					gfx2::draw_gaussian(conv_to<fvec>::from(COLORS.row(i).t()),
										(*stochastic_mu)[n][i][perspective],
										model.sigma[i][perspective]);
				}
			}
		}
	}

	glClear(GL_DEPTH_BUFFER_BIT);

	// Draw the demonstrations
	int color_index = 0;
	for (auto iter = demonstrations.begin(); iter != demonstrations.end(); ++iter) {
		arma::mat datapoints = iter->points_in_coordinate_systems[perspective](span(0, 1), span::all);

		arma::fvec color = arma::conv_to<arma::fvec>::from(COLORS.row(color_index));

		gfx2::draw_line(color, datapoints);

		++color_index;
		if (color_index >= COLORS.n_rows)
			color_index = 0;
	}

	// Draw the handler
	Handler handler(&viewport, ImVec2(0, 0), ImVec2(-30, 0), perspective,
					(window_size.fb_width == window_size.win_width));
	handler.update();
	handler.draw_anchor();

	for (auto iter = demonstrations.begin(); iter != demonstrations.end(); ++iter) {
		for (int n = 0; n < model.parameters.nb_frames; ++n) {
			if (n == perspective)
				continue;

			fvec color = HANDLER_COLORS.row(n).t();

			vec pos = iter->convert_to_coordinate_system(iter->coordinate_systems[n].position,
														 perspective);

			gfx2::draw_rectangle(color, 10, 10, fvec({ (float) pos(0), (float) pos(1), 0.0f }));
		}
	}


	// Draw the axes
	arma::mat x_axis({{(float) -viewport.width, (float) viewport.width, 0.0f},
					  {0.0f, 0.0f, 0.0f}});

	gfx2::draw_line(fvec({0.0f, 0.0f, 0.0f}), x_axis);

	arma::mat y_axis({{0.0f, 0.0f, 0.0f},
					  {(float) -viewport.height, (float) viewport.height, 0.0f}});

	gfx2::draw_line(fvec({0.0f, 0.0f, 0.0f}), y_axis);
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

	// Draw the reproductions (if any)
	int color_index = 0;
	for (auto iter = reproductions.begin(); iter != reproductions.end(); ++iter) {
		arma::fvec color = arma::conv_to<arma::fvec>::from(COLORS.row(color_index));

		gfx2::draw_line(color, *iter);

		++color_index;
		if (color_index >= COLORS.n_rows)
			color_index = 0;
	}

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


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	// Model
	model_t model;

	// Parameters
	model.parameters.nb_states		 = 3;
	model.parameters.nb_frames		 = 2;
	model.parameters.nb_deriv		 = 2;
	model.parameters.nb_data		 = 200;
	model.parameters.rfactor		 = 0.01;
	model.parameters.dt				 = 0.1f;
	model.parameters.nb_stoch_repros = 10;


	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 1200;
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
		"Demo - Linear quadratic optimal control",
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
	viewport_t viewport_repros;
	viewport_t viewport_model;
	viewport_t viewport_new_repros;
	viewport_t viewport_stochastic_model;
	viewport_t viewport_stochastic_repros;


	// GUI state
	gui_state_t gui_state;
	gui_state.can_draw_demonstration = false;
	gui_state.is_drawing_demonstration = false;
	gui_state.is_parameters_dialog_displayed = false;
	gui_state.are_parameters_modified = false;
	gui_state.must_recompute_reproductions = false;
	gui_state.parameter_nb_states = model.parameters.nb_states;
	gui_state.parameter_nb_frames = model.parameters.nb_frames;
	gui_state.parameter_nb_data = model.parameters.nb_data;
	gui_state.parameter_nb_stoch_repros = model.parameters.nb_stoch_repros;
	gui_state.displayed_frame = 1;


	// Demonstration and reproduction handlers
	std::vector<handler_list_t> fixed_demonstration_handlers;
	handler_list_t current_demonstration_handlers;
	handler_list_t reproduction_handlers;

	// List of demonstrations and reproductions
	demonstration_list_t demos;
	matrix_list_t reproductions;
	matrix_list_t new_reproductions;
	matrix_list_t stochastic_reproductions;
	std::vector< std::vector< vector_list_t > > stochastic_mu;


	// Main loop
	vector_list_t current_trajectory;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		if ((window_result == gfx2::WINDOW_READY) || (window_result == gfx2::WINDOW_RESIZED)) {

			viewport_width = window_size.fb_width / 3 - 1;
			viewport_height = window_size.fb_height / 2 - 1;

			// Update all the viewports
			setup_viewport(&viewport_demos, 0, window_size.fb_height - viewport_height,
						   viewport_width, viewport_height);

			setup_viewport(&viewport_repros, 0, 0,
						   viewport_width, viewport_height);

			setup_viewport(&viewport_model, viewport_width + 2,
						   window_size.fb_height - viewport_height,
						   viewport_width, viewport_height, -1.0f, 1.0f,
						   compute_centered_view_matrix(demos, gui_state.displayed_frame - 1));

			setup_viewport(&viewport_new_repros, viewport_width + 2, 0,
						   viewport_width, viewport_height);

			setup_viewport(&viewport_stochastic_model, window_size.fb_width - viewport_width,
						   window_size.fb_height - viewport_height,
						   viewport_width, viewport_height, -1.0f, 1.0f,
						   viewport_model.view);

			setup_viewport(&viewport_stochastic_repros, window_size.fb_width - viewport_width, 0,
						   viewport_width, viewport_height);

			// Update all the handlers
			if (window_result == gfx2::WINDOW_RESIZED) {
				for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
					current_demonstration_handlers[i]->viewport_resized(window_size);

				for (size_t i = 0; i < reproduction_handlers.size(); ++i)
					reproduction_handlers[i]->viewport_resized(window_size);
			}

			// At the very first frame: load initial data from files (can't be done
			// outside the loop because we need to know the size of the OpenGL front
			// buffer)
			else if (window_result == gfx2::WINDOW_READY) {
				cube loaded_trajectories;
				mat loaded_frames;
				loaded_trajectories.load("data/data_4_trajectories.txt");
				loaded_frames.load("data/data_4_frames.txt");

				for (int n = 0; n < loaded_frames.n_cols / 2; ++n) {
					Handler* handler1 = new Handler(
						&viewport_demos,
						ImVec2(loaded_frames(0, n * 2) * window_size.win_width,
							   loaded_frames(1, n * 2) * window_size.win_height),
						ImVec2(loaded_frames(2, n * 2) * window_size.win_width,
							   loaded_frames(3, n * 2) * window_size.win_height),
						0,
						(window_size.fb_width == window_size.win_width)
					);
					handler1->update(window_size);
					handler1->fix();

					Handler* handler2 = new Handler(
						&viewport_demos,
						ImVec2(loaded_frames(0, n * 2 + 1) * window_size.win_width,
							   loaded_frames(1, n * 2 + 1) * window_size.win_height),
						ImVec2(loaded_frames(2, n * 2 + 1) * window_size.win_width,
							   loaded_frames(3, n * 2 + 1) * window_size.win_height),
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
					&viewport_new_repros,
					fixed_demonstration_handlers[0][0]->ui_position +
					ImVec2(window_size.win_width / 3, window_size.win_height / 2),
					fixed_demonstration_handlers[0][0]->ui_y,
					0,
					(window_size.fb_width == window_size.win_width)
				);
				handler_reproduction_1->update(window_size);
				reproduction_handlers.push_back(handler_reproduction_1);

				Handler* handler_reproduction_2 = new Handler(
					&viewport_new_repros,
					fixed_demonstration_handlers[0][1]->ui_position +
					ImVec2(window_size.win_width / 3, window_size.win_height / 2),
					fixed_demonstration_handlers[0][1]->ui_y,
					1,
					(window_size.fb_width == window_size.win_width)
				);
				handler_reproduction_2->update(window_size);
				reproduction_handlers.push_back(handler_reproduction_2);

				for (int n = 0; n < loaded_trajectories.n_slices; ++n) {
					vector_list_t trajectory;
					for (int i = 0; i < loaded_trajectories.n_cols; ++i) {
						mat t = loaded_trajectories(span::all, span(i), span(n));
						t(0, span::all) *= window_size.fb_width;
						t(1, span::all) *= window_size.fb_height;
						trajectory.push_back(t);
					}

					coordinate_system_list_t coordinate_systems;
					convert(fixed_demonstration_handlers[n], coordinate_systems, model.parameters);

					Demonstration demonstration(coordinate_systems, trajectory, model.parameters);
					demos.push_back(demonstration);
				}

				// Initial learning from the loaded data
				learn(demos, model);

				std::vector<coordinate_system_list_t> coordinate_systems_list;
				convert(demos, coordinate_systems_list);

				for (int n = 0; n < demos.size(); ++n) {
					mat reproduction = batch_lqr_reproduction(
						model, coordinate_systems_list[n],
						demos[n].points_original(span::all, 0), n);

					reproductions.push_back(reproduction);
				}

				gui_state.must_recompute_reproductions = true;
			}
		}


		// If the parameters changed, recompute
		if (gui_state.are_parameters_modified) {

			// If the number of frames changed, clear everything
			if (model.parameters.nb_frames != gui_state.parameter_nb_frames) {
				demos.clear();

				for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
					delete current_demonstration_handlers[i];

				for (size_t n = 0; n < fixed_demonstration_handlers.size(); ++n) {
					for (size_t i = 0; i < fixed_demonstration_handlers[n].size(); ++i)
						delete fixed_demonstration_handlers[n][i];
				}

				for (size_t i = 0; i < reproduction_handlers.size(); ++i)
					delete reproduction_handlers[i];

				current_demonstration_handlers.clear();
				fixed_demonstration_handlers.clear();
				reproduction_handlers.clear();

				reproductions.clear();
				new_reproductions.clear();
				stochastic_reproductions.clear();

				create_new_demonstration_handlers(viewport_demos, window_size,
												  gui_state.parameter_nb_frames,
												  current_demonstration_handlers
				);

				create_reproduction_handlers(viewport_new_repros, window_size,
											 gui_state.parameter_nb_frames,
											 reproduction_handlers);

				model.parameters.nb_frames = gui_state.parameter_nb_frames;
				gui_state.displayed_frame = 2;
				gui_state.can_draw_demonstration = true;

				viewport_model.view = compute_centered_view_matrix(
						demos, gui_state.displayed_frame - 1
				);

				viewport_stochastic_model.view = viewport_model.view;
			}

			// If the number of states or datapoints changed, recompute the model
			if ((model.parameters.nb_data != gui_state.parameter_nb_data) ||
				(model.parameters.nb_states != gui_state.parameter_nb_states)) {

				model.parameters.nb_data = gui_state.parameter_nb_data;
				model.parameters.nb_states = gui_state.parameter_nb_states;

				for (auto iter = demos.begin(); iter != demos.end(); ++iter)
					iter->update(model.parameters);

				if (!demos.empty()) {
					learn(demos, model);

					std::vector<coordinate_system_list_t> coordinate_systems;
					convert(demos, coordinate_systems);

					reproductions.clear();
					for (int n = 0; n < demos.size(); ++n) {
						reproductions.push_back(
							batch_lqr_reproduction(model, coordinate_systems[n],
												   demos[n].points_original(span::all, 0),
												   n)
						);
					}

					gui_state.must_recompute_reproductions = true;

					viewport_model.view = compute_centered_view_matrix(
							demos, gui_state.displayed_frame - 1
					);

					viewport_stochastic_model.view = viewport_model.view;
				}
			}

			// If the number of stochastic reproductions changed, recompute them
			if (model.parameters.nb_stoch_repros != gui_state.parameter_nb_stoch_repros) {
				model.parameters.nb_stoch_repros = gui_state.parameter_nb_stoch_repros;
				gui_state.must_recompute_reproductions = true;
			}

			gui_state.are_parameters_modified = false;
		}

		// Recompute the new reproductions (if necessary)
		if (!demos.empty() && gui_state.must_recompute_reproductions) {

			coordinate_system_list_t coordinate_systems;
			convert(reproduction_handlers, coordinate_systems, model.parameters);

			new_reproductions.clear();

			new_reproductions.push_back(
				batch_lqr_reproduction(
					model, coordinate_systems,
					conv_to<vec>::from(reproduction_handlers[0]->transforms.position)
				)
			);

			batch_lqr_stochastic_reproduction(
				model, coordinate_systems,
				conv_to<vec>::from(reproduction_handlers[0]->transforms.position),
				stochastic_reproductions,
				stochastic_mu
			);

			gui_state.must_recompute_reproductions = false;

			for (size_t i = 0; i < reproduction_handlers.size(); ++i)
				reproduction_handlers[i]->moved = false;

			viewport_model.view = compute_centered_view_matrix(
					demos, gui_state.displayed_frame - 1
			);

			viewport_stochastic_model.view = viewport_model.view;
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

		draw_reproductions_viewport(viewport_repros, fixed_demonstration_handlers,
									reproductions);

		draw_model_viewport(viewport_model, window_size, demos,
							gui_state.displayed_frame - 1, model);

		draw_reproductions_viewport(viewport_new_repros, reproduction_handlers,
									new_reproductions);

		draw_model_viewport(viewport_stochastic_model, window_size, demos,
							gui_state.displayed_frame - 1, model, &stochastic_mu);

		draw_reproductions_viewport(viewport_stochastic_repros, reproduction_handlers,
									stochastic_reproductions);


		// Draw the UI controls of the handlers
		ui::begin("handlers");

		bool hovering_ui = false;

		for (size_t i = 0; i < reproduction_handlers.size(); ++i) {
			hovering_ui = reproduction_handlers[i]->draw(window_size) || hovering_ui;

			if (!ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)) {
				gui_state.must_recompute_reproductions = gui_state.must_recompute_reproductions ||
														 reproduction_handlers[i]->moved;
			}
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

		ImGui::Text("Demonstrations      ");
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

			for (size_t i = 0; i < current_demonstration_handlers.size(); ++i)
				delete current_demonstration_handlers[i];

			for (size_t n = 0; n < fixed_demonstration_handlers.size(); ++n) {
				for (size_t i = 0; i < fixed_demonstration_handlers[n].size(); ++i)
					delete fixed_demonstration_handlers[n][i];
			}

			current_demonstration_handlers.clear();
			fixed_demonstration_handlers.clear();

			reproductions.clear();
			new_reproductions.clear();
			stochastic_reproductions.clear();

			create_new_demonstration_handlers(viewport_demos, window_size,
											  model.parameters.nb_frames,
											  current_demonstration_handlers
			);

			gui_state.can_draw_demonstration = true;
		}

		ImGui::SameLine();
		ImGui::Text("    ");
		ImGui::SameLine();

		if (ImGui::Button("Parameters"))
			gui_state.is_parameters_dialog_displayed = true;

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

		ImGui::End();


		// Window: Reproductions
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(0, window_size.win_height / 2));
		ImGui::Begin("Reproductions", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Reproductions");

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

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
		ImGui::SameLine();
		ImGui::Text("Frame:");
		ImGui::SameLine();

		int previous_displayed_frame = gui_state.displayed_frame;

		ImGui::SliderInt("", &gui_state.displayed_frame, 1, model.parameters.nb_frames);

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

		ImGui::End();

		if (previous_displayed_frame != gui_state.displayed_frame) {
			viewport_model.view = compute_centered_view_matrix(
					demos, gui_state.displayed_frame - 1
			);

			viewport_stochastic_model.view = viewport_model.view;
		}


		// Window: New reproductions
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width / 3, window_size.win_height / 2));
		ImGui::Begin("New reproductions", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("New reproductions");

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

		ImGui::End();


		// Window: Stochastic model
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width - window_size.win_width / 3, 0));
		ImGui::Begin("Stochastic model", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Stochastic model");

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

		ImGui::End();


		// Window: Stochastic reproductions
		ImGui::SetNextWindowSize(ImVec2(window_size.win_width / 3, 36));
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width - window_size.win_width / 3, window_size.win_height / 2));
		ImGui::Begin("Stochastic reproductions", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
					 ImGuiWindowFlags_NoTitleBar
		);

		ImGui::Text("Stochastic reproductions");

		hovering_ui = ImGui::IsWindowHovered() || hovering_ui;

		ImGui::End();


		// Window: Parameters
		ImGui::SetNextWindowSize(ImVec2(600, 146));
		ImGui::SetNextWindowPos(ImVec2((window_size.win_width - 600) / 2, (window_size.win_height - 146) / 2));
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 255));

		if (gui_state.is_parameters_dialog_displayed)
			ImGui::OpenPopup("Parameters");

		if (ImGui::BeginPopupModal("Parameters", NULL,
								   ImGuiWindowFlags_NoResize |
								   ImGuiWindowFlags_NoSavedSettings)) {

			ImGui::SliderInt("Nb states", &gui_state.parameter_nb_states, 1, 10);
			ImGui::SliderInt("Nb frames", &gui_state.parameter_nb_frames, 2, 5);
			ImGui::SliderInt("Nb data", &gui_state.parameter_nb_data, 100, 300);
			ImGui::SliderInt("Nb stochastic reproductions", &gui_state.parameter_nb_stoch_repros, 2, 10);

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

		// When the 'S' key is pressed, save the demonstrations in files
		if (ImGui::IsKeyPressed(GLFW_KEY_S)) {
			mat frames(4, fixed_demonstration_handlers.size() * gui_state.parameter_nb_frames);
			for (size_t i = 0; i < fixed_demonstration_handlers.size(); ++i) {
				for (size_t j = 0; j < fixed_demonstration_handlers[i].size(); ++j) {
					frames(0, i * gui_state.parameter_nb_frames + j) = (float) fixed_demonstration_handlers[i][j]->ui_position.x / window_size.win_width;
					frames(1, i * gui_state.parameter_nb_frames + j) = (float) fixed_demonstration_handlers[i][j]->ui_position.y / window_size.win_height;
					frames(2, i * gui_state.parameter_nb_frames + j) = (float) fixed_demonstration_handlers[i][j]->ui_y.x / window_size.win_width;
					frames(3, i * gui_state.parameter_nb_frames + j) = (float) fixed_demonstration_handlers[i][j]->ui_y.y / window_size.win_height;
				}
			}

			frames.save("data_tpbatch_lqr_frames.txt", arma_ascii);

			cube trajectories(2, gui_state.parameter_nb_data, demos.size());
			for (size_t i = 0; i < demos.size(); ++i) {
				for (size_t j = 0; j < gui_state.parameter_nb_data; ++j) {
					trajectories(0, j, i) = demos[i].points(0, j) / window_size.fb_width;
					trajectories(1, j, i) = demos[i].points(1, j) / window_size.fb_height;
				}
			}

			trajectories.save("data_tpbatch_lqr_trajectories.txt", arma_ascii);
		}

		// Left click: start a new demonstration (only if not on the UI and in the
		// demonstrations viewport)
		if (!gui_state.is_drawing_demonstration && !gui_state.is_parameters_dialog_displayed) {
			if (ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1) && gui_state.can_draw_demonstration) {
				double mouse_x, mouse_y;
				glfwGetCursorPos(window, &mouse_x, &mouse_y);

				if (!hovering_ui && (mouse_x <= window_size.win_width / 3) &&
					(mouse_y <= window_size.win_height / 2))
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

					std::vector<coordinate_system_list_t> coordinate_systems_list;
					convert(demos, coordinate_systems_list);

					reproductions.clear();

					for (int n = 0; n < demos.size(); ++n) {
						mat reproduction = batch_lqr_reproduction(
							model, coordinate_systems_list[n],
							demos[n].points_original(span::all, 0), n);

						reproductions.push_back(reproduction);
					}

					gui_state.must_recompute_reproductions = true;
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
