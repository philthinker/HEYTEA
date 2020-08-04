/*
 * demo_Riemannian_SPD_GMR_Mandel01.cpp
 *
 * GMR with time as input and covariance data as output by relying on Riemannian manifold.
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
 * Authors: Sylvain Calinon, Philip Abbet, Noemie Jaquier
 */


#include <stdio.h>
#include <armadillo>
#include <iomanip>

#include <mvn.h>

#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>

using namespace arma;


/******************************** CONSTANTS **********************************/

const arma::fmat COLORS({
	{ 0.00f, 0.00f, 1.00f, 1.0f },
	{ 0.00f, 0.50f, 0.00f, 1.0f },
	{ 1.00f, 0.00f, 0.00f, 1.0f },
	{ 0.00f, 0.75f, 0.75f, 1.0f },
	{ 0.75f, 0.00f, 0.75f, 1.0f },
	{ 0.75f, 0.75f, 0.00f, 1.0f },
	{ 0.25f, 0.25f, 0.25f, 1.0f },
	{ 0.00f, 0.00f, 1.00f, 1.0f },
	{ 0.00f, 0.50f, 0.00f, 1.0f },
	{ 1.00f, 0.00f, 0.00f, 1.0f },
	{ 0.00f, 0.75f, 0.75f, 1.0f },
	{ 0.75f, 0.00f, 0.75f, 1.0f },
	{ 0.75f, 0.75f, 0.00f, 1.0f },
	{ 0.25f, 0.25f, 0.25f, 1.0f },
	{ 0.00f, 0.00f, 1.00f, 1.0f },
	{ 0.00f, 0.50f, 0.00f, 1.0f },
	{ 1.00f, 0.00f, 0.00f, 1.0f },
	{ 0.00f, 0.75f, 0.75f, 1.0f },
	{ 0.75f, 0.00f, 0.75f, 1.0f },
	{ 0.75f, 0.75f, 0.00f, 1.0f },
});


/*********************************** TYPES ***********************************/

struct parameters_t {
	int	  nb_states;		// Number of components in the GMM
	int	  nb_var;			// Dimension of the manifold and tangent space (here:
							// 1D input + 2^2 output)
	int	  nb_var_vec;		// Dimension of the manifold and tangent space in vector form
	int	  nb_data;			// Number of datapoints
	float dt;				// Time step duration
	int	  nb_iter_EM;		// Number of iterations for the EM algorithm
	int	  nb_iter;			// Number of iterations for the Gauss Newton algorithm
	float diag_reg_fact;	// Regularization term to avoid numerical instability
};

//-----------------------------------------------

struct model_t {
	// These lists contain one element per GMM state
	vec				 priors;
	std::vector<vec> mu_man;
	std::vector<vec> mu;
	std::vector<mat> sigma;
	std::vector<vec> xhat;
	std::vector<vec> H;
};


/****************************** HELPER FUNCTIONS *****************************/

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error %d: %s\n", error, description);
}


//-----------------------------------------------------------------------------
// Creates a MxNx1 cube from a MxN matrix
//-----------------------------------------------------------------------------
template <typename T>
Cube<T> mat2cube(const Mat<T>& m) {
	return Cube<T>(m.memptr(), m.n_rows, m.n_cols, 1);
}


//-------------------------------------------------------------------------
// Vectorization of a tensor of symmetric matrices
//-------------------------------------------------------------------------
arma::vec symMat2vec(const arma::mat& data) {
	const int D = data.n_rows;
	int ind = D;

	arma::vec v = arma::zeros<vec>(D*D/2 + D/2);
	for (int d = 0; d < D; ++d){
		v(d) = data(d,d);
	}

	for (int d = 1; d < D; ++d){
		for (int i = 0; i < D-d; ++i){
		  v(ind) = sqrtf(2.0f) * data(i, d);
		  ind++;
		}
	}
	return v;
}


//-------------------------------------------------------------------------
// Vectorization of a tensor of symmetric matrices
//-------------------------------------------------------------------------
arma::mat symMat2vec(const arma::cube& data) {
	const int D = data.n_rows;
	const int N = data.n_slices;
	arma::mat V = arma::zeros<mat>(D*D/2 + D/2, N);

	for (int n = 0; n < N; ++n) {
	  arma::mat M = mat(data.slice(n));
	  V(span::all, n) = symMat2vec(M);
	}
	return V;
}


//-------------------------------------------------------------------------
// Transforms a matrix of vector to a tensor of symmetric matrices
//-------------------------------------------------------------------------
arma::mat vec2symMat(const arma::vec& data) {
	const int D = data.n_rows;
	const int newD = (-1 + sqrtf(1 + 8*D))/2;

	arma::mat M = arma::zeros<mat>(newD, newD);
	for (int d = 0; d < newD; ++d){
		M(d,d) = data(d);
	}

	int ind = newD;
	for (int d = 1; d < newD; ++d){
		for (int i = 0; i < newD-d; ++i){
			M(i, d) = data(ind)/sqrtf(2.0f);
			M(d, i) = M(i, d);
			ind++;
		}
	}
	return M;
}


//-------------------------------------------------------------------------
// Transforms a matrix of vector to a tensor of symmetric matrices
//-------------------------------------------------------------------------
arma::cube vec2symMat(const arma::mat& data) {
	const int D = data.n_rows;
	const int N = data.n_cols;
	const int newD = (-1 + sqrtf(1 + 8*D))/2;

	arma::cube S = arma::zeros<cube>(newD, newD, N);

	for (int n = 0; n < N; ++n) {
		S.slice(n) = mat2cube(vec2symMat(vec(data(span::all,n))));
	}
	return S;
}


//-----------------------------------------------------------------------------
// Logarithm map (SPD manifold)
//-----------------------------------------------------------------------------
arma::cube logmap(const arma::cube& X, const arma::mat& S) {
	arma::cube U(size(X));

	for (int n = 0; n < X.n_slices; ++n) {
		U(span::all, span::all, span(n)) =
			real(sqrtmat(S) * logmat(inv(sqrtmat(S)) * mat(X(span::all, span::all, span(n))) *
									 inv(sqrtmat(S))
			) * sqrtmat(S)
		);
	}

	return U;
}


//-----------------------------------------------------------------------------
// Logarithm map (SPD manifold)
//-----------------------------------------------------------------------------
arma::mat logmap(const arma::mat& X, const arma::mat& S) {
	return real(sqrtmat(S) * logmat(inv(sqrtmat(S)) * X * inv(sqrtmat(S))) * sqrtmat(S));
}


//-----------------------------------------------------------------------------
// Exponential map (SPD manifold)
//-----------------------------------------------------------------------------
arma::mat expmap(const arma::mat& U, const arma::mat& S) {
	return real(sqrtmat(S) * expmat(inv(sqrtmat(S)) * U * inv(sqrtmat(S))) * sqrtmat(S));
}


//-----------------------------------------------------------------------------
// Logarithm map (SPD manifold) for data in vector form
//-----------------------------------------------------------------------------
arma::vec logmap_vec(const arma::vec& x, const arma::vec& s) {
	arma::mat X = vec2symMat(x);
	arma::mat S = vec2symMat(s);
	arma::mat U = logmap(X, S);
	return symMat2vec(U);
}


//-----------------------------------------------------------------------------
// Logarithm map (SPD manifold) for data in matrix form
//-----------------------------------------------------------------------------
arma::mat logmap_vec(const arma::mat& x, const arma::vec& s) {
	arma::cube X = vec2symMat(x);
	arma::mat S = vec2symMat(s);
	arma::cube U = logmap(X, S);
	return symMat2vec(U);
}


//-----------------------------------------------------------------------------
// Exponential map (SPD manifold) for data in vector form
//-----------------------------------------------------------------------------
arma::vec expmap_vec(const arma::vec& u, const arma::vec& s) {
	arma::mat U = vec2symMat(u);
	arma::mat S = vec2symMat(s);
	arma::mat X = expmap(U, S);
	return symMat2vec(X);
}


//-----------------------------------------------------------------------------
// Parallel transport (SPD manifold)
//
// Transportation operator Ac to move V from S1 to S2: VT = Ac * V * Ac'
//-----------------------------------------------------------------------------
mat transp(const mat& S1, const mat& S2) {
	mat U = logmap(S2, S1);
	return real(sqrtmat(S1) * expmat(0.5 * inv(sqrtmat(S1)) * U * inv(sqrtmat(S1))) * inv(sqrtmat(S1)));
}


//-----------------------------------------------------------------------------
// Mean of SPD matrices on the manifold
//-----------------------------------------------------------------------------
mat spdMean(const cube& setS, unsigned int nb_iterations = 10) {
	mat M = setS(span::all, span::all, span(0));

	for (unsigned int i = 0; i < nb_iterations; ++i) {
		mat L = zeros<mat>(setS.n_rows, setS.n_cols);

		mat sqrt_M = real(sqrtmat(M));
		mat inv_sqrt_M = inv(M) * sqrt_M;

		for (unsigned int n = 0; n < setS.n_slices; ++n) {
			L = L + real(logmat(inv_sqrt_M * mat(setS(span::all, span::all, span(n))) *
								inv_sqrt_M));
		}

		M = sqrt_M * expmat(L / setS.n_slices) * sqrt_M;
	}

	return M;
}


//-----------------------------------------------------------------------------
// K-Bins initialisation by relying on SPD manifold
//-----------------------------------------------------------------------------
void spd_init_GMM_kbins(const mat& data, const span& spd_data_id,
						const parameters_t& parameters, model_t &model) {
	// Parameters
	int nb_var = data.n_rows;
	int nb_data = data.n_cols;

	model.priors = vec(parameters.nb_states);
	model.mu_man.clear();
	model.sigma.clear();

	// Delimit the cluster bins
	uvec t_sep = conv_to<uvec>::from(round(linspace<vec>(0, parameters.nb_data, parameters.nb_states + 1)));

	// Compute statistics for each bin
	for (int i = 0; i < parameters.nb_states; ++i) {
		span id(t_sep(i), t_sep(i + 1) - 1);

		model.priors(i) = id.b - id.a + 1;

		// Mean computed on SPD manifold for parts of the data belonging to the
		// manifold
		vec mu_man = mean(data(span::all, id), 1);
		mu_man(spd_data_id) = symMat2vec(spdMean(vec2symMat(mat(data(spd_data_id, id))), 3));

		// Parts of data belonging to SPD manifold projected to tangent space at
		// the mean to compute the covariance tensor in the tangent space
		mat data_tgt = data(span::all, id);
		data_tgt(spd_data_id, span::all) = logmap_vec(mat(data(spd_data_id, id)), mu_man(spd_data_id));

		mat sigma = cov(data_tgt.t()) + arma::eye(nb_var,nb_var) * parameters.diag_reg_fact;

		model.mu_man.push_back(mu_man);
		model.sigma.push_back(sigma);
	}

	model.priors /= sum(model.priors);
}


/********************************* FUNCTIONS *********************************/

void process(const parameters_t& parameters, cube &X, model_t &model) {
	//_____ Generate covariance data from rotating covariance __________

	mat v_data = eye(parameters.nb_var - 1, parameters.nb_var - 1);
	mat d_data = eye(parameters.nb_var - 1, parameters.nb_var - 1);

	mat x = zeros(parameters.nb_var_vec, parameters.nb_data);

	X = zeros(parameters.nb_var, parameters.nb_var, parameters.nb_data);
	X.tube(0, 0) = linspace<vec>(
		1, parameters.nb_data, parameters.nb_data
	) * parameters.dt;

	for (int t = 1; t <= parameters.nb_data; ++t) {
		d_data(0, 0) = parameters.dt * t;

		mat R(2, 2);
		double a = datum::pi / 2.0 * t * parameters.dt;
		R(0, 0) = cos(a);
		R(0, 1) = -sin(a);
		R(1, 0) = sin(a);
		R(1, 1) = cos(a);

		mat V = R * v_data;
		X.subcube(1, 1, t-1, 2, 2, t-1) = V * d_data * V.t();

		x.submat(0, t-1, 0, t-1) = mat(X.subcube(0,0,t-1,0,0,t-1));
		x.submat(1, t-1, x.n_rows-1, t-1) = symMat2vec(mat(X.subcube(1, 1, t-1, 2, 2, t-1)));
	}

	//_____ GMM parameters estimation __________

	// Initialisation on the manifold
	span in(0);
	span out(1, parameters.nb_var_vec - 1);
	span out_mat(1, parameters.nb_var - 1);

	spd_init_GMM_kbins(x, out, parameters, model);

	for (size_t i = 0; i < model.mu_man.size(); ++i)
		model.mu.push_back(zeros(size(model.mu_man[i])));

	mat L = zeros(parameters.nb_states, parameters.nb_data);
	for (int nb = 0; nb < parameters.nb_iter_EM; ++nb) {
		// E-step
		for (int i = 0; i < parameters.nb_states; ++i) {
			mat xts = zeros(parameters.nb_var_vec, parameters.nb_data);
			mat mu_man = vec(model.mu_man[i](in));
			for (int j = 0; j < parameters.nb_data; ++j)
				xts(in, span(j)) = x(in, span(j)) - mu_man;

			xts(out, span::all) = logmap_vec(mat(x(out, span::all)), model.mu_man[i](out));

			L(i, span::all) = trans(model.priors(i) * mvn::getPDFValue(model.mu[i],
													   model.sigma[i], xts));
		}

		mat GAMMA = L / repmat(sum(L, 0) + DBL_MIN, parameters.nb_states, 1);
		mat H = GAMMA / repmat(sum(GAMMA, 1) + DBL_MIN, 1, parameters.nb_data);

		// M-step
		for (int i = 0; i < parameters.nb_states; ++i) {
			// Update priors
			model.priors(i) = mat(sum(GAMMA(span(i), span::all), 1))(0, 0) / parameters.nb_data;

			// Update mu_man
			mat uTmp = zeros(parameters.nb_var_vec, parameters.nb_data);
			for (int n = 0; n < parameters.nb_iter; ++n) {
				vec mu_man = vec(model.mu_man[i](in));
				for (int j = 0; j < parameters.nb_data; ++j)
					uTmp(in, span(j)) = x(in, span(j)) - mu_man;

				uTmp(out, span::all) = logmap_vec(mat(x(out, span::all)), model.mu_man[i](out));

				vec uTmpTot = zeros(parameters.nb_var_vec);

				for (int k = 0; k < parameters.nb_data; ++k)
					uTmpTot = uTmpTot + mat(uTmp(span::all, span(k))) * H(i, k);

				model.mu_man[i](in) = uTmpTot(in) + model.mu_man[i](in);
				model.mu_man[i](out) = expmap_vec(uTmpTot(out), model.mu_man[i](out));
			}

			// Update Sigma
			mat sigma(parameters.nb_var_vec, parameters.nb_var_vec);
			sigma = uTmp * diagmat(H(i,span::all)) * uTmp.t();
			model.sigma[i] = sigma + eye(parameters.nb_var_vec,parameters.nb_var_vec) * parameters.diag_reg_fact;
		}
	}
	// Eigendecomposition of sigma
	std::vector<mat> V;
	std::vector<vec> D;

	for (int i = 0; i < parameters.nb_states; ++i) {
		cx_mat V_;
		cx_vec D_;

		eig_gen(D_, V_, model.sigma[i]);

		V.push_back(real(V_));
		D.push_back(real(D_));
	}

	//_____ GMR (version with single optimization loop) __________

	vec xIn = linspace<vec>(
		1, parameters.nb_data, parameters.nb_data
	) * parameters.dt;

	int nb_var_out = out.b - out.a + 1;
	//int nb_var_out_vec = parameters.nb_var + parameters.nb_var * (parameters.nb_var - 1) / 2;

	model.xhat.clear();
	model.H.clear();
	for (unsigned int t = 0; t < parameters.nb_data; ++t) {
		// Compute activation weight
		vec H_(parameters.nb_states);

		for (unsigned int i = 0; i < parameters.nb_states; ++i) {
			H_(i) = model.priors(i) * mvn::getPDFValue(model.mu[i](in),
													   model.sigma[i](in, in),
													   xIn(t) - model.mu_man[i](in))(0);
		}
		H_ = H_ / sum(H_ + DBL_MIN);

		model.H.push_back(H_);


		// Compute conditional mean (with covariance transportation)
		mat xhat;

		if (t == 0) {
			uword id = index_max(H_);
			xhat = model.mu_man[id](out);	// Initial point
		}
		else {
			xhat = model.xhat[t - 1];
		}

		std::vector<mat> pSigma;

		for (int n = 0; n < parameters.nb_iter; ++n) {
			vec uhat = zeros(nb_var_out);

			for (int i = 0; i < parameters.nb_states; ++i) {
				// Transportation of covariance from model.mu_man to xhat
				mat Ac = zeros(parameters.nb_var, parameters.nb_var);
				Ac(0, 0) = 1.0;
				Ac(out_mat, out_mat) = transp(vec2symMat(vec(model.mu_man[i](out))), vec2symMat(xhat));

				// Parallel transport of eigenvectors
				mat pV(V[i].n_rows, V[i].n_cols);

				for (int j = 0; j < V[i].n_cols; ++j) {
					double sqrt_d = sqrt(D[i](j));
					if (std::isnan(sqrt_d))
						sqrt_d = 0.0;
					mat M_tmp = zeros(parameters.nb_var, parameters.nb_var);
					M_tmp.submat(0,0,0,0) = V[i](in, span(j));
					M_tmp.submat(1, 1, parameters.nb_var-1, parameters.nb_var-1) = vec2symMat(vec(V[i](out, span(j))));

					mat pM_tmp = real(Ac * sqrt_d * M_tmp * Ac.t());

					pV(in, span(j)) = pM_tmp(0, 0);
					pV(out, span(j)) = symMat2vec(pM_tmp.submat(1, 1, parameters.nb_var-1, parameters.nb_var -1));
				}

				// Parallel transported sigma (reconstruction from eigenvectors)
				mat pSigma(model.sigma[i].n_rows, model.sigma[i].n_cols);
				pSigma = pV * pV.t();

				// Gaussian conditioning on the tangent space
				int dim = out.b - out.a + 1;

				mat uOut = logmap_vec(vec(model.mu_man[i](out)), xhat) +
						   pSigma(out, in) * inv(pSigma(in, in)) * (xIn(t) - model.mu_man[i](in));

				uhat = uhat + uOut * H_(i);
			}

			xhat = expmap_vec(uhat, xhat);
		}

		model.xhat.push_back(xhat);
	}
}

//-----------------------------------------------

int main(int argc, char **argv) {
	arma::arma_version ver;

	arma_rng::set_seed_random();


	// Parameters
	parameters_t parameters;
	parameters.nb_states = 10;
	parameters.nb_data = 50;
	parameters.nb_var = 3;
	parameters.nb_var_vec = 4;
	parameters.dt = 0.1;
	parameters.nb_iter_EM = 5;
	parameters.nb_iter = 5;
	parameters.diag_reg_fact = 1e-4f;


	// Initial computation
	cube X;
	model_t model;
	process(parameters, X, model);

	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 1200;
	window_size.win_height = 400;
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
		"Demo Gaussian Mixture Regression",
		window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	float workspace_size = (parameters.nb_data + 1) * parameters.dt;

	// Setup OpenGL
	gfx2::init();
	glEnable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Setup ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);


	// Main loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;


		// Detect if the parameters have changed
		if ((parameters.nb_data != X.n_slices) || (parameters.nb_states != model.mu_man.size())) {
			parameters.nb_states = std::min(parameters.nb_states, parameters.nb_data / 2);

			process(parameters, X, model);

			workspace_size = (parameters.nb_data + 1) * parameters.dt;
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();
		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_BLEND);

		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		glOrtho(-window_size.fb_width / 2, window_size.fb_width / 2,
				-window_size.fb_height / 2, window_size.fb_height / 2, -1., 1.);
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		// Draw the datapoints
		for (int i = 0; i < parameters.nb_data; ++i) {
			gfx2::draw_gaussian(
				fvec({ 0.5f, 0.5f, 0.5f, 0.25f }),
				vec({ X(0, 0, i) * window_size.fb_width / workspace_size - window_size.fb_width / 2,
					  ((double) window_size.win_height / 2 - 140.0) * window_size.scale_y()
				}),
				X.subcube(1, 1, i, 2, 2, i) * window_size.fb_width / workspace_size
			);
		}

		// Draw the GMM states
		for (int i = 0; i < parameters.nb_states; ++i) {
			fvec color = COLORS.row(i).t();

			gfx2::draw_gaussian(
				fvec({ color(0), color(1), color(2), 0.5f }),
				vec({ model.mu_man[i](0) * window_size.fb_width / workspace_size - window_size.fb_width / 2,
					  ((double) window_size.win_height / 2 - 140.0) * window_size.scale_y()
				}),
				vec2symMat(vec(model.mu_man[i].subvec(1, 3))) * window_size.fb_width / workspace_size
			);
		}

		// Draw the GMR results
		vec xIn = linspace<vec>(
			1, parameters.nb_data, parameters.nb_data
		) * parameters.dt;

		for (int t = 0; t < parameters.nb_data; ++t) {
			gfx2::draw_gaussian(
				fvec({ 0.0f, 1.0f, 0.0f, 0.5f }),
				vec({ xIn[t] * window_size.fb_width / workspace_size - window_size.fb_width / 2,
					  ((double) window_size.win_height / 2 - 240.0) * window_size.scale_y()
				}),
				vec2symMat(model.xhat[t]) * window_size.fb_width / workspace_size
			);
		}

		glDisable(GL_BLEND);

		// Draw the lines
		double plot_height = ((double) window_size.win_height - 290.0 - 100) * window_size.scale_y();

		for (int i = 0; i < parameters.nb_states; ++i) {
			mat points(2, parameters.nb_data);

			points(span(0), span::all) = xIn.t() * window_size.fb_width / workspace_size - window_size.fb_width / 2;

			for (int t = 0; t < parameters.nb_data; ++t) {
				points(1, t) = model.H[t](i) * plot_height + (50 - window_size.win_height / 2) * window_size.scale_y();
			}

			gfx2::draw_line(fvec(COLORS.row(i).t()), points);
		}


		// Parameter window
		ImGui::SetNextWindowSize(ImVec2(400, 80));
		ImGui::Begin("Parameters", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove
		);
		ImGui::SliderInt("Nb datapoints", &parameters.nb_data, 10, 100);
		ImGui::SliderInt("Nb GMM states", &parameters.nb_states, 3, std::min(20, parameters.nb_data / 2));
		ImGui::End();


		// GUI rendering
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		// Swap buffers
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
