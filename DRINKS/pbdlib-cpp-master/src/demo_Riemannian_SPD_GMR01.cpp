/*
 * demo_Riemannian_SPD_GMR01.cpp
 *
 * GMR with time as input and covariance data as output by relying on Riemannian manifold.
 *
 * If this code is useful for your research, please cite the related publication:
 * @article{Jaquier17IROS,
 *   author="Jaquier, N. and Calinon, S.",
 *   title="Gaussian Mixture Regression on Symmetric Positive Definite Matrices Manifolds:
 *   Application to Wrist Motion Estimation with s{EMG}",
 *   year="2017, submitted for publication",
 *   booktitle = "{IEEE/RSJ} Intl. Conf. on Intelligent Robots and Systems ({IROS})",
 *   address = "Vancouver, Canada"
 * }
 *
 * Authors: Sylvain Calinon, Philip Abbet
 */


#include <stdio.h>
#include <armadillo>

#include <tensor.h>
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

// 4-dimensional tensor of doubles
typedef TensorDouble Tensor4D;

//-----------------------------------------------

struct parameters_t {
	int	  nb_states;		// Number of components in the GMM
	int	  nb_var;			// Dimension of the manifold and tangent space (here:
							// 1D input + 2^2 output)
	int	  nb_data;			// Number of datapoints
	float dt;				// Time step duration
	int	  nb_iter_EM;		// Number of iterations for the EM algorithm
	int	  nb_iter;			// Number of iterations for the Gauss Newton algorithm
	float diag_reg_fact;	// Regularization term to avoid numerical instability
};

//-----------------------------------------------

struct model_t {
	// These lists contain one element per GMM state
	vec						priors;
	std::vector<mat>		mu_man;
	std::vector<mat>		mu;
	std::vector<Tensor4D>	sigma;
	std::vector<mat>		xhat;
	std::vector<vec>		H;
};


//-----------------------------------------------------------------------------
// Computation of the outer product between two complex matrices
//-----------------------------------------------------------------------------
template <typename type>
Tensor4D outerprod(const Mat<type>& U, const Mat<type>& V) {
	Tensor4D T(U.n_rows, U.n_cols, V.n_rows, V.n_cols);

	T.data = real(vectorise(vectorise(U) * strans(vectorise(V))));

	return T;
}


//-----------------------------------------------------------------------------
// Set the factors necessary to simplify a 4th-order covariance of symmetric
// matrix to a 2nd-order covariance. The dimension ofthe 4th-order
// covariance is dim x dim x dim x dim.
//-----------------------------------------------------------------------------
class CovOrder {

public:
	CovOrder(int dim)
	: dim(dim)
	{
		new_dim = dim + dim * (dim - 1) / 2;

		// Computation of the indices and coefficients to transform 4th-order
		// covariances to 2nd-order covariances
		Tensor4D sizeS(dim, dim, dim, dim);
		SizeMat sizeSred = size(new_dim, new_dim);

		//---- left-up part

		id.insert_rows(0, dim * dim);
		id_red.insert_rows(0, dim * dim);
		coeffs.insert_rows(0, dim * dim);

		int offset = 0;
		for (int k = 0; k < dim; ++k) {
			for (int m = 0; m < dim; ++m) {
				id(offset) = sizeS.indice(k, k, m, m);
				id_red(offset) = sub2ind(sizeSred, k, m);
				coeffs(offset) = 1.0;
				++offset;
			}
		}

		//---- right-down part

		int row = dim;
		int col = dim;

		for (int k = 0; k < dim - 1; ++k) {
			for (int m = k + 1; m < dim; ++m) {
				for (int p = 0; p < dim - 1; ++p) {
					id.insert_rows(offset, dim - (p + 1));
					id_red.insert_rows(offset, dim - (p + 1));
					coeffs.insert_rows(offset, dim - (p + 1));

					for (int q = p + 1; q < dim; ++q) {
						id(offset) = sizeS.indice({k, m, p, q});
						id_red(offset) = sub2ind(sizeSred, row, col);
						coeffs(offset) = 2.0;
						++offset;
						++col;
					}
				}

				++row;
				col = dim;
			}
		}

		//---- side-parts

		for (int k = 0; k < dim; ++k) {
			col = dim;

			for (int p = 0; p < dim - 1; ++p) {
				id.insert_rows(offset, 2 * (dim - (p + 1)));
				id_red.insert_rows(offset, 2 * (dim - (p + 1)));
				coeffs.insert_rows(offset, 2 * (dim - (p + 1)));

				for (int q = p + 1; q < dim; ++q) {
					id(offset) = sizeS.indice({k, k, p, q});
					id(offset + 1) = sizeS.indice({k, k, p, q});

					id_red(offset) = sub2ind(sizeSred, k, col);
					id_red(offset + 1) = sub2ind(sizeSred, col, k);

					coeffs(offset) = sqrt(2.0);
					coeffs(offset+1) = sqrt(2.0);

					offset += 2;
					++col;
				}
			}
		}

		// Computation of the indices and coefficients to transform eigenvectors
		// to eigentensors
		SizeCube sizeV = size(dim, dim, new_dim);
		SizeMat sizeVred = size(new_dim, new_dim);

		offset = 0;
		for (int n = 0; n < new_dim; ++n) {

			id_eigen.insert_rows(offset, dim);
			id_red_eigen.insert_rows(offset, dim);
			coeffs_eigen.insert_rows(offset, dim);

			// diagonal part
			for (int j = 0; j < dim; ++j) {
				id_eigen(offset) = sub2ind(sizeV, j, j, n);
				id_red_eigen(offset) = sub2ind(sizeVred, j, n);
				coeffs_eigen(offset) = 1.0;
				++offset;
			}

			// side part
			int j = dim;
			for (int k = 0; k < dim - 1; ++k) {
				id_eigen.insert_rows(offset, 2 * (dim - (k + 1)));
				id_red_eigen.insert_rows(offset, 2 * (dim - (k + 1)));
				coeffs_eigen.insert_rows(offset, 2 * (dim - (k + 1)));

				for (int m = k + 1; m < dim; ++m) {
					id_eigen(offset) = sub2ind(sizeV, k, m, n);
					id_eigen(offset + 1) = sub2ind(sizeV, m, k, n);

					id_red_eigen(offset) = sub2ind(sizeVred, j, n);
					id_red_eigen(offset + 1) = sub2ind(sizeVred, j, n);

					coeffs_eigen(offset) = 1.0 / sqrt(2.0);
					coeffs_eigen(offset + 1) = 1.0 / sqrt(2.0);

					++j;
					offset += 2;
				}
			}
		}
	}


	void conv4to2(const Tensor4D& S, mat &Sred, cx_cube &V, cx_vec &D) const {
		Sred = zeros(new_dim, new_dim);
		Sred(id_red) = S(id) % coeffs;

		cx_mat v;
		eig_gen(D, v, Sred);

		V.set_size(dim, dim, new_dim);
		V.zeros();
		V(id_eigen) = v(id_red_eigen) % coeffs_eigen;
	}


	void conv2to4(const mat& Sred, Tensor4D &S, cx_cube &V, cx_vec &D) const {
		cx_mat v;
		eig_gen(D, v, Sred);

		V.set_size(dim, dim, new_dim);
		V(id_eigen) = v(id_red_eigen) % coeffs_eigen;

		S = Tensor4D(dim, dim, dim, dim);

		cx_vec S_complex(S.size, fill::zeros);
		for (int i = 0; i < new_dim; ++i)
			S_complex = S_complex + D(i) * outerprod(V.slice(i), V.slice(i)).data;

		S = vec(real(S_complex));
	}

private:
	int	  dim;
	int	  new_dim;
	uvec  id;
	uvec  id_red;
	vec	  coeffs;
	uvec  id_eigen;
	uvec  id_red_eigen;
	vec	  coeffs_eigen;
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


//-----------------------------------------------------------------------------
// Creates a MxN matrix from a MxNx1 cube
//-----------------------------------------------------------------------------
template <typename T>
Mat<T> cube2mat(const Cube<T>& c) {
	assert(c.n_slices == 1);
	return Mat<T>(c.memptr(), c.n_rows, c.n_cols);
}


//-----------------------------------------------------------------------------
// Reduced vectorisation of a symmetric matrix
//-----------------------------------------------------------------------------
template <typename T>
Col<T> sym_mat_to_vec(const Mat<T>& S) {

	Col<T> v = zeros<Col<T> >(S.n_rows + S.n_rows * (S.n_rows - 1) / 2);

	v.subvec(0, S.n_rows - 1) = S.diag();

	int row = S.n_rows;
	for (int i = 0; i < S.n_rows - 1; ++i) {
		v.subvec(row, row + S.n_rows - 2 - i) = sqrtf(2.0f) *
												S.submat(i+1, i, S.n_rows - 1, i);
		row = row + S.n_rows - 1 - i;
	}

	return v;
}


//-----------------------------------------------------------------------------
// Return locations of non-zero elements in a block diagonal matrix
//-----------------------------------------------------------------------------
uvec find_non_zero_indices(const span& in, const span& out) {
	int in_size = in.b - in.a + 1;
	int out_size = out.b - out.a + 1;

	fmat number_mat = zeros<fmat>(out_size + 1, out_size + 1);

	number_mat.submat(in, in) = ones<fmat>(in_size, in_size);
	number_mat.submat(out, out) = ones<fmat>(out_size, out_size);

	fvec sym_number_vec = sym_mat_to_vec(number_mat);

	uvec number_vec = zeros<uvec>(sym_number_vec.n_elem);
	number_vec(find(sym_number_vec)).ones();

	number_vec = number_vec % linspace<uvec>(1, number_vec.n_elem, number_vec.n_elem);

	return find(number_vec);
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
// Parallel transport (SPD manifold)
//
// Transportation operator Ac to move V from S1 to S2: VT = Ac * V * Ac'
//-----------------------------------------------------------------------------
mat transp(const mat& S1, const mat& S2) {
	mat U = cube2mat(logmap(mat2cube(S2), S1));
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
// Compute the 4th-order covariance of matrix data
//-----------------------------------------------------------------------------
Tensor4D compute_cov(const cube& data) {
	const int d = data.n_rows;
	const int N = data.n_slices;

	cube data_mean = mean(data, 2);

	cube data2(size(data));

	for (int n = 0; n < N; ++n) {
		data2(span::all, span::all, span(n)) = data(span::all, span::all, span(n)) - data_mean;
	}

	Tensor4D S(d, d, d, d);
	for (int n = 0; n < N; ++n) {
		mat m = mat(data2(span::all, span::all, span(n)));
		S.data = S.data + outerprod(m, m).data;
	}

	S.data /= (N - 1);

	return S;
}


//-------------------------------------------------------------------------
// Matricisation of a tensor
//
// The rows, respectively columns of the matrix are 'rows', respectively
// 'cols' of the tensor
//-------------------------------------------------------------------------
mat tensor2mat(const Tensor4D& T) {

	const uvec rows = { 0, 1 };
	const uvec cols = { 2, 3 };

	return mat(T.data.memptr(), prod(T.dims(rows)), prod(T.dims(cols)));
}


//-----------------------------------------------------------------------------
// K-Bins initialisation by relying on SPD manifold
//-----------------------------------------------------------------------------
void spd_init_GMM_kbins(const cube& data, const span& spd_data_id,
						const parameters_t& parameters, model_t &model) {
	// Parameters
	int nb_data = data.n_slices;

	model.priors = vec(parameters.nb_states);
	model.mu_man.clear();
	model.sigma.clear();

	Tensor4D e0tensor(parameters.nb_var, parameters.nb_var, parameters.nb_var, parameters.nb_var);
	for (int i = 0; i < parameters.nb_var; ++i)
		e0tensor.set(e0tensor.indices(i, i, i, i), 1.0);

	// Delimit the cluster bins
	uvec t_sep = conv_to<uvec>::from(round(linspace<vec>(0, parameters.nb_data, parameters.nb_states + 1)));

	// Compute statistics for each bin
	for (int i = 0; i < parameters.nb_states; ++i) {
		span id(t_sep(i), t_sep(i + 1) - 1);

		model.priors(i) = id.b - id.a + 1;

		// Mean computed on SPD manifold for parts of the data belonging to the
		// manifold
		mat mu_man = mean(data(span::all, span::all, id), 2);

		mu_man(spd_data_id, spd_data_id) = spdMean(data(spd_data_id, spd_data_id, id), 3);

		// Parts of data belonging to SPD manifold projected to tangent space at
		// the mean to compute the covariance tensor in the tangent space
		cube data_tgt = data(span::all, span::all, id);

		data_tgt(spd_data_id, spd_data_id, span::all) =
			logmap(cube(data(spd_data_id, spd_data_id, id)),
				   mat(mu_man(spd_data_id, spd_data_id))
		);

		Tensor4D sigma = compute_cov(data_tgt) + e0tensor * parameters.diag_reg_fact;

		model.mu_man.push_back(mu_man);
		model.sigma.push_back(sigma);
	}

	model.priors /= sum(model.priors);
}


/********************************* FUNCTIONS *********************************/

void process(const parameters_t& parameters, cube &X, model_t &model) {

	CovOrder cov_order(parameters.nb_var);

	//_____ Generate covariance data from rotating covariance __________

	mat v_data = eye(parameters.nb_var - 1, parameters.nb_var - 1);
	mat d_data = eye(parameters.nb_var - 1, parameters.nb_var - 1);

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
	}


	//_____ GMM parameters estimation __________

	// Initialisation on the manifold
	span in(0);
	span out(1, parameters.nb_var - 1);
	uvec id = find_non_zero_indices(in, out);

	spd_init_GMM_kbins(X, out, parameters, model);

	for (size_t i = 0; i < model.mu_man.size(); ++i)
		model.mu.push_back(zeros(size(model.mu_man[i])));

	mat L = zeros(parameters.nb_states, parameters.nb_data);

	for (int nb = 0; nb < parameters.nb_iter_EM; ++nb) {
		// E-step
		for (int i = 0; i < parameters.nb_states; ++i) {
			cube Xts = zeros(parameters.nb_var, parameters.nb_var, parameters.nb_data);

			cube mu_man = mat2cube(mat(model.mu_man[i](in, in)));
			for (int j = 0; j < parameters.nb_data; ++j)
				Xts(in, in, span(j)) = X(in, in, span(j)) - mu_man;

			Xts(out, out, span::all) = logmap(cube(X(out, out, span::all)), model.mu_man[i](out, out));

			// Compute probabilities using the reduced form (computationally less
			// expensive than complete form)
			mat sym_xts;

			for (int j = 0; j < parameters.nb_data; ++j) {
				vec v = sym_mat_to_vec(cube2mat(cube(Xts(span::all, span::all, span(j)))));

				if (sym_xts.n_rows == 0)
					sym_xts.set_size(v.n_rows, parameters.nb_data);

				sym_xts(span::all, span(j)) = v;
			}

			vec mu_vec = sym_mat_to_vec(model.mu[i]);

			mat sigma_vec;
			cx_cube V;
			cx_vec D;

			cov_order.conv4to2(model.sigma[i], sigma_vec, V, D);

			L(i, span::all) = trans(model.priors(i) *
							  mvn::getPDFValue(vec(mu_vec(id)), sigma_vec(id, id), sym_xts.rows(id)));
		}

		mat GAMMA = L / repmat(sum(L, 0) + DBL_MIN, parameters.nb_states, 1);
		mat H = GAMMA / repmat(sum(GAMMA, 1) + DBL_MIN, 1, parameters.nb_data);

		// M-step
		for (int i = 0; i < parameters.nb_states; ++i) {
			// Update priors
			model.priors(i) = mat(sum(GAMMA(span(i), span::all), 1))(0, 0) / parameters.nb_data;

			// Update mu_man
			cube uTmp = zeros(parameters.nb_var, parameters.nb_var, parameters.nb_data);
			for (int n = 0; n < parameters.nb_iter; ++n) {
				cube mu_man = mat2cube(mat(model.mu_man[i](in, in)));
				for (int j = 0; j < parameters.nb_data; ++j)
					uTmp(in, in, span(j)) = X(in, in, span(j)) - mu_man;

				uTmp(out, out, span::all) = logmap(cube(X(out, out, span::all)), model.mu_man[i](out, out));

				mat uTmpTot = zeros(parameters.nb_var, parameters.nb_var);

				for (int k = 0; k < parameters.nb_data; ++k)
					uTmpTot = uTmpTot + cube2mat(cube(uTmp(span::all, span::all, span(k)))) * H(i, k);

				model.mu_man[i](in, in) = uTmpTot(in, in) + model.mu_man[i](in, in);
				model.mu_man[i](out, out) = expmap(uTmpTot(out, out), model.mu_man[i](out, out));
			}

			// Update Sigma
			Tensor4D sigma(parameters.nb_var, parameters.nb_var, parameters.nb_var, parameters.nb_var);
			for (int k = 0; k < parameters.nb_data; ++k) {
				sigma = sigma + outerprod(mat(uTmp(span::all, span::all, span(k))),
										  mat(uTmp(span::all, span::all, span(k)))
								) * H(i, k);
			}

			model.sigma[i] = sigma + parameters.diag_reg_fact * parameters.diag_reg_fact;
		}
	}

	// Eigendecomposition of sigma
	std::vector<cube> V;
	std::vector<vec> D;

	for (int i = 0; i < parameters.nb_states; ++i) {
		mat Sred;
		cx_cube V_;
		cx_vec D_;

		cov_order.conv4to2(model.sigma[i], Sred, V_, D_);
		V.push_back(real(V_));
		D.push_back(real(D_));
	}


	//_____ GMR (version with single optimization loop) __________

	vec xIn = linspace<vec>(
		1, parameters.nb_data, parameters.nb_data
	) * parameters.dt;

	int nb_var_out = out.b - out.a + 1;
	int nb_var_cov_out = parameters.nb_var + parameters.nb_var * (parameters.nb_var - 1) / 2;

	model.xhat.clear();
	model.H.clear();

	for (unsigned int t = 0; t < parameters.nb_data; ++t) {
		// Compute activation weight
		vec H_(parameters.nb_states);

		for (unsigned int i = 0; i < parameters.nb_states; ++i) {
			H_(i) = model.priors(i) * mvn::getPDFValue(model.mu[i](in, in),
													   model.sigma[i](in, in, in, in).data,
													   xIn(t) - model.mu_man[i](in, in))(0);
		}
		H_ = H_ / sum(H_ + DBL_MIN);

		model.H.push_back(H_);


		// Compute conditional mean (with covariance transportation)
		mat xhat;

		if (t == 0) {
			uword id = index_max(H_);
			xhat = model.mu_man[id](out, out);	 // Initial point
		}
		else {
			xhat = model.xhat[t - 1];
		}

		std::vector<Tensor4D> pSigma;

		for (int n = 0; n < parameters.nb_iter; ++n) {
			mat uhat = zeros(nb_var_out, nb_var_out);

			for (int i = 0; i < parameters.nb_states; ++i) {
				// Transportation of covariance from model.mu_man to xhat
				mat Ac = zeros(parameters.nb_var, parameters.nb_var);
				Ac(0, 0) = 1.0;
				Ac(out, out) = transp(model.mu_man[i](out, out), xhat);

				// Parallel transport of eigenvectors
				cube pV(V[i].n_rows, V[i].n_cols, V[i].n_slices);

				for (int j = 0; j < V[i].n_slices; ++j) {
					double sqrt_d = sqrt(D[i](j));
					if (std::isnan(sqrt_d))
						sqrt_d = 0.0;

					pV(span::all, span::all, span(j)) =
						real(Ac * sqrt_d * mat(V[i](span::all, span::all, span(j))) * Ac.t());
				}

				// Parallel transported sigma (reconstruction from eigenvectors)
				Tensor4D pSigma(model.sigma[i].dims);
				for (int j = 0; j < pV.n_slices; ++j) {
					mat pV2 = pV(span::all, span::all, span(j));
					pSigma = pSigma + outerprod(pV2, pV2);
				}

				// Gaussian conditioning on the tangent space
				int dim = out.b - out.a + 1;

				mat uOut = logmap(model.mu_man[i](out, out), xhat(span::all, span::all)) +
						   mat(reshape(tensor2mat(pSigma(out, out, in, in) / pSigma(in, in, in, in)[0]) *
									   (xIn(t) - model.mu_man[i](in, in))[0], dim, dim));

				uhat = uhat + uOut * H_(i);
			}

			xhat = expmap(uhat, xhat);
		}

		model.xhat.push_back(xhat);
	}
}

//-----------------------------------------------

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	// Parameters
	parameters_t parameters;
	parameters.nb_states = 10;
	parameters.nb_data = 50;
	parameters.nb_var = 3;
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
				vec({ model.mu_man[i](0, 0) * window_size.fb_width / workspace_size - window_size.fb_width / 2,
					  ((double) window_size.win_height / 2 - 140.0) * window_size.scale_y()
				}),
				model.mu_man[i].submat(1, 1, 2, 2) * window_size.fb_width / workspace_size
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
				model.xhat[t] * window_size.fb_width / workspace_size
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
