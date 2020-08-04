/*
 * demo_ergodicControl_nD01.cpp
 *
 * nD ergodic control, inspired by G. Mathew and I. Mezic, "Spectral Multiscale  
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
#include <tensor.h>

#include <gfx2.h>
#include <gfx_ui.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>
#include <ImGuizmo.h>
#include <GLFW/glfw3.h>

using namespace arma;


/***************************** ALGORITHM SECTION *****************************/

typedef std::vector<vec> vector_list_t;
typedef std::vector<mat> matrix_list_t;
typedef std::vector<TensorUInt> tensor_list_t;


//-----------------------------------------------------------------------------
// Contains all the parameters used by the algorithm. Some of them are
// modifiable through the UI, others are hard-coded.
//-----------------------------------------------------------------------------
struct parameters_t {
	int   nb_gaussians; // Number of gaussians that control the trajectory
	int   nb_data;      // Number of datapoints
	int   nb_fct;    	// Number of basis functions along each dimension
	int   nb_var;    	// Dimension of datapoint
	float dt;           // Time step
};


//-----------------------------------------------------------------------------
// Represents a 3D gaussian
//-----------------------------------------------------------------------------
struct gaussian_t {
	vec mu;
	mat sigma;
	
	fmat transforms;	// Used to display the 3D widget
};


//-----------------------------------------------------------------------------
// Represents a list of 3D gaussians
//-----------------------------------------------------------------------------
typedef std::vector<gaussian_t> gaussian_list_t;


//-----------------------------------------------------------------------------
// Given a vector representing indices in a N-D tensor (with all dimensions
// having the same number of elements, <size>), update it to contain the next
// indices (assuming an iteration over all possible combinations). If
// <fixed_dim> contains a valid dimension number, that dimension isn't
// incremented.
//
// Returns false when the iteration is done.
//-----------------------------------------------------------------------------
bool increment_subscripts(ivec &subscripts, int size, int fixed_dim) {
	int dim = 0;

	while (dim < subscripts.n_elem) {
		if (dim == fixed_dim) {
			++dim;
			continue;
		}

		subscripts(dim) += 1;
		if (subscripts(dim) >= size) {
			subscripts(dim) = 0;
			++dim;
			continue;
		}
		
		return true;
	}

	return false;
}


//-----------------------------------------------------------------------------
// Returns a list of <nb_var> <nb_var>-D tensors, each dimension having a size
// of <lst.n_elem>.
//
// The values of <lst> are used to fill the tensors, like in the following
// example.
//
// With nb_var = 3 and lst = [0..2]:
//
//   result[0]:
//     slice 0 = [        slice 1 = [        slice 2 = [
//         0 0 0              0 0 0              0 0 0
//         1 1 1              1 1 1              1 1 1
//         2 2 2              2 2 2              2 2 2
//     ]                  ]                  ]
//
//   result[1]:
//     slice 0 = [        slice 1 = [        slice 2 = [
//         0 1 2              0 1 2              0 1 2
//         0 1 2              0 1 2              0 1 2
//         0 1 2              0 1 2              0 1 2
//     ]                  ]                  ]
//
//   result[2]:
//     slice 0 = [        slice 1 = [        slice 2 = [
//         0 0 0              1 1 1              2 2 2
//         0 0 0              1 1 1              2 2 2
//         0 0 0              1 1 1              2 2 2
//     ]                  ]                  ]
//
//-----------------------------------------------------------------------------
tensor_list_t ndarr(const uvec& lst, int nb_var) {
	tensor_list_t result;

	ivec dims = ones<ivec>(nb_var) * lst.n_elem;
	
	for (int n = 0; n < nb_var; ++n) {
		TensorUInt tensor(dims);

		for (int i = 0; i < lst.n_elem; ++i) {
		
			ivec subscripts = zeros<ivec>(nb_var);
			subscripts(n) = i;

			do {
				tensor.set(tensor.indice(subscripts), lst(i));
			} while (increment_subscripts(subscripts, lst.n_elem, n));
		}

		result.push_back(tensor);
	}

	return result;
}


//-----------------------------------------------------------------------------
// Returns an N-by-N Hadamard matrix (where N is the form 2^k)
//-----------------------------------------------------------------------------
mat hadamard(int N) {
	const mat H2({
		{ 1, 1 },
		{ 1, -1 }
	});

	if (N == 2)
		return H2;

	return kron(H2, hadamard(N / 2));
}


//-----------------------------------------------------------------------------
// Implementation of the algorithm
//-----------------------------------------------------------------------------
mat compute(const parameters_t& parameters, const mat& mu, const cube& sigma) {

	const float sp = ((float) parameters.nb_var + 1.0f) / 2.0f; // Sobolev norm parameter

	const float xsize = 1.0;   // Domain size

	// Initial position
	vec x = zeros(parameters.nb_var);
	x(span(0, 2)) = vec({ 0.7f, 0.1f, 0.5f });


	// Basis functions (Fourier coefficients of coverage distribution)
	//----------------------------------------------------------------
	tensor_list_t Karr = ndarr(
		linspace<uvec>(0, parameters.nb_fct - 1, parameters.nb_fct),
		parameters.nb_var
	);
	
	mat rg = repmat(
		linspace<rowvec>(0, parameters.nb_fct - 1, parameters.nb_fct),
		parameters.nb_var, 1
	);

	TensorDouble stmp(Karr[0].dims);
	for (int n = 0; n < parameters.nb_var; ++n)
		stmp.data += pow(conv_to<vec>::from(Karr[n].data), 2);

	// Weighting matrix
	TensorDouble LK(stmp.dims);
	LK.data = pow(stmp.data + 1, -sp);

	vec hk = join_vert(vec({1.0}), sqrt(0.5) * ones(parameters.nb_fct - 1));

	TensorDouble HK(Karr[0].dims);
	HK.data = ones(HK.size);
	for (int n = 0; n < parameters.nb_var; ++n)
		HK.data = HK.data % hk.elem(Karr[n].data);

	
	// Desired spatial distribution as mixture of Gaussians
	mat w(parameters.nb_var, Karr[0].size);
	for (int n = 0; n < parameters.nb_var; ++n)
		w(n, span::all) = conv_to<vec>::from(Karr[n].data).t() * datum::pi / xsize;


	// Enumerate symmetry operations for <nb_var>-D signal
	mat op = hadamard(pow(2, parameters.nb_var - 1));
	op = op(span(0, parameters.nb_var - 1), span::all);


	// Compute phi_k
	TensorDouble phi(HK.dims);
	phi.data = zeros(phi.size);
	for (int k = 0; k < mu.n_cols; ++k) {
		for (int n = 0; n < op.n_cols; ++n) {
			mat MuTmp = diagmat(op(span::all, n)) * mu(span::all, k);
			mat SigmaTmp = diagmat(op(span::all, n)) * sigma.slice(k) * diagmat(op(span::all, n)).t();
			phi.data = phi.data + cos(w.t() * MuTmp) % diagvec(exp(-0.5 * w.t() * SigmaTmp * w));
		}
	}
	phi.data = phi.data / HK.data / mu.n_cols / op.n_cols;


	// Ergodic control with spectral multiscale coverage (SMC) algorithm
	//------------------------------------------------------------------
	TensorDouble Ck(HK.dims);
	Ck.data = zeros(Ck.size);

	mat result = zeros(parameters.nb_var, parameters.nb_data);

	for (int t = 0; t < parameters.nb_data; ++t) {

		// Log data
		result(span::all, t) = x;

		// Updating Fourier coefficients of coverage distribution
		mat cx = cos(rg * datum::pi / xsize % repmat(x, 1, parameters.nb_fct));
		mat dcx = sin(rg * datum::pi / xsize % repmat(x, 1, parameters.nb_fct)) % rg * datum::pi / xsize;
		
		vec Mtmp = ones(pow(parameters.nb_fct, parameters.nb_var));
		for (int n = 0; n < parameters.nb_var; ++n)
			Mtmp = Mtmp % rowvec(cx.row(n)).elem(Karr[n].data);

		// Fourier coefficients along trajectory
		Ck.data = Ck.data + Mtmp / HK.data * parameters.dt;

		// SMC feedback control law
		vec dx(parameters.nb_var);

		for (int n = 0; n < parameters.nb_var; ++n) {
			Mtmp = ones(pow(parameters.nb_fct, parameters.nb_var));
			for (int m = 0; m < parameters.nb_var; ++m) {
				if (m == n)
					Mtmp = Mtmp % rowvec(dcx.row(m)).elem(Karr[m].data);
				else
					Mtmp = Mtmp % rowvec(cx.row(m)).elem(Karr[m].data);
			}

			Mtmp = LK.data / HK.data % (Ck.data - phi.data * t * parameters.dt) % Mtmp;
			dx(n) = sum(Mtmp);
		}

		x = x + dx * parameters.dt;
	}

	return result;
}


/****************************** HELPER FUNCTIONS *****************************/

static void error_callback(int error, const char* description){
	fprintf(stderr, "Error %d: %s\n", error, description);
}


//-----------------------------------------------------------------------------
// Colors of the displayed gaussians
//-----------------------------------------------------------------------------
const fmat COLORS({
	{ 0.0f,  0.0f,  1.0f  },
	{ 0.0f,  0.5f,  0.0f  },
	{ 1.0f,  0.0f,  0.0f  },
	{ 0.0f,  0.75f, 0.75f },
	{ 0.75f, 0.0f,  0.75f },
	{ 0.75f, 0.75f, 0.0f  },
});


//-----------------------------------------------


gaussian_list_t create_random_gaussians(
	const parameters_t& parameters, const gfx2::window_size_t& window_size) {

	gaussian_list_t gaussians;

	for (int i = 0; i < parameters.nb_gaussians; ++i) {
		gaussian_t gaussian;
		gaussian.mu = randu(3);

		fmat rotation = gfx2::rotate(fvec({ 0.0f, 0.0f, 1.0f }), randu() * 2 * datum::pi) *
						gfx2::rotate(fvec({ 1.0f, 0.0f, 0.0f }), randu() * 2 * datum::pi);

		mat RG = zeros(3, 3);
		RG(span::all, 0) = vec(rotation * vec({ randu() * 0.2 + 0.05, 0.0, 0.0, 0.0 })).rows(0, 2);
		RG(span::all, 1) = vec(rotation * vec({ 0.0, randu() * 0.2 + 0.05, 0.0, 0.0 })).rows(0, 2);
		RG(span::all, 2) = vec(rotation * vec({ 0.0, 0.0, randu() * 0.2 + 0.05, 0.0 })).rows(0, 2);

		gaussian.sigma = RG * RG.t();

		gaussian.transforms = eye<fmat>(4, 4);
		gaussian.transforms(span(0, 2), span(0, 2)) = conv_to<fmat>::from(RG);
		gaussian.transforms(span(0, 2), 3) = conv_to<fvec>::from(gaussian.mu);
	
		gaussians.push_back(gaussian);
	}

	return gaussians;
}


//-----------------------------------------------------------------------------
// Render the 3D scene
//-----------------------------------------------------------------------------
bool draw_scene(const gfx2::window_size_t& window_size, const fmat& projection,
				const gfx2::camera_t& camera, const mat& result, float t,
				gaussian_list_t &gaussians, int &current_gaussian,
				ImGuizmo::OPERATION transforms_operation) {

	glViewport(0, 0, window_size.fb_width, window_size.fb_height);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMultMatrixf(projection.memptr());

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	fmat view = gfx2::view_matrix(camera);
	glMultMatrixf(view.memptr());


	// Draw the axes
	gfx2::draw_line(
		fvec({ 1.0f, 0.0f, 0.0f }),
		mat({
			{ 0.0, 1.0 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.0f, 1.0f, 0.0f }),
		mat({
			{ 0.0, 0.0 },
			{ 0.0, 1.0 },
			{ 0.0, 0.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.0f, 0.0f, 1.0f }),
		mat({
			{ 0.0, 0.0 },
			{ 0.0, 0.0 },
			{ 0.0, 1.0 },
		})
	);


	// Draw the boundaries
	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 0.0, 1.0 },
			{ 0.0, 0.0 },
			{ 1.0, 1.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 0.0, 1.0 },
			{ 1.0, 1.0 },
			{ 0.0, 0.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 0.0, 1.0 },
			{ 1.0, 1.0 },
			{ 1.0, 1.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 1.0, 1.0 },
			{ 0.0, 1.0 },
			{ 0.0, 0.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 1.0, 1.0 },
			{ 0.0, 1.0 },
			{ 1.0, 1.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 0.0, 0.0 },
			{ 0.0, 1.0 },
			{ 1.0, 1.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 1.0, 1.0 },
			{ 0.0, 0.0 },
			{ 0.0, 1.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 0.0, 0.0 },
			{ 1.0, 1.0 },
			{ 0.0, 1.0 },
		})
	);

	gfx2::draw_line(
		fvec({ 0.8f, 0.8f, 0.8f }),
		mat({
			{ 1.0, 1.0 },
			{ 1.0, 1.0 },
			{ 0.0, 1.0 },
		})
	);


	// Draw the gaussians
	if (!gaussians.empty()) {
		for (int i = 0; i < gaussians.size(); ++i) {
			gfx2::draw_gaussian_3D(COLORS.row(i % COLORS.n_rows).t(), gaussians[i].mu,
				   	  gaussians[i].sigma);
		}
	}


	// Draw the results
	if (result.n_cols > 0) {
		glClear(GL_DEPTH_BUFFER_BIT);
		gfx2::draw_line(fvec({ 0.0f, 0.0f, 1.0f }), result(span(0, 2), span::all));

		int current_index = t * result.n_cols;

		if (current_index > 0) {
			glClear(GL_DEPTH_BUFFER_BIT);
			glLineWidth(4.0f);
			gfx2::draw_line(fvec({ 1.0f, 0.0f, 0.0f }), result(span(0, 2), span(0, current_index)));
			glLineWidth(1.0f);
		}
	}


	// Draw the gizmo allowing to manipulate the gaussians
	if (!gaussians.empty()) {

		// Draw a selection circle at the center of each gaussian
        ImGui::PushID("gaussians");
        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.0);
 
        ImGui::Begin("gaussians", 0, ImGuiWindowFlags_NoTitleBar |
        			 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        			 ImGuiWindowFlags_NoSavedSettings |
        			 ImGuiWindowFlags_NoBringToFrontOnFocus);
        
        ImGui::SetNextWindowBgAlpha(1.0); 

		for (int i = 0; i < gaussians.size(); ++i) {
			if (i == current_gaussian)
				continue;

			fvec coords_world(4);
			coords_world(span(0, 2)) = conv_to<fvec>::from(gaussians[i].mu);
			coords_world(3) = 1.0f;
				
			fvec coords_camera = view * coords_world;

			fvec coords_screen = projection * coords_camera;
			coords_screen /= coords_screen(3);

			ivec coords_ui = conv_to<ivec>::from(
				(coords_screen(span(0, 1)) + 1.0f) / 2.0f %
				fvec({ (float) window_size.win_width, (float) window_size.win_height })
			);

			coords_ui(1) = window_size.win_height - coords_ui(1);

			ImDrawList* draw_list = ImGui::GetWindowDrawList();

			const int radius = 6;

			bool mouse_is_over = ImGui::IsMouseHoveringRect(
				ImVec2(coords_ui(0) - radius, coords_ui(1) - radius),
				ImVec2(coords_ui(0) + radius, coords_ui(1) + radius)
			);

			draw_list->AddCircleFilled(ImVec2(coords_ui(0), coords_ui(1)), radius,
									   (mouse_is_over ? 0xFF00A5FF : 0xFFFFFFFF));
			draw_list->AddCircle(ImVec2(coords_ui(0), coords_ui(1)), radius, 0xFF000000);

			if (mouse_is_over && ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1)) {
				current_gaussian = i;
		        ImGui::End();
    		    ImGui::PopID();
				return false;
			}
		}

        ImGui::End();
        ImGui::PopID();
 

		// Draw the 3D manipulator
		glClear(GL_DEPTH_BUFFER_BIT);
		
		ImGuizmo::Enable(true);
		ImGuizmo::SetRect(0, 0, window_size.win_width, window_size.win_height);

		fmat diff(4, 4);
		ImGuizmo::Manipulate(
			view.memptr(), projection.memptr(), transforms_operation,
			ImGuizmo::LOCAL, gaussians[current_gaussian].transforms.memptr(), diff.memptr()
		);


		// Convert the transforms to mu and sigma
		if (any(any(diff - eye(4, 4)))) {
			gaussians[current_gaussian].mu = conv_to<vec>::from(
				gaussians[current_gaussian].transforms(span(0, 2), 3)
			);
		}

		//-- Ensures mu stays in the boundaries
		if (ImGui::IsMouseReleased(GLFW_MOUSE_BUTTON_1)) {
			if (gaussians[current_gaussian].mu(0) > 1.0f)
				gaussians[current_gaussian].mu(0) = 1.0f;
			else if (gaussians[current_gaussian].mu(0) < 0.0f)
				gaussians[current_gaussian].mu(0) = 0.0f;

			if (gaussians[current_gaussian].mu(1) > 1.0f)
				gaussians[current_gaussian].mu(1) = 1.0f;
			else if (gaussians[current_gaussian].mu(1) < 0.0f)
				gaussians[current_gaussian].mu(1) = 0.0f;

			if (gaussians[current_gaussian].mu(2) > 1.0f)
				gaussians[current_gaussian].mu(2) = 1.0f;
			else if (gaussians[current_gaussian].mu(2) < 0.0f)
				gaussians[current_gaussian].mu(2) = 0.0f;

			gaussians[current_gaussian].transforms(span(0, 2), 3) =
				conv_to<fvec>::from(gaussians[current_gaussian].mu);
		}

		if (any(any(diff - eye(4, 4)))) {
			mat rot_scale = conv_to<mat>::from(gaussians[current_gaussian].transforms(span(0, 2), span(0, 2)));

			mat RG = zeros(3, 3);
			RG(span::all, 0) = rot_scale * vec({ 1.0, 0.0, 0.0 });
			RG(span::all, 1) = rot_scale * vec({ 0.0, 1.0, 0.0 });
			RG(span::all, 2) = rot_scale * vec({ 0.0, 0.0, 1.0 });

			gaussians[current_gaussian].sigma = RG * RG.t();

			return true;
		}
	}

	return false;
}


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv){

	arma_rng::set_seed_random();

	// Parameters
	parameters_t parameters;
	parameters.nb_gaussians = 2;
	parameters.nb_data      = 2000;
	parameters.nb_fct       = 10;
	parameters.nb_var       = 3;
	parameters.dt           = 0.01f;


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
		"Demo nD ergodic control",
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


	// Projection matrix
	fmat projection;


	// Camera
	gfx2::camera_t camera(fvec({0.5f, 0.5f, 0.5f}), 2.0f);
	gfx2::yaw(camera, gfx2::deg2rad(30.0f));
	gfx2::pitch(camera, gfx2::deg2rad(-20.0f));


	// Mouse control
	double mouse_x, mouse_y, previous_mouse_x, previous_mouse_y;
	bool rotating = false;
	GLFWcursor* crosshair_cursor = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
	GLFWcursor* arrow_cursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

	glfwGetCursorPos(window, &previous_mouse_x, &previous_mouse_y);


	// 3D gaussians
	gaussian_list_t gaussians;

	// Main loop
	mat result;
	const float speed = 1.0f / 20.0f;
	float t = 0.0f;
	bool must_recompute = false;
	ImGuizmo::OPERATION transforms_operation = ImGuizmo::TRANSLATE;
	int current_gaussian = 0;

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

			// Update the projection matrix
			projection = gfx2::perspective(
				gfx2::deg2rad(60.0f),
				(float) window_size.fb_width / (float) window_size.fb_height,
				0.1f, 10.0f
			);

			// At the very first frame: random initialisation of the gaussians (taking 4K
			// screens into account)
			if (window_result == gfx2::WINDOW_READY)
				gaussians = create_random_gaussians(parameters, window_size);

			must_recompute = true;
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();
		ImGuizmo::BeginFrame();

		must_recompute = draw_scene(window_size, projection, camera, result, t,
									gaussians, current_gaussian, transforms_operation
						 ) || must_recompute;


		// Control panel GUI
		ImGui::SetNextWindowSize(ImVec2(500, 106));
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("Control Panel", NULL,
					 ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
					 ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings);

		int previous_nb_gaussians = parameters.nb_gaussians;
		int previous_nb_data = parameters.nb_data;
		int previous_nb_fct = parameters.nb_fct;

		ImGui::SliderInt("Nb gaussians", &parameters.nb_gaussians, 1, 10);
		ImGui::SliderInt("Nb data", &parameters.nb_data, 500, 10000);
		ImGui::SliderInt("Nb basis functions", &parameters.nb_fct, 5, 20);

		ImGui::Text("Current mode: ");
		ImGui::SameLine();
		ImGui::RadioButton("(T)ranslate", (int*) &transforms_operation, (int) ImGuizmo::TRANSLATE);
		ImGui::SameLine();
		ImGui::RadioButton("(R)otate", (int*) &transforms_operation, (int) ImGuizmo::ROTATE);
		ImGui::SameLine();
		ImGui::RadioButton("(S)cale", (int*) &transforms_operation, (int) ImGuizmo::SCALE);

		if ((parameters.nb_gaussians != previous_nb_gaussians) ||
			(parameters.nb_data != previous_nb_data) ||
			(parameters.nb_fct != previous_nb_fct)) {

			must_recompute = true;
		}

		ImGui::End();


		// Gaussian widgets
		if (parameters.nb_gaussians != gaussians.size()) {
			gaussians = create_random_gaussians(parameters, window_size);
			must_recompute = true;
		}


		// Redo the computation when necessary
		if (must_recompute && !ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)) {
			mat mu = zeros(parameters.nb_var, parameters.nb_gaussians);
			cube sigma = zeros(parameters.nb_var, parameters.nb_var, parameters.nb_gaussians);

			for (int i = 0; i < parameters.nb_gaussians; ++i) {
				mu(span(0, 2), i) = gaussians[i].mu;
				sigma(span(0, 2), span(0, 2), span(i)) = gaussians[i].sigma;
			}

			result = compute(parameters, mu, sigma);
			t = 0.0f;

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

		if (ImGui::IsKeyPressed(GLFW_KEY_T))
			transforms_operation = ImGuizmo::TRANSLATE;

		else if (ImGui::IsKeyPressed(GLFW_KEY_R))
			transforms_operation = ImGuizmo::ROTATE;

		else if (ImGui::IsKeyPressed(GLFW_KEY_S))
			transforms_operation = ImGuizmo::SCALE;


		// Mouse input
		glfwGetCursorPos(window, &mouse_x, &mouse_y);

		//-- Right mouse button: rotation of the meshes while held down
		if (ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_2)) {
			if (!rotating) {
				rotating = true;
				glfwSetCursor(window, crosshair_cursor);
			}

			gfx2::yaw(camera, -0.2f * gfx2::deg2rad(mouse_x - previous_mouse_x));
			gfx2::pitch(camera, -0.2f * gfx2::deg2rad(mouse_y - previous_mouse_y));

		} else if (rotating) {
			rotating = false;
			glfwSetCursor(window, arrow_cursor);
		}

		previous_mouse_x = mouse_x;
		previous_mouse_y = mouse_y;


		t += speed * ImGui::GetIO().DeltaTime;
		if (t >= 1.0f)
			t = 0.0f;
	}


	// Cleanup
	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}
