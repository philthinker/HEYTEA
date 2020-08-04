/*
 * demo_MPC_semitied01.cpp
 *
 * Interactive MPC demo, demonstrates a GUI to edit semi-tied covariances.
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
 * Fabien Crépon, Philip Abbet, Sylvain Calinon, 2017
 */

#include <stdio.h>
#include <armadillo>
#include <mpc_utils.h>

#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>
#include <gl2ps.h>

using namespace arma;


/***************************** ALGORITHM SECTION *****************************/

//-----------------------------------------------------------------------------
// Contains all the parameters used by the algorithm. Some of them are
// modifiable through the UI, others are hard-coded.
//-----------------------------------------------------------------------------
struct parameters_t {
	int    nb_targets;
	int    nb_dimensions;
	int    order;
	double global_scale;         // global scale, avoids numerical issues in batch method
	double end_weight;           // forces movement to a stop (see stepwiseReference function)
	float  maximum_displacement; // maximum displacement, used to compute R diagonal
	float  stroke_duration;      // duration of a stroke
	double dt;
};

//-----------------------------------------------

mat compute_LQR(const parameters_t& parameters, const mat& Mu, const cube& Sigma) {

	const double duration = parameters.stroke_duration * parameters.nb_targets;
	const int n = (int) (duration / parameters.dt);
	const int cDim = parameters.nb_dimensions * parameters.order;

	// Integration with higher order Taylor series expansion
	mat A, B;
	makeIntegratorChain(&A, &B, parameters.order);
	discretizeSystem(&A, &B, A, B, parameters.dt);
	A = kron(A, eye(parameters.nb_dimensions, parameters.nb_dimensions));
	B = kron(B, eye(parameters.nb_dimensions, parameters.nb_dimensions));

	// Reference
	mat Q, MuQ;
	stepwiseReference(&MuQ, &Q, Mu, Sigma, n, parameters.order,
					  parameters.nb_dimensions, parameters.end_weight);
	MuQ *= parameters.global_scale;
	Q /= parameters.global_scale * parameters.global_scale;

	// r based on oscillatory movement displacement
	double r = SHM_r(parameters.maximum_displacement, parameters.stroke_duration,
					 parameters.order);

	mat R = kron(eye(n - 1, n - 1), eye(parameters.nb_dimensions, parameters.nb_dimensions) * r);

	////////////////////////////////////
	// Batch LQR

	// Sx and Su matrices for batch LQR
	mat Su = zeros(cDim * n, parameters.nb_dimensions * (n - 1));
	mat Sx = kron(ones(n, 1),
				  eye(parameters.nb_dimensions * parameters.order,
					  parameters.nb_dimensions * parameters.order)
	);
	mat M = B;
	for (int i = 1; i < n; i++) {
		Sx.rows(i * cDim, n * cDim - 1) = Sx.rows(i * cDim, n * cDim - 1) * A;
		Su.submat(i * cDim, 0, (i + 1) * cDim - 1, i * parameters.nb_dimensions - 1) = M;
		M = join_horiz(A * M.cols(0, parameters.nb_dimensions - 1), M);
	}

	arma::vec x0 = MuQ.col(0);

	// Flatten Mu's
	mat Xi_hat = reshape(MuQ, cDim * n, 1);

	mat SuInvSigmaQ = Su.t() * Q;

	mat Rq, rq;

	Rq = SuInvSigmaQ * Su + R;
	rq = SuInvSigmaQ * (Xi_hat - Sx * x0);

	// Least squares solution
	vec u = pinv(Rq) * rq;;
	mat Y = reshape(Sx * x0 + Su * u, cDim, n);

	return Y.rows(0, parameters.nb_dimensions - 1) / parameters.global_scale;
}


/****************************** HELPER FUNCTIONS *****************************/

static void error_callback(int error, const char* description){
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//-----------------------------------------------

arma::mat gradient_2d(arma::mat X, int order = 1) {
	mat df = diff(X, 1, 1);
	mat db = fliplr(-diff(fliplr(X), 1, 1));

	mat d = (df + db) / 2;
	d = join_horiz(zeros(2, 1), d);
	d.col(0) = df.col(0);
	d.col(d.n_cols - 1) = db.col(db.n_cols - 1);

	if(order > 1)
		return gradient_2d(d, order - 1);

	return d;
}

//-----------------------------------------------

void plot(float x, float y, float w, float h, const vec& v, const fvec& color) {
	float v_min = v.min();
	float v_max = v.max();

	mat points(2, v.n_rows);
	points(0, span::all) = linspace<vec>(0, w, v.n_rows).t() + x;
	points(1, span::all) = (v.t() - v_min) / (v_max - v_min) * h + y;

	gfx2::draw_line(color, points);
}

//-----------------------------------------------

arma::mat semitiedBasis(const arma::vec& params) {
	float t1 = params[0];
	float t2 = params[1];
	return {{cos(t1), cos(t2)}, {sin(t1), sin(t2)}};
}

//-----------------------------------------------

arma::mat semitiedSigma(const arma::vec& params) {
	float h = params[2];
	arma::mat H = semitiedBasis(params);

	arma::mat sigma = H * H.t() * h * h;

	// This is needed to compensate the fact that the coordinates system of the UI
	// and OpenGL are inverted on the Y axis
	sigma(0, 1) = -sigma(0, 1);
	sigma(1, 0) = -sigma(1, 0);

	return sigma;
}

//-----------------------------------------------

arma::cube semitiedCovariances(const arma::mat& Mu, const arma::vec& params) {
	int m = Mu.n_cols;
	arma::cube Sigma = arma::zeros(2,2,m);
	arma::mat sigm = semitiedSigma(params);

	for (int i = 0; i < m; i++)
		Sigma.slice(i) = sigm;

	return Sigma;
}

//-----------------------------------------------

arma::vec semitiedWidget(int id, arma::vec params, const arma::vec& pos,
						 const gfx2::window_size_t& window_size, float minLength=2.0,
						 float maxLength=500.0) {

	float h = params[2] * (float) window_size.win_height / window_size.fb_height;
	arma::vec theta = params.subvec(0,1);

	for (int i = 0; i < 2; i++) {
		ImVec2 theta_length = ui::lengthHandle(
			id + i, ImVec2(theta[i], h), 0.0, pos,
			ImVec2(-1000.0, minLength), ImVec2(1000.0, maxLength)
		);
		params[i] = theta_length.x;
		h = theta_length.y;
	}

	h = h * (float) window_size.fb_height / window_size.win_height;

	params[2] = h;

	// draw ellipse
	static arma::mat circ(2, 35);
	static bool circ_init = true;
	if (circ_init) {
		circ = join_cols(cos(linspace<rowvec>(0, 2 * datum::pi, 35)),
						 sin(linspace<rowvec>(0, 2 * datum::pi, 35)));
		circ_init = false;
	}

	arma::mat basis = semitiedBasis(params);

	arma::mat pts = basis * circ * h;

	// This is needed to compensate the fact that the coordinates system of the UI
	// and OpenGL are inverted on the Y axis
	pts.row(1) = -pts.row(1);

	pts = pts + arma::repmat(ui2fb_centered(pos, window_size), 1, 35);

	gfx2::draw_line(fvec({ 1.0f, 0.0f, 0.0f }), pts);

	return params;
}


/******************************* MAIN FUNCTION *******************************/

int main(int argc, char **argv){

	arma_rng::set_seed_random();

	// Parameters
	parameters_t parameters;
	parameters.nb_targets           = 4;
	parameters.nb_dimensions        = 2;
	parameters.order                = 4;
	parameters.global_scale         = 0.001;
	parameters.end_weight           = 1.;
	parameters.maximum_displacement = 0.05;
	parameters.stroke_duration      = 0.3;
	parameters.dt                   = 0.01;


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
		"Demo Semitied MPC", window_size.win_width, window_size.win_height
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


	// Targets
	mat Mu;
	cube Sigma;

	// Semi-tied parameters (theta1, theta2, h)
	arma::vec semitied_params;

	// Main loop
	FILE* eps_file = NULL;
	bool saving_eps = false;
	bool must_recompute = true;
	mat trajectory;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		// At the very first frame: random initialisation of the gaussians (taking 4K
		// screens into account)
		if (window_result == gfx2::WINDOW_READY) {
			Mu = randu(2, parameters.nb_targets);
			Mu.row(0) = Mu.row(0) * (window_size.fb_width - 200) - (window_size.fb_width / 2 - 100);
			Mu.row(1) = Mu.row(1) * (window_size.fb_height - 200) - (window_size.fb_height / 2 - 100);

			semitied_params = {
				0.0,
				datum::pi / 2.0,
				100.0 * (float) window_size.fb_height / window_size.win_height
			};
		}


		// Start of rendering
		ImGui_ImplGlfwGL2_NewFrame();

		if (saving_eps)
		{
			eps_file = fopen("out.eps", "wb");
			gl2psBeginPage("grab", "gl2psTestSimple", NULL, GL2PS_EPS, GL2PS_SIMPLE_SORT,
						   GL2PS_DRAW_BACKGROUND | GL2PS_USE_CURRENT_VIEWPORT,
						   GL_RGBA, 0, NULL, 0, 0, 0, 0, eps_file, "out.eps");
		}

		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-window_size.fb_width / 2, window_size.fb_width / 2,
				-window_size.fb_height / 2, window_size.fb_height / 2, -1.0f, 1.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glPushMatrix();


		// Gaussian UI
		ui::begin("Gaussian");
		arma::vec mu;
		arma::mat Sigm;

		// Drag only target means
		for (int i = 0; i < parameters.nb_targets; i++) {
			vec mu = ui2fb_centered((arma::vec) ui::dragger(i, fb2ui_centered(Mu.col(i), window_size)), window_size);
			must_recompute = must_recompute || (norm(mu - Mu.col(i)) > 1e-6);
			Mu.col(i) = mu;
		}

		// Manipulate semitied covariance
		semitied_params = semitiedWidget(parameters.nb_targets, semitied_params, {100, 100}, window_size);

		// Setup cov states
		cube new_sigma = semitiedCovariances(Mu, semitied_params);

		for (int i = 0; !must_recompute && (i < parameters.nb_targets); i++)
			must_recompute = (norm(new_sigma.slice(i) - Sigma.slice(i)) > 1e-6);

		Sigma = new_sigma;

		ui::end();


		// Recompute the LQR when needed
		if (must_recompute && !ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)) {
			trajectory = compute_LQR(parameters, Mu, Sigma);
			must_recompute = false;
		}


		// Draw the gaussians
		for( int i = 0; i < parameters.nb_targets; i++ ) {
			glClear(GL_DEPTH_BUFFER_BIT);
			gfx2::draw_gaussian_background(fvec({ 0.0f, 0.5f, 1.0f }), Mu.col(i), Sigma.slice(i));
		}
		glClear(GL_DEPTH_BUFFER_BIT);

		// Draw the motor plan
		glColor3f(0.5, 0.5 ,0.5);
		gfx2::draw_line(fvec({ 0.5f, 0.5f, 0.5f }), Mu);

		// Draw the trajectory
		gfx2::draw_line(fvec({ 0.0f, 0.0f, 1.0f }), trajectory);

		// Plot derivatives magnitude
		int plot_width = 200 * window_size.fb_width / window_size.win_width;
		int plot_height = 100 * window_size.fb_height / window_size.win_height;
		int plot_x = -window_size.fb_width / 2;
		int plot_y = -window_size.fb_height / 2 + 25 * window_size.fb_height / window_size.win_height;

		mat dx = gradient_2d(trajectory, 1);
		vec speed = sqrt(sum(dx % dx, 0)).t();
		plot(plot_x, plot_y, plot_width, plot_height, speed, {1.0f, 0.0f, 0.0f});

		ui::begin("Text");
		ui::text(ImVec2(10, window_size.win_height - 25), "velocity magnitude",
				 ImVec4(1, 0, 0, 1));
		ui::end();

		dx = gradient_2d(trajectory, 2);
		speed = sqrt(sum(dx % dx, 0)).t();
		plot(plot_x + plot_width, plot_y, plot_width, plot_height, speed, {0.0f, 0.0f, 1.0f});

		ui::begin("Text");
		ui::text(ImVec2(200 + 10, window_size.win_height - 25), "acceleration magnitude",
				 ImVec4(0, 0, 1, 1));
		ui::end();

		dx = gradient_2d(trajectory, 3);
		speed = sqrt(sum(dx % dx, 0)).t();
		plot(plot_x + 2 * plot_width, plot_y, plot_width, plot_height, speed, {0.0f, 0.5f, 0.0f});

		ui::begin("Text");
		ui::text(ImVec2(400 + 10, window_size.win_height - 25), "jerk magnitude",
				 ImVec4(0, 0.5, 0, 1));
		ui::end();

		glPopMatrix();


		// Parameters window
		int winw = 360;
		ImGui::SetNextWindowPos(ImVec2(window_size.win_width - winw, 2));
		ImGui::Begin("Params", NULL, ImVec2(winw, 86), 0.5f,
					 ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
					 ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings
		);

		if (ImGui::Button("Save EPS"))
			saving_eps = true;

		int previous_order = parameters.order;
		float previous_maximum_displacement = parameters.maximum_displacement;

		ImGui::SliderInt("Order", &parameters.order, 2, 7);
		ImGui::SliderFloat("Max Displacement", &parameters.maximum_displacement, 0.01, 5.);
		ImGui::End();

		must_recompute = must_recompute ||
						 (parameters.order != previous_order) ||
						 !gfx2::is_close(parameters.maximum_displacement, previous_maximum_displacement);


		// Render UI
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		if(eps_file)
		{
			glFlush();
			gl2psEndPage();
			fclose(eps_file);
			eps_file = NULL;
			saving_eps = false;
		}

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
