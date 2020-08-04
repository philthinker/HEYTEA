/*
 * demo_Riemannian_S2_infHorLQR01.cpp
 *
 * Linear quadratic regulation on a sphere by relying on Riemannian manifold and
 * infinite-horizon LQR.
 *
 * If this code is useful for your research, please cite the related publication:
 * @article{Calinon19,
 * 	author="Calinon, S. and Jaquier, N.",
 * 	title="Gaussians on {R}iemannian Manifolds for Robot Learning and Adaptive Control",
 * 	journal="arXiv:1909.05946",
 * 	year="2019",
 * 	pages="1--10"
 * }
 *
 * Authors: Sylvain Calinon, Fabien Crépon, Philip Abbet
 */

#include <stdio.h>
#include <armadillo>

#include <lqr.h>
#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>

using namespace arma;


/****************************** HELPER FUNCTIONS *****************************/

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//-----------------------------------------------

int factorial(int n) {
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

//-----------------------------------------------

arma::mat trans2cov(const ui::Trans2d& trans) {
	arma::mat RG = {
		{ trans.x.x / 200.0, trans.y.x / 200.0 },
		{ -trans.x.y / 200.0, -trans.y.y / 200.0 }
	};

	return RG * RG.t();
}

//-----------------------------------------------

arma::mat rotM(arma::vec v) {
	arma::vec st = v / norm(v);
	float acangle = st(2);
	float cosa = acangle;
	float sina = sqrt(1 - pow(acangle, 2));


	if ((1 - acangle) > 1E-16) {
		v << st(1) << endr << -st(0) << endr << 0 << endr;
		v = v / sina;
	}
	else {
		v = zeros(3);
	}

	float vera = 1 - cosa;
	float x = v(0);
	float y = v(1);
	float z = v(2);

	arma::mat rot;
	rot = {{cosa + pow(x, 2) * vera, x * y * vera - z * sina, x * z * vera + y * sina},
		   {x * y * vera + z * sina, cosa + pow(y, 2) * vera, y * z * vera - x * sina},
		   {x * z * vera - y * sina, y * z * vera + x * sina, cosa + pow(z, 2) * vera}};

	return rot;
}

//-----------------------------------------------

arma::mat expfct(arma::mat u) {
	arma::mat normv = sqrt(pow(u.row(0), 2) + pow(u.row(1), 2));
	arma::mat Exp(3, u.n_cols);

	Exp.row(0) = u.row(0) % sin(normv) / normv;
	Exp.row(1) = u.row(1) % sin(normv) / normv;
	Exp.row(2) = cos(normv);

	return Exp;
}

//-----------------------------------------------

arma::mat logfct(arma::mat x) {
	arma::mat fullone;
	fullone.ones(size(x.row(2)));
	arma::mat scale(1, x.n_cols);
	scale = acos(x.row(2)) / sqrt(fullone - pow(x.row(2), 2));

	arma::mat Log(2, x.n_cols);
	Log.row(0) = x.row(0) % scale;
	Log.row(1) = x.row(1) % scale;

	return Log;
}

//-----------------------------------------------

arma::mat expmap(arma::mat u, arma::vec mu) {
	arma::mat x = trans(rotM(mu)) * expfct(u);
	return x;
}

//-----------------------------------------------

arma::mat logmap(arma::mat x, arma::vec mu) {
	arma::mat pole;
	pole = {0, 0, 1};
	arma::mat R(3, 3, fill::ones);

	if (norm(mu - trans(pole)) < 1E-6) {
		R = {{1, 0,	 0},
			 {0, -1, 0},
			 {0, 0,	 -1}};
	}
	else {
		R = rotM(mu);
	}

	arma::mat u;
	u = logfct(R * x);

	return u;
}

//-----------------------------------------------

arma::mat sample_gaussian_points(const ui::Trans2d& gaussian, const arma::vec& target) {

	arma::mat uCov = trans2cov(gaussian);

	arma::mat V;
	arma::vec d;
	arma::eig_sym(d, V, uCov);

	arma::mat sample_points({
		{ 0.0f, 0.0f },
		{ 1.0f, 0.0f },
		{ 0.0f, 1.0f },
		{ -1.0f, 0.0f },
		{ 0.0f, -1.0f },
	});

	arma::mat points = expmap(V * diagmat(sqrt(d)) * sample_points.t(), target);

	points.col(0) = target;

	return points;
}


/********************************* RENDERING *********************************/

//-----------------------------------------------------------------------------
// Create a gaussian mesh, colored (no lightning)
//-----------------------------------------------------------------------------
gfx2::model_t create_gaussian(const arma::fvec& color, const ui::Trans2d& gaussian,
							  const arma::vec& target) {

	arma::mat uCov = trans2cov(gaussian);

	arma::rowvec tl = arma::linspace<arma::rowvec>(-arma::datum::pi, arma::datum::pi, 100);

	arma::mat V;
	arma::vec d;
	arma::eig_sym(d, V, uCov);

	arma::mat points = expmap(V * diagmat(sqrt(d)) * join_cols(cos(tl), sin(tl)), target);

	return gfx2::create_line(color, points);
}


/********************************* FUNCTIONS *********************************/

void compute(const ui::Trans2d& gaussian, const arma::vec& target,
			 const std::vector<arma::vec>& start_points,
			 std::vector<arma::mat> &reproductions) {

	//_____ Setup __________

	// Parameters
	const int nbData = 200;				  // Number of datapoints
	const int nbVarPos = 2;				  // Dimension of position data (here: x1,x2)
	const int nbDeriv = 2;				  // Number of static & dynamic features (D=2 for [x,dx])
	const int nbVar = nbVarPos * nbDeriv; // Dimension of state vector in the tangent space
	const int nbVarMan = nbVarPos + 1;	  // Dimension of the manifold
	const double dt = 1E-3;				  // Time step duration
	const float rfactor = 1E-8;			  // Control cost in LQR

	// Control Cost matrix
	arma::mat R = eye(nbVarPos, nbVarPos) * rfactor;

	// Desired covariance
	arma::mat uCov = trans2cov(gaussian);


	//_____ Discrete dynamical system settings (in tangent space) __________

	arma::mat A, B;

	arma::mat A1d(zeros(nbDeriv, nbDeriv));
	for (int i = 0; i <= nbDeriv - 1; i++) {
		A1d = A1d + diagmat(ones(nbDeriv - i), i) * pow(dt, i) * 1.0 / factorial(i);
	}

	arma::mat B1d(zeros(nbDeriv, 1));
	for (int i = 1; i <= nbDeriv; i++) {
		B1d(nbDeriv - i, 0) = pow(dt, i) * 1.0 / factorial(i);
	}

	A = kron(A1d, eye(nbVarPos, nbVarPos));
	B = kron(B1d, eye(nbVarPos, nbVarPos));


	//_____ Iterative discrete LQR with infinite horizon __________

	arma::vec duTar = zeros(nbVarPos * (nbDeriv - 1));

	arma::mat Q = inv(uCov);
	Q.resize(nbVar, nbVar);
	arma::mat P = lqr::solve_algebraic_Riccati_discrete(A, B, Q, R);
	arma::mat L = (trans(B) * P * B + R).i() * trans(B) * P * A;

	arma::vec x(nbVarMan);
	arma::mat U;
	arma::mat x_y(zeros(3, nbData));
	arma::vec ddu(nbVarPos);

	for (int n = 0; n < start_points.size(); n++) {
		x = start_points[n];

		U = -logmap(x, target);
		U.resize(4, U.n_cols);

		for (int t = 0; t < nbData; t++) {
			x_y.col(t) = x.col(0);

			ddu.rows(0, 1) = logmap(x, target);
			ddu.resize(4);
			ddu.rows(2, 3) = duTar - U.rows(2, 3);

			ddu = L * ddu;
			U = A * U + B * ddu;
			x = expmap(-U.rows(0, nbVarPos - 1), target);
		}

		reproductions.push_back(x_y);
	}
}

//-----------------------------------------------

int main(int argc, char **argv) {
	arma_rng::set_seed_random();

	// Parameters
	int nb_repros = 5;	 // Initial number of reproductions

	// Initial LQR computation
	std::vector<arma::vec> start_points;
	std::vector<arma::mat> reproductions;

	arma::vec target({ 0.0, 0.2, 0.6 });
	target = target / norm(target);

	ui::Trans2d gaussian(ImVec2(50, 0), ImVec2(0, 100), ImVec2(50, 680));

	for (int i = 0; i < nb_repros; ++i) {
		arma::vec x(3);
		x(0) = -1.0 + randn() * 0.9;
		x(1) = -1.0 + randn() * 0.9;
		x(2) = randn() * 0.9;
		start_points.push_back(x / norm(x));
	}

	compute(gaussian, target, start_points, reproductions);


	// Initialise GLFW
	gfx2::window_size_t window_size;
	window_size.win_width = 800;
	window_size.win_height = 800;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;

	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	// Open a window and create its OpenGL context
	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Demo Riemannian sphere", window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	// Setup OpenGL
	gfx2::init();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LINE_SMOOTH);
	glDepthFunc(GL_LESS);

	// Setup ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);

	ui::config.color = 0x770000ff;
	ui::config.lineColor = 0x770000ff;


	// Projection matrix
	arma::fmat projection;

	// Camera matrix
	arma::fmat view = gfx2::lookAt(
		arma::fvec({0, 0, 3}),	// Position of the camera
		arma::fvec({0, 0, 0}),	// Look at the origin
		arma::fvec({0, 1, 0})	// Head is up
	);


	// Creation of the models
	//
	// The following hierarchy is used:
	//	 - transforms node ("node")
	//	  |__ sphere ("sphere")
	//	  |__ reproduction lines ("reproductions_models")
	//	  |__ gaussian ("gaussian_model")
	//	  |__ transforms node ("target_model_node")
	//		|__ target point ("target_model")
	//
	// This allows to rotate everything by just changing the rotation of the
	// root transforms node. The "target_model_node" transforms is necessary
	// correctly place the target point, always tangent to the sphere.

	//-- The root transforms node
	gfx2::transforms_t node;
	node.position.zeros(3);
	node.rotation.eye(4, 4);

	//-- The sphere
	gfx2::model_t sphere = gfx2::create_sphere(0.99f);
	sphere.transforms.parent = &node;

	arma::mat points1 = sample_gaussian_points(gaussian, arma::vec({1.0f, 0.0f, 0.0f}));
	arma::mat points2 = sample_gaussian_points(gaussian, target);

	arma::fmat rotation;
	arma::fvec translation;
	gfx2::rigid_transform_3D(conv_to<fmat>::from(points1),
							 conv_to<fmat>::from(points2),
							 rotation, translation);

	sphere.transforms.rotation.submat(0, 0, 2, 2) = rotation;

	//-- The reproduction lines
	std::vector<gfx2::model_t> reproductions_models;
	for (size_t i = 0; i < reproductions.size(); ++i) {
		gfx2::model_t line = gfx2::create_line(
			arma::fvec({0.0f, 0.5f, 0.0f}), reproductions[i]
		);

		line.transforms.parent = &node;

		reproductions_models.push_back(line);
	}

	//-- The gaussian
	gfx2::model_t gaussian_model = create_gaussian(
		arma::fvec({0.8f, 0.0f, 0.0f}), gaussian, target
	);

	gaussian_model.transforms.parent = &node;

	//-- The intermediate transforms node of the target
	gfx2::transforms_t target_model_node;
	target_model_node.position.zeros(3);
	target_model_node.rotation = gfx2::rotation(arma::fvec({1.0f, 0.0f, 0.0f}),
												conv_to<fvec>::from(target));
	target_model_node.parent = &node;

	//-- The target point
	gfx2::model_t target_model = gfx2::create_square(
		arma::fvec({0.8f, 0.0f, 0.0f}), 0.01f
	);

	target_model.transforms.position = arma::fvec({1.0f, 0.0f, 0.0f});
	target_model.transforms.rotation = gfx2::rotate(arma::fvec({0.0f, 1.0f, 0.0f}),
													gfx2::deg2rad(90.0f));
	target_model.transforms.parent = &target_model_node;


	// Creation of the light
	gfx2::point_light_t light;
	light.transforms.position = arma::fvec({0.0f, 0.0f, 5.0f, 1.0f});
	light.diffuse_color = arma::fvec({1.0f, 1.0f, 1.0f});
	light.ambient_color = arma::fvec({0.1f, 0.1f, 0.1f});

	gfx2::light_list_t lights;
	lights.push_back(light);


	// Mouse control
	double mouse_x, mouse_y, previous_mouse_x, previous_mouse_y;
	bool rotating = false;
	GLFWcursor* crosshair_cursor = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
	GLFWcursor* arrow_cursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

	glfwGetCursorPos(window, &previous_mouse_x, &previous_mouse_y);


	// Main loop
	ui::Trans2d previous_gaussian = gaussian;
	bool must_recompute = false;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		if ((window_result == gfx2::WINDOW_READY) || (window_result == gfx2::WINDOW_RESIZED)) {

			// Update the projection matrix
			projection = gfx2::perspective(
				gfx2::deg2rad(45.0f),
				(float) window_size.fb_width / (float) window_size.fb_height,
				0.1f, 10.0f
			);

			// Ensure that the gaussian widgets are still visible
			gaussian.pos.x = 50;
			gaussian.pos.y = window_size.win_height - 120;
		}


		// Ensure that the number of desired reproductions hasn't changed
		//	-> trigger a recomputation or the recreation of the meshes
		bool must_recreate_lines = false;

		if (nb_repros > reproductions.size()) {
			for (int i = reproductions.size(); i < nb_repros; ++i) {
				arma::vec x(3);
				x(0) = -1.0 + randn() * 0.9;
				x(1) = -1.0 + randn() * 0.9;
				x(2) = randn() * 0.9;
				start_points.push_back(x / norm(x));
			}

			must_recompute = true;
		}
		else if (nb_repros < reproductions.size()) {
			start_points.resize(nb_repros);
			reproductions.resize(nb_repros);

			must_recreate_lines = true;
		}


		// Ensure that the gaussian parameters have not changed
		//	-> trigger a recomputation
		if (!gfx2::is_close(previous_gaussian.x.x, gaussian.x.x) ||
			!gfx2::is_close(previous_gaussian.x.y, gaussian.x.y) ||
			!gfx2::is_close(previous_gaussian.y.x, gaussian.y.x) ||
			!gfx2::is_close(previous_gaussian.y.y, gaussian.y.y)) {

			must_recompute = true;
			previous_gaussian = gaussian;
		}


		// If needed, recompute the LQR
		if (must_recompute) {
			reproductions.clear();
			compute(gaussian, target, start_points, reproductions);

			must_recreate_lines = true;
			must_recompute = false;
		}


		// If needed, recreate the models
		if (must_recreate_lines) {
			for (size_t i = 0; i < reproductions_models.size(); ++i) {
				gfx2::destroy(reproductions_models[i]);
			}

			reproductions_models.clear();

			for (size_t i = 0; i < reproductions.size(); ++i) {
				gfx2::model_t line = gfx2::create_line(
					arma::fvec({0.0f, 0.5f, 0.0f}), reproductions[i]
				);

				line.transforms.parent = &node;

				reproductions_models.push_back(line);
			}

			gfx2::destroy(gaussian_model);

			gaussian_model = create_gaussian(
				arma::fvec({0.8f, 0.0f, 0.0f}), gaussian, target
			);

			gaussian_model.transforms.parent = &node;
		}


		// Start the rendering
		ImGui_ImplGlfwGL2_NewFrame();
		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glMultMatrixf(projection.memptr());

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glMultMatrixf(view.memptr());


		// Drawing
		gfx2::draw(sphere, lights);

		for (size_t i = 0; i < reproductions_models.size(); ++i)
			gfx2::draw(reproductions_models[i], lights);

		gfx2::draw(gaussian_model, lights);
		gfx2::draw(target_model, lights);


		// Gaussian UI widget
		ui::begin("Gaussian");
		gaussian = ui::affineSimple(0, gaussian);
		ui::end();


		// Parameter window
		ImGui::SetNextWindowSize(ImVec2(400, 90));
		ImGui::Begin("Parameters", NULL,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
					 ImGuiWindowFlags_NoMove
		);
		ImGui::SliderInt("Nb reproductions", &nb_repros, 1, 10);
		ImGui::Text("Hold the right mouse button to rotate the sphere");
		ImGui::Text("Left-click on the sphere to change the target point");
		ImGui::End();


		// GUI rendering
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		// Swap buffers
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;


		// Mouse input
		glfwGetCursorPos(window, &mouse_x, &mouse_y);

		//-- Left click
		if (ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1)) {

			// Determine if the click was on the sphere, and where exactly
			gfx2::ray_t ray = gfx2::create_ray(
				arma::fvec({0, 0, 3}), (int) mouse_x, (int) mouse_y,
				view, projection, window_size.win_width, window_size.win_height
			);

			arma::fvec intersection;
			if (gfx2::intersects(ray, node.position, 1.0f, intersection)) {
				// Change the target position
				arma::fvec p(4);
				p.rows(0, 2) = intersection;
				p(3) = 1.0f;

				p = arma::inv(gfx2::worldRotation(&node)) * p;

				target = arma::conv_to<arma::vec>::from(p.rows(0, 2));

				target_model_node.rotation = gfx2::rotation(arma::fvec({1.0f, 0.0f, 0.0f}),
															conv_to<fvec>::from(target));

				arma::mat points1 = sample_gaussian_points(gaussian, arma::vec({1.0f, 0.0f, 0.0f}));
				arma::mat points2 = sample_gaussian_points(gaussian, target);

				arma::fmat rotation;
				arma::fvec translation;
				gfx2::rigid_transform_3D(conv_to<fmat>::from(points1),
										 conv_to<fmat>::from(points2),
										 rotation, translation);

				sphere.transforms.rotation.submat(0, 0, 2, 2) = rotation;

				must_recompute = true;
			}
		}

		//-- Right mouse button: rotation of the meshes while held down
		if (ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_2)) {
			if (!rotating) {
				rotating = true;
				glfwSetCursor(window, crosshair_cursor);
			}

			arma::fmat rotation =
				gfx2::rotate(arma::fvec({ 0.0f, 1.0f, 0.0f }), 0.2f * gfx2::deg2rad(mouse_x - previous_mouse_x)) *
				gfx2::rotate(arma::fvec({ 1.0f, 0.0f, 0.0f }), 0.2f * gfx2::deg2rad(mouse_y - previous_mouse_y));

			node.rotation = rotation * node.rotation;
		}
		else if (rotating) {
			rotating = false;
			glfwSetCursor(window, arrow_cursor);
		}

		previous_mouse_x = mouse_x;
		previous_mouse_y = mouse_y;
	}


	// Cleanup
	glfwDestroyCursor(crosshair_cursor);
	glfwDestroyCursor(arrow_cursor);

	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}
