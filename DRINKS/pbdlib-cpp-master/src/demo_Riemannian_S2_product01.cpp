/*
 * demo_Riemannian_S2_product01.cpp
 *
 * Gaussian product on sphere.
 * 3 Gaussians are displayed:
 * - Red is a Gaussian whose position and variance can be altered by the user by left clicking on the sphere and by changing the eigenvalues/vectors of the covariance in the bottom left of the window.
 * - Blue is a constant Gaussian.
 * - Black is the product of Red and Blue
 * By right clicking on the sphere the user can rotate it.
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
 * Author: Andras Kupcsik
 */

#include <stdio.h>
#include <armadillo>

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
	} else {
		v = zeros(3);
	}

	float vera = 1 - cosa;
	float x = v(0);
	float y = v(1);
	float z = v(2);

	arma::mat rot;
	rot = {
		{ cosa + pow(x, 2) * vera, x * y * vera - z * sina, x * z * vera + y * sina },
		{ x * y * vera + z * sina, cosa + pow(y, 2) * vera, y * z * vera - x * sina },
		{ x * z * vera - y * sina, y * z * vera + x * sina, cosa + pow(z, 2) * vera }
	};

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
		R = { {	1, 0, 0},
			{	0, -1, 0},
			{	0, 0, -1}};
	}
	else {
		R = rotM(mu);
	}

	arma::mat u;
	u = logfct(R * x);

	return u;
}

//-----------------------------------------------

arma::mat transp(vec g, vec h) {
	mat E;
	E << 1.0 << 0.0 << endr << 0.0 << 1.0 << endr << 0.0 << 0.0;
	vec logmapTmp = logmap(h, g);
	vec logmapTmp_bottom = zeros(1, 1);

	vec vm = rotM(g).t() * arma::join_vert(logmapTmp, logmapTmp_bottom);
	double mn = norm(vm, 2);
	vec uv = vm / (mn);
	mat Rpar = arma::eye(3, 3) - sin(mn) * (g * uv.t()) - (1 - cos(mn)) * (uv * uv.t());
	mat Ac = E.t() * rotM(h) * Rpar * rotM(g).t() * E; //Transportation operator from g to h
	return Ac;

}

//-----------------------------------------------

arma::mat sample_gaussian_points(const ui::Trans2d& gaussian, const arma::vec& target) {

	arma::mat uCov = trans2cov(gaussian);

	arma::mat V;
	arma::vec d;
	arma::eig_sym(d, V, uCov);

	arma::mat sample_points( {
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

int main(int argc, char **argv) {
	cout << "Gaussian product on S2 Riemannian manifold." << endl;
	cout << "3 Gaussians are displayed:" << endl;
	cout
			<< " - Red is a Gaussian whose position and variance can be altered by the user by left clicking on the sphere and by changing the eigenvalues/vectors of the covariance in the bottom left of the window."
			<< endl;
	cout << " - Blue is a constant Gaussian." << endl;
	cout << " - Black is the product of Red and Blue." << endl;
	cout << "By right clicking on the sphere the user can rotate it." << endl;
	arma_rng::set_seed_random();

	arma::vec target( { 0.0, 0.2, 0.6 });
	target = target / norm(target);

	arma::vec target2( { 0.2, 0.2, 0.2 });
	target2 = target2 / norm(target2);

	arma::vec target_product( { -1.0, -1.0, 0.0 });
	target_product = target_product / norm(target_product);

	ui::Trans2d gaussian(ImVec2(50, 0), ImVec2(0, 100), ImVec2(50, 680));
	ui::Trans2d gaussian2(ImVec2(50, 0), ImVec2(0, 100), ImVec2(50, 300));
	ui::Trans2d gaussian_product(ImVec2(50, 0), ImVec2(0, 100), ImVec2(50, 300));

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
		"Demo Riemannian sphere product", window_size.win_width, window_size.win_height
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
		arma::fvec( { 0, 0, 3 }),  // Position of the camera
		arma::fvec( { 0, 0, 0 }),  // Look at the origin
		arma::fvec( { 0, 1, 0 })   // Head is up
	);


	// Creation of the models
	//
	// The following hierarchy is used:
	//   - transforms node ("node")
	//    |__ sphere ("sphere")
	//    |__ user-modifiable gaussian ("gaussian_model")
	//    |__ constant gaussian ("gaussian_model2")
	//    |__ gaussians product ("gaussian_product_model")
	//    |__ transforms node ("target_model_node")
	//      |__ target point ("target_model")
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

	arma::mat points1 = sample_gaussian_points(gaussian_product,
											   arma::vec( { 1.0f, 0.0f, 0.0f }));
	arma::mat points2 = sample_gaussian_points(gaussian_product, target_product);

	arma::fmat rotation;
	arma::fvec translation;
	gfx2::rigid_transform_3D(conv_to<fmat>::from(points1), conv_to<fmat>::from(points2),
							 rotation, translation);

	sphere.transforms.rotation.submat(0, 0, 2, 2) = rotation;

	//-- The gaussian
	gfx2::model_t gaussian_model = create_gaussian(
		arma::fvec( { 0.8f, 0.0f, 0.0f }), gaussian, target
	);

	gfx2::model_t gaussian_model2 = create_gaussian(
		arma::fvec( { 0.0f, 0.0f, 0.8f }), gaussian2, target2
	);

	gfx2::model_t gaussian_product_model = create_gaussian(
		arma::fvec( { 0.0f, 0.0f, 0.8f }), gaussian_product, target_product
	);

	gaussian_model.transforms.parent = &node;
	gaussian_model2.transforms.parent = &node;
	gaussian_product_model.transforms.parent = &node;

	//-- The intermediate transforms node of the target

	gfx2::transforms_t target_model_node;
	target_model_node.position.zeros(3);
	target_model_node.rotation = gfx2::rotation(arma::fvec( { 1.0f, 0.0f, 0.0f }),
												conv_to<fvec>::from(target_product));
	target_model_node.parent = &node;

	//-- The target point
	gfx2::model_t target_model = gfx2::create_square(
		arma::fvec( { 0.0f, 0.0f, 0.0f }), 0.01f
	);

	target_model.transforms.position = arma::fvec( { 1.0f, 0.0f, 0.0f });
	target_model.transforms.rotation = gfx2::rotate(arma::fvec( { 0.0f, 1.0f, 0.0f }),
													gfx2::deg2rad(90.0f));
	target_model.transforms.parent = &target_model_node;

	// Creation of the light
	gfx2::point_light_t light;
	light.transforms.position = arma::fvec({ 0.0f, 0.0f, 5.0f, 1.0f });
	light.diffuse_color = arma::fvec({ 1.0f, 1.0f, 1.0f });
	light.ambient_color = arma::fvec({ 0.1f, 0.1f, 0.1f });

	gfx2::light_list_t lights;
	lights.push_back(light);

	// Mouse control
	double mouse_x, mouse_y, previous_mouse_x, previous_mouse_y;
	bool rotating = false;
	GLFWcursor* crosshair_cursor = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
	GLFWcursor* arrow_cursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

	glfwGetCursorPos(window, &previous_mouse_x, &previous_mouse_y);

	// Main loop
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

		mat componentsMu = zeros(3, 2); // current means of the components
		cube U0 = zeros(2, 2, 2);  //current covariances of the components

		vec tmpVec;
		mat tmpCov;
		vec tmpEigenValues;
		mat tmpEigenVectors;

		componentsMu.col(0) = target;
		tmpCov = trans2cov(gaussian);
		eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);
		U0.slice(0) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

		componentsMu.col(1) = target2;
		tmpCov = trans2cov(gaussian2);
		eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);
		U0.slice(1) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

		vec MuMan; // starting point on the manifold
		MuMan << -1.0 << endr << -1.0 << endr << 0.0;
		MuMan /= norm(MuMan, 2);
		mat Sigma = zeros(2, 2);

		mat MuTmp = zeros(2, 2); //nbVar (on tangent space) x components
		cube SigmaTmp = zeros(2, 2, 2); // nbVar (on tangent space) x nbVar (on tangent space) x components

		for (int n = 0; n < 10; n++) { // compute the Gaussian product
			// nbVar = 2 for S2 sphere tangent space
			colvec Mu = zeros(2, 1);
			mat SigmaSum = zeros(2, 2);

			for (int i = 0; i < 2; i++) { // we have two states
				mat Ac = transp(componentsMu.col(i), MuMan);
				mat U1 = Ac * U0.slice(i);
				SigmaTmp.slice(i) = U1 * U1.t();
				//Tracking component for Gaussian i
				SigmaSum += inv(SigmaTmp.slice(i));
				MuTmp.col(i) = logmap(componentsMu.col(i), MuMan);
				Mu += inv(SigmaTmp.slice(i)) * MuTmp.col(i);
			}

			Sigma = inv(SigmaSum);
			//Gradient computation
			Mu = Sigma * Mu;

			MuMan = expmap(Mu, MuMan);
		}

		eig_sym(tmpEigenValues, tmpEigenVectors, Sigma);
		mat tmpMat = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5) * 200.0);

		if (tmpMat(0, 1) >= 0) {
			gaussian_product.x.x = -tmpMat(0, 0);
			gaussian_product.x.y = tmpMat(1, 0);
			gaussian_product.y.x = -tmpMat(0, 1);
			gaussian_product.y.y = tmpMat(1, 1);
		} else {
			gaussian_product.x.x = tmpMat(0, 0);
			gaussian_product.x.y = -tmpMat(1, 0);
			gaussian_product.y.x = tmpMat(0, 1);
			gaussian_product.y.y = -tmpMat(1, 1);
		}

		target_product = MuMan;

		// Recreate the gaussian 3D models
		gfx2::destroy(gaussian_model);
		gaussian_model = create_gaussian(
			arma::fvec( { 0.8f, 0.0f, 0.0f }), gaussian, target
		);
		gaussian_model.transforms.parent = &node;

		gfx2::destroy(gaussian_product_model);
		gaussian_product_model = create_gaussian(
			arma::fvec( { 0.0f, 0.0f, 0.0f }), gaussian_product, MuMan
		);
		gaussian_product_model.transforms.parent = &node;

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

		gfx2::draw(gaussian_model, lights);
		gfx2::draw(gaussian_model2, lights);
		gfx2::draw(gaussian_product_model, lights);
		gfx2::draw(target_model, lights);

		// Gaussian UI widget
		ui::begin("Gaussian");
		gaussian = ui::affineSimple(0, gaussian);
		ui::end();

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
				arma::fvec( { 0, 0, 3 }), (int) mouse_x, (int) mouse_y, view,
				projection, window_size.win_width, window_size.win_height
			);

			arma::fvec intersection;
			if (gfx2::intersects(ray, node.position, 1.0f, intersection)) {
				// Change the target position
				arma::fvec p(4);
				p.rows(0, 2) = intersection;
				p(3) = 1.0f;

				p = arma::inv(gfx2::worldRotation(&node)) * p;

				target = arma::conv_to<arma::vec>::from(p.rows(0, 2));
			}
		}

		target_model_node.rotation = gfx2::rotation(
			arma::fvec( { 1.0f, 0.0f, 0.0f }), conv_to<fvec>::from(target_product)
		);

		arma::mat points1 = sample_gaussian_points(gaussian_product,
												   arma::vec( { 1.0f, 0.0f, 0.0f }));
		arma::mat points2 = sample_gaussian_points(gaussian_product, target_product);

		arma::fmat rotation;
		arma::fvec translation;
		gfx2::rigid_transform_3D(conv_to<fmat>::from(points1), conv_to<fmat>::from(points2),
								 rotation, translation);

		sphere.transforms.rotation.submat(0, 0, 2, 2) = rotation;

		//-- Right mouse button: rotation of the meshes while held down
		if (ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_2)) {
			if (!rotating) {
				rotating = true;
				glfwSetCursor(window, crosshair_cursor);
			}

			arma::fmat rotation = gfx2::rotate(
				arma::fvec( { 0.0f, 1.0f, 0.0f }),
				0.2f * gfx2::deg2rad(mouse_x - previous_mouse_x)) *
					gfx2::rotate(arma::fvec( { 1.0f, 0.0f, 0.0f }),
				0.2f * gfx2::deg2rad(mouse_y - previous_mouse_y)
			);

			node.rotation = rotation * node.rotation;
		} else if (rotating) {
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
