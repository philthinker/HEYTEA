/*
 * gfx_common.cpp
 *
 * Common functions usable with both gfx2 (OpenGL 2) and gfx3 (OpenGL 3.3+)
 *
 * Authors: Philip Abbet
 */

#ifndef GFX_NAMESPACE
#error "Don't compile 'gfx_common.cpp' yourself, it is included by 'gfx2.cpp' or 'gfx3.cpp'"
#endif


namespace GFX_NAMESPACE {

GLFWmonitor* get_current_monitor(GLFWwindow* window)
{
	int win_x, win_y, win_width, win_height;
	glfwGetWindowPos(window, &win_x, &win_y);
	glfwGetWindowSize(window, &win_width, &win_height);

	int nb_monitors;
	GLFWmonitor** monitors;
	monitors = glfwGetMonitors(&nb_monitors);

	int best_overlap = 0;
	GLFWmonitor* best_monitor = 0;

	for (int i = 0; i < nb_monitors; ++i) {
		const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);

		int monitor_x, monitor_y;
		glfwGetMonitorPos(monitors[i], &monitor_x, &monitor_y);

		int monitor_width = mode->width;
		int monitor_height = mode->height;

		int overlap =
			std::max(0, std::min(win_x + win_width, monitor_x + monitor_width) - std::max(win_x, monitor_x)) *
			std::max(0, std::min(win_y + win_height, monitor_y + monitor_height) - std::max(win_y, monitor_y)
		);

		if (best_overlap < overlap) {
			best_overlap = overlap;
			best_monitor = monitors[i];
		}
	}

	return best_monitor;
}

//-----------------------------------------------

GLFWwindow* create_window_at_optimal_size(const char* title, int &width, int &height) {

	// First create a window of the desired size
	GLFWwindow* window = glfwCreateWindow(
		width, height, title, NULL, NULL
	);

	// Next determine on which monitor the window is located
	GLFWmonitor* monitor = get_current_monitor(window);

	// Then compute the "optimal" size of the window on that monitor
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	int original_width = width;
	int original_height = height;

	if (mode->height >= 2000)
		height = 1800;
	else if (mode->height >= 1440)
		height = 1200;
	else if (mode->height >= 1200)
		height = 1000;
	else if (mode->height >= 1000)
		height = 900;
	else if (mode->height >= 900)
		height = 800;
	else
		height = 600;

	width = original_width * height / original_height;

	if (width >= mode->width)
	{
		width = mode->width - 100;
		height = original_height * width / original_width;
	}

	// Finally, resize the window and center it
	glfwSetWindowSize(window, width, height);
	glfwSetWindowPos(window, (mode->width - width) / 2, (mode->height - height) / 2);

	return window;
}

//-----------------------------------------------

window_result_t handle_window_resizing(GLFWwindow* window, window_size_t* window_size,
									   window_size_t* previous_size) {

	// Retrieve the current size of the window
	int width, height;
	glfwGetWindowSize(window, &width, &height);

	if ((width == -1) || (height == -1))
		return INVALID_SIZE;

	// Detect when the window was resized
	if ((width != window_size->win_width) || (height != window_size->win_height) ||
		(window_size->fb_width == -1)) {

		bool first = (window_size->win_width == -1) || (window_size->fb_width == -1);

		if (previous_size) {
			previous_size->win_width  = window_size->win_width;
			previous_size->win_height = window_size->win_height;
			previous_size->fb_width   = window_size->fb_width;
			previous_size->fb_height  = window_size->fb_height;
		}

		window_size->win_width = width;
		window_size->win_height = height;

		// Retrieve the new framebuffer size
		glfwGetFramebufferSize(window, &window_size->fb_width, &window_size->fb_height);

#if __APPLE__
		// Workaround for a bug in macOS 10.14 (Mojave): the window is black
		// until moved or resized
		if (first) {
			int window_x, window_y;
			glfwGetWindowPos(window, &window_x, &window_y);
			glfwSetWindowPos(window, window_x - 1, window_y);
		}
#endif

		return (first ? WINDOW_READY : WINDOW_RESIZED);
	}

	if (window_size->fb_width == -1)
		return INVALID_SIZE;

	return NO_CHANGE;
}


/****************************** UTILITY FUNCTIONS ****************************/

void init()
{
	glewExperimental = GL_TRUE;
	glewInit();
}

//-----------------------------------------------

double deg2rad(double deg)
{
	return deg / 360.0 * (2.0 * M_PI);
}

//-----------------------------------------------

double sin_deg(double deg)
{
	return sin(deg2rad(deg));
}

//-----------------------------------------------

double cos_deg(double deg)
{
	return cos(deg2rad(deg));
}

//-----------------------------------------------

bool is_close(float a, float b, float epsilon)
{
	return fabs(a - b) < epsilon;
}

//-----------------------------------------------

arma::vec ui2fb(const arma::vec& coords, int win_width, int win_height,
				int fb_width, int fb_height) {
	arma::vec result = coords;

	result(0) = coords(0) * (float) fb_width / (float) win_width;
	result(1) = ((float) win_height - coords(1)) * (float) fb_height / (float) win_height;

	return result;
}

//-----------------------------------------------

arma::vec ui2fb(const arma::vec& coords, const window_size_t& window_size) {
	arma::vec result = coords;

	result(0) = coords(0) * (float) window_size.fb_width / (float) window_size.win_width;

	result(1) = ((float) window_size.win_height - coords(1)) *
				(float) window_size.fb_height / (float) window_size.win_height;

	return result;
}

//-----------------------------------------------

arma::vec ui2fb_centered(const arma::vec& coords, int win_width, int win_height,
						 int fb_width, int fb_height) {
	arma::vec result = coords;

	result(0) = (coords(0) - (float) win_width * 0.5f) * (float) fb_width / (float) win_width;
	result(1) = ((float) win_height * 0.5f - coords(1)) * (float) fb_height / (float) win_height;

	return result;
}

//-----------------------------------------------

arma::vec ui2fb_centered(const arma::vec& coords, const window_size_t& window_size) {
	arma::vec result = coords;

	result(0) = (coords(0) - (float) window_size.win_width * 0.5f) *
				(float) window_size.fb_width / (float) window_size.win_width;

	result(1) = ((float) window_size.win_height * 0.5f - coords(1)) *
				(float) window_size.fb_height / (float) window_size.win_height;

	return result;
}

//-----------------------------------------------

arma::vec fb2ui(const arma::vec& coords, int win_width, int win_height,
				int fb_width, int fb_height) {
	arma::vec result = coords;

	result(0) = coords(0) * (float) win_width / (float) fb_width;
	result(1) = -(coords(1) * (float) win_height / (float) fb_height - (float) win_height);

	return result;
}

//-----------------------------------------------

arma::vec fb2ui(const arma::vec& coords, const window_size_t& window_size) {
	arma::vec result = coords;

	result(0) = coords(0) * (float) window_size.win_width / (float) window_size.fb_width;
	result(1) = -(coords(1) * (float) window_size.win_height / (float) window_size.fb_height -
				(float) window_size.win_height);

	return result;
}

//-----------------------------------------------

arma::vec fb2ui_centered(const arma::vec& coords, int win_width, int win_height,
						 int fb_width, int fb_height) {
	arma::vec result = coords;

	result(0) = coords(0) * (float) win_width / (float) fb_width + (float) win_width * 0.5f;
	result(1) = -(coords(1) * (float) win_height / (float) fb_height - (float) win_height * 0.5f);

	return result;
}

//-----------------------------------------------

arma::vec fb2ui_centered(const arma::vec& coords, const window_size_t& window_size) {
	arma::vec result = coords;

	result(0) = coords(0) * (float) window_size.win_width / (float) window_size.fb_width +
				(float) window_size.win_width * 0.5f;
	result(1) = -(coords(1) * (float) window_size.win_height / (float) window_size.fb_height -
				(float) window_size.win_height * 0.5f);

	return result;
}


/************************* PROJECTION & VIEW MATRICES ************************/

arma::fmat perspective(float fovy, float aspect, float zNear, float zFar)
{
	const float top = zNear * tan(fovy / 2.0f);
	const float bottom = -top;
	const float right = top * aspect;
	const float left = -right;

	arma::fmat projection = arma::zeros<arma::fmat>(4,4);

	projection(0, 0) = 2.0f * zNear / (right - left);
	projection(0, 2) = (right + left) / (right - left);
	projection(1, 1) = 2.0f * zNear / (top - bottom);
	projection(1, 2) = (top + bottom) / (top - bottom);
	projection(2, 2) = -(zFar + zNear) / (zFar - zNear);
	projection(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
	projection(3, 2) = -1.0f;

	return projection;
}

//-----------------------------------------------

arma::fmat orthographic(float width, float height, float zNear, float zFar)
{
	const float top = height / 2.0f;
	const float bottom = -top;
	const float right = width / 2.0f;
	const float left = -right;

	arma::fmat projection = arma::zeros<arma::fmat>(4,4);

	projection(0, 0) = 2.0f / (right - left);
	projection(0, 3) = -(right + left) / (right - left);
	projection(1, 1) = 2.0f / (top - bottom);
	projection(1, 3) = -(top + bottom) / (top - bottom);
	projection(2, 2) = -2.0f / (zFar - zNear);
	projection(2, 3) = -(zFar + zNear) / (zFar - zNear);
	projection(3, 3) = 1.0f;

	return projection;
}

//-----------------------------------------------

arma::fmat lookAt(const arma::fvec& position, const arma::fvec& target,
				  const arma::fvec& up)
{
	const arma::fvec f(arma::normalise(target - position));
	const arma::fvec s(arma::normalise(arma::cross(f, up)));
	const arma::fvec u(arma::cross(s, f));

	arma::fmat result = arma::zeros<arma::fmat>(4,4);

	result(0, 0) = s(0);
	result(0, 1) = s(1);
	result(0, 2) = s(2);
	result(1, 0) = u(0);
	result(1, 1) = u(1);
	result(1, 2) = u(2);
	result(2, 0) =-f(0);
	result(2, 1) =-f(1);
	result(2, 2) =-f(2);
	result(0, 3) =-arma::dot(s, position);
	result(1, 3) =-arma::dot(u, position);
	result(2, 3) = arma::dot(f, position);
	result(3, 3) = 1.0f;

	return result;
}


/****************************** TRANSFORMATIONS ******************************/

arma::fmat rotate(const arma::fvec& axis, float angle)
{
	float rcos = cos(angle);
	float rsin = sin(angle);

	arma::fmat matrix = arma::zeros<arma::fmat>(4, 4);

	matrix(0, 0) =			  rcos + axis(0) * axis(0) * (1.0f - rcos);
	matrix(1, 0) =	axis(2) * rsin + axis(1) * axis(0) * (1.0f - rcos);
	matrix(2, 0) = -axis(1) * rsin + axis(2) * axis(0) * (1.0f - rcos);
	matrix(0, 1) = -axis(2) * rsin + axis(0) * axis(1) * (1.0f - rcos);
	matrix(1, 1) =			  rcos + axis(1) * axis(1) * (1.0f - rcos);
	matrix(2, 1) =	axis(0) * rsin + axis(2) * axis(1) * (1.0f - rcos);
	matrix(0, 2) =	axis(1) * rsin + axis(0) * axis(2) * (1.0f - rcos);
	matrix(1, 2) = -axis(0) * rsin + axis(1) * axis(2) * (1.0f - rcos);
	matrix(2, 2) =			  rcos + axis(2) * axis(2) * (1.0f - rcos);
	matrix(3, 3) = 1.0f;

	return matrix;
}

//-----------------------------------------------

arma::fmat rotation(const arma::fvec& from, const arma::fvec& to)
{
	const float dot = arma::dot(from, to);
	const arma::fvec cross = arma::cross(from, to);
	const float norm = arma::norm(cross);

	arma::fmat g({
		{ dot,	-norm, 0.0f },
		{ norm,	 dot,  0.0f },
		{ 0.0f,	 0.0f, 1.0f },
	});

	arma::fmat fi(3, 3);
	fi.rows(0, 0) = from.t();
	fi.rows(1, 1) = arma::normalise(to - dot * from).t();
	fi.rows(2, 2) = arma::cross(to, from).t();

	arma::fmat result = arma::eye<arma::fmat>(4, 4);

	arma::fmat u;
	if (arma::inv(u, fi))
	{
		u = u * g * fi;
		result.submat(0, 0, 2, 2) = u;
	}

	return result;
}

//-----------------------------------------------

void rigid_transform_3D(const arma::fmat& A, const arma::fmat& B,
						arma::fmat &rotation, arma::fvec &translation) {

	arma::fvec centroidsA = arma::mean(A, 1);
	arma::fvec centroidsB = arma::mean(B, 1);

	int n = A.n_cols;

	arma::fmat H = (A - repmat(centroidsA, 1, n)) * (B - repmat(centroidsB, 1, n)).t();

	arma::fmat U, V;
	arma::fvec s;
	arma::svd(U, s, V, H);

	rotation = V * U.t();

	if (arma::det(rotation) < 0.0f)
		rotation.col(2) *= -1.0f;

	translation = -rotation * centroidsA + centroidsB;
}

//-----------------------------------------------

arma::fmat worldTransforms(const transforms_t* transforms)
{
	arma::fmat result = arma::eye<arma::fmat>(4, 4);
	result(0, 3, arma::size(3, 1)) = worldPosition(transforms);

	result = result * worldRotation(transforms);

	return result;
}


//-----------------------------------------------

arma::fvec worldPosition(const transforms_t* transforms)
{
	if (transforms->parent)
	{
		arma::fvec position(4);
		position.rows(0, 2) = transforms->position;
		position(3) = 1.0f;

		position = worldRotation(transforms->parent) * position;

		return worldPosition(transforms->parent) + position.rows(0, 2);
	}

	return transforms->position;
}


//-----------------------------------------------

arma::fmat worldRotation(const transforms_t* transforms)
{
	if (transforms->parent)
	{
		arma::fmat result = worldRotation(transforms->parent) * transforms->rotation;
		return arma::normalise(result);
	}

	return transforms->rotation;
}


/*********************************** CAMERAS *********************************/

void yaw(camera_t &camera, float delta) {
	camera.target.rotation = rotate(arma::fvec({ 0.0f, 1.0f, 0.0f }), delta) * camera.target.rotation;
}

//-----------------------------------------------

void pitch(camera_t &camera, float delta) {
	camera.rotator.rotation = rotate(arma::fvec({ 1.0f, 0.0f, 0.0f }), delta) * camera.rotator.rotation;
}

//-----------------------------------------------

arma::fmat view_matrix(const camera_t& camera) {

	arma::fvec target_position = worldPosition(&camera.target);

	arma::fvec position = worldPosition(&camera.transforms);

	arma::fmat rotation = worldRotation(&camera.transforms);

	arma::fmat view = lookAt(
		position,               // Position of the camera
		target_position,        // Look at the origin
		arma::fvec(rotation * arma::fvec({0, 1, 0, 1}))(arma::span(0, 2))   // Head is up
	);

	return view;
}


/******************************** RAY CASTING ********************************/

ray_t create_ray(const arma::fvec& origin, int mouse_x, int mouse_y,
				 const arma::fmat& view, const arma::fmat& projection,
				 int window_width, int window_height)
{
	ray_t ray;

	ray.origin = origin;

	// Compute the ray in homogeneous clip coordinates (range [-1:1, -1:1, -1:1, -1:1])
	arma::fvec ray_clip(4);
	ray_clip(0) = (2.0f * mouse_x) / window_width - 1.0f;
	ray_clip(1) = 1.0f - (2.0f * mouse_y) / window_height;
	ray_clip(2) = -1.0f;
	ray_clip(3) = 1.0f;

	// Compute the ray in camera coordinates
	arma::fvec ray_eye = arma::inv(projection) * ray_clip;
	ray_eye(2) = -1.0f;
	ray_eye(3) = 0.0f;

	// Compute the ray in world coordinates
	arma::fvec ray_world = arma::inv(view) * ray_eye;
	ray.direction = arma::fvec(arma::normalise(ray_world)).rows(0, 2);

	return ray;
}

//-----------------------------------------------

bool intersects(const ray_t& ray, const arma::fvec& center, float radius,
				arma::fvec &result)
{
	arma::fvec O_C = ray.origin - center;
	float b = arma::dot(ray.direction, O_C);
	float c = arma::dot(O_C, O_C) - radius * radius;

	float det = b * b - c;

	if (det < 0.0f)
		return false;

	float t;

	if (det > 0.0f)
	{
		float t1 = -b + sqrtf(det);
		float t2 = -b - sqrtf(det);

		t = (t1 < t2 ? t1 : t2);
	}
	else
	{
		t = -b + sqrtf(det);
	}

	result = ray.origin + ray.direction * t;

	return true;
}


/********************************** OTHERS ***********************************/

arma::mat get_gaussian_background_vertices(const arma::vec& mu, const arma::mat& sigma,
										   int nb_points)
{
	arma::mat pts = get_gaussian_border_vertices(mu, sigma, nb_points, true);

	arma::mat vertices(2, nb_points * 3);

	// We need to ensure that the vertices will be in a counter-clockwise order
	arma::vec v1 = pts(arma::span::all, 0) - mu(arma::span(0, 1));
	arma::vec v2 = pts(arma::span::all, 1) - mu(arma::span(0, 1));

	if (atan2(v1(1), v1(0)) - atan2(v2(1), v2(0)) > 0.0) {
		for (int i = 0; i < nb_points - 1; ++i)
		{
			vertices(arma::span::all, i * 3) = mu(arma::span(0, 1));
			vertices(arma::span::all, i * 3 + 1) = pts(arma::span::all, i + 1);
			vertices(arma::span::all, i * 3 + 2) = pts(arma::span::all, i);
		}

		vertices(arma::span::all, (nb_points - 1) * 3) = mu(arma::span(0, 1));
		vertices(arma::span::all, (nb_points - 1) * 3 + 1) = pts(arma::span::all, 0);
		vertices(arma::span::all, (nb_points - 1) * 3 + 2) = pts(arma::span::all, nb_points - 1);
	} else {
		for (int i = 0; i < nb_points - 1; ++i)
		{
			vertices(arma::span::all, i * 3) = mu(arma::span(0, 1));
			vertices(arma::span::all, i * 3 + 1) = pts(arma::span::all, i);
			vertices(arma::span::all, i * 3 + 2) = pts(arma::span::all, i + 1);
		}

		vertices(arma::span::all, (nb_points - 1) * 3) = mu(arma::span(0, 1));
		vertices(arma::span::all, (nb_points - 1) * 3 + 1) = pts(arma::span::all, nb_points - 1);
		vertices(arma::span::all, (nb_points - 1) * 3 + 2) = pts(arma::span::all, 0);
	}

	return vertices;
}

//-----------------------------------------------

arma::mat get_gaussian_border_vertices(const arma::vec& mu, const arma::mat& sigma,
									   int nb_points, bool line_strip)
{
	arma::mat pts0 = arma::join_cols(arma::cos(arma::linspace<arma::rowvec>(0, 2 * arma::datum::pi, nb_points)),
									 arma::sin(arma::linspace<arma::rowvec>(0, 2 * arma::datum::pi, nb_points))
	);

	arma::vec eigval(2);
	arma::mat eigvec(2, 2);
	eig_sym(eigval, eigvec, sigma(arma::span(0, 1), arma::span(0, 1)));

	arma::mat R = eigvec * diagmat(sqrt(eigval));

	arma::mat pts = R * pts0 + arma::repmat(mu(arma::span(0, 1)), 1, nb_points);

	if (line_strip)
		return pts;

	arma::mat vertices(2, nb_points * 2);

	for (int i = 0; i < nb_points - 1; ++i)
	{
		vertices(arma::span::all, i * 2) = pts(arma::span::all, i);;
		vertices(arma::span::all, i * 2 + 1) = pts(arma::span::all, i + 1);
	}

	vertices(arma::span::all, (nb_points - 1) * 2) = pts(arma::span::all, 0);;
	vertices(arma::span::all, (nb_points - 1) * 2 + 1) = pts(arma::span::all, nb_points - 1);

	return vertices;
}

}
