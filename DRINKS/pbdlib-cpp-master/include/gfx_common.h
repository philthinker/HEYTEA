/*
 * window_utils.h
 *
 * Helper functions to manage windows taking multiple monitors
 * into account.
 *
 * Authors: Philip Abbet
 */

#pragma once

#ifndef GFX_NAMESPACE
#error "Don't include 'gfx_common.h' directly, it must be done by 'gfx2.h' or 'gfx3.h'"
#endif

#include <GLFW/glfw3.h>
#include <algorithm>


namespace GFX_NAMESPACE
{

	/************************** WINDOW-RELATED FUNCTIONS ************************/

	//----------------------------------------------------------------------------
	// Holds the sizes of the window and of the OpenGL front-buffer (they might
	// be different, for instance on a 4K screen)
	//----------------------------------------------------------------------------
	struct window_size_t {
		int win_width;
		int win_height;
		int fb_width;
		int fb_height;

		inline float scale_x() const {
			return (float) fb_width / (float) win_width;
		}

		inline float scale_y() const {
			return (float) fb_height / (float) win_height;
		}
	};

	//----------------------------------------------------------------------------
	// Returns the monitor on which the provided window is located.
	//
	// This functionality doesn't exists in GLFW.
	//----------------------------------------------------------------------------
	GLFWmonitor* get_current_monitor(GLFWwindow* window);

	//----------------------------------------------------------------------------
	// Create a window as large as possiblem but adapted to the current monitor.
	// On input, the provided dimensions are used as an hint for the desired aspect
	// ratio. On output, the real dimensions of the window is returned.
	//----------------------------------------------------------------------------
	GLFWwindow* create_window_at_optimal_size(const char* title, int &width, int &height);

	//----------------------------------------------------------------------------
	// Result codes for the function 'handle_window_resizing()'
	//----------------------------------------------------------------------------
	enum window_result_t {
		INVALID_SIZE,   // The window size is invalid (happens at the beginning of the
		                // program)
		NO_CHANGE,      // Nothing has changed
		WINDOW_READY,   // The window is ready (only returned once)
		WINDOW_RESIZED, // The window was resized
	};

	//----------------------------------------------------------------------------
	// Handle the resizing of the window
	//
	// Call it periodically (each frame)
	//----------------------------------------------------------------------------
	window_result_t handle_window_resizing(GLFWwindow* window,
										   window_size_t* window_size,
										   window_size_t* previous_size = 0
										  );


	/***************************** UTILITY FUNCTIONS ****************************/

	//----------------------------------------------------------------------------
	// Initialisations
	//----------------------------------------------------------------------------
	void init();

	//----------------------------------------------------------------------------
	// Convert radian to degrees
	//----------------------------------------------------------------------------
	double deg2rad(double deg);

	//----------------------------------------------------------------------------
	// Returns the sinus of an angle in degrees
	//----------------------------------------------------------------------------
	double sin_deg(double deg);

	//----------------------------------------------------------------------------
	// Returns the cosinus of an angle in degrees
	//----------------------------------------------------------------------------
	double cos_deg(double deg);

	//----------------------------------------------------------------------------
	// Indicates if two values are close enough
	//----------------------------------------------------------------------------
	bool is_close(float a, float b, float epsilon = 1e-6f);

	//----------------------------------------------------------------------------
	// Converts some coordinates from UI-space to OpenGL-space
	//
	// UI coordinates range from (0, 0) to (win_width, win_height)
	// OpenGL coordinates range from (0, fb_height) to (fb_width, 0)
	//----------------------------------------------------------------------------
	arma::vec ui2fb(const arma::vec& coords, const window_size_t& window_size);

	//----------------------------------------------------------------------------
	// Converts some coordinates from UI-space to OpenGL-space
	//
	// UI coordinates range from (0, 0) to (win_width, win_height)
	// OpenGL coordinates range from (0, fb_height) to (fb_width, 0)
	//----------------------------------------------------------------------------
	arma::vec ui2fb(const arma::vec& coords, int win_width, int win_height,
					int fb_width, int fb_height);

	//----------------------------------------------------------------------------
	// Converts some coordinates from UI-space to OpenGL-space
	//
	// UI coordinates range from (0, 0) to (win_width, win_height)
	// OpenGL coordinates range from (-fb_width / 2, fb_height / 2) to
	// (fb_width / 2, -fb_height / 2)
	//----------------------------------------------------------------------------
	arma::vec ui2fb_centered(const arma::vec& coords, const window_size_t& window_size);

	//----------------------------------------------------------------------------
	// Converts some coordinates from UI-space to OpenGL-space
	//
	// UI coordinates range from (0, 0) to (win_width, win_height)
	// OpenGL coordinates range from (-fb_width / 2, fb_height / 2) to
	// (fb_width / 2, -fb_height / 2)
	//----------------------------------------------------------------------------
	arma::vec ui2fb_centered(const arma::vec& coords, int win_width, int win_height,
							 int fb_width, int fb_height);

	//----------------------------------------------------------------------------
	// Converts some coordinates from OpenGL-space to UI-space
	//
	// OpenGL coordinates range from (0, fb_height) to (fb_width, 0)
	// UI coordinates range from (0, 0) to (win_width, win_height)
	//----------------------------------------------------------------------------
	arma::vec fb2ui(const arma::vec& coords, int win_width, int win_height,
					int fb_width, int fb_height);

	//----------------------------------------------------------------------------
	// Converts some coordinates from OpenGL-space to UI-space
	//
	// OpenGL coordinates range from (0, fb_height) to (fb_width, 0)
	// UI coordinates range from (0, 0) to (win_width, win_height)
	//----------------------------------------------------------------------------
	arma::vec fb2ui(const arma::vec& coords, const window_size_t& window_size);

	//----------------------------------------------------------------------------
	// Converts some coordinates from OpenGL-space to UI-space
	//
	// OpenGL coordinates range from (-fb_width / 2, fb_height / 2) to
	// (fb_width / 2, -fb_height / 2)
	// UI coordinates range from (0, 0) to (win_width, win_height)
	//----------------------------------------------------------------------------
	arma::vec fb2ui_centered(const arma::vec& coords, int win_width, int win_height,
							 int fb_width, int fb_height);

	//----------------------------------------------------------------------------
	// Converts some coordinates from OpenGL-space to UI-space
	//
	// OpenGL coordinates range from (-fb_width / 2, fb_height / 2) to
	// (fb_width / 2, -fb_height / 2)
	// UI coordinates range from (0, 0) to (win_width, win_height)
	//----------------------------------------------------------------------------
	arma::vec fb2ui_centered(const arma::vec& coords, const window_size_t& window_size);


	/************************ PROJECTION & VIEW MATRICES ************************/

	//----------------------------------------------------------------------------
	// Returns a perspective projection matrix
	//----------------------------------------------------------------------------
	arma::fmat perspective(float fovy, float aspect, float zNear, float zFar);

	//----------------------------------------------------------------------------
	// Returns a orthographic projection matrix
	//----------------------------------------------------------------------------
	arma::fmat orthographic(float width, float height, float zNear, float zFar);

	//----------------------------------------------------------------------------
	// Returns a view matrix
	//----------------------------------------------------------------------------
	arma::fmat lookAt(const arma::fvec& position, const arma::fvec& target,
					  const arma::fvec& up);


	/***************************** TRANSFORMATIONS ******************************/

	//----------------------------------------------------------------------------
	// Holds all the transformations needed for a 3D entity
	//
	// Can be organised in a hierarchy, where the parent transforms affect the
	// children ones
	//----------------------------------------------------------------------------
	struct transforms_t {

		// Constructor
		transforms_t()
		: parent(0)
		{
			position.zeros(3);
			rotation.eye(4, 4);
		}

		const transforms_t* parent;

		arma::fvec position;
		arma::fmat rotation;
	};

	//----------------------------------------------------------------------------
	// Returns a rotation matrix given an axis and an angle (in radian)
	//----------------------------------------------------------------------------
	arma::fmat rotate(const arma::fvec& axis, float angle);

	//----------------------------------------------------------------------------
	// Returns the rotation matrix to go from one direction to another one
	//----------------------------------------------------------------------------
	arma::fmat rotation(const arma::fvec& from, const arma::fvec& to);

	//----------------------------------------------------------------------------
	// Compute the translation and rotation to apply to a list of 3D points A to
	// obtain the list of 3D points B.
	//
	// Points are organised in columns:
	//	[ x0 x1 x2 ...
	//	  y0 y1 y2
	//	  z0 z1 z2 ]
	//----------------------------------------------------------------------------
	void rigid_transform_3D(const arma::fmat& A, const arma::fmat& B,
							arma::fmat &rotation, arma::fvec &translation);

	//----------------------------------------------------------------------------
	// Returns the full world transformation matrix corresponding to the given
	// transforms structure, taking all its parent hierarchy into account
	//----------------------------------------------------------------------------
	arma::fmat worldTransforms(const transforms_t* transforms);

	//----------------------------------------------------------------------------
	// Returns the full world position corresponding to the given transforms
	// structure, taking all its parent hierarchy into account
	//----------------------------------------------------------------------------
	arma::fvec worldPosition(const transforms_t* transforms);

	//----------------------------------------------------------------------------
	// Returns the full world rotation matrix corresponding to the given
	// transforms structure, taking all its parent hierarchy into account
	//----------------------------------------------------------------------------
	arma::fmat worldRotation(const transforms_t* transforms);


	/******************************** CAMERAS ********************************/

	//----------------------------------------------------------------------------
	// Represents a camera in the 3D scene
	//
	// A camera points to a target point, looking at it from a distance
	//----------------------------------------------------------------------------
	struct camera_t {

		// Constructor
		camera_t()
		{
			transforms.position(2) = 5.0f;
			rotator.parent = &this->target;
			transforms.parent = &this->rotator;
		}

		// Constructor
		camera_t(const arma::fvec& target, float distance)
		{
			transforms.position(2) = distance;

			this->target.position = target;
			rotator.parent = &this->target;
			transforms.parent = &this->rotator;
		}

		transforms_t target;
		transforms_t rotator;
		transforms_t transforms;
	};

	//----------------------------------------------------------------------------
	// Rotate the camera around the Y-axis of its target (left or right)
	//
	// The 'delta' is an angle in radian
	//----------------------------------------------------------------------------
	void yaw(camera_t &camera, float delta);

	//----------------------------------------------------------------------------
	// Rotate the camera around the X-axis of its target (up or down)
	//
	// The 'delta' is an angle in radian
	//----------------------------------------------------------------------------
	void pitch(camera_t &camera, float delta);

	//----------------------------------------------------------------------------
	// Returns the view matrix corresponding to the camera
	//----------------------------------------------------------------------------
	arma::fmat view_matrix(const camera_t& camera);


	/******************************* RAY CASTING ********************************/

	//----------------------------------------------------------------------------
	// Represents a 3D ray (in world coordinates)
	//----------------------------------------------------------------------------
	struct ray_t {
		arma::fvec origin;
		arma::fvec direction;
	};

	ray_t create_ray(const arma::fvec& origin, int mouse_x, int mouse_y,
					 const arma::fmat& view, const arma::fmat& projection,
					 int window_width, int window_height);

	bool intersects(const ray_t& ray, const arma::fvec& center, float radius,
					arma::fvec &result);


	/********************************* OTHERS ***********************************/

	//----------------------------------------------------------------------------
	// Returns the vertices needed to create a mesh representing the background
	// a gaussian
	//
	// The result is a matrix of shape (2, nb_points * 3)
	//----------------------------------------------------------------------------
	arma::mat get_gaussian_background_vertices(const arma::vec& mu, const arma::mat& sigma,
											   int nb_points = 60);

	//----------------------------------------------------------------------------
	// Returns the vertices needed to create a line representing the border of
	// a gaussian
	//
	// If line_strip is true:
	//  - The result is a matrix of shape (2, nb_points)
	//  - The rendering mode must be GL_LINE_STRIP
	//
	// If line_strip is false:
	//  - The result is a matrix of shape (2, nb_points * 2)
	//  - The rendering mode must be GL_LINES
	//----------------------------------------------------------------------------
	arma::mat get_gaussian_border_vertices(const arma::vec& mu, const arma::mat& sigma,
										   int nb_points = 60, bool line_strip = true);

}
