/*
 * gfx2.h
 *
 * Rendering utility structures and functions based on OpenGL 2 (no shader)
 *
 * Authors: Philip Abbet
 */

#pragma once

#ifdef _WIN32
	#define _USE_MATH_DEFINES
#endif

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include <map>

// Detect the platform
#ifdef _WIN32
	#define GFX_WINDOWS
	#define NOMINMAX
	#include <windows.h>
#elif __APPLE__
	#define GFX_OSX
#elif __linux__ || __unix__ || defined(_POSIX_VERSION)
	#define GFX_LINUX
#else
	#error "Unknown platform"
#endif

// OpenGL includes
#ifdef GFX_WINDOWS
	#include <windows.h>
	#ifndef __glew_h__
		#define GL_GLEXT_PROTOTYPES
		#define GLEW_STATIC
		#include <GL/glew.h>
	#endif
	#include <GL/glu.h>
	#ifndef GL_CLAMP_TO_EDGE
		#define GL_CLAMP_TO_EDGE 0x812F
	#endif
#elif defined GFX_LINUX
	#ifndef __glew_h__
		#define GL_GLEXT_PROTOTYPES
		#include <GL/glew.h>
		#include <GL/glu.h>
		#include <GL/gl.h>
		#include <GL/glx.h>
	#endif
#elif defined GFX_OSX
	#ifndef __glew_h__
		#define GL_GLEXT_PROTOTYPES
		#include <GL/glew.h>
	#endif
	#include <OpenGL/glu.h>
	#include <OpenGL/OpenGL.h>
	#include <OpenGL/gl.h>
	#include <OpenGL/glext.h>
#endif


#define GFX_NAMESPACE gfx2
#include "gfx_common.h"
#undef GFX_NAMESPACE


namespace gfx2
{
	/******************************* LIGHTNING *******************************/

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a point light
	//-------------------------------------------------------------------------
	struct point_light_t {

		// Constructor
		point_light_t()
		: diffuse_color({1.0f, 1.0f, 1.0f, 1.0f}),
		  ambient_color({0.0f, 0.0f, 0.0f, 1.0f}),
		  specular_color({1.0f, 1.0f, 1.0f, 1.0f})
		{
		}

		transforms_t transforms;
		arma::fvec   diffuse_color;
		arma::fvec   ambient_color;
		arma::fvec   specular_color;
	};

	//-------------------------------------------------------------------------
	// A list of lights
	//-------------------------------------------------------------------------
	typedef std::vector<point_light_t> light_list_t;


	/******************************** TEXTURES *******************************/

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a texture
	//-------------------------------------------------------------------------
	struct texture_t {
		GLuint id;
		GLuint width;
		GLuint height;
		GLenum format;
		GLenum type;

		union {
			float* pixels_f;
			unsigned char* pixels_b;
		};
	};

	//-------------------------------------------------------------------------
	// Create a texture
	//-------------------------------------------------------------------------
	texture_t create_texture(int width, int height, GLenum format, GLenum type);

	//-------------------------------------------------------------------------
	// Create a texture
	//-------------------------------------------------------------------------
	void destroy(texture_t &texture);


	/********************************* MESHES ********************************/

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a mesh
	//-------------------------------------------------------------------------
	struct model_t {
		GLenum			mode;

		// Vertex data
		GLuint			nb_vertices;
		GLfloat*		vertex_buffer;
		GLfloat*		normal_buffer;
		GLfloat*		uv_buffer;

		// Transforms
		transforms_t	transforms;

		// Material
		arma::fvec		ambiant_color;
		arma::fvec		diffuse_color;
		arma::fvec		specular_color;
		float			specular_power;
		texture_t		texture;

		// Other
		bool			lightning_enabled;
		bool			use_one_minus_src_alpha_blending;
	};

	//-------------------------------------------------------------------------
	// Represent a list of models
	//-------------------------------------------------------------------------
	typedef std::vector<gfx2::model_t> model_list_t;

	//-------------------------------------------------------------------------
	// Create a rectangular mesh, colored (no lightning)
	//-------------------------------------------------------------------------
	model_t create_rectangle(const arma::fvec& color, float width, float height,
							 const arma::fvec& position = arma::zeros<arma::fvec>(3),
							 const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
							 const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a rectangular mesh, textured (no lightning)
	//-------------------------------------------------------------------------
	model_t create_rectangle(const texture_t& texture, float width, float height,
							 const arma::fvec& position = arma::zeros<arma::fvec>(3),
							 const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
							 const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a square mesh, colored (no lightning)
	//-------------------------------------------------------------------------
	model_t create_square(const arma::fvec& color, float size,
						  const arma::fvec& position = arma::zeros<arma::fvec>(3),
						  const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						  const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a sphere mesh, lighted
	//-------------------------------------------------------------------------
	model_t create_sphere(float radius = 1.0f,
						  const arma::fvec& position = arma::zeros<arma::fvec>(3),
						  const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						  const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a line mesh, colored (no lightning), from a matrix containing the
	// point coordinates
	//-------------------------------------------------------------------------
	model_t create_line(const arma::fvec& color, const arma::mat& points,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0,
						bool line_strip = true);

	//-------------------------------------------------------------------------
	// Create a line mesh, colored (no lightning), from an array containing the
	// point coordinates
	//-------------------------------------------------------------------------
	model_t create_line(const arma::fvec& color, const std::vector<arma::vec>& points,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0,
						bool line_strip = true);

	//-------------------------------------------------------------------------
	// Create a mesh, colored (no lightning), from a matrix containing the
	// vertex coordinates
	//-------------------------------------------------------------------------
	model_t create_mesh(const arma::fvec& color, const arma::mat& vertices,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a mesh representing a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	model_t create_gaussian_background(const arma::fvec& color, const arma::vec& mu,
									   const arma::mat& sigma,
									   const arma::fvec& position = arma::zeros<arma::fvec>(3),
									   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
									   const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a line mesh representing a gaussian border, colored (no lightning)
	//-------------------------------------------------------------------------
	model_t create_gaussian_border(const arma::fvec& color, const arma::vec& mu,
								   const arma::mat& sigma,
								   const arma::fvec& position = arma::zeros<arma::fvec>(3),
								   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
								   const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Release the OpenGL resources used by the model
	//-------------------------------------------------------------------------
	void destroy(model_t &model);


	/******************************* RENDERING *******************************/

	//-------------------------------------------------------------------------
	// Render a mesh
	//-------------------------------------------------------------------------
	bool draw(const model_t& model, const light_list_t& lights);

	//-------------------------------------------------------------------------
	// Render a mesh (shortcut when lights aren't used)
	//-------------------------------------------------------------------------
	inline bool draw(const model_t& model)
	{
		light_list_t lights;
		return draw(model, lights);
	}

	//-------------------------------------------------------------------------
	// Render a rectangular mesh, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_rectangle(const arma::fvec& color, float width, float height,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a rectangular mesh, textured (no lightning)
	//-------------------------------------------------------------------------
	bool draw_rectangle(const texture_t& texture, float width, float height,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a line, colored (no lightning), from a matrix containing the
	// point coordinates
	//-------------------------------------------------------------------------
	bool draw_line(const arma::fvec& color, const arma::mat& points,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a line, colored (no lightning), from an array containing the
	// point coordinates
	//-------------------------------------------------------------------------
	bool draw_line(const arma::fvec& color, const std::vector<arma::vec>& points,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a mesh, colored (no lightning), from a matrix containing the
	// vertex coordinates
	//-------------------------------------------------------------------------
	bool draw_mesh(const arma::fvec& color, const arma::mat& vertices,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_gaussian(const arma::fvec& color, const arma::vec& mu,
					   const arma::mat& sigma, bool background = true,
					   bool border = true);

	//-------------------------------------------------------------------------
	// Render the border of a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	inline bool draw_gaussian_border(const arma::fvec& color, const arma::vec& mu,
									 const arma::mat& sigma) {

		return draw_gaussian(color, mu, sigma, false, true);
	}

	//-------------------------------------------------------------------------
	// Render the background of a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	inline bool draw_gaussian_background(const arma::fvec& color, const arma::vec& mu,
										 const arma::mat& sigma) {

		return draw_gaussian(color, mu, sigma, true, false);
	}

	//-------------------------------------------------------------------------
	// Render a 3D gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_gaussian_3D(const arma::fvec& color, const arma::vec& mu,
						  const arma::mat& sigma);

};
