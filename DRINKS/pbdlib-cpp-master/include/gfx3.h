/*
 * gfx3.h
 *
 * Rendering utility structures and functions based on OpenGL 3.3+
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


#define GFX_NAMESPACE gfx3
#include "gfx_common.h"
#undef GFX_NAMESPACE


namespace gfx3
{
	/**************************** UTILITY FUNCTIONS **************************/

	//-------------------------------------------------------------------------
	// Converts some coordinates from UI-space to shader-space
	//
	// UI coordinates range from (0, 0) to (win_width, win_height)
	// Shader coordinates range from (sh_left, sh_top) to (sh_right, sh_bottom)
	// by default: (-1, 1) to (1, -1)
	//-------------------------------------------------------------------------
	arma::vec ui2shader(const arma::vec& coords, int win_width, int win_height,
						int fb_width, int fb_height, float sh_left = -1.0f,
						float sh_top = 1.0f, float sh_right = 1.0f,
						float sh_bottom = -1.0f);

	//-------------------------------------------------------------------------
	// Converts some coordinates from UI-space to shader-space
	//
	// UI coordinates range from (0, 0) to (win_width, win_height)
	// Shader coordinates range from (sh_left, sh_top) to (sh_right, sh_bottom)
	// by default: (-1, 1) to (1, -1)
	//-------------------------------------------------------------------------
	arma::vec ui2shader(const arma::vec& coords, const window_size_t& window_size,
						float sh_left = -1.0f, float sh_top = 1.0f,
						float sh_right = 1.0f, float sh_bottom = -1.0f);

	//-------------------------------------------------------------------------
	// Converts some coordinates from OpenGL-space to shader-space
	//
	// OpenGL coordinates range from (-fb_width / 2, fb_height / 2) to
	// (fb_width / 2, -fb_height / 2)
	// Shader coordinates range from (sh_left, sh_top) to (sh_right, sh_bottom)
	// by default: (-1, 1) to (1, -1)
	//-------------------------------------------------------------------------
	arma::vec fb2shader_centered(const arma::vec& coords, const window_size_t& window_size,
								 float sh_left = -1.0f, float sh_top = 1.0f,
								 float sh_right = 1.0f, float sh_bottom = -1.0f);


	/******************************** SHADERS ********************************/

	struct shader_fmat_uniform_t {
		GLint	   handle;
		arma::fmat value;
	};


	struct shader_fvec_uniform_t {
		GLint	   handle;
		arma::fvec value;
	};


	struct shader_float_uniform_t {
		GLint handle;
		float value;
	};


	struct shader_bool_uniform_t {
		GLint handle;
		bool value;
	};


	struct shader_int_uniform_t {
		GLint handle;
		int value;
	};


	struct shader_texture_uniform_t {
		GLint handle;
		GLuint texture;
		int nb_dimensions;
	};


	//-------------------------------------------------------------------------
	// Holds all the needed informations about a GLSL program
	//-------------------------------------------------------------------------
	struct shader_t {
		GLuint		id;

		bool		use_lightning;

		// Matrices
		GLint		model_matrix_handle;
		GLint		view_matrix_handle;
		GLint		projection_matrix_handle;

		// Material
		GLint		ambiant_color_handle;	// valid if lightning is used
		GLint		diffuse_color_handle;
		GLint		specular_color_handle;	// valid if lightning is used
		GLint		specular_power_handle;	// valid if lightning is used

		// Textures
		GLint		diffuse_texture_handle; // valid if textures are used

		// Light
		GLint		light_position_handle;	// valid if lightning is used
		GLint		light_color_handle;		// valid if lightning is used
		GLint		light_power_handle;		// valid if lightning is used

		// RTT-specific
		GLint		backbuffer_handle;

		// Application-dependant uniforms
		std::map<std::string, shader_fmat_uniform_t>    fmat_uniforms;
		std::map<std::string, shader_fvec_uniform_t>    fvec_uniforms;
		std::map<std::string, shader_float_uniform_t>   float_uniforms;
		std::map<std::string, shader_bool_uniform_t>    bool_uniforms;
		std::map<std::string, shader_int_uniform_t>     int_uniforms;
		std::map<std::string, shader_texture_uniform_t> texture_uniforms;

		void setUniform(const std::string& name, const arma::fmat& value);
		void setUniform(const std::string& name, const arma::mat& value);
		void setUniform(const std::string& name, const arma::fvec& value);
		void setUniform(const std::string& name, const arma::vec& value);
		void setUniform(const std::string& name, float value);
		void setUniform(const std::string& name, bool value);
		void setUniform(const std::string& name, int value);
		void setTexture(const std::string& name, GLuint texture, int nb_dimensions = 2);
	};


	//-------------------------------------------------------------------------
	// Convert to string (handy to create a shader, but messes up line numbers,
	// which makes it tricky with errors)
	//-------------------------------------------------------------------------
	#define STRINGIFY( expr ) #expr

	//-------------------------------------------------------------------------
	// Load and compile a vertex and a fragment shaders into a GLSL program
	//-------------------------------------------------------------------------
	shader_t loadShader(const std::string& vertex_shader,
						const std::string& fragment_shader,
						const std::string& version = "330 core");

	//-------------------------------------------------------------------------
	// Vertex and fragment shaders for a mesh with normals and one point light
	// source
	//-------------------------------------------------------------------------
	extern const char* VERTEX_SHADER_ONE_LIGHT;
	extern const char* FRAGMENT_SHADER_ONE_LIGHT;

	//-------------------------------------------------------------------------
	// Vertex and fragment shaders without any lightning
	//-------------------------------------------------------------------------
	extern const char* VERTEX_SHADER_COLORED;
	extern const char* FRAGMENT_SHADER_COLORED;

	//-------------------------------------------------------------------------
	// Vertex and fragment shaders with a texture, without any lightning
	//-------------------------------------------------------------------------
	extern const char* VERTEX_SHADER_TEXTURED;
	extern const char* FRAGMENT_SHADER_ONE_TEXTURE;
	extern const char* FRAGMENT_SHADER_GAUSSIAN;

	//-------------------------------------------------------------------------
	// Vertex and fragment shaders for 3D gaussians
	//-------------------------------------------------------------------------
	extern const char* VERTEX_SHADER_3D_GAUSSIAN;
	extern const char* FRAGMENT_SHADER_3D_GAUSSIAN;

	//-------------------------------------------------------------------------
	// Vertex and fragment shaders to do render-to-texture
	//-------------------------------------------------------------------------
	extern const char* RTT_VERTEX_SHADER;
	extern const char* RTT_FRAGMENT_SHADER_GAUSSIAN;
	extern const char* RTT_FRAGMENT_SHADER_TRANSPARENT_GAUSSIAN;
	extern const char* RTT_FRAGMENT_SHADER_FLOW_FIELD;
	extern const char* RTT_FRAGMENT_SHADER_LIC;
	extern const char* RTT_FRAGMENT_SHADER_LIC_COLORED_MEAN;


	/******************************* LIGHTNING *******************************/

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a point light
	//-------------------------------------------------------------------------
	struct point_light_t {
		transforms_t	transforms;
		arma::fvec		color;
		float			power;
	};

	//-------------------------------------------------------------------------
	// A list of lights
	//-------------------------------------------------------------------------
	typedef std::vector<point_light_t> light_list_t;


	/******************************** TEXTURE ********************************/

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a texture
	//-------------------------------------------------------------------------
	struct texture_t {
		GLuint 			id;
		unsigned int	width;
		unsigned int	height;
		unsigned int	depth;
	};

	//-------------------------------------------------------------------------
	// Represent a list of textures
	//-------------------------------------------------------------------------
	typedef std::vector<gfx3::texture_t> texture_list_t;

	//-------------------------------------------------------------------------
	// Create a 2D texture object initialised with an array of pixels
	//
	// The texture will be stored on the GPU as unsigned bytes
	//-------------------------------------------------------------------------
	texture_t create_texture(unsigned int width, unsigned int height,
							 unsigned int nb_channels, const unsigned char* pixels);

	//-------------------------------------------------------------------------
	// Create a 2D texture object initialised with an array of pixels
	//
	// The texture will be stored on the GPU as floats
	//-------------------------------------------------------------------------
	texture_t create_texture(unsigned int width, unsigned int height,
							 unsigned int nb_channels, const float* pixels);

	//-------------------------------------------------------------------------
	// Create a 3D texture object initialised with an array of pixels
	//
	// The texture will be stored on the GPU as floats
	//-------------------------------------------------------------------------
	texture_t create_texture(unsigned int width, unsigned int height,
							 unsigned int depth, unsigned int nb_channels,
							 const float* pixels);

	//-------------------------------------------------------------------------
	// Release the OpenGL resources used by the texture
	//-------------------------------------------------------------------------
	void destroy(const texture_t& texture);


	/*************************** RENDER-TO-TEXTURE ***************************/

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a texture and its associated
	// framebuffer (to be used by a render-to-texture object)
	//-------------------------------------------------------------------------
	struct render_to_texture_buffer_t {
		GLuint    framebuffer;
		texture_t texture;
	};

	//-------------------------------------------------------------------------
	// Represent a list of render-to-texture buffers
	//-------------------------------------------------------------------------
	typedef std::vector<gfx3::render_to_texture_buffer_t> render_to_texture_buffer_list_t;

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a render-to-texture object
	//-------------------------------------------------------------------------
	struct render_to_texture_t {

		// Textures
		render_to_texture_buffer_list_t	buffers;
		unsigned int					nb_buffers;
		unsigned int					current_buffer;
		unsigned int					width;
		unsigned int					height;

		// Rectangular mesh
		GLuint nb_vertices;
		GLuint vertex_buffer;

		// Shader
		shader_t const* shader;

		render_to_texture_t()
		: nb_buffers(0), current_buffer(0), width(0), height(0), nb_vertices(0),
		  vertex_buffer(-1), shader(0)
		{
		}

		inline GLuint texture() const {
			return buffers[current_buffer].texture.id;
		};

		inline GLuint previous_texture() const {
			int previous_buffer = current_buffer - 1;
			if (previous_buffer < 0)
				previous_buffer = nb_buffers - 1;

			return buffers[previous_buffer].texture.id;
		};
	};

	//-------------------------------------------------------------------------
	// Represent a list of render-to-texture objects
	//-------------------------------------------------------------------------
	typedef std::vector<gfx3::render_to_texture_t> render_to_texture_list_t;

	//-------------------------------------------------------------------------
	// Create a render-to-texture object
	//
	// The number of channels of the texture (RGB or RGBA) depends on the
	// number of dimensions of the provided initial color
	//
	// channel_size can be 8 (unsigned byte) or 32 (float)
	//-------------------------------------------------------------------------
	render_to_texture_t createRTT(const shader_t& shader, unsigned int width,
								  unsigned int height, const arma::fvec& color,
								  unsigned int nb_buffers = 1,
								  unsigned int channel_size = 8);

	//-------------------------------------------------------------------------
	// Create a render-to-texture object initialised with an array of pixels
	//
	// The texture will be stored on the GPU as unsigned bytes
	//-------------------------------------------------------------------------
	render_to_texture_t createRTT(const shader_t& shader, unsigned int width,
								  unsigned int height, unsigned int nb_channels,
								  const unsigned char* pixels,
								  unsigned int nb_buffers = 1);

	//-------------------------------------------------------------------------
	// Create a render-to-texture object initialised with an array of pixels
	//
	// The texture will be stored on the GPU as floats
	//-------------------------------------------------------------------------
	render_to_texture_t createRTT(const shader_t& shader, unsigned int width,
								  unsigned int height, unsigned int nb_channels,
								  const float* pixels,
								  unsigned int nb_buffers = 1);

	//-------------------------------------------------------------------------
	// Render a render-to-texture object
	//-------------------------------------------------------------------------
	bool draw(render_to_texture_t &rtt);

	//-------------------------------------------------------------------------
	// Release the OpenGL resources used by the render-to-texture object
	//-------------------------------------------------------------------------
	void destroy(const render_to_texture_t& rtt);


	/********************************* MESHES ********************************/

	//-------------------------------------------------------------------------
	// Holds all the needed informations about a mesh
	//-------------------------------------------------------------------------
	struct model_t {

		model_t()
		: mode(0), nb_vertices(0), vertex_buffer(0), normal_buffer(0),
		  uv_buffer(0), uvw_buffer(0), shader(0), diffuse_texture(0),
		  use_transparency(false)
		{}


		GLenum			mode;

		// Vertex data
		GLuint			nb_vertices;
		GLuint			vertex_buffer;
		GLuint			normal_buffer;
		GLuint			uv_buffer;
		GLuint			uvw_buffer;

		// Transforms
		transforms_t	transforms;

		// Shader
		shader_t const* shader;

		// Material
		arma::fvec		ambiant_color;
		arma::fvec		diffuse_color;
		arma::fvec		specular_color;
		float			specular_power;

		// Textures
		GLuint			diffuse_texture;

		// Other
		bool			use_transparency;
	};

	//-------------------------------------------------------------------------
	// Represent a list of models
	//-------------------------------------------------------------------------
	typedef std::vector<gfx3::model_t> model_list_t;

	//-------------------------------------------------------------------------
	// Create a rectangular mesh, colored (no lightning)
	//-------------------------------------------------------------------------
	model_t create_rectangle(const shader_t& shader, const arma::fvec& color,
							 float width, float height,
							 const arma::fvec& position = arma::zeros<arma::fvec>(3),
							 const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
							 const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a square mesh, colored (no lightning)
	//-------------------------------------------------------------------------
	model_t create_square(const shader_t& shader, const arma::fvec& color, float size,
						  const arma::fvec& position = arma::zeros<arma::fvec>(3),
						  const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						  const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a sphere mesh, either colored or that can be lighted (it depends
	// on the shader)
	//-------------------------------------------------------------------------
	model_t create_sphere(const shader_t& shader, float radius = 1.0f,
						  const arma::fvec& position = arma::zeros<arma::fvec>(3),
						  const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						  const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a line mesh, colored (no lightning), from a matrix containing the
	// point coordinates
	//-------------------------------------------------------------------------
	model_t create_line(const shader_t& shader, const arma::fvec& color,
						const arma::mat& points,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0,
						bool line_strip = true);

	//-------------------------------------------------------------------------
	// Create a line mesh, colored (no lightning), from a matrix containing the
	// point coordinates, with the specified width
	//-------------------------------------------------------------------------
	model_t create_line(const shader_t& shader, const arma::fvec& color,
						const arma::mat& points, float width,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a line mesh, colored (no lightning), from an array containing the
	// point coordinates
	//-------------------------------------------------------------------------
	model_t create_line(const shader_t& shader, const arma::fvec& color,
						const std::vector<arma::vec>& points,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0,
						bool line_strip = true);

	//-------------------------------------------------------------------------
	// Create a line mesh, colored (no lightning), from an array containing the
	// point coordinates, with the specified width
	//-------------------------------------------------------------------------
	model_t create_line(const shader_t& shader, const arma::fvec& color,
						const std::vector<arma::vec>& points, float width,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a mesh, colored (no lightning), from a matrix containing the
	// vertex coordinates
	//-------------------------------------------------------------------------
	model_t create_mesh(const shader_t& shader, const arma::fvec& color,
						const arma::mat& vertices,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
						const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a line mesh representing a gaussian border, colored (no lightning)
	//-------------------------------------------------------------------------
	model_t create_gaussian_border(const shader_t& shader, const arma::fvec& color,
								   const arma::vec& mu, const arma::mat& sigma,
								   const arma::fvec& position = arma::zeros<arma::fvec>(3),
								   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
								   const transforms_t* parent_transforms = 0);

	//-------------------------------------------------------------------------
	// Create a plane mesh, attached to the camera, used to display 3D
	// gaussians, using draw_gaussian_3D()
	//-------------------------------------------------------------------------
	model_t create_gaussian_plane(const shader_t& shader, const camera_t& camera);

	//-------------------------------------------------------------------------
	// Release the OpenGL resources used by the model
	//-------------------------------------------------------------------------
	void destroy(const model_t& model);


	/******************************* RENDERING *******************************/

	//-------------------------------------------------------------------------
	// Render a mesh
	//-------------------------------------------------------------------------
	bool draw(const model_t& model, const arma::fmat& view,
			  const arma::fmat& projection,
			  const light_list_t& lights);

	//-------------------------------------------------------------------------
	// Render a mesh (shortcut when lights aren't used by the shaders)
	//-------------------------------------------------------------------------
	inline bool draw(const model_t& model, const arma::fmat& view,
					 const arma::fmat& projection)
	{
		light_list_t lights;
		return draw(model, view, projection, lights);
	}

	//-------------------------------------------------------------------------
	// Render a rectangular mesh, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_rectangle(const shader_t& shader, const arma::fvec& color,
						float width, float height, const arma::fmat& view,
						const arma::fmat& projection,
						const arma::fvec& position = arma::zeros<arma::fvec>(3),
						const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a line, colored (no lightning), from a matrix containing the
	// point coordinates
	//-------------------------------------------------------------------------
	bool draw_line(const shader_t& shader, const arma::fvec& color,
				   const arma::mat& points, const arma::fmat& view,
				   const arma::fmat& projection,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
				   bool line_strip = true);

	//-------------------------------------------------------------------------
	// Render a line, colored (no lightning), from a matrix containing the
	// point coordinates, with the specified width
	//-------------------------------------------------------------------------
	bool draw_line(const shader_t& shader, const arma::fvec& color,
				   const arma::mat& points, float width,
				   const arma::fmat& view, const arma::fmat& projection,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a line, colored (no lightning), from an array containing the
	// point coordinates
	//-------------------------------------------------------------------------
	bool draw_line(const shader_t& shader, const arma::fvec& color,
				   const std::vector<arma::vec>& points, const arma::fmat& view,
				   const arma::fmat& projection,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4),
				   bool line_strip = true);

	//-------------------------------------------------------------------------
	// Render a line, colored (no lightning), from an array containing the
	// point coordinates, with the specified width
	//-------------------------------------------------------------------------
	bool draw_line(const shader_t& shader, const arma::fvec& color,
				   const std::vector<arma::vec>& points, float width,
				   const arma::fmat& view, const arma::fmat& projection,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a mesh, colored (no lightning), from a matrix containing the
	// vertex coordinates
	//-------------------------------------------------------------------------
	bool draw_mesh(const shader_t& shader, const arma::fvec& color,
				   const arma::mat& vertices,const arma::fmat& view,
				   const arma::fmat& projection,
				   const arma::fvec& position = arma::zeros<arma::fvec>(3),
				   const arma::fmat& rotation = arma::eye<arma::fmat>(4, 4));

	//-------------------------------------------------------------------------
	// Render a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_gaussian(shader_t* shader, const arma::fvec& color,
					   const arma::vec& mu, const arma::mat& sigma,
					   const arma::fmat& view, const arma::fmat& projection,
					   float viewport_width, float viewport_height);

	//-------------------------------------------------------------------------
	// Render the border of a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_gaussian_border(shader_t* shader, const arma::fvec& color,
							  const arma::vec& mu, const arma::mat& sigma,
							  const arma::fmat& view,
							  const arma::fmat& projection,
							  float viewport_width, float viewport_height);

	//-------------------------------------------------------------------------
	// Render the border of a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_gaussian_border(shader_t* shader, const arma::fvec& color,
							  const arma::vec& mu, const arma::mat& sigma,
							  float width, const arma::fmat& view,
							  const arma::fmat& projection,
							  float viewport_width, float viewport_height);

	//-------------------------------------------------------------------------
	// Render a gaussian, colored (no lightning)
	//-------------------------------------------------------------------------
	bool draw_gaussian_3D(model_t* plane, shader_t* shader,
						  const arma::fvec& color,
						  const arma::vec& mu, const arma::mat& sigma,
						  const arma::fmat& view, const arma::fmat& projection);

};
