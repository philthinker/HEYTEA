// Minimalist OpenGL utilities
// adapted from https://github.com/colormotor/colormotor

#pragma once

#ifdef _WIN32
	#define _USE_MATH_DEFINES
#endif

#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>	//for ostringsream
#include <iomanip>	//for setprecision
#include <errno.h>
#include <functional>
#include <fstream>
#include <thread>
#include <complex>

#include "armadillo"

// Detect platform
#ifdef _WIN32
	#define GFX_WINDOWS
	#define NOMINMAX
	#include <windows.h>
#elif __APPLE__
	#define GFX_OSX
#elif __linux__
	#define GFX_LINUX
#elif __unix__
	#define GFX_LINUX
#elif defined(_POSIX_VERSION)
	#define GFX_LINUX
#else
	#error "Unknown platform"
#endif

// GL includes
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

namespace gfx
{

void init();

int checkGlExtension(char const *name);

//////////////////////////////////////////////////
/// Projections/view/transformations
//@{
/// Returns a perspective projection matrix
arma::fmat perspective(float fovy, float aspect, float zNear, float zFar);
/// Sets perspective projection (as in MESA standard)
void setPerspectiveProjection(float fovy, float aspect, float zNear, float zFar);
/// Set orthographic projection with origin at (0,0) top-left
void setOrtho( float w, float h );
/// Wrapper around glMultMatrix with armadillo matrices
/// handles 2d and 3d affine matrices (3x3 and 4x4)
void multMatrix( const arma::mat& mat );
/// Returns a view (or camera) matrix
arma::fmat lookAt(const arma::fvec& position, const arma::fvec& target, const arma::fvec& up);
/// Returns a rotation matrix for the provided axis and angle
arma::fmat rotate(const arma::fvec& axis, float angle);
//@}

//////////////////////////////////////////////////
/// Drawing
//@{
/// Draw a 2d quad providing texture coordinates (u,v)
void drawUVQuad( float x  , float y	 , float w , float h, float maxU=0.0f, float maxV=1.0f, bool flip=true );
/// Draw a 2d quad
void drawQuad( float x, float y, float w, float h );
/// Draw a line between a and b
void drawLine( const arma::vec& a, const arma::vec& b );
/// draw a set of points (columns of the matrix) with a given GL primitive defined with @prim,
/// e.g. GL_LINE_STRIP, GL_LINE_LOOP, GL_POINTS
void draw( const arma::mat & P, int prim=GL_LINE_STRIP );
/// Draw 3d axes defined by a 4x4 matrix (have to test here)
void drawAxis( const arma::mat& mat, float scale );
//@}

//////////////////////////////////////
/// Shader interface (needs further testing)
//@{
/// Convert to string (handy to create a shader, but messes up line numbers, which makes it tricky with errors)
#define STRINGIFY( expr ) #expr

void removeShader( int id );
void deleteShaderProgram( int id );
void deleteAllShaders();

void setShaderVersion( const std::string& version );

int loadShader( const std::string& vs, const std::string& ps );
int reloadShader( int id, const std::string& vs, const std::string& ps );
std::string shaderString( const std::string& path );
bool setTexture( const std::string& handle, int sampler );
void bindShader( int id );
void unbindShader();

bool setInt( const std::string& handle, int v );
bool setBool( const std::string& handle, bool v );
bool setVector( const std::string& handle, const arma::vec& v );
bool setMatrix( const std::string& handle, const arma::mat& v );
//@}

//////////////////////////////////////
/// Texture interface
/// Example combinations of parameters:
///	   { GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE,3 }, //R8G8B8,
///	   { GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE,4 }, //A8R8G8B8
///	   { GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE,1 }, //L8,
//@{
int createTexture(int w, int h, int glFormat=GL_RGB8, int dataFormat=GL_BGRA, int dataType=GL_UNSIGNED_BYTE );
int createTexture(void* data, int w, int h, int glFormat=GL_RGB8, int dataFormat=GL_BGRA, int dataType=GL_UNSIGNED_BYTE );
void bindTexture( int id, int sampler=0 );
void unbindTexture( int sampler=0 );
void grabFrameBuffer( int texId, int w, int h );
//@}

} // end gfx


//////////////////////////////////////
// OpenGL transformation matrix utils

/// 3d transformations, when @affine is set to true a 4x4 matrix is created to handle homogeneous coords
//@{
arma::mat rotX3d( double theta, bool affine=true );
arma::mat rotY3d( double theta, bool affine=true );
arma::mat rotZ3d( double theta, bool affine=true );
arma::mat trans3d( const arma::vec& xyz );
arma::mat trans3d( double x, double y, double z );
arma::mat scaling3d( const arma::vec& xyz, bool affine=true );
arma::mat scaling3d( double s, bool affine=true );
arma::mat scaling3d( double x, double y, double z, bool affine=true );
//@}

//////////////////////////////////////
/// 2d transformations, when @affine is set to true a 3x3 matrix is created to handle homogeneous coords
//@{
arma::mat rot2d( double theta, bool affine=true );
arma::mat trans2d( const arma::vec& xy, bool affine=true );
arma::mat trans2d( double x, double y, bool affine=true );
arma::mat scaling2d( const arma::vec& xy, bool affine=true );
arma::mat scaling2d( double s, bool affine=true );
arma::mat scaling2d( double x, double y, bool affine=true );
//@}
