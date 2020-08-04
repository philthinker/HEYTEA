#include "gfx.h"

using namespace arma;
#define PI				3.14159265358979323846

namespace gfx
{

void init()
{
	glewInit();
}

int checkGlExtension(char const *name)
{
	const char *glExtensions = (const char *)glGetString(GL_EXTENSIONS);
	return (strstr(glExtensions, name) != NULL);
}

bool getGLError()
{
	GLenum err = glGetError();
	//glClearError();
	if( err != GL_NO_ERROR )
	{
		#if !defined(CM_GLES) && !defined(GFX_OSX)
			printf("GL ERROR %d %s",(int)err,gluErrorString(err));
			#endif
		return true;
	}

	return false;
}


arma::fmat perspective(float fovy, float aspect, float zNear, float zFar)
{
	const float top = zNear * tan(fovy / 2.0f);
	const float bottom = -top;
	const float right = top * aspect;
	const float left = -right;

	arma::fmat projection = arma::zeros<fmat>(4,4);

	projection(0, 0) = 2.0f * zNear / (right - left);
	projection(0, 2) = (right + left) / (right - left);
	projection(1, 1) = 2.0f * zNear / (top - bottom);
	projection(1, 2) = (top + bottom) / (top - bottom);
	projection(2, 2) = -(zFar + zNear) / (zFar - zNear);
	projection(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
	projection(3, 2) = -1.0f;

	return projection;
}


void setPerspectiveProjection(float fovy, float aspect, float zNear, float zFar)
{
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();

	arma::fmat projection = perspective(fovy, aspect, zNear, zFar);

	// Add to current matrix
	glMultMatrixf(projection.memptr());

	glMatrixMode( GL_MODELVIEW );
}

void setOrtho( float w, float h )
{
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glOrtho( 0, 0+w, 0+h, 0, -1., 1.);
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
}

void multMatrix( const arma::mat& m_ )
{
	arma::mat m = m_;
	if(m.n_rows==3)
	{
		m.insert_cols(2, 1);
		m.insert_rows(2, 1);
	}
	glMultMatrixd(m.memptr());
}

arma::fmat lookAt(const arma::fvec& position, const arma::fvec& target, const arma::fvec& up)
{
	const arma::fvec f(arma::normalise(target - position));
	const arma::fvec s(arma::normalise(arma::cross(f, up)));
	const arma::fvec u(arma::cross(s, f));

	arma::fmat result = arma::zeros<fmat>(4,4);
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


///////////////////////////////////////////////////////////////

void drawUVQuad( float x  , float y	 , float w , float h, float maxU, float maxV, bool flip )
{
	GLfloat tex_coords[8];

	if(flip)
	{
		tex_coords[0] = 0.0f; tex_coords[1] = maxV;
		tex_coords[2] = 0.0f; tex_coords[3] = 0.0f;
		tex_coords[4] = maxU; tex_coords[5] =  0.0f;
		tex_coords[6] = maxU; tex_coords[7] = maxV;
	}
	else
	{
		tex_coords[0] = 0.0f; tex_coords[1] = 0.0f;
		tex_coords[2] = 0.0f; tex_coords[3] = maxV;
		tex_coords[4] = maxU; tex_coords[5] = maxV;
		tex_coords[6] = maxU; tex_coords[7] = 0.0f;
	}

	GLfloat verts[] = {
		x,y,
		x,y+h,
		x+w,y+h,
		x+w,y
	};

	glEnableClientState( GL_TEXTURE_COORD_ARRAY );
	glTexCoordPointer(2, GL_FLOAT, 0, tex_coords );

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, verts );
	glDrawArrays( GL_TRIANGLE_FAN, 0, 4 );
	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
}

void drawQuad( float x	, float y  , float w , float h )
{
	GLfloat verts[] = {
		x+w,y,
		x+w,y+h,
		x,y+h,
		x,y
	};

	glBegin(GL_TRIANGLE_FAN);
	for( int i = 0; i < 8; i+=2 )
	{
		glVertex2f(verts[i],verts[i+1]);
	}
	glEnd();
}


void enableAntiAliasing( bool aa )
{

	if ( aa )
	{
		glHint( GL_POINT_SMOOTH_HINT, GL_NICEST );
		glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
		glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST );
		glEnable( GL_POINT_SMOOTH );
		glEnable( GL_LINE_SMOOTH );
		glEnable( GL_POLYGON_SMOOTH );
		glLineWidth( 0.5 );
	}	else	{
		glHint( GL_POINT_SMOOTH_HINT, GL_FASTEST );
		glHint( GL_LINE_SMOOTH_HINT, GL_FASTEST );
		glHint( GL_POLYGON_SMOOTH_HINT, GL_FASTEST );
		glDisable( GL_POINT_SMOOTH );
		glDisable( GL_LINE_SMOOTH );
		glDisable( GL_POLYGON_SMOOTH );
	}
}

void vertex( const arma::vec& v	 )
{
	switch(v.size())
	{
		case 2:
		glVertex2f(v[0], v[1]);
		break;
		case 4:
		glVertex4f(v[0], v[1], v[2], v[3]);
		break;
		default:
		glVertex3f(v[0], v[1], v[2]);
		break;
	}
}

void draw( const arma::mat& P, int prim )
{
	if(!P.size())
		return;

	glBegin(prim);
	switch(P.n_rows)
	{
		case 2:
		{
			for( int i = 0; i < P.n_cols; i++ )
			{
				glVertex2f(P.at(0,i), P.at(1,i)); //  )P[i][0], P[i][1] );
			}
			break;
		}
		default:
		{
			for( int i = 0; i < P.n_cols; i++ )
			{
				glVertex3f(P.at(0, i), P.at(1, i), P.at(2, i));//P[i][0], P[i][1], P[i][2] );
			}
		break;
		}
	}
	glEnd();
}

void drawLine(	const arma::vec&  a,  const arma::vec&	b )
{
	glBegin(GL_LINES);
	vertex(a);
	vertex(b);
	glEnd();
}


void drawAxis( const arma::vec& m, float scale	)
{
	arma::vec pos = m.submat(0, 3, 2, 3);
	arma::vec x = pos + m.submat(0, 0, 2, 0)*scale;
	arma::vec y = pos + m.submat(0, 1, 2, 1)*scale;
	arma::vec z = pos + m.submat(0, 2, 2, 2)*scale;

	glBegin(GL_LINE_STRIP);
	glColor4f(1,0,0,1);
	vertex(pos);
	vertex(x);
	glColor4f(0,1,0,1);
	vertex(pos);
	vertex(y);
	glColor4f(0,0,1,1);
	vertex(pos);
	vertex(z);
	glColor4f(1,1,1,1);
	glEnd();
}

///////////////////////////////////////////////////////////////

// Handle release
struct GLObj
{
	enum TYPE
	{
		TEXTURE = 0,
		SHADER = 1,
		PROGRAM,
		VB,
		IB
	};

	GLObj( GLuint glId, TYPE type )
	:
	glId(glId),
	type(type)
	{
	}

	GLObj()
	{
	}

	GLuint glId;
	TYPE type;
};

static void releaseGLObject( GLObj o )
{
	switch(o.type)
	{
		case GLObj::TEXTURE:
			glDeleteTextures(1,&o.glId);
			break;

		case GLObj::SHADER:
			glDeleteShader(o.glId);
			break;

		case GLObj::PROGRAM:
			glDeleteProgram(o.glId);
			break;

		case GLObj::VB:
			glDeleteBuffers(1,&o.glId);
			break;

		case GLObj::IB:
			glDeleteBuffers(1,&o.glId);
			break;
	}
}

struct Shader : public GLObj
{
	Shader( int id=-1 )
	:
	GLObj(id, GLObj::SHADER)
	{

	}

	int refcount=0;

	GLuint glId;
};

struct ShaderProgram : public GLObj
{
	ShaderProgram( int id=-1, int vs=-1, int ps=-1 )
	:
	GLObj(id, GLObj::PROGRAM ),
	ps(ps),
	vs(vs)
	{

	}

	int ps=-1;
	int vs=-1;
};

typedef std::map<int, Shader> ShaderMap;
typedef std::map<int, ShaderProgram> ShaderProgramMap;

static ShaderProgramMap shaderProgramMap;
static ShaderMap shaderMap;

static int curProgram=-1;

static std::string shaderVersion="120";

static void incShaderRefCount(int id)
{
	auto it = shaderMap.find(id);
	if(it != shaderMap.end())
		it->second.refcount++;
}


void removeShader( int id )
{
	auto it = shaderMap.find(id);
	if(it == shaderMap.end())
		return;
	it->second.refcount--;
	if( it->second.refcount <= 0 )
	{
		releaseGLObject(it->second);
		shaderMap.erase(it);
	}
}

void deleteShaderProgram( int id )
{
	auto it = shaderProgramMap.find(id);
	if(it == shaderProgramMap.end())
		return;
	releaseGLObject(it->second);
	removeShader(it->second.vs);
	removeShader(it->second.ps);
	shaderProgramMap.erase(it);
}

void deleteAllShaders()
{
	{
		auto it = shaderMap.begin();
		while( it != shaderMap.end() )
		{
			releaseGLObject(it->second);
			it++;
		}
	}

	{
		auto it = shaderProgramMap.begin();
		while( it != shaderProgramMap.end() )
		{
			releaseGLObject(it->second);
			it++;
		}
	}
}

static bool compileShader( GLuint id, const char* shaderStr )
{
	char errors[1024];

	glShaderSource(id, 1, (const GLchar**)&shaderStr,NULL);

	GLsizei len;
	GLint compiled;

	glCompileShader(id);
	glGetShaderiv(id, GL_COMPILE_STATUS, &compiled);

	if (!compiled)
	{
		glGetShaderInfoLog( id, 1024, &len, errors );
		printf("shader errors: %s\n",errors);
		return false;
	}

	return true;
}

void setShaderVersion( const std::string& version )
{
	shaderVersion = version;
}

static std::string addVersion( const std::string& str )
{
	return std::string("\n#version ") + shaderVersion + "\n\n" + str;
}

int loadVertexShader( const std::string& vs_ )
{
	int id = glCreateShader(GL_VERTEX_SHADER);
	std::string vs = addVersion(vs_);
	if(!compileShader(id, vs.c_str()))
	{
		return -1;
	}

	shaderMap[id] = Shader(id);

	return id;
}

int loadPixelShader( const std::string& ps_ )
{
	int id = glCreateShader(GL_FRAGMENT_SHADER);
	std::string ps = addVersion(ps_);
	if(!compileShader(id, ps.c_str()))
	{
		return -1;
	}

	shaderMap[id] = Shader(id);

	return id;
}

int linkShader( int vs, int ps )
{
	int id = glCreateProgram();

	glAttachShader(id, vs );
	glAttachShader(id, ps );


	// bind attributes
	/*
	for( u32 i = 0; i < _vertexAttributes.size() ; i++ )
	{
		VertexAttribute * a = _vertexAttributes[i];

		glBindAttribLocation(this->_glId, GENERIC_ATTRIB(a->index), a->name.str);

		if(getGLGfx()->getGLError())
		{
			debugPrint("In GLShaderProgram::link");
			return false;
		}

		delete _vertexAttributes[i];
	}

	_vertexAttributes.clear();
	*/

	glLinkProgram(id);

	GLint linked;
	char errors[1024];
	GLsizei len;

	glGetProgramiv(id, GL_LINK_STATUS, &linked);
	if (!linked)
	{
	   glGetProgramInfoLog(id, 1024, &len, errors );
	   printf("GLSL Shader linker error:%s\n",errors);
	   assert(0);
	   return -1;
	}

	shaderProgramMap[id] = ShaderProgram(id, vs, ps);
	incShaderRefCount(vs);
	incShaderRefCount(ps);

	return id;
}

int loadShader( const std::string& vs, const std::string& ps )
{
	int vsid = loadVertexShader(vs);
	if(vsid < 0)
		return -1;

	int psid = loadPixelShader(ps);
	if(psid < 0)
		return -1;

	return linkShader( vsid, psid );
}

int reloadShader( int id, const std::string& vs, const std::string& ps )
{
	int newid = loadShader(vs, ps);
	if(newid==-1)
		return id;

	deleteShaderProgram(id);
	return newid;
}

static int getUniformLocation( const std::string& handle )
{
	int loc = glGetUniformLocation( curProgram, handle.c_str() );
	if(loc==-1)
	{
		printf("shader error: Could not get uniform %s\n", handle.c_str());
	}
	return loc;
}

bool setTexture( const std::string& handle, int sampler )
{
	int id = getUniformLocation(handle);
	if(id == -1)
		return false;

	glUniform1i(id, sampler);
	return true;
}

void bindShader( int id )
{
	glUseProgram(id);
	curProgram = id;
}

void unbindShader()
{
	glUseProgram(0);
	curProgram = -1;
}

bool setInt( const std::string& handle, int v )
{
	int id = getUniformLocation(handle);
	if(id == -1)
		return false;

	glUniform1i(id, v);
	return true;
}

bool setFloat( const std::string& handle, float v )
{
	int id = getUniformLocation(handle);
	if(id == -1)
		return false;
	glUniform1f(id, v);

	return true;
}

bool setVector( const std::string& handle, const arma::vec& v )
{
	int id = getUniformLocation(handle);
	if(id == -1)
		return false;
	fmat fv = conv_to<fvec>::from(v);
	switch(fv.n_rows)
	{
		case 2:
			//glUniform2fv(id, 1, (GLfloat*)fv.memptr());
			glUniform2f(id, fv[0], fv[1]);//(GLfloat*)fv.memptr());
			break;
		case 3:
			glUniform3fv(id, 1, (GLfloat*)fv.memptr());
			break;
		case 4:
			glUniform4fv(id, 1, (GLfloat*)fv.memptr());
			break;
	}

	return true;
}

bool setMatrix( const std::string& handle, const mat& v )
{
	int id = getUniformLocation(handle);
	if(id == -1)
		return false;

	fmat fm = conv_to<fmat>::from(v);

	switch(v.n_rows)
	{
		case 2:
			glUniformMatrix2fv(id, 1, GL_FALSE, (GLfloat*)fm.memptr());
			break;
		case 3:
			glUniformMatrix3fv(id, 1, GL_FALSE, (GLfloat*)fm.memptr());
			break;
		case 4:
			glUniformMatrix4fv(id, 1, GL_FALSE, (GLfloat*)fm.memptr());
			break;
		default:
			return false;
	}

	return true;
}

/*
// Hack, limited number of texture samplers, using a larger number will creash
static int textureSamplers[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

int createTexture(int w, int h, int glFormat, int dataFormat, int dataType )
{
	unsigned char * data = new unsigned char[w*h*4*4];
	int id = createTexture(data, w, h, glFormat, dataFormat, dataType);
	delete [] data;
	return id;
}

int createTexture(void* data, int w, int h, int glFormat, int dataFormat, int dataType )
{
	glDisable( GL_TEXTURE_RECTANGLE );
	glEnable( GL_TEXTURE_2D );

	GLint prevAlignment;
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlignment);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	int id;
	glGenTextures(1, (GLint*)&id);					// Create 1 Texture
	glBindTexture(GL_TEXTURE_2D, id);			// Bind The Texture

	// Does not handle mip maps at the moment
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	int hwWidth=width;
	int hwHeight=height;

	// Check if non power of two
	if(!checkGlExtension("GL_ARB_texture_non_power_of_two"))
	{
		hwWidth = NPOT(width);
		hwHeight = NPOT(height);

		if(hwWidth != width)
		{
			// realign data
			int hww = info.hwWidth;
			int hwh = info.hwHeight;

			int sz = glFormatDescs[fmt].bytesPerPixel;

			unsigned char * src = (unsigned char*)data;
			unsigned char * dst = new unsigned char[ hww*hwh*sz ];
			data = dst;

			memset( dst, 0, hww*hwh*sz );
			for( int y = 0; y < h; y++ )
			{
				memcpy(&dst[y*hww*sz],&src[y*w*sz],w*sz);
			}
		}
	}

	glTexImage2D(	GL_TEXTURE_2D,
					 0,
					 info.glFormat,
					 info.hwWidth,
					 info.hwHeight,
					 0,
					 info.glDataFormat,
					 info.glDataType,
					 data );

	glPixelStorei(GL_UNPACK_ALIGNMENT, prevAlignment);
	glDisable(GL_TEXTURE_2D);

	if(hwWidth != width)
	{
		delete [] (unsigned char*)data;
	}

	return id;
}

void bindTexture( int id, int sampler )
{
	glActiveTexture( GL_TEXTURE0+sampler );
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, id);

	textureSamplers[sampler] = id;
}

void unbindTexture( int sampler )
{
	if(textureSamplers[sampler] > 0)
	{
		glActiveTexture( GL_TEXTURE0+sampler);
		glDisable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
		textureSamplers[sampler] = 0;
	}
}

void grabFrameBuffer( int texId, int w, int h )
{
	bindTexture(id);
	glCopyTexSubImage2D(GL_TEXTURE_2D, //target
					0, //level
					0, //xoffset
					0, //yoffset
					0, //x
					0, //y
					w, //width
					h //height
					);
	unbindTexture();
}
*/
}

arma::mat rot2d( double theta, bool affine )
{
	int d = affine?3:2;
	arma::mat m = arma::eye(d,d);

	double ct = cos(theta);
	double st = sin(theta);

	m(0,0) = ct; m(0,1) = -st;
	m(1,0) = st; m(1,1) = ct;

	return m;
}

arma::mat trans2d( const arma::vec& xy, bool affine )
{
	arma::mat m = arma::eye(3,3);
	m(0,2) = xy[0];
	m(1,2) = xy[1];
	return m;
}

arma::mat trans2d( double x, double y, bool affine )
{
	return trans2d( arma::vec({x, y}) );
}

arma::mat scaling2d( const arma::vec& xy, bool affine )
{
	int d = affine?3:2;
	arma::mat m = arma::eye(d,d);

	m(0,0) = xy[0];
	m(1,1) = xy[1];
	return m;
}

arma::mat scaling2d( double s, bool affine )
{
	return scaling2d( arma::vec({s, s}), affine );
}

arma::mat scaling2d( double x, double y, bool affine )
{
	return scaling2d( arma::vec({x, y}), affine );
}


arma::mat rotX3d( double theta, bool affine )
{
	int d = affine?4:3;
	arma::mat m = arma::eye(d,d);

	double ct = cos(theta);
	double st = sin(theta);

	m(1,1) = ct; m(1,2) = -st;
	m(2,1) = st; m(2,2) = ct;

	return m;
}

arma::mat rotY3d( double theta, bool affine )
{
	int d = affine?4:3;
	arma::mat m = arma::eye(d,d);

	double ct = cos(theta);
	double st = sin(theta);

	m(0,0) = ct; m(0,2) = st;
	m(2,0) = -st; m(2,2) = ct;

	return m;
}

arma::mat rotZ3d( double theta, bool affine )
{
	int d = affine?4:3;
	arma::mat m = arma::eye(d,d);

	double ct = cos(theta);
	double st = sin(theta);

	m(0,0) = ct; m(0,1) = -st;
	m(1,0) = st; m(1,1) = ct;

	return m;
}

arma::mat trans3d( const arma::vec& xyz )
{
	arma::mat m = arma::eye(4,4);
	m(0,3) = xyz[0];
	m(1,3) = xyz[1];
	m(2,3) = xyz[2];
	return m;
}

arma::mat trans3d( double x, double y, double z )
{
	return trans3d( arma::vec({x, y, z}) );
}

arma::mat scaling3d( const arma::vec& xyz, bool affine )
{
	int d = affine?4:3;
	arma::mat m = arma::eye(d,d);

	m(0,0) = xyz[0];
	m(1,1) = xyz[1];
	m(2,2) = xyz[2];
	return m;
}

arma::mat scaling3d( double s, bool affine )
{
	return scaling3d( arma::vec({s, s, s}), affine );
}

arma::mat scaling3d( double x, double y, double z, bool affine )
{
	return scaling3d( arma::vec({x, y, z}), affine );
}



