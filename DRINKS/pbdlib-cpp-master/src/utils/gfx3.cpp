/*
 * gfx3.cpp
 *
 * Rendering utility structures and functions based on OpenGL 3.3+
 *
 * Authors: Philip Abbet
 */

#include <gfx3.h>

#define GFX_NAMESPACE gfx3
#include "gfx_common.cpp"
#undef GFX_NAMESPACE


namespace gfx3 {

/****************************** UTILITY FUNCTIONS ****************************/

arma::vec ui2shader(const arma::vec& coords, int win_width, int win_height,
					int fb_width, int fb_height, float sh_left, float sh_top,
					float sh_right, float sh_bottom) {
	arma::vec result = ui2fb_centered(coords, win_width, win_height, fb_width, fb_height);

	result(0) = result(0) * (sh_right - sh_left) / (float) fb_width + (1.0f + sh_left);
	result(1) = result(1) * (sh_top - sh_bottom) / (float) fb_height + (1.0f + sh_bottom);

	return result;
}

//-----------------------------------------------

arma::vec ui2shader(const arma::vec& coords, const window_size_t& window_size,
					float sh_left, float sh_top, float sh_right, float sh_bottom) {
	arma::vec result = ui2fb_centered(coords, window_size);

	result(0) = result(0) * (sh_right - sh_left) / (float) window_size.fb_width + (1.0f + sh_left);
	result(1) = result(1) * (sh_top - sh_bottom) / (float) window_size.fb_height + (1.0f + sh_bottom);

	return result;
}

//-----------------------------------------------

arma::vec fb2shader_centered(const arma::vec& coords, const window_size_t& window_size,
							 float sh_left, float sh_top, float sh_right, float sh_bottom) {
	arma::vec result(2);

	result(0) = coords(0) * (sh_right - sh_left) / (float) window_size.fb_width + (1.0f + sh_left);
	result(1) = coords(1) * (sh_top - sh_bottom) / (float) window_size.fb_height + (1.0f + sh_bottom);

	return result;
}


/********************************** SHADERS **********************************/

void shader_t::setUniform(const std::string& name, const arma::fmat& value)
{
	auto iter = fmat_uniforms.find(name);

	if (iter == fmat_uniforms.end())
	{
		shader_fmat_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.value = value;
		fmat_uniforms[name] = entry;
	}
	else
	{
		iter->second.value = value;
	}
}

//-----------------------------------------------

void shader_t::setUniform(const std::string& name, const arma::mat& value)
{
	auto iter = fmat_uniforms.find(name);

	if (iter == fmat_uniforms.end())
	{
		shader_fmat_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.value = arma::conv_to<arma::fmat>::from(value);
		fmat_uniforms[name] = entry;
	}
	else
	{
		iter->second.value = arma::conv_to<arma::fmat>::from(value);
	}
}

//-----------------------------------------------

void shader_t::setUniform(const std::string& name, const arma::fvec& value)
{
	auto iter = fvec_uniforms.find(name);

	if (iter == fvec_uniforms.end())
	{
		shader_fvec_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.value = value;
		fvec_uniforms[name] = entry;
	}
	else
	{
		iter->second.value = value;
	}
}

//-----------------------------------------------

void shader_t::setUniform(const std::string& name, const arma::vec& value)
{
	auto iter = fvec_uniforms.find(name);

	if (iter == fvec_uniforms.end())
	{
		shader_fvec_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.value = arma::conv_to<arma::fvec>::from(value);
		fvec_uniforms[name] = entry;
	}
	else
	{
		iter->second.value = arma::conv_to<arma::fvec>::from(value);
	}
}

//-----------------------------------------------

void shader_t::setUniform(const std::string& name, float value)
{
	auto iter = float_uniforms.find(name);

	if (iter == float_uniforms.end())
	{
		shader_float_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.value = value;
		float_uniforms[name] = entry;
	}
	else
	{
		iter->second.value = value;
	}
}

//-----------------------------------------------

void shader_t::setUniform(const std::string& name, bool value)
{
	auto iter = bool_uniforms.find(name);

	if (iter == bool_uniforms.end())
	{
		shader_bool_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.value = value;
		bool_uniforms[name] = entry;
	}
	else
	{
		iter->second.value = value;
	}
}

//-----------------------------------------------

void shader_t::setUniform(const std::string& name, int value)
{
	auto iter = int_uniforms.find(name);

	if (iter == int_uniforms.end())
	{
		shader_int_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.value = value;
		int_uniforms[name] = entry;
	}
	else
	{
		iter->second.value = value;
	}
}

//-----------------------------------------------

void shader_t::setTexture(const std::string& name, GLuint texture, int nb_dimensions)
{
	auto iter = texture_uniforms.find(name);

	if (iter == texture_uniforms.end())
	{
		shader_texture_uniform_t entry;
		entry.handle = glGetUniformLocation(this->id, name.c_str());
		if (entry.handle == -1)
			return;

		entry.texture = texture;
		entry.nb_dimensions = nb_dimensions;
		texture_uniforms[name] = entry;
	}
	else
	{
		iter->second.texture = texture;
	}
}

//-----------------------------------------------

GLuint compileShader(GLenum shader_type, const char* shader_source)
{
	GLuint id = glCreateShader(shader_type);
	GLint compiled;

	glShaderSource(id, 1, (const GLchar**) &shader_source, NULL);
	glCompileShader(id);
	glGetShaderiv(id, GL_COMPILE_STATUS, &compiled);

	if (!compiled)
	{
		char errors[1024];
		GLsizei len;

		glGetShaderInfoLog(id, 1024, &len, errors);
		printf("shader errors: %s\n", errors);
		return 0;
	}

	return id;
}

//-----------------------------------------------

GLuint linkShader(GLuint vertex_shader, GLuint fragment_shader)
{
	GLuint id = glCreateProgram();

	glAttachShader(id, vertex_shader);
	glAttachShader(id, fragment_shader);
	glLinkProgram(id);

	GLint linked;

	glGetProgramiv(id, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		char errors[1024];
		GLsizei len;

		glGetProgramInfoLog(id, 1024, &len, errors);
		printf("GLSL Shader linker error:%s\n",errors);
		return 0;
	}

	return id;
}

//-----------------------------------------------

shader_t loadShader(const std::string& vertex_shader,
					const std::string& fragment_shader,
					const std::string& version)
{
	shader_t shader = { 0 };
	shader.id = 0;

	// Compile the vertex shader
	GLuint vs_id = compileShader(
		GL_VERTEX_SHADER, (std::string("#version ") + version + "\n\n" + vertex_shader).c_str());

	if (vs_id == 0)
		return shader;

	// Compile the fragment shader
	GLuint fg_id = compileShader(
		GL_FRAGMENT_SHADER, (std::string("#version ") + version + "\n\n" + fragment_shader).c_str());

	if (fg_id == 0)
		return shader;

	// Link the shaders into a GLSL program
	shader.id = linkShader(vs_id, fg_id);

	// Transformation matrices-related uniforms
	shader.model_matrix_handle		= glGetUniformLocation(shader.id, "ModelMatrix");
	shader.view_matrix_handle		= glGetUniformLocation(shader.id, "ViewMatrix");
	shader.projection_matrix_handle = glGetUniformLocation(shader.id, "ProjectionMatrix");

	// Material-related uniforms
	shader.ambiant_color_handle		= glGetUniformLocation(shader.id, "AmbiantColor");
	shader.diffuse_color_handle		= glGetUniformLocation(shader.id, "DiffuseColor");
	shader.specular_color_handle	= glGetUniformLocation(shader.id, "SpecularColor");
	shader.specular_power_handle	= glGetUniformLocation(shader.id, "SpecularPower");

	// Texture-related uniforms
	shader.diffuse_texture_handle	= glGetUniformLocation(shader.id, "DiffuseTexture");

	// Light-related uniforms
	shader.light_position_handle	= glGetUniformLocation(shader.id, "LightPosition");
	shader.light_color_handle		= glGetUniformLocation(shader.id, "LightColor");
	shader.light_power_handle		= glGetUniformLocation(shader.id, "LightPower");

	// Determine if lightning is used by the shaders
	shader.use_lightning = (shader.light_position_handle != -1);

	// Backbuffer
	shader.backbuffer_handle = glGetUniformLocation(shader.id, "BackBuffer");

	return shader;
}

//-----------------------------------------------

void sendApplicationUniforms(const shader_t* shader)
{
	// Send the application-specific uniforms to the shader
	for (auto iter = shader->fmat_uniforms.begin(), iterEnd = shader->fmat_uniforms.end();
		 iter != iterEnd; ++iter)
	{
		if (iter->second.value.n_rows == 4)
		{
			if (iter->second.value.n_cols == 4)
				glUniformMatrix4fv(iter->second.handle, 1, GL_FALSE, iter->second.value.memptr());
			else if (iter->second.value.n_cols == 3)
				glUniformMatrix2x3fv(iter->second.handle, 1, GL_FALSE, iter->second.value.memptr());
			else if (iter->second.value.n_cols == 2)
				glUniformMatrix2x4fv(iter->second.handle, 1, GL_FALSE, iter->second.value.memptr());
		}
		else if (iter->second.value.n_rows == 2)
		{
			if (iter->second.value.n_cols == 2)
				glUniformMatrix2fv(iter->second.handle, 1, GL_FALSE, iter->second.value.memptr());
			else if (iter->second.value.n_cols == 3)
				glUniformMatrix3x2fv(iter->second.handle, 1, GL_FALSE, iter->second.value.memptr());
			else if (iter->second.value.n_cols == 4)
				glUniformMatrix4x2fv(iter->second.handle, 1, GL_FALSE, iter->second.value.memptr());
		}
		else if ((iter->second.value.n_rows == 3) && (iter->second.value.n_cols == 3)) {
			glUniformMatrix3fv(iter->second.handle, 1, GL_FALSE, iter->second.value.memptr());
		}
	}

	for (auto iter = shader->fvec_uniforms.begin(), iterEnd = shader->fvec_uniforms.end();
		 iter != iterEnd; ++iter)
	{
		if (iter->second.value.n_rows == 4)
			glUniform4fv(iter->second.handle, 1, iter->second.value.memptr());
		else if (iter->second.value.n_rows == 3)
			glUniform3fv(iter->second.handle, 1, iter->second.value.memptr());
		else if (iter->second.value.n_rows == 2)
			glUniform2fv(iter->second.handle, 1, iter->second.value.memptr());
	}

	for (auto iter = shader->float_uniforms.begin(), iterEnd = shader->float_uniforms.end();
		 iter != iterEnd; ++iter)
	{
		glUniform1f(iter->second.handle, iter->second.value);
	}

	for (auto iter = shader->bool_uniforms.begin(), iterEnd = shader->bool_uniforms.end();
		 iter != iterEnd; ++iter)
	{
		glUniform1i(iter->second.handle, iter->second.value);
	}

	for (auto iter = shader->int_uniforms.begin(), iterEnd = shader->int_uniforms.end();
		 iter != iterEnd; ++iter)
	{
		glUniform1i(iter->second.handle, iter->second.value);
	}
}


//-----------------------------------------------

void sendTextures(const shader_t* shader, GLuint next_texture = GL_TEXTURE0)
{
	if (shader->backbuffer_handle >= 0)
		++next_texture;

	for (auto iter = shader->texture_uniforms.begin(), iterEnd = shader->texture_uniforms.end();
		 iter != iterEnd; ++iter)
	{
		glActiveTexture(next_texture);

		if (iter->second.nb_dimensions == 2)
			glBindTexture(GL_TEXTURE_2D, iter->second.texture);
		else
			glBindTexture(GL_TEXTURE_3D, iter->second.texture);

		glUniform1i(iter->second.handle, next_texture - GL_TEXTURE0);
		++next_texture;
	}
}

//-----------------------------------------------

const char* VERTEX_SHADER_ONE_LIGHT = STRINGIFY(
	// Input vertex data
	layout(location = 0) in vec3 vertex_position;
	layout(location = 1) in vec3 normal;

	// Values that stay constant for the whole mesh
	uniform mat4 ModelMatrix;
	uniform mat4 ViewMatrix;
	uniform mat4 ProjectionMatrix;
	uniform vec3 LightPosition;

	// Output data ; will be interpolated for each fragment
	out vec3 position_worldspace;
	out vec3 eye_direction_cameraspace;
	out vec3 light_direction_cameraspace;
	out vec3 normal_cameraspace;

	void main() {
		// Position of the vertex, in clip space
		gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vertex_position, 1);

		// Position of the vertex, in worldspace
		position_worldspace = (ModelMatrix * vec4(vertex_position, 1)).xyz;

		// Vector that goes from the vertex to the camera, in camera space.
		// In camera space, the camera is at the origin (0,0,0).
		vec3 vertex_position_cameraspace = (ViewMatrix * vec4(position_worldspace, 1)).xyz;
		eye_direction_cameraspace = vec3(0,0,0) - vertex_position_cameraspace;

		// Vector that goes from the vertex to the light, in camera space
		vec3 light_position_cameraspace = (ViewMatrix * vec4(LightPosition, 1)).xyz;
		light_direction_cameraspace = light_position_cameraspace + eye_direction_cameraspace;

		// Normal of the vertex, in camera space (Only correct if ModelMatrix does
		// not scale the model)
		normal_cameraspace = (ViewMatrix * ModelMatrix * vec4(normal, 0)).xyz;
	}
);

//-----------------------------------------------

const char* FRAGMENT_SHADER_ONE_LIGHT = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec3 AmbiantColor;
	uniform vec4 DiffuseColor;
	uniform vec3 SpecularColor;
	uniform float SpecularPower;
	uniform vec3 LightPosition;
	uniform vec3 LightColor;
	uniform float LightPower;

	// Interpolated values from the vertex shaders
	in vec3 position_worldspace;
	in vec3 eye_direction_cameraspace;
	in vec3 light_direction_cameraspace;
	in vec3 normal_cameraspace;

	// Output data
	out vec4 color;

	void main() {
		// Normal of the computed fragment, in camera space
		vec3 n = normalize(normal_cameraspace);

		// Direction of the light (from the fragment to the light)
		vec3 l = normalize(light_direction_cameraspace);

		// Eye vector (towards the camera)
		vec3 E = normalize(eye_direction_cameraspace);

		// Direction in which the triangle reflects the light
		vec3 R = reflect(-l, n);

		// Cosine of the angle between the normal and the light direction,
		// clamped above 0
		//	- light is at the vertical of the triangle -> 1
		//	- light is perpendicular to the triangle -> 0
		//	- light is behind the triangle -> 0
		float cos_theta = clamp(dot(n, l), 0, 1);

		// Cosine of the angle between the Eye vector and the Reflect vector,
		// clamped to 0
		//	- Looking into the reflection -> 1
		//	- Looking elsewhere -> < 1
		float cos_alpha = clamp(dot(E, R), 0, 1);

		// Distance to the light
		float distance = length(LightPosition - position_worldspace);

		// Computation of the color of the fragment
		color.rgb = AmbiantColor +
					DiffuseColor.rgb * LightColor * LightPower * cos_theta / (distance * distance) +
					SpecularColor * LightColor * LightPower * pow(cos_alpha, SpecularPower) / (distance * distance);

		color.a = DiffuseColor.a;
	}
);

//-----------------------------------------------

const char* VERTEX_SHADER_COLORED = STRINGIFY(
	// Input vertex data
	layout(location = 0) in vec3 vertex_position;

	// Values that stay constant for the whole mesh
	uniform mat4 ModelMatrix;
	uniform mat4 ViewMatrix;
	uniform mat4 ProjectionMatrix;

	void main() {
		// Position of the vertex, in clip space
		gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vertex_position, 1);
	}
);

//-----------------------------------------------

const char* FRAGMENT_SHADER_COLORED = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec4 DiffuseColor;

	// Output data
	out vec4 color;

	void main() {
		color = DiffuseColor;
	}
);

//-----------------------------------------------

const char* VERTEX_SHADER_TEXTURED = STRINGIFY(
	// Input vertex data
	layout(location = 0) in vec3 vertex_position;
	layout(location = 2) in vec2 vertex_UV;

	// Values that stay constant for the whole mesh
	uniform mat4 ModelMatrix;
	uniform mat4 ViewMatrix;
	uniform mat4 ProjectionMatrix;

	// Output data ; will be interpolated for each fragment
	out vec2 UV;

	void main() {
		// Position of the vertex, in clip space
		gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vertex_position, 1);

		UV = vertex_UV;
	}
);

//-----------------------------------------------

const char* FRAGMENT_SHADER_ONE_TEXTURE = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform sampler2D DiffuseTexture;

	// Input data
	in vec2 UV;

	// Output data
	out vec3 color;

	void main() {
		color = texture(DiffuseTexture, UV).rgb;
	}
);

//-----------------------------------------------

char const* FRAGMENT_SHADER_GAUSSIAN = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec2 Mu;
	uniform mat2 InvSigma;
	uniform vec4 GaussianColor;

	// Input data
	in vec2 UV;

	// Output data
	out vec4 color;

	void main() {
		vec2 e = UV - Mu;

		float p = clamp(exp(-(InvSigma[0][0] * e.x * e.x + 2 * InvSigma[0][1] * e.x * e.y +
							  InvSigma[1][1] * e.y * e.y)), 0, 0.5f);

		color = vec4(GaussianColor.x, GaussianColor.y, GaussianColor.z,
					 clamp(GaussianColor.a * 2 * p, 0.0, 1.0));
	}
);

//-----------------------------------------------

const char* VERTEX_SHADER_3D_GAUSSIAN = STRINGIFY(
	// Input vertex data
	layout(location = 0) in vec3 vertex_position;

	// Values that stay constant for the whole mesh
	uniform mat4 ModelMatrix;
	uniform mat4 ViewMatrix;
	uniform mat4 ProjectionMatrix;

	// Output data ; will be interpolated for each fragment
	out vec3 vertex_position_cameraspace;

	void main() {
		// Position of the vertex, in clip space
		gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vertex_position, 1);

		// Position of the vertex, in worldspace
		vec3 position_worldspace = (ModelMatrix * vec4(vertex_position, 1)).xyz;

		// Vector that goes from the vertex to the camera, in camera space.
		// In camera space, the camera is at the origin (0,0,0).
		vertex_position_cameraspace = (ViewMatrix * vec4(position_worldspace, 1)).xyz;
	}
);

//-----------------------------------------------

char const* FRAGMENT_SHADER_3D_GAUSSIAN = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec3 Mu;
	uniform mat3 InvSigma;
	uniform vec4 DiffuseColor;

	// Interpolated values from the vertex shaders
	in vec3 vertex_position_cameraspace;

	// Output data
	out vec4 color;

	void main() {
		vec3 eye_dir = normalize(vertex_position_cameraspace);

		vec3 dir_step = eye_dir * 0.01;

		vec3 position = vertex_position_cameraspace;

		color.a = 0.0f;

		for (int i = 0; i < 200; ++i) {
			vec3 e = position - Mu;

			float alpha = clamp(exp(-(InvSigma[0][0] * e.x * e.x + 2 * InvSigma[0][1] * e.x * e.y +
									  InvSigma[1][1] * e.y * e.y + 2 * InvSigma[0][2] * e.x * e.z +
									  InvSigma[2][2] * e.z * e.z + 2 * InvSigma[1][2] * e.y * e.z)),
								0, 1.0f
			);

			if (alpha > color.a)
				color.a = alpha;

			// Stop when the alpha becomes significantly smaller than the maximum
			// value seen so far
			else if (alpha < color.a * 0.9f)
				break;

			// Stop when the alpha becomes very large
			if (color.a >= 0.999f)
				break;

			position = position + dir_step;
		}

		color.rgb = DiffuseColor.rgb;
		color.a = color.a * DiffuseColor.a;
	}
);

//-----------------------------------------------

const char* RTT_VERTEX_SHADER = STRINGIFY(
	// Input vertex data
	layout(location = 0) in vec3 vertex_position;

	// Output data ; will be interpolated for each fragment
	out vec2 coords;

	void main() {
		gl_Position = vec4(vertex_position, 1);
		coords = vertex_position.xy;
	}
);

//-----------------------------------------------

char const* RTT_FRAGMENT_SHADER_GAUSSIAN = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec2 Scale;
	uniform vec2 Mu;
	uniform mat2 InvSigma;
	uniform vec3 GaussianColor;
	uniform vec3 BackgroundColor;

	// Input data
	in vec2 coords;

	// Output data
	layout(location = 0) out vec3 color;

	void main() {
		vec2 e = coords * Scale - Mu;
		float a = clamp(exp(-(InvSigma[0][0] * e.x * e.x + 2 * InvSigma[0][1] * e.x * e.y +
							  InvSigma[1][1] * e.y * e.y)), 0, 1.0f);

		color = GaussianColor * a + (1.0f - a) * BackgroundColor;
	}
);

//-----------------------------------------------

char const* RTT_FRAGMENT_SHADER_TRANSPARENT_GAUSSIAN = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec2 Scale;
	uniform vec2 Mu;
	uniform mat2 InvSigma;
	uniform vec3 GaussianColor;

	// Input data
	in vec2 coords;

	// Output data
	layout(location = 0) out vec4 color;

	void main() {
		vec2 e = coords * Scale - Mu;
		float a = clamp(exp(-(InvSigma[0][0] * e.x * e.x + 2 * InvSigma[0][1] * e.x * e.y +
							  InvSigma[1][1] * e.y * e.y)), 0, 1.0f);

		color.rgb = GaussianColor;
		color.a = a;
	}
);

//-----------------------------------------------

const char* RTT_FRAGMENT_SHADER_FLOW_FIELD = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec2 Target;
	uniform mat2 InvSigma;

	// Input data
	in vec2 coords;

	// Output data
	layout(location = 0) out vec2 color;

	void main() {
		vec2 dTarget = -InvSigma * (coords - Target);
		dTarget = dTarget / length(dTarget);

		color = dTarget;
	}
);

//-----------------------------------------------

const char* RTT_FRAGMENT_SHADER_LIC = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform vec2 Resolution;
	uniform float Time;
	uniform sampler2D FlowField;
	uniform sampler2D BackBuffer;

	// Input data
	in vec2 coords;

	// Output data
	layout(location = 0) out vec3 color;

	// Constants
	const int Length = 10;
	const int nbPass = 5;

	void main() {
		vec2 uvUnit = 1.0 / Resolution.xy;
		vec2 uv = (coords.xy + 1.0) * 0.5;

		color = vec3(0);

		// Random noise
		color += vec3(fract(sin(dot(uv.xy, vec2(12.9898, 78.233) * sin(Time))) * 43758.5453));

		for (int n = 0; n < nbPass; n++) {
			for (int i = 0; i < Length; i++) {
				color += texture(BackBuffer, uv - texture(FlowField, uv).rg * uvUnit * float(i + 4)).rgb;
			}
		}
		color /= float(Length) * float(nbPass) + 1.0;
	}
);

//-----------------------------------------------

const char* RTT_FRAGMENT_SHADER_LIC_COLORED_MEAN = STRINGIFY(
	// Values that stay constant for the whole mesh
	uniform sampler2D FlowField;
	uniform sampler2D Image1;
	uniform sampler2D Image2;
	uniform sampler2D Image3;
	uniform sampler2D Image4;
	uniform sampler2D Image5;

	// Input data
	in vec2 coords;

	// Output data
	layout(location = 0) out vec3 color;

	void main() {
		const float pi = 3.1415926538;

		vec2 uv = (coords.xy + 1.0) * 0.5;

		color = (texture(Image1, uv).rgb + texture(Image2, uv).rgb +
				 texture(Image3, uv).rgb + texture(Image4, uv).rgb +
				 texture(Image5, uv).rgb) / 5.0;

		color = clamp((color - 0.49) * 20.0 + 0.49, 0, 1);

		vec2 dir = texture(FlowField, uv).rg;
		float angle = atan(dir.y, dir.x);

		const float inc = 2 * pi / 6;

		vec3 start;
		vec3 end;

		if (angle < -pi + inc) {
			start = vec3(0.9, 0.1, 0.1);
			end = vec3(0.9, 0.9, 0.1);
			angle = (angle + pi) / inc;
		}
		else if (angle < -pi + 2 * inc) {
			start = vec3(0.9, 0.9, 0.1);
			end = vec3(0.1, 0.9, 0.1);
			angle = (angle + pi - inc) / inc;
		}
		else if (angle < -pi + 3 * inc) {
			start = vec3(0.1, 0.9, 0.1);
			end = vec3(0.1, 0.9, 0.9);
			angle = (angle + pi - 2 * inc) / inc;
		}
		else if (angle < -pi + 4 * inc) {
			start = vec3(0.1, 0.9, 0.9);
			end = vec3(0.1, 0.1, 0.9);
			angle = (angle + pi - 3 * inc) / inc;
		}
		else if (angle < -pi + 5 * inc) {
			start = vec3(0.1, 0.1, 0.9);
			end = vec3(0.9, 0.1, 0.9);
			angle = (angle + pi - 4 * inc) / inc;
		}
		else {
			start = vec3(0.9, 0.1, 0.9);
			end = vec3(0.9, 0.1, 0.1);
			angle = (angle + pi - 5 * inc) / inc;
		}

		color.r += mix(start.r, end.r, angle) * 0.4;
		color.g += mix(start.g, end.g, angle) * 0.4;
		color.b += mix(start.b, end.b, angle) * 0.4;
	}
);


/********************************** TEXTURE **********************************/

texture_t create_texture(unsigned int width, unsigned int height,
						 unsigned int nb_channels, const unsigned char* pixels)
{
	texture_t texture = { 0 };
	texture.width = width;
	texture.height = height;
	texture.depth = 1;

	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &texture.id);
	glBindTexture(GL_TEXTURE_2D, texture.id);

	if (nb_channels == 4)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	else if (nb_channels == 3)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	else if (nb_channels == 2)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, width, height, 0, GL_RG, GL_UNSIGNED_BYTE, pixels);
	else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, pixels);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	return texture;
}

//-----------------------------------------------

texture_t create_texture(unsigned int width, unsigned int height,
						 unsigned int nb_channels, const float* pixels)
{
	texture_t texture = { 0 };
	texture.width = width;
	texture.height = height;
	texture.depth = 1;

	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &texture.id);
	glBindTexture(GL_TEXTURE_2D, texture.id);

	if (nb_channels == 4)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, pixels);
	else if (nb_channels == 3)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, pixels);
	else if (nb_channels == 2)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, pixels);
	else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, pixels);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	return texture;
}

//-----------------------------------------------

texture_t create_texture(unsigned int width, unsigned int height,
						 unsigned int depth, unsigned int nb_channels,
						 const float* pixels)
{
	texture_t texture = { 0 };
	texture.width = width;
	texture.height = height;
	texture.depth = depth;

	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &texture.id);
	glBindTexture(GL_TEXTURE_3D, texture.id);

	if (nb_channels == 4)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, width, height, depth, 0, GL_RGBA, GL_FLOAT, pixels);
	else if (nb_channels == 3)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, pixels);
	else if (nb_channels == 2)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, width, height, depth, 0, GL_RG, GL_FLOAT, pixels);
	else
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, width, height, depth, 0, GL_RED, GL_FLOAT, pixels);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	return texture;
}

//-----------------------------------------------

void destroy(const texture_t& texture)
{
	glDeleteTextures(1, &texture.id);
}


/***************************** RENDER-TO-TEXTURE *****************************/

render_to_texture_t initRTT(const shader_t& shader, unsigned int width,
							unsigned int height, unsigned int nb_buffers)
{
	render_to_texture_t rtt;
	rtt.width = width;
	rtt.height = height;

	rtt.shader = &shader;

	rtt.current_buffer = 0;

	if (nb_buffers > 1)
		rtt.nb_buffers = nb_buffers;
	else
		rtt.nb_buffers = (shader.backbuffer_handle >= 0 ? 2 : 1);

	// The rectangular mesh on which the shader is applied
	rtt.nb_vertices = 6;

	const GLfloat vertex_buffer_data[] = {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f,
	};

	// Vertex buffer
	glGenBuffers(1, &rtt.vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, rtt.vertex_buffer);

	glBufferData(GL_ARRAY_BUFFER, rtt.nb_vertices * 3 * sizeof(GLfloat),
				 vertex_buffer_data, GL_STATIC_DRAW);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return rtt;
}

//-----------------------------------------------

render_to_texture_t createRTT(const shader_t& shader, unsigned int width,
							  unsigned int height, const arma::fvec& color,
							  unsigned int nb_buffers, unsigned int channel_size)
{
	render_to_texture_t rtt;

	int nb_channels = color.n_elem;

	if (channel_size == 8)
	{
		// Preparation of the initial color buffer
		unsigned char* pixels = new unsigned char[width * height * nb_channels];
		unsigned char* pDst = pixels;
		for (unsigned int y = 0; y < height; ++y)
		{
			for (unsigned int x = 0; x < width; ++x)
			{
				pDst[0] = (unsigned char) (color(0) * 255);

				if (nb_channels >= 2)
				{
					pDst[1] = (unsigned char) (color(1) * 255);

					if (nb_channels >= 3)
					{
						pDst[2] = (unsigned char) (color(2) * 255);

						if (nb_channels == 4)
							pDst[3] = (unsigned char) (color(3) * 255);
					}
				}

				pDst += nb_channels;
			}
		}

		rtt = createRTT(shader, width, height, nb_channels, pixels, nb_buffers);

		delete[] pixels;
	}
	else  if (channel_size == 32)
	{
		// Preparation of the initial color buffer
		float* pixels = new float[width * height * nb_channels];
		float* pDst = pixels;
		for (unsigned int y = 0; y < height; ++y)
		{
			for (unsigned int x = 0; x < width; ++x)
			{
				pDst[0] = color(0);

				if (nb_channels >= 2)
				{
					pDst[1] = color(1);

					if (nb_channels >= 3)
					{
						pDst[2] = color(2);

						if (nb_channels == 4)
							pDst[3] = color(3);
					}
				}

				pDst += nb_channels;
			}
		}

		rtt = createRTT(shader, width, height, nb_channels, pixels, nb_buffers);

		delete[] pixels;
	}

	return rtt;
}

//-----------------------------------------------

render_to_texture_t createRTT(const shader_t& shader, unsigned int width,
							  unsigned int height, unsigned int nb_channels,
							  const unsigned char* pixels,
							  unsigned int nb_buffers)
{
	render_to_texture_t rtt = initRTT(shader, width, height, nb_buffers);

	for (unsigned int i = 0; i < rtt.nb_buffers; ++i) {
		render_to_texture_buffer_t buffer;

		// Creates the framebuffer
		glGenFramebuffers(1, &buffer.framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, buffer.framebuffer);

		// Creates the texture we're going to render to
		buffer.texture = create_texture(width, height, nb_channels, pixels);

		// Link the two
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, buffer.texture.id, 0);

		rtt.buffers.push_back(buffer);
	}

	return rtt;
}

//-----------------------------------------------

render_to_texture_t createRTT(const shader_t& shader, unsigned int width,
							  unsigned int height, unsigned int nb_channels,
							  const float* pixels, unsigned int nb_buffers)
{
	render_to_texture_t rtt = initRTT(shader, width, height, nb_buffers);

	for (unsigned int i = 0; i < rtt.nb_buffers; ++i) {
		render_to_texture_buffer_t buffer;

		// Creates the framebuffer
		glGenFramebuffers(1, &buffer.framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, buffer.framebuffer);

		// Creates the texture we're going to render to
		buffer.texture = create_texture(width, height, nb_channels, pixels);

		// Link the two
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, buffer.texture.id, 0);

		rtt.buffers.push_back(buffer);
	}

	return rtt;
}

//-----------------------------------------------

bool draw(render_to_texture_t &rtt)
{
	// Various checks
	if (rtt.shader->id == 0)
		return false;

	// Retrieve the current viewport
	GLint previous_viewport[4];
	glGetIntegerv(GL_VIEWPORT, previous_viewport);

	// Activate the GLSL program
	glUseProgram(rtt.shader->id);

	// Backbuffer management
	if (rtt.nb_buffers > 1) {
		unsigned int previous_buffer = rtt.current_buffer;

		rtt.current_buffer++;
		if (rtt.current_buffer >= rtt.nb_buffers)
			rtt.current_buffer = 0;

		if (rtt.shader->backbuffer_handle >= 0)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, rtt.buffers[previous_buffer].texture.id);
			glUniform1i(rtt.shader->backbuffer_handle, 0);
		}
	}

	// Send the textures to the shader
	sendTextures(rtt.shader);

	glBindFramebuffer(GL_FRAMEBUFFER, rtt.buffers[rtt.current_buffer].framebuffer);

	glViewport(0, 0, rtt.width, rtt.height);

	// Send the application-specific uniforms to the shader
	sendApplicationUniforms(rtt.shader);

	// Specify the vertices for the shader
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, rtt.vertex_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);

	// Set the list of draw buffers.
	GLenum draw_buffers[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, draw_buffers);

	// Draw the mesh
	glDrawArrays(GL_TRIANGLES, 0, rtt.nb_vertices);

	// Cleanup
	glDisableVertexAttribArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Restore the previous viewport
	glViewport(previous_viewport[0], previous_viewport[1], previous_viewport[2],
			   previous_viewport[3]);

	return true;
}

//-----------------------------------------------

void destroy(const render_to_texture_t& rtt)
{
	for (auto iter = rtt.buffers.begin(); iter != rtt.buffers.end(); ++iter)
	{
		glDeleteFramebuffers(1, &iter->framebuffer);
		destroy(iter->texture);
	}

	glDeleteBuffers(1, &rtt.vertex_buffer);
}


/*********************************** MESHES **********************************/

model_t create_rectangle(const shader_t& shader, const arma::fvec& color,
						 float width, float height, const arma::fvec& position,
						 const arma::fmat& rotation, const transforms_t* parent_transforms)
{
	model_t model;

	if (shader.use_lightning)
		return model;

	model.mode = GL_TRIANGLES;
	model.shader = &shader;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.diffuse_color = color;

	// Create the mesh
	model.nb_vertices = 6;

	//-- Vertex buffer
	float half_width = 0.5f * width;
	float half_height = 0.5f * height;

	const GLfloat vertex_buffer_data[] = {
		 half_width,  half_height, 0.0f,
		-half_width,  half_height, 0.0f,
		-half_width, -half_height, 0.0f,
		-half_width, -half_height, 0.0f,
		 half_width, -half_height, 0.0f,
		 half_width,  half_height, 0.0f,
	};

	glGenBuffers(1, &model.vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, model.vertex_buffer);

	glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 3 * sizeof(GLfloat),
				 vertex_buffer_data, GL_STATIC_DRAW);

	//-- UVs buffer
	const GLfloat uv_buffer_data[] = {
		1.0f,  1.0f,
		0.0f,  1.0f,
		0.0f,  0.0f,
		0.0f,  0.0f,
		1.0f,  0.0f,
		1.0f,  1.0f,
	};

	glGenBuffers(1, &model.uv_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, model.uv_buffer);

	glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 2 * sizeof(GLfloat),
				 uv_buffer_data, GL_STATIC_DRAW);

	return model;
}

//-----------------------------------------------

model_t create_square(const shader_t& shader, const arma::fvec& color, float size,
					  const arma::fvec& position, const arma::fmat& rotation,
					  const transforms_t* parent_transforms)
{
	model_t model;

	if (shader.use_lightning)
		return model;

	model.mode = GL_TRIANGLES;
	model.shader = &shader;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.diffuse_color = color;

	// Create the mesh
	model.nb_vertices = 6;

	//-- Vertex buffer
	float half_size = 0.5f * size;

	const GLfloat vertex_buffer_data[] = {
		 half_size,	 half_size, 0.0f,
		-half_size,	 half_size, 0.0f,
		-half_size, -half_size, 0.0f,
		-half_size, -half_size, 0.0f,
		 half_size, -half_size, 0.0f,
		 half_size,	 half_size, 0.0f,
	};

	glGenBuffers(1, &model.vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, model.vertex_buffer);

	glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 3 * sizeof(GLfloat),
				 vertex_buffer_data, GL_STATIC_DRAW);

	//-- UVs buffer
	const GLfloat uv_buffer_data[] = {
		1.0f,  1.0f,
		0.0f,  1.0f,
		0.0f,  0.0f,
		0.0f,  0.0f,
		1.0f,  0.0f,
		1.0f,  1.0f,
	};

	glGenBuffers(1, &model.uv_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, model.uv_buffer);

	glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 2 * sizeof(GLfloat),
				 uv_buffer_data, GL_STATIC_DRAW);

	return model;
}

//-----------------------------------------------

model_t create_sphere(const shader_t& shader, float radius, const arma::fvec& position,
					  const arma::fmat& rotation, const transforms_t* parent_transforms)
{
	model_t model;

	model.mode = GL_TRIANGLES;
	model.shader = &shader;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.ambiant_color = arma::fvec({0.2f, 0.2f, 0.2f});
	model.diffuse_color = arma::fvec({0.8f, 0.8f, 0.8f});
	model.specular_color = arma::fvec({0.0f, 0.0f, 0.0f});
	model.specular_power = 5;

	// Create the mesh
	const int NB_STEPS = 72;
	const float STEP_SIZE = 360.0f / NB_STEPS;

	model.nb_vertices = NB_STEPS / 2 * NB_STEPS * 6;

	GLfloat* vertex_buffer_data = new GLfloat[model.nb_vertices * 3];

	GLfloat* normal_buffer_data = (shader.use_lightning ?
										new GLfloat[model.nb_vertices * 3] : 0);

	GLfloat* dst_vertex = vertex_buffer_data;
	GLfloat* dst_normal = normal_buffer_data;

	for (int i = 0; i < NB_STEPS / 2; ++i)
	{
		GLfloat latitude_lo = (float) i * STEP_SIZE;
		GLfloat latitude_hi = latitude_lo + STEP_SIZE;

		for (int j = 0; j < NB_STEPS; ++j)
		{
			GLfloat longitude_lo = (float) j * STEP_SIZE;
			GLfloat longitude_hi = longitude_lo + STEP_SIZE;

			arma::fvec vert_ne(3);
			arma::fvec vert_nw(3);
			arma::fvec vert_sw(3);
			arma::fvec vert_se(3);

			// Assign each X,Z with sin,cos values scaled by latitude radius indexed by longitude.
			vert_ne(1) = vert_nw(1) = (float) -cos_deg(latitude_hi) * radius;
			vert_sw(1) = vert_se(1) = (float) -cos_deg(latitude_lo) * radius;

			vert_nw(0) = (float) cos_deg(longitude_lo) * (radius * (float) sin_deg(latitude_hi));
			vert_sw(0) = (float) cos_deg(longitude_lo) * (radius * (float) sin_deg(latitude_lo));
			vert_ne(0) = (float) cos_deg(longitude_hi) * (radius * (float) sin_deg(latitude_hi));
			vert_se(0) = (float) cos_deg(longitude_hi) * (radius * (float) sin_deg(latitude_lo));

			vert_nw(2) = (float) -sin_deg(longitude_lo) * (radius * (float) sin_deg(latitude_hi));
			vert_sw(2) = (float) -sin_deg(longitude_lo) * (radius * (float) sin_deg(latitude_lo));
			vert_ne(2) = (float) -sin_deg(longitude_hi) * (radius * (float) sin_deg(latitude_hi));
			vert_se(2) = (float) -sin_deg(longitude_hi) * (radius * (float) sin_deg(latitude_lo));

			dst_vertex[0] = vert_ne(0); dst_vertex[1] = vert_ne(1); dst_vertex[2] = vert_ne(2); dst_vertex += 3;
			dst_vertex[0] = vert_nw(0); dst_vertex[1] = vert_nw(1); dst_vertex[2] = vert_nw(2); dst_vertex += 3;
			dst_vertex[0] = vert_sw(0); dst_vertex[1] = vert_sw(1); dst_vertex[2] = vert_sw(2); dst_vertex += 3;

			dst_vertex[0] = vert_sw(0); dst_vertex[1] = vert_sw(1); dst_vertex[2] = vert_sw(2); dst_vertex += 3;
			dst_vertex[0] = vert_se(0); dst_vertex[1] = vert_se(1); dst_vertex[2] = vert_se(2); dst_vertex += 3;
			dst_vertex[0] = vert_ne(0); dst_vertex[1] = vert_ne(1); dst_vertex[2] = vert_ne(2); dst_vertex += 3;

			if (shader.use_lightning)
			{
				arma::fvec normal_ne = arma::normalise(vert_ne);
				arma::fvec normal_nw = arma::normalise(vert_nw);
				arma::fvec normal_sw = arma::normalise(vert_sw);
				arma::fvec normal_se = arma::normalise(vert_se);

				dst_normal[0] = normal_ne(0); dst_normal[1] = normal_ne(1); dst_normal[2] = normal_ne(2); dst_normal += 3;
				dst_normal[0] = normal_nw(0); dst_normal[1] = normal_nw(1); dst_normal[2] = normal_nw(2); dst_normal += 3;
				dst_normal[0] = normal_sw(0); dst_normal[1] = normal_sw(1); dst_normal[2] = normal_sw(2); dst_normal += 3;

				dst_normal[0] = normal_sw(0); dst_normal[1] = normal_sw(1); dst_normal[2] = normal_sw(2); dst_normal += 3;
				dst_normal[0] = normal_se(0); dst_normal[1] = normal_se(1); dst_normal[2] = normal_se(2); dst_normal += 3;
				dst_normal[0] = normal_ne(0); dst_normal[1] = normal_ne(1); dst_normal[2] = normal_ne(2); dst_normal += 3;
			}
		}
	}

	// Vertex buffer
	glGenBuffers(1, &model.vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, model.vertex_buffer);

	glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 3 * sizeof(GLfloat),
				 vertex_buffer_data, GL_STATIC_DRAW);

	// Normal buffer
	if (shader.use_lightning)
	{
		glGenBuffers(1, &model.normal_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, model.normal_buffer);

		glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 3 * sizeof(GLfloat),
					 normal_buffer_data, GL_STATIC_DRAW);
	}

	// Cleanup
	delete[] vertex_buffer_data;

	if (shader.use_lightning)
		delete[] normal_buffer_data;

	return model;
}

//-----------------------------------------------

model_t create_line(const shader_t& shader, const arma::fvec& color,
					const arma::mat& points, const arma::fvec& position,
					const arma::fmat& rotation, const transforms_t* parent_transforms,
					bool line_strip)
{
	model_t model;

	if (shader.use_lightning)
		return model;

	model.mode = (line_strip ? GL_LINE_STRIP : GL_LINES);
	model.shader = &shader;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.diffuse_color = color;

	// Create the mesh
	model.nb_vertices = points.n_cols;

	GLfloat* vertex_buffer_data = new GLfloat[model.nb_vertices * 3];

	GLfloat* dst = vertex_buffer_data;

	for (int i = 0; i < points.n_cols; ++i) {
		dst[0] = (float) points(0, i);
		dst[1] = (float) points(1, i);

		if (points.n_rows == 3)
			dst[2] = (float) points(2, i);
		else
			dst[2] = 0.0f;

		dst += 3;
	}

	// Vertex buffer
	glGenBuffers(1, &model.vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, model.vertex_buffer);

	glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 3 * sizeof(GLfloat),
				 vertex_buffer_data, GL_STATIC_DRAW);

	// Cleanup
	delete[] vertex_buffer_data;

	return model;
}

//-----------------------------------------------

model_t create_line(const shader_t& shader, const arma::fvec& color,
					const arma::mat& points, float width, const arma::fvec& position,
					const arma::fmat& rotation, const transforms_t* parent_transforms)
{
	arma::mat vertices(2, (points.n_cols - 1) * 12 - 6);

	int n = 0;
	for (int i = 0; i < points.n_cols - 1; ++i)
	{
		arma::vec p1 = points.col(i);
		arma::vec p2 = points.col(i + 1);

		float dx = p2(0) - p1(0);
		float dy = p2(1) - p1(1);

		arma::vec normal = arma::normalise(arma::vec({ -dy, dx }));

		arma::vec v1 = p1 + normal * width / 2;
		arma::vec v2 = p2 + normal * width / 2;
		arma::vec v3 = p1 - normal * width / 2;
		arma::vec v4 = p2 - normal * width / 2;

		if (n > 0)
		{
			arma::vec previous_v2 = vertices.col(n - 1);
			arma::vec previous_v4 = vertices.col(n - 2);

			vertices.col(n) = previous_v2;
			vertices.col(n + 1) = previous_v4;
			vertices.col(n + 2) = v1;
			vertices.col(n + 3) = previous_v4;
			vertices.col(n + 4) = v3;
			vertices.col(n + 5) = v1;

			n += 6;
		}

		vertices.col(n) = v1;
		vertices.col(n + 1) = v3;
		vertices.col(n + 2) = v2;
		vertices.col(n + 3) = v3;
		vertices.col(n + 4) = v4;
		vertices.col(n + 5) = v2;

		n += 6;
	}

	return create_mesh(shader, color, vertices, position, rotation, parent_transforms);
}

//-----------------------------------------------

model_t create_line(const shader_t& shader, const arma::fvec& color,
					const std::vector<arma::vec>& points, const arma::fvec& position,
					const arma::fmat& rotation, const transforms_t* parent_transforms,
					bool line_strip)
{
	arma::mat points_mat(points[0].n_rows, points.size());

	for (size_t i = 0; i < points.size(); ++i)
		points_mat.col(i) = points[i];

	return create_line(shader, color, points_mat, position, rotation,
					   parent_transforms, line_strip);
}

//-----------------------------------------------

model_t create_line(const shader_t& shader, const arma::fvec& color,
					const std::vector<arma::vec>& points, float width,
					const arma::fvec& position, const arma::fmat& rotation,
					const transforms_t* parent_transforms)
{
	arma::mat points_mat(points[0].n_rows, points.size());

	for (size_t i = 0; i < points.size(); ++i)
		points_mat.col(i) = points[i];

	return create_line(shader, color, points_mat, width, position, rotation,
					   parent_transforms);
}

//-----------------------------------------------

model_t create_mesh(const shader_t& shader, const arma::fvec& color,
					const arma::mat& vertices,
					const arma::fvec& position, const arma::fmat& rotation,
					const transforms_t* parent_transforms)
{
	model_t model;

	if (shader.use_lightning)
		return model;

	model.mode = GL_TRIANGLES;
	model.shader = &shader;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.diffuse_color = color;

	// Create the mesh
	model.nb_vertices = vertices.n_cols;

	GLfloat* vertex_buffer_data = new GLfloat[model.nb_vertices * 3];

	GLfloat* dst = vertex_buffer_data;

	for (int i = 0; i < vertices.n_cols; ++i) {
		dst[0] = (float) vertices(0, i);
		dst[1] = (float) vertices(1, i);

		if (vertices.n_rows == 3)
			dst[2] = (float) vertices(2, i);
		else
			dst[2] = 0.0f;

		dst += 3;
	}

	// Vertex buffer
	glGenBuffers(1, &model.vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, model.vertex_buffer);

	glBufferData(GL_ARRAY_BUFFER, model.nb_vertices * 3 * sizeof(GLfloat),
				 vertex_buffer_data, GL_STATIC_DRAW);

	// Cleanup
	delete[] vertex_buffer_data;

	return model;
}

//-----------------------------------------------

model_t create_gaussian_border(const shader_t& shader, const arma::fvec& color,
							   const arma::vec& mu, const arma::mat& sigma,
							   const arma::fvec& position, const arma::fmat& rotation,
							   const transforms_t* parent_transforms)
{
	arma::mat pts = get_gaussian_border_vertices(mu, sigma, 60, true);

	return create_line(shader, color, pts, position, rotation, parent_transforms);
}


//-----------------------------------------------

model_t create_gaussian_plane(const shader_t& shader, const camera_t& camera)
{
	model_t plane = create_rectangle(
		shader, arma::fvec({ 1.0f, 0.0f, 0.0f, 1.0f }), 1.0f, 1.0f,
		arma::fvec({ 0.0f, 0.0f, 1.5f }),
		rotate(arma::fvec({ 0.0f, 1.0f, 0.0f }), deg2rad(0.0f)),
		&camera.rotator
	);

	plane.use_transparency = true;

	return plane;
}

//-----------------------------------------------

void destroy(const model_t& model)
{
	if (model.vertex_buffer)
		glDeleteBuffers(1, &model.vertex_buffer);

	if (model.normal_buffer)
		glDeleteBuffers(1, &model.normal_buffer);

	if (model.uv_buffer)
		glDeleteBuffers(1, &model.uv_buffer);
}


/********************************** RENDERING ********************************/

inline arma::fvec check_color(const arma::fvec& color) {

	if (color.n_rows < 4) {
		arma::fvec color4({ 0.0, 0.0, 0.0, 1.0 });
		color4(arma::span(0, 2)) = color;
		return color4;
	}

	return color;
}

//-----------------------------------------------

bool draw(const model_t& model, const arma::fmat& view,
		  const arma::fmat& projection, const light_list_t& lights)
{
	// Various checks
	if (model.nb_vertices == 0)
		return false;

	if (!model.shader || (model.shader->id == 0))
		return false;

	if (model.shader->use_lightning && lights.empty())
		return false;

	// Activate the GLSL program
	glUseProgram(model.shader->id);

	// Send the model, view and projection matrices to the shader
	arma::fmat model_matrix = worldTransforms(&model.transforms);

	glUniformMatrix4fv(model.shader->model_matrix_handle, 1, GL_FALSE, model_matrix.memptr());
	glUniformMatrix4fv(model.shader->view_matrix_handle, 1, GL_FALSE, view.memptr());
	glUniformMatrix4fv(model.shader->projection_matrix_handle, 1, GL_FALSE, projection.memptr());

	// Send the material parameters to the shader
	glUniform4fv(model.shader->diffuse_color_handle, 1, check_color(model.diffuse_color).memptr());

	if (model.shader->use_lightning) {
		glUniform3fv(model.shader->ambiant_color_handle, 1, model.ambiant_color.memptr());
		glUniform3fv(model.shader->specular_color_handle, 1, model.specular_color.memptr());
		glUniform1f(model.shader->specular_power_handle, model.specular_power);
	}

	// Send the texture parameters to the shader
	if (model.shader->diffuse_texture_handle > -1) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, model.diffuse_texture);
		glUniform1i(model.shader->diffuse_texture_handle, 0);
	}

	// Send the light parameters to the shader
	if (model.shader->use_lightning) {
		glUniform3fv(model.shader->light_position_handle, 1, lights[0].transforms.position.memptr());
		glUniform3fv(model.shader->light_color_handle, 1, lights[0].color.memptr());
		glUniform1f(model.shader->light_power_handle, lights[0].power);
	}

	// Send the textures to the shader
	sendTextures(model.shader, (model.shader->diffuse_texture_handle > -1 ? GL_TEXTURE1 : GL_TEXTURE0));

	// Send the application-specific uniforms to the shader
	sendApplicationUniforms(model.shader);

	// Specify the vertices for the shader
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, model.vertex_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);

	// Specify the normals for the shader
	if (model.shader->use_lightning) {
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, model.normal_buffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
	}

	// Specify the UVs for the shader
	if (model.uv_buffer) {
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, model.uv_buffer);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*) 0);
	}

	// Specify the UVWs for the shader
	if (model.uvw_buffer) {
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, model.uvw_buffer);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
	}

	if (model.use_transparency) {
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
	}

	// Draw the mesh
	glDrawArrays(model.mode, 0, model.nb_vertices);

	if (model.use_transparency) {
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
	}

	glDisableVertexAttribArray(0);

	if (model.shader->use_lightning)
		glDisableVertexAttribArray(1);

	if (model.uv_buffer)
		glDisableVertexAttribArray(2);

	if (model.uvw_buffer)
		glDisableVertexAttribArray(2);

	return true;
}

//-----------------------------------------------

bool draw_rectangle(const shader_t& shader, const arma::fvec& color, float width,
					float height, const arma::fmat& view, const arma::fmat& projection,
					const arma::fvec& position, const arma::fmat& rotation)
{
	model_t rect = create_rectangle(shader, color, width, height, position, rotation);

	bool result = draw(rect, view, projection);

	destroy(rect);

	return result;
}

//-----------------------------------------------

bool draw_line(const shader_t& shader, const arma::fvec& color, const arma::mat& points,
			   const arma::fmat& view, const arma::fmat& projection,
			   const arma::fvec& position, const arma::fmat& rotation, bool line_strip)
{
	model_t line = create_line(shader, color, points, position, rotation, 0, line_strip);

	bool result = draw(line, view, projection);

	destroy(line);

	return result;
}

//-----------------------------------------------

bool draw_line(const shader_t& shader, const arma::fvec& color, const arma::mat& points,
			   float width, const arma::fmat& view, const arma::fmat& projection,
			   const arma::fvec& position, const arma::fmat& rotation)
{
	model_t line = create_line(shader, color, points, width, position, rotation);

	bool result = draw(line, view, projection);

	destroy(line);

	return result;
}

//-----------------------------------------------

bool draw_line(const shader_t& shader, const arma::fvec& color,
			   const std::vector<arma::vec>& points,
			   const arma::fmat& view, const arma::fmat& projection,
			   const arma::fvec& position, const arma::fmat& rotation,
			   bool line_strip)
{
	model_t line = create_line(shader, color, points, position, rotation, 0, line_strip);

	bool result = draw(line, view, projection);

	destroy(line);

	return result;
}

//-----------------------------------------------

bool draw_line(const shader_t& shader, const arma::fvec& color,
			   const std::vector<arma::vec>& points, float width,
			   const arma::fmat& view, const arma::fmat& projection,
			   const arma::fvec& position, const arma::fmat& rotation)
{
	model_t line = create_line(shader, color, points, width, position, rotation);

	bool result = draw(line, view, projection);

	destroy(line);

	return result;
}

//-----------------------------------------------

bool draw_mesh(const shader_t& shader, const arma::fvec& color, const arma::mat& vertices,
			   const arma::fmat& view, const arma::fmat& projection,
			   const arma::fvec& position, const arma::fmat& rotation)
{
	model_t mesh = create_mesh(shader, color, vertices, position, rotation);

	bool result = draw(mesh, view, projection);

	destroy(mesh);

	return result;
}

//-----------------------------------------------

bool draw_gaussian(shader_t* shader, const arma::fvec& color,
				   const arma::vec& mu, const arma::mat& sigma,
				   const arma::fmat& view, const arma::fmat& projection,
				   float viewport_width, float viewport_height)
{
	float square_size = fmax(viewport_width, viewport_height) +
						fmax(fabs(fabs(mu(0)) - fabs(view(0, 3))),
							 fabs(fabs(mu(1)) - fabs(view(1, 3)))) * 2;

	gfx3::model_t square = gfx3::create_rectangle(*shader, color, square_size, square_size);

	square.transforms.position = { (float) mu(0), (float) mu(1), 0.0f };

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);

	arma::mat scaling({
		{ 1.0 / square_size, 0.0 },
		{ 0.0, 1.0 / square_size }
	});

	arma::mat scaled_sigma = scaling * sigma(arma::span(0, 1), arma::span(0, 1)) * scaling.t();

	arma::vec gaussian_color;
	if (color.n_rows == 4)
		gaussian_color = arma::conv_to<arma::vec>::from(color);
	else
		gaussian_color = arma::vec({ color(0), color(1), color(2), 0.5 });

	shader->setUniform("Mu", arma::vec{0.5, 0.5});
	shader->setUniform("InvSigma", arma::mat(arma::inv(scaled_sigma)));
	shader->setUniform("GaussianColor", gaussian_color);

	bool result = gfx3::draw(square, view, projection);

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	gfx3::destroy(square);

	return result;
}

//-----------------------------------------------

bool draw_gaussian_border(shader_t* shader, const arma::fvec& color,
						  const arma::vec& mu, const arma::mat& sigma,
						  const arma::fmat& view, const arma::fmat& projection,
						  float viewport_width, float viewport_height)
{
	arma::mat pts = get_gaussian_border_vertices(mu, sigma, 60, true);

	arma::fvec gaussian_color = color(arma::span(0, 2));

	return draw_line(*shader, gaussian_color, pts, view, projection);
}

//-----------------------------------------------

bool draw_gaussian_border(shader_t* shader, const arma::fvec& color,
						  const arma::vec& mu, const arma::mat& sigma,
						  float width, const arma::fmat& view,
						  const arma::fmat& projection,
						  float viewport_width, float viewport_height)
{
	arma::mat pts = get_gaussian_border_vertices(mu, sigma, 60, true);

	arma::fvec gaussian_color = color(arma::span(0, 2));

	return draw_line(*shader, gaussian_color, pts, width, view, projection);
}

//-----------------------------------------------

bool draw_gaussian_3D(model_t* plane, shader_t* shader,
					  const arma::fvec& color,
					  const arma::vec& mu, const arma::mat& sigma,
					  const arma::fmat& view, const arma::fmat& projection)
{
	plane->diffuse_color = color;

	// We need to compute mu and sigma in camera space
	arma::vec mu_cameraspace(4);
	mu_cameraspace(arma::span(0, 2)) = mu;
	mu_cameraspace(3) = 1.0;

	mu_cameraspace = view * mu_cameraspace;

	arma::mat sigma_cameraspace =
		view(arma::span(0, 2), arma::span(0, 2)) * sigma * view(arma::span(0, 2), arma::span(0, 2)).t();

	shader->setUniform("Mu", arma::vec(mu_cameraspace(arma::span(0, 2))));
	shader->setUniform("InvSigma", arma::mat(arma::inv(sigma_cameraspace)));


	glClear(GL_DEPTH_BUFFER_BIT);

	return gfx3::draw(*plane, view, projection);
}

}
