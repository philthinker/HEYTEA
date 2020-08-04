/*
 * gfx3.cpp
 *
 * Rendering utility structures and functions based on OpenGL 2 (no shader)
 *
 * Authors: Philip Abbet
 */

#include <gfx2.h>

#define GFX_NAMESPACE gfx2
#include "gfx_common.cpp"
#undef GFX_NAMESPACE


namespace gfx2 {

/********************************** TEXTURES *********************************/

texture_t create_texture(int width, int height, GLenum format, GLenum type)
{
	texture_t texture = { 0 };

	texture.width = (GLuint) width;
	texture.height = (GLuint) height;
	texture.format = format;
	texture.type = type;

	glGenTextures(1, &texture.id);

	glBindTexture(GL_TEXTURE_2D, texture.id);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	if (texture.type == GL_FLOAT)
		texture.pixels_f = new float[width * height * 3];
	else
		texture.pixels_b = new unsigned char[width * height * 3];

	return texture;
}

//-----------------------------------------------

void destroy(texture_t &texture)
{
	if (texture.type == GL_FLOAT)
		delete[] texture.pixels_f;
	else
		delete[] texture.pixels_b;

	glDeleteTextures(1, &texture.id);

	texture = {0};
}


/*********************************** MESHES **********************************/

model_t create_rectangle(const arma::fvec& color, float width, float height,
						 const arma::fvec& position, const arma::fmat& rotation,
						 const transforms_t* parent_transforms)
{
	model_t model = { 0 };

	model.mode = GL_TRIANGLES;
	model.lightning_enabled = false;
	model.use_one_minus_src_alpha_blending = false;

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

	model.vertex_buffer = new GLfloat[model.nb_vertices * 3];
	memcpy(model.vertex_buffer, vertex_buffer_data,
		   model.nb_vertices * 3 * sizeof(GLfloat));

	//-- UVs buffer
	const GLfloat uv_buffer_data[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
	};

	model.uv_buffer = new GLfloat[model.nb_vertices * 2];
	memcpy(model.uv_buffer, uv_buffer_data, model.nb_vertices * 2 * sizeof(GLfloat));

	return model;
}

//-----------------------------------------------

model_t create_rectangle(const texture_t& texture, float width, float height,
						 const arma::fvec& position, const arma::fmat& rotation,
						 const transforms_t* parent_transforms)
{
	model_t model = create_rectangle(arma::fvec({1.0f, 1.0f, 1.0f, 1.0f}),
									 width, height, position, rotation,
						 			 parent_transforms);

	model.texture = texture;

	return model;
}

//-----------------------------------------------

model_t create_square(const arma::fvec& color, float size, const arma::fvec& position,
					  const arma::fmat& rotation, const transforms_t* parent_transforms)
{
	model_t model = { 0 };

	model.mode = GL_TRIANGLES;
	model.lightning_enabled = false;
	model.use_one_minus_src_alpha_blending = false;

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

	model.vertex_buffer = new GLfloat[model.nb_vertices * 3];
	memcpy(model.vertex_buffer, vertex_buffer_data,
		   model.nb_vertices * 3 * sizeof(GLfloat));

	//-- UVs buffer
	const GLfloat uv_buffer_data[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
	};

	model.uv_buffer = new GLfloat[model.nb_vertices * 2];
	memcpy(model.uv_buffer, uv_buffer_data, model.nb_vertices * 2 * sizeof(GLfloat));

	return model;
}

//-----------------------------------------------

model_t create_sphere(float radius, const arma::fvec& position,
					  const arma::fmat& rotation, const transforms_t* parent_transforms)
{
	model_t model = { 0 };

	model.mode = GL_TRIANGLES;
	model.lightning_enabled = true;
	model.use_one_minus_src_alpha_blending = false;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.ambiant_color = arma::fvec({0.2f, 0.2f, 0.2f, 1.0f});
	model.diffuse_color = arma::fvec({0.8f, 0.8f, 0.8f, 1.0f});
	model.specular_color = arma::fvec({0.0f, 0.0f, 0.0f, 1.0f});
	model.specular_power = 5;

	// Create the mesh
	const int NB_STEPS = 72;
	const float STEP_SIZE = 360.0f / NB_STEPS;

	model.nb_vertices = NB_STEPS / 2 * NB_STEPS * 6;

	model.vertex_buffer = new GLfloat[model.nb_vertices * 3];
	model.normal_buffer = new GLfloat[model.nb_vertices * 3];

	GLfloat* dst_vertex = model.vertex_buffer;
	GLfloat* dst_normal = model.normal_buffer;

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

			// Normals
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

	return model;
}

//-----------------------------------------------

model_t create_line(const arma::fvec& color, const arma::mat& points,
					const arma::fvec& position, const arma::fmat& rotation,
					const transforms_t* parent_transforms, bool line_strip)
{
	model_t model = { 0 };

	model.mode = (line_strip ? GL_LINE_STRIP : GL_LINES);
	model.lightning_enabled = false;
	model.use_one_minus_src_alpha_blending = false;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.diffuse_color = color;

	// Create the mesh
	model.nb_vertices = points.n_cols;

	model.vertex_buffer = new GLfloat[model.nb_vertices * 3];

	GLfloat* dst = model.vertex_buffer;

	for (int i = 0; i < points.n_cols; ++i) {
		dst[0] = (float) points(0, i);
		dst[1] = (float) points(1, i);

		if (points.n_rows == 3)
			dst[2] = (float) points(2, i);
		else
			dst[2] = 0.0f;

		dst += 3;
	}

	return model;
}

//-----------------------------------------------

model_t create_line(const arma::fvec& color, const std::vector<arma::vec>& points,
					const arma::fvec& position, const arma::fmat& rotation,
					const transforms_t* parent_transforms, bool line_strip)
{
	arma::mat points_mat(points[0].n_rows, points.size());

	for (size_t i = 0; i < points.size(); ++i)
		points_mat.col(i) = points[i];

	return create_line(color, points_mat, position, rotation, parent_transforms, line_strip);
}

//-----------------------------------------------

model_t create_mesh(const arma::fvec& color, const arma::mat& vertices,
					const arma::fvec& position, const arma::fmat& rotation,
					const transforms_t* parent_transforms)
{
	model_t model = { 0 };

	model.mode = GL_TRIANGLES;
	model.lightning_enabled = false;
	model.use_one_minus_src_alpha_blending = false;

	// Position & rotation
	model.transforms.position = position;
	model.transforms.rotation = rotation;
	model.transforms.parent = parent_transforms;

	// Material
	model.diffuse_color = color;

	// Create the mesh
	model.nb_vertices = vertices.n_cols;

	model.vertex_buffer = new GLfloat[model.nb_vertices * 3];

	GLfloat* dst = model.vertex_buffer;

	for (int i = 0; i < vertices.n_cols; ++i) {
		dst[0] = (float) vertices(0, i);
		dst[1] = (float) vertices(1, i);

		if (vertices.n_rows == 3)
			dst[2] = (float) vertices(2, i);
		else
			dst[2] = 0.0f;

		dst += 3;
	}

	return model;
}

//-----------------------------------------------

model_t create_gaussian_background(const arma::fvec& color, const arma::vec& mu,
								   const arma::mat& sigma,
								   const arma::fvec& position, const arma::fmat& rotation,
								   const transforms_t* parent_transforms)
{
	arma::mat vertices = get_gaussian_background_vertices(mu, sigma, 60);

	model_t model = create_mesh(color, vertices, position, rotation, parent_transforms);

	model.use_one_minus_src_alpha_blending = true;

	return model;
}

//-----------------------------------------------

model_t create_gaussian_border(const arma::fvec& color, const arma::vec& mu,
							   const arma::mat& sigma,
							   const arma::fvec& position, const arma::fmat& rotation,
							   const transforms_t* parent_transforms)
{
	arma::mat pts = get_gaussian_border_vertices(mu, sigma, 60, true);

	return create_line(color, pts, position, rotation, parent_transforms);
}

//-----------------------------------------------

void destroy(model_t &model)
{
	if (model.vertex_buffer)
		delete[] model.vertex_buffer;

	if (model.normal_buffer)
		delete[] model.normal_buffer;

	if (model.uv_buffer)
		delete[] model.uv_buffer;

	model = {0};
}


/********************************** RENDERING ********************************/

bool draw(const model_t& model, const light_list_t& lights)
{
	// Various checks
	if (model.lightning_enabled && lights.empty())
		return false;

	// Specify material parameters for the lighting model
	if (model.lightning_enabled) {
		glMaterialfv(GL_FRONT, GL_AMBIENT, model.ambiant_color.memptr());
		glMaterialfv(GL_FRONT, GL_DIFFUSE, model.diffuse_color.memptr());
		glMaterialfv(GL_FRONT, GL_SPECULAR, model.specular_color.memptr());
		glMaterialf(GL_FRONT, GL_SHININESS, model.specular_power);
	} else {
		if (model.texture.width > 0)
			glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		else if (model.diffuse_color.n_rows == 3)
			glColor3fv(model.diffuse_color.memptr());
		else
			glColor4fv(model.diffuse_color.memptr());
	}

	// Specify the light parameters
	glDisable(GL_LIGHTING);
	if (model.lightning_enabled) {
		glEnable(GL_LIGHTING);

		glEnable(GL_LIGHT0);
		glLightfv(GL_LIGHT0, GL_POSITION, lights[0].transforms.position.memptr());
		glLightfv(GL_LIGHT0, GL_AMBIENT, lights[0].ambient_color.memptr());
		glLightfv(GL_LIGHT0, GL_DIFFUSE, lights[0].diffuse_color.memptr());
		glLightfv(GL_LIGHT0, GL_SPECULAR, lights[0].specular_color.memptr());
	}

	// Set vertex data
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, model.vertex_buffer);

	// Set normal data
	if (model.lightning_enabled) {
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, 0, model.normal_buffer);
	}

	// Set UV data
	if (model.uv_buffer) {
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(2, GL_FLOAT, 0, model.uv_buffer);
	}

	// Texturing
	if (model.texture.width > 0) {
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, model.texture.id);

		if (model.texture.type == GL_FLOAT) {
			glTexImage2D(GL_TEXTURE_2D, 0, model.texture.format,
						 model.texture.width, model.texture.height,
						 0, model.texture.format, GL_FLOAT,
						 model.texture.pixels_f
			);
		} else {
			glTexImage2D(GL_TEXTURE_2D, 0, model.texture.format,
						 model.texture.width, model.texture.height,
						 0, model.texture.format, GL_UNSIGNED_BYTE,
						 model.texture.pixels_b
			);
		}
	}

	// Apply the model matrix
	arma::fmat model_matrix = worldTransforms(&model.transforms);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glMultMatrixf(model_matrix.memptr());

	GLboolean is_blending_enabled = glIsEnabled(GL_BLEND);
	GLint blend_src_rgb, blend_src_alpha, blend_dst_rgb, blend_dst_alpha;

	if (model.use_one_minus_src_alpha_blending)
	{
		if (is_blending_enabled)
		{
			glGetIntegerv(GL_BLEND_SRC_RGB, &blend_src_rgb);
			glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_src_alpha);
			glGetIntegerv(GL_BLEND_DST_RGB, &blend_dst_rgb);
			glGetIntegerv(GL_BLEND_DST_ALPHA, &blend_dst_alpha);
		}
		else
		{
			glEnable(GL_BLEND);
		}

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	// Draw the mesh
	glDrawArrays(model.mode, 0, model.nb_vertices);

	if (model.use_one_minus_src_alpha_blending)
	{
		if (is_blending_enabled)
		{
			glBlendFunc(GL_SRC_COLOR, blend_src_rgb);
			glBlendFunc(GL_SRC_ALPHA, blend_src_alpha);
			glBlendFunc(GL_DST_COLOR, blend_dst_rgb);
			glBlendFunc(GL_DST_ALPHA, blend_dst_alpha);
		}
		else
		{
			glDisable(GL_BLEND);
		}
	}

	glPopMatrix();

	// Cleanup
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	glDisable(GL_LIGHTING);

	if (model.texture.width > 0)
		glDisable(GL_TEXTURE_2D);

	return true;
}

//-----------------------------------------------

bool draw_rectangle(const arma::fvec& color, float width, float height,
					const arma::fvec& position, const arma::fmat& rotation)
{
	model_t rect = create_rectangle(color, width, height, position, rotation);

	bool result = draw(rect);

	destroy(rect);

	return result;
}

//-----------------------------------------------

bool draw_rectangle(const texture_t& texture, float width, float height,
					const arma::fvec& position, const arma::fmat& rotation)
{
	model_t rect = create_rectangle(texture, width, height, position, rotation);

	bool result = draw(rect);

	destroy(rect);

	return result;
}

//-----------------------------------------------

bool draw_line(const arma::fvec& color, const arma::mat& points,
			   const arma::fvec& position, const arma::fmat& rotation)
{
	model_t line = create_line(color, points, position, rotation);

	bool result = draw(line);

	destroy(line);

	return result;
}

//-----------------------------------------------

bool draw_line(const arma::fvec& color, const std::vector<arma::vec>& points,
			   const arma::fvec& position, const arma::fmat& rotation)
{
	model_t line = create_line(color, points, position, rotation);

	bool result = draw(line);

	destroy(line);

	return result;
}

//-----------------------------------------------

bool draw_mesh(const arma::fvec& color, const arma::mat& vertices,
			   const arma::fvec& position, const arma::fmat& rotation)
{
	model_t mesh = create_mesh(color, vertices, position, rotation);

	bool result = draw(mesh);

	destroy(mesh);

	return result;
}

//-----------------------------------------------

bool draw_gaussian(const arma::fvec& color, const arma::vec& mu, const arma::mat& sigma,
				   bool background, bool border)
{
	const int NB_POINTS = 60;

	arma::mat pts = get_gaussian_border_vertices(mu, sigma, NB_POINTS, true);

	arma::mat vertices(2, NB_POINTS * 3);

	if (background)
	{
		for (int i = 0; i < NB_POINTS - 1; ++i)
		{
			vertices(arma::span::all, i * 3) = mu(arma::span(0, 1));
			vertices(arma::span::all, i * 3 + 1) = pts(arma::span::all, i + 1);
			vertices(arma::span::all, i * 3 + 2) = pts(arma::span::all, i);
		}

		vertices(arma::span::all, (NB_POINTS - 1) * 3) = mu(arma::span(0, 1));
		vertices(arma::span::all, (NB_POINTS - 1) * 3 + 1) = pts(arma::span::all, 0);
		vertices(arma::span::all, (NB_POINTS - 1) * 3 + 2) = pts(arma::span::all, NB_POINTS - 1);
	}


	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	arma::fvec gaussian_color;
	if (color.n_rows == 4)
		gaussian_color = color;
	else
		gaussian_color = arma::fvec({color(0), color(1), color(2), 0.1f});

	bool result = false;

	if (background)
		result = draw_mesh(gaussian_color, vertices);

	glDisable(GL_BLEND);

	if (border) {
		arma::fvec darker_color = gaussian_color(arma::span(0, 2)) * 0.5f;
		result &= gfx2::draw_line(darker_color, pts);
	}

	return result;
}

//-----------------------------------------------

bool draw_gaussian_3D(const arma::fvec& color, const arma::vec& mu,
					  const arma::mat& sigma)
{
	gfx2::model_t sphere = gfx2::create_sphere();
	sphere.lightning_enabled = false;
	sphere.use_one_minus_src_alpha_blending = true;

	sphere.diffuse_color(3) = 0.1f;
	sphere.specular_power = 0.0f;

	sphere.ambiant_color(arma::span(0, 2)) = color;
	sphere.diffuse_color(arma::span(0, 2)) = color;

	arma::mat V;
	arma::vec d;
	arma::eig_sym(d, V, sigma);
	arma::mat VD = V * arma::diagmat(sqrt(d));

	sphere.transforms.position = arma::conv_to<arma::fmat>::from(mu);
	arma::fmat rot_scale = arma::conv_to<arma::fmat>::from(VD);

	bool result = false;

	for (int j = 1; j <= 20; ++j) {
		glClear(GL_DEPTH_BUFFER_BIT);

		arma::fmat scaling = arma::eye<arma::fmat>(3, 3) * (float) j / 20.0f;

		arma::fmat scaled_transforms = arma::eye<arma::fmat>(4, 4);

		scaled_transforms(arma::span(0, 2), arma::span(0, 2)) = scaling * rot_scale;

		sphere.transforms.rotation = scaled_transforms;

		result &= gfx2::draw(sphere);
	}

	gfx2::destroy(sphere);

	return result;
}

}
