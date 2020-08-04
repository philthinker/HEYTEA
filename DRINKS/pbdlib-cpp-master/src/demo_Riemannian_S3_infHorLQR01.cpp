/*
 * demo_Riemannian_S3_infHorLQR01.cpp
 *
 * Linear quadratic regulation of unit quaternions by relying on Riemannian manifold and infinite-horizon LQR
 * Change display by setting REFERENTIALS to 1 (referentials in sphere) or 0 (LQR for quaternions).
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
 * Author: Fabien Crépon
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw_gl2.h>

namespace linmath{
#include <linmath.h>
}

#include <armadillo>
#include <lqr.h>

using namespace arma;


#define REFERENTIALS	0		//Change REFERENTIALS to 1 (referentials in sphere) or 0 (LQR for quaternions)

#define RADIUS          1.f
#define STEP_LONGITUDE  10.5f
#define STEP_LATITUDE   10.5f

#define DIST_BALL       (RADIUS * 2.f + RADIUS * 0.1f)
#define VIEW_SCENE_DIST (DIST_BALL * 3.f + 200.f)

#define MARGIN 50



/* Vertex type */
typedef struct {float x; float y; float z;} vertex_t;


void DrawSphereBand( GLfloat long_lo, GLfloat long_hi );
void DrawSphere( void );

//-----------------------------------------------

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//-----------------------------------------------

int factorial(int n)
{
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

//-----------------------------------------------

double deg2rad( double deg )
{
	return deg / 360 * (2 * M_PI);
}

//-----------------------------------------------

double sin_deg( double deg )
{
	return sin( deg2rad( deg ) );
}

//-----------------------------------------------

double cos_deg( double deg )
{
	return cos(deg2rad ( deg ) );
}

//-----------------------------------------------

void CrossProduct( vertex_t a, vertex_t b, vertex_t c, vertex_t *n )
{
	GLfloat u1, u2, u3;
	GLfloat v1, v2, v3;

	u1 = b.x - a.x;
	u2 = b.y - a.y;
	u3 = b.y - a.z;

	v1 = c.x - a.x;
	v2 = c.y - a.y;
	v3 = c.z - a.z;

	n->x = u2 * v3 - v2 * v3;
	n->y = u3 * v1 - v3 * u1;
	n->z = u1 * v2 - v1 * u2;
}

//-----------------------------------------------

arma::mat QuatMatrix(arma::mat q)
{
	arma::mat Q;
	Q = {{q(0),-q(1),-q(2),-q(3)},
		 {q(1), q(0),-q(3), q(2)},
		 {q(2), q(3), q(0),-q(1)},
		 {q(3),-q(2), q(1), q(0)}};

	return Q;
}

//-----------------------------------------------

arma::mat acoslog(arma::mat x)
{
	arma::mat acosx(1,x.size());

	for (int n=0; n <= x.size()-1; n++)
	{
		if (x(0,n) >= 1.0)
			x(0,n) = 1.0;
		if (x(0,n) <= -1.0)
			x(0,n) = -1.0;
		if (x(0,n) < 0)
		{
			acosx(0,n) = acos(x(0,n))-M_PI;
		}
		else
			acosx(0,n) = acos(x(0,n));
	}

	return acosx;
}

//-----------------------------------------------

arma::mat expfct(arma::mat u)
{
	arma::mat normv = sqrt(pow(u.row(0),2)+pow(u.row(1),2)+pow(u.row(2),2));
	arma::mat Exp(4,1);

	Exp.row(0) = cos(normv);
	Exp.row(1) = u.row(0)%sin(normv)/normv;
	Exp.row(2) = u.row(1)%sin(normv)/normv;
	Exp.row(3) = u.row(2)%sin(normv)/normv;

	return Exp;
}

//-----------------------------------------------

arma::mat logfct(arma::mat x)
{
	arma::mat fullone;
	fullone.ones(size(x.row(0)));
	arma::mat scale(1,x.size()/4);
	scale = acoslog(x.row(0))/sqrt(fullone-pow(x.row(0),2));

	arma::mat Log(3,x.size()/4);
	Log.row(0) = x.row(1) % scale;
	Log.row(1) = x.row(2) % scale;
	Log.row(2) = x.row(3) % scale;

	return Log;
}

//-----------------------------------------------

arma::mat expmap(arma::mat u, arma::vec mu)
{
	arma::mat x = QuatMatrix(mu)*expfct(u);
	return x;
}

//-----------------------------------------------

arma::mat logmap(arma::mat x, arma::vec mu)
{
	arma::mat pole;
	arma::mat Q(4,4,fill::ones);

	pole = {1,0,0,0};

	if(norm(mu-trans(pole))< 1E-6)
		Q = {{1,0,0,0},
			 {0,1,0,0},
			 {0,0,1,0},
			 {0,0,0,1}};
	else
		Q = QuatMatrix(mu);

	arma::mat u;
	u = logfct(trans(Q)*x);

	return u;
}

//-----------------------------------------------

void DrawResults(int nbData, int nbRepros, const arma::mat& x_y,
				 const gfx2::window_size_t& window_size)
{
	const int plot_width = (window_size.win_width - 3 * MARGIN) / 2;
	const int plot_height = (window_size.win_height - 3 * MARGIN) / 2;

	const float scale_x = (float) plot_width / (nbData - 1);
	const float scale_y = (float) plot_height / 2;

	// Draw the plot backgrounds
	glColor3f(0.95f, 0.95f, 0.95f);

	glBegin (GL_QUADS);
	glVertex2f(MARGIN, window_size.win_height - MARGIN);
	glVertex2f(MARGIN, window_size.win_height - (MARGIN + plot_height));
	glVertex2f(MARGIN + plot_width, window_size.win_height - (MARGIN + plot_height));
	glVertex2f(MARGIN + plot_width, window_size.win_height - MARGIN);
	glEnd();

	glBegin (GL_QUADS);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), window_size.win_height - MARGIN);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), window_size.win_height - (MARGIN + plot_height));
	glVertex2f(window_size.win_width - (MARGIN + plot_width) + plot_width, window_size.win_height - (MARGIN + plot_height));
	glVertex2f(window_size.win_width - (MARGIN + plot_width) + plot_width, window_size.win_height - MARGIN);
	glEnd();

	glBegin (GL_QUADS);
	glVertex2f(MARGIN, MARGIN + plot_height);
	glVertex2f(MARGIN, MARGIN);
	glVertex2f(MARGIN + plot_width, MARGIN);
	glVertex2f(MARGIN + plot_width, MARGIN + plot_height);
	glEnd();

	glBegin (GL_QUADS);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), MARGIN + plot_height);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), MARGIN);
	glVertex2f(window_size.win_width - (MARGIN + plot_width) + plot_width, MARGIN);
	glVertex2f(window_size.win_width - (MARGIN + plot_width) + plot_width, MARGIN + plot_height);
	glEnd();

	// Draw the axes
	glColor3f(0.0f, 0.0f, 0.0f);
	glLineWidth(2);

	glBegin(GL_LINES);
	glVertex2f(MARGIN, window_size.win_height - MARGIN);
	glVertex2f(MARGIN, window_size.win_height - (MARGIN + plot_height));
	glVertex2f(MARGIN, window_size.win_height - (MARGIN + plot_height / 2));
	glVertex2f(MARGIN + plot_width, window_size.win_height - (MARGIN + plot_height / 2));
	glEnd();

	glBegin(GL_LINES);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), window_size.win_height - MARGIN);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), window_size.win_height - (MARGIN + plot_height));
	glVertex2f(window_size.win_width - (MARGIN + plot_width), window_size.win_height - (MARGIN + plot_height / 2));
	glVertex2f(window_size.win_width - MARGIN, window_size.win_height - (MARGIN + plot_height / 2));
	glEnd();

	glBegin(GL_LINES);
	glVertex2f(MARGIN, MARGIN + plot_height);
	glVertex2f(MARGIN, MARGIN);
	glVertex2f(MARGIN, MARGIN + plot_height / 2);
	glVertex2f(MARGIN + plot_width, MARGIN + plot_height / 2);
	glEnd();

	glBegin(GL_LINES);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), MARGIN + plot_height);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), MARGIN);
	glVertex2f(window_size.win_width - (MARGIN + plot_width), MARGIN + plot_height / 2);
	glVertex2f(window_size.win_width - MARGIN, MARGIN + plot_height / 2);
	glEnd();

	// Draw the targets
	glColor3f(0.98, 0.349, 0.329);
	glLineWidth(4);

	glBegin(GL_LINE_STRIP);
	for (int t = 0; t < nbData; t++) {
		glVertex2f(MARGIN + t * scale_x,
				   window_size.win_height - (MARGIN + plot_height / 2) + 1 * scale_y
		);
	}
	glEnd();
	glBegin(GL_LINE_STRIP);
	for (int t = 0; t < nbData; t++) {
		glVertex2f(window_size.win_width - (MARGIN + plot_width) + t * scale_x,
				   window_size.win_height - (MARGIN + plot_height / 2) + 0 * scale_y
		);
	}
	glEnd();
	glBegin(GL_LINE_STRIP);
	for (int t = 0; t < nbData; t++) {
		glVertex2f(MARGIN + t * scale_x,
				   MARGIN + plot_height / 2 + 0 * scale_y
		);
	}
	glEnd();
	glBegin(GL_LINE_STRIP);
	for (int t = 0; t < nbData; t++) {
		glVertex2f(window_size.win_width - (MARGIN + plot_width) + t * scale_x,
				   MARGIN + plot_height / 2 + 0 * scale_y
		);
	}
	glEnd();

	// Draw the results
	const arma::fmat COLORS({
		{ 0.00f, 0.00f, 1.00f },
		{ 0.00f, 0.50f, 0.00f },
		{ 1.00f, 0.00f, 0.00f },
		{ 0.00f, 0.75f, 0.75f },
		{ 0.75f, 0.00f, 0.75f },
		{ 0.75f, 0.75f, 0.00f },
		{ 0.25f, 0.25f, 0.25f },
		{ 0.00f, 0.00f, 1.00f },
		{ 0.00f, 0.50f, 0.00f },
		{ 1.00f, 0.00f, 0.00f },
	});

	glLineWidth(1);

	for (int n = 0; n < nbRepros; n++) {
		glColor3f(COLORS(n, 0), COLORS(n, 1), COLORS(n, 2));

		glBegin(GL_LINE_STRIP);
		for (int t = 0; t < nbData; t++) {
			glVertex2f(MARGIN + t * scale_x,
					   window_size.win_height - (MARGIN + plot_height / 2) + x_y(0, n * nbData + t) * scale_y
			);
		}
		glEnd();
	}

	for (int n = 0; n < nbRepros; n++) {
		glColor3f(COLORS(n, 0), COLORS(n, 1), COLORS(n, 2));

		glBegin(GL_LINE_STRIP);
		for (int t = 0; t < nbData; t++) {
			glVertex2f(window_size.win_width - (MARGIN + plot_width) + t * scale_x,
					   window_size.win_height - (MARGIN + plot_height / 2) + x_y(1, n * nbData + t) * scale_y
			);
		}
		glEnd();
	}

	for (int n = 0; n < nbRepros; n++) {
		glColor3f(COLORS(n, 0), COLORS(n, 1), COLORS(n, 2));

		glBegin(GL_LINE_STRIP);
		for (int t = 0; t < nbData; t++) {
			glVertex2f(MARGIN + t * scale_x,
					   MARGIN + plot_height / 2 + x_y(2, n * nbData + t) * scale_y
			);
		}
		glEnd();
	}

	for (int n = 0; n < nbRepros; n++) {
		glColor3f(COLORS(n, 0), COLORS(n, 1), COLORS(n, 2));

		glBegin(GL_LINE_STRIP);
		for (int t = 0; t < nbData; t++) {
			glVertex2f(window_size.win_width - (MARGIN + plot_width) + t * scale_x,
					   MARGIN + plot_height / 2 + x_y(3, n * nbData + t) * scale_y
			);
		}
		glEnd();
	}
}

//-----------------------------------------------

void printQuat(arma::mat q,int black)
{
	arma::mat referential = {{1,0,0},{0,1,0},{0,0,1}};
	arma::mat rotation;
	rotation	<< 2*(pow(q(0),2)+pow(q(1),2))-1	<< 2*(q(1)*q(2)-q(0)*q(3))	<< 2*(q(1)*q(3)+q(0)*q(2))	<< endr
				<< 2*(q(1)*q(2)+q(0)*q(3))	<< 2*(pow(q(0),2)+pow(q(2),2))-1	<< 2*(q(2)*q(3)-q(0)*q(1))	<< endr
				<< 2*(q(1)*q(3)-q(0)*q(2))	<< 2*(q(2)*q(3)+q(0)*q(1))	<< 2*(pow(q(0),2)+pow(q(3),2))-1	<< endr;
	arma::mat orientation;
	orientation = referential*rotation;


	if(black)
		glColor3f(0.,0.,0.);
	else
		glColor3f(1.,0.,0.);
	glLineWidth(1.4);
	glBegin(GL_LINE_STRIP);
		glVertex3f(0,0,0);
		glVertex3f(orientation(0,0),orientation(0,1),orientation(0,2));
	glEnd();

	if(black)
		glColor3f(0.,0.,0.);
	else
		glColor3f(0.,1.,0.);
	glLineWidth(1.4);
	glBegin(GL_LINE_STRIP);
		glVertex3f(0,0,0);
		glVertex3f(orientation(1,0),orientation(1,1),orientation(1,2));
	glEnd();

	if(black)
		glColor3f(0.,0.,0.);
	else
		glColor3f(0.,0.,1.);
	glLineWidth(1.4);
	glBegin(GL_LINE_STRIP);
		glVertex3f(0,0,0);
		glVertex3f(orientation(2,0),orientation(2,1),orientation(2,2));
	glEnd();

	return;
}

//-----------------------------------------------

void reshape( GLFWwindow* window, int w, int h )
{
	linmath::mat4x4 projection, view;
	glViewport( 0, 0, (GLsizei)w, (GLsizei)h );

	glMatrixMode( GL_PROJECTION );
	linmath::mat4x4_perspective( projection,
								 2.f * (float) atan2( RADIUS, 200.f ),
								 (float)w / (float)h,
								 1.f, VIEW_SCENE_DIST );
	glLoadMatrixf((const GLfloat*) projection);
	glMatrixMode( GL_MODELVIEW );
	{
		linmath::vec3 eye = { 0.f, 0.f, VIEW_SCENE_DIST };
		linmath::vec3 center = { 0.f, 0.f, 0.f };
		linmath::vec3 up = { 0.f, -1.f, 0.f };
		linmath::mat4x4_look_at( view, eye, center, up );
	}
	glLoadMatrixf((const GLfloat*) view);
}

//-----------------------------------------------

int main(int argc, char **argv){

	arma_rng::set_seed_random();

	//------------------------ Setup parameters ------------------------

	int nbData = 100;				//Number of datapoints
	int nbRepros = 5;				//Number of reproductions

	int nbVarPos = 3;				//Dimension of position data (here: x1,x2)
	int nbDeriv = 2;				//Number of static & dynamic features (D=2 for [x,dx])
	int nbVar = nbVarPos * nbDeriv;	//Dimension of state vector in the tangent space
	int nbVarMan = nbVarPos + 1;	//Dimension of the manifold
	double dt = 1E-3;				//Time step duration
	float rfactor = 1E-6;			//Control cost in LQR

	//Control Cost matrix
	arma::mat R = eye(nbVarPos,nbVarPos) * rfactor;

	//Target and desired covariance
	arma::vec xTar;
	xTar << 1 << endr << 0 << endr << 0 << endr << 0 << endr;

	arma::vec vecCov(3);
	vecCov << 1 << endr << 1E-3 << endr << 1E-3 << endr;
	arma::mat uCov = diagmat(vecCov);


	//------------ Discrete dynamical system settings (in tangent space) ------------

	arma::mat A, B;

	arma::mat A1d(zeros(nbDeriv,nbDeriv));
	for (int i = 0; i <= nbDeriv -1 ; i++) {
		A1d = A1d + diagmat(ones(nbDeriv-i),i) * pow(dt,i) * 1/factorial(i);
	}
	arma::mat B1d(zeros(nbDeriv,1));
	for (int i = 1; i <= nbDeriv ; i++) {
		B1d(nbDeriv-i,0) = pow(dt,i) * 1./factorial(i);
	}

	A = kron(A1d, eye(nbVarPos,nbVarPos));
	B = kron(B1d, eye(nbVarPos,nbVarPos));

	//---------------- Iterative discrete LQR with infinite horizon ----------------

	arma::mat duTar = zeros(nbVarPos*(nbDeriv-1),1);

	arma::mat Q = inv(uCov);
	Q.resize(6,6);
	arma::mat P = lqr::solve_algebraic_Riccati_discrete(A, B, Q, R);
	arma::mat L = (trans(B)*P*B + R).i() * trans(B)*P*A;

	arma::mat x(nbVarMan,1);
	arma::mat U;
	arma::mat x_y(zeros(4,nbRepros*nbData));
	arma::mat ddu(3,1);

	for (int n = 0; n < nbRepros ; n++) {
		x(0,0) = -1 + randn()*9E-1;
		x(1,0) = -1 + randn()*9E-1;
		x(2,0) =  1 + randn()*9E-1;
		x(3,0) = randn()*9E-1;

		x = x/norm(x);
		U = -logmap(x,xTar);
		U.resize(6,1);

		for (int t = 0; t <nbData ; t++) {
			x_y.col(n*nbData+t) = x;

			ddu.rows(0,2) = logmap(x,xTar);
			ddu.resize(6,1);
			ddu.rows(3,5) = duTar-U.rows(3,5);
			ddu = L * ddu;

			U = A*U + B*ddu;
			x = expmap(-1*U.rows(0,nbVarPos-1),xTar);
		}
	}

	//------------------------ Plots ------------------------

	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 500;
	window_size.win_height = 500;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;

	//Setup GUI
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		exit(1);

	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Quaternions", window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	// Setup ImGui binding
	if (!REFERENTIALS)
	{
		ImGui::CreateContext();
		ImGui_ImplGlfwGL2_Init(window, true);
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);

	while (!glfwWindowShouldClose(window)) {

		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;


		if (!REFERENTIALS)
		{
			ImGui_ImplGlfwGL2_NewFrame();

			ui::begin("demo");
			ImGuiWindow *win = ImGui::GetCurrentWindow();
			ui::end();

			ImVec4 clip_rect(0, 0, window_size.win_width, window_size.win_height);

			const int plot_height = (window_size.win_height - 3 * MARGIN) / 2;

			win->DrawList->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.5f,
								   ImVec2(10, MARGIN + plot_height / 2 - 10),
								   ImColor(0, 0, 0, 255), "qw", NULL, 0.0f, &clip_rect);

			win->DrawList->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.5f,
								   ImVec2(window_size.win_width / 2 - 10,
										  MARGIN + plot_height / 2 - 10),
								   ImColor(0, 0, 0, 255), "qx", NULL, 0.0f, &clip_rect);

			win->DrawList->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.5f,
								   ImVec2(10, MARGIN * 2 + plot_height + plot_height / 2 - 10),
								   ImColor(0, 0, 0, 255), "qy", NULL, 0.0f, &clip_rect);

			win->DrawList->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.5f,
								   ImVec2(window_size.win_width / 2 - 10,
										  MARGIN * 2 + plot_height + plot_height / 2 - 10),
								   ImColor(0, 0, 0, 255), "qz", NULL, 0.0f, &clip_rect);
		}

		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(1.f, 1.f, 1.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		if (!REFERENTIALS)
			glOrtho(0, window_size.win_width, 0, window_size.win_height, 0, 1);

		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		if (!REFERENTIALS)
		{
			ImGui::Render();
			ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());
		}

		glPushMatrix();

		//------------ Draw referentials in sphere ---------------
		if (REFERENTIALS)
		{
			printQuat(x_y.col(1), 0);
			printQuat(x_y.col(nbData - 1), 1);

			DrawSphere();
		}

		//--------------- Draw Q1,Q2,Q3,Q4 ---------------
		else
		{
			DrawResults(nbData, nbRepros, x_y, window_size);
		}

		glPopMatrix();
		glfwSwapBuffers(window);

		// Keyboard input
		int state = glfwGetKey(window, GLFW_KEY_ESCAPE);
		if (state == GLFW_PRESS)
			break;
	}

	//Cleanup
	if (!REFERENTIALS)
		ImGui_ImplGlfwGL2_Shutdown();

	glfwTerminate();

	return 0;
}

//-----------------------------------------------

void DrawSphere( void )
{
	GLfloat lon_deg;	  //degree of longitude
	double dt_total, dt2;

	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);

	glCullFace( GL_FRONT);
	glEnable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);


	//Build a faceted latitude slice of the sphere,
	//stepping same-sized vertical bands of the sphere.
	for (lon_deg = 0; lon_deg < 180; lon_deg += STEP_LONGITUDE )
	{
		//Draw a latitude circle at this longitude.
		DrawSphereBand(lon_deg, lon_deg + STEP_LONGITUDE);
	}

	glPopMatrix();

	return;
}

//-----------------------------------------------

void DrawSphereBand( GLfloat long_lo, GLfloat long_hi ) {
	vertex_t vert_ne;			  //"ne" means south-east, so on
	vertex_t vert_nw;
	vertex_t vert_sw;
	vertex_t vert_se;
	vertex_t vert_norm;
	GLfloat lat_deg;

	// Iterate thru the points of a latitude circle.
	// A latitude circle is a 2D set of X,Z points.

	for (lat_deg = 0; lat_deg <= (360 - STEP_LATITUDE);
		 lat_deg += STEP_LATITUDE) {

		glColor3f(0.55f, 0.55f, 0.55f);
		//Assign each Y.

		vert_ne.y = vert_nw.y = (float) cos_deg(long_hi) * RADIUS;
		vert_sw.y = vert_se.y = (float) cos_deg(long_lo) * RADIUS;


		// Assign each X,Z with sin,cos values scaled by latitude radius indexed by longitude.
		//Eg, long=0 and long=180 are at the poles, so zero scale is sin(longitude),
		//while long=90 (sin(90)=1) is at equator.

		vert_ne.x = (float) cos_deg(lat_deg) * (RADIUS * (float) sin_deg(long_lo + STEP_LONGITUDE));
		vert_se.x = (float) cos_deg(lat_deg) * (RADIUS * (float) sin_deg(long_lo));
		vert_nw.x = (float) cos_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float) sin_deg(long_lo + STEP_LONGITUDE));
		vert_sw.x = (float) cos_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float) sin_deg(long_lo));

		vert_ne.z = (float) sin_deg(lat_deg) * (RADIUS * (float) sin_deg(long_lo + STEP_LONGITUDE));
		vert_se.z = (float) sin_deg(lat_deg) * (RADIUS * (float) sin_deg(long_lo));
		vert_nw.z = (float) sin_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float) sin_deg(long_lo + STEP_LONGITUDE));
		vert_sw.z = (float) sin_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float) sin_deg(long_lo));


		//Draw the facet.
		glBegin(GL_LINE_STRIP);

		CrossProduct(vert_ne, vert_nw, vert_sw, &vert_norm);
		glNormal3f(vert_norm.x, vert_norm.y, vert_norm.z);

		glVertex3f(vert_ne.x, vert_ne.y, vert_ne.z);
		glVertex3f(vert_nw.x, vert_nw.y, vert_nw.z);
		glVertex3f(vert_sw.x, vert_sw.y, vert_sw.z);
		glVertex3f(vert_se.x, vert_se.y, vert_se.z);

		glEnd();

#if SPHERE_DEBUG
		printf( "----------------------------------------------------------- \n" );
		printf( "lat = %f  long_lo = %f	 long_hi = %f \n", lat_deg, long_lo, long_hi );
		printf( "vert_ne  x = %.8f	y = %.8f  z = %.8f \n", vert_ne.x, vert_ne.y, vert_ne.z );
		printf( "vert_nw  x = %.8f	y = %.8f  z = %.8f \n", vert_nw.x, vert_nw.y, vert_nw.z );
		printf( "vert_se  x = %.8f	y = %.8f  z = %.8f \n", vert_se.x, vert_se.y, vert_se.z );
		printf( "vert_sw  x = %.8f	y = %.8f  z = %.8f \n", vert_sw.x, vert_sw.y, vert_sw.z );
#endif
	}

	return;
}
