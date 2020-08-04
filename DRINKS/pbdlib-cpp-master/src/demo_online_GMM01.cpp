/*
 * demo_online_gmm01.cpp
 *
 * Online GMM learning.
 *
 * If this code is useful for your research, please cite the related publication:
 * @incollection{Calinon19chapter,
 * 	author="Calinon, S. and Lee, D.",
 * 	title="Learning Control",
 * 	booktitle="Humanoid Robotics: a Reference",
 * 	publisher="Springer",
 * 	editor="Vadakkepat, P. and Goswami, A.", 
 * 	year="2019",
 * 	doi="10.1007/978-94-007-7194-9_68-1",
 * 	pages="1--52"
 * }
 *
 * Author: Sylvain Calinon
 */

#include <stdio.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>
#include <gfx2.h>
#include <GLFW/glfw3.h>

#include <armadillo>
#include <mvn.h>

using namespace arma;


//-----------------------------------------------------------------------------
// Contains the parameters of a gaussian
//-----------------------------------------------------------------------------
struct gaussian_t {

	gaussian_t() {}

	gaussian_t(const vec& mu, const mat& sigma)
	: mu(mu), sigma(sigma) {}

	vec mu;
	mat sigma;
};


//---------------------------------------------------------


typedef std::vector<mat> matrix_list_t;
typedef std::vector<gaussian_t> gaussian_list_t;


//---------------------------------------------------------


class GMM_Model
{
public:
	// Constructor
	GMM_Model(unsigned int _nSTATES, unsigned int _nVARS) {
		nVARS = _nVARS;
		nSTATES = _nSTATES;
		PRIORS = rowvec(_nSTATES);

		// Initialize Components
		COMPONENTS.resize(nSTATES);
	}


	inline unsigned int getNumSTATES() const {
		return nSTATES;
	}

	inline gaussian_t& getCOMPONENTS(int id) {
		return COMPONENTS[id];
	}

	inline const colvec& getMU(int id) const {
		return COMPONENTS[id].mu;
	}

	inline const mat& getSIGMA(int id) const {
		return COMPONENTS[id].sigma;
	}

	inline void setCOMPONENTS(gaussian_list_t& components) {
		COMPONENTS = components;
	}


	//----------------------------------------------------------------------------
	// Performs online GMM clustering by using an updated DP-Means algorithms
	// Ref:
	//	 Kulis, B. and Jordan, M. I. (2012). Revisiting k-means: New algorithms
	//	 via bayesian nonparametrics. In Proc. Intl Conf. on Machine Learning
	//	 (ICML)
	//
	// Input:
	//	 N: Number of points processed
	//	 P: current point being added to GMM
	//	 lambda: splitting distance
	//	 nimSigma: minimum covariance for regularization
	//----------------------------------------------------------------------------
	void online_EMDP(int N, const colvec& P, double lambda, double minSigma) {
		// Initialized distance of P from first cluster
		double d = arma::norm(this->getMU(0)-P,2);

		int minIndex = 0;	//Index corresponding to current cluster

		// Find cluster corresponding to minimum distance
		for(int k=1;k<nSTATES;k++){
			double Dist = arma::norm(this->getMU(k) - P,2);
			if (Dist < d){
					d = Dist;
					minIndex = k; // Index corresponding to minimum distance
			}
		}

		// Allocate new cluster if distance of each component higher than lambda
		if (lambda < d){
			minIndex = nSTATES;
			mat SigmaTmp = minSigma*eye(nVARS,nVARS); // Sigma of new component is the minimum cov

			COMPONENTS.push_back(gaussian_t(P, SigmaTmp)); // Mean of new component is P

			rowvec priorsTmp = zeros(1, nSTATES + 1);
			priorsTmp.cols(0, nSTATES - 1) = PRIORS;
			priorsTmp(nSTATES) = 1. / N; //prior for new component inversely proportional to #P
			priorsTmp = priorsTmp / arma::norm(priorsTmp, 1); //evaluate new priors
			nSTATES = nSTATES +1; //update number of states
			PRIORS = priorsTmp;
		}
		else{
			/*
			*Update components belonging to P by using MAP estimate
			Ref:
			Gauvain, J.-L. and Lee, C.-H. (1994). Maximum a pos- teriori estimation for
			multivariate gaussian mixture observations of markov chians. IEE Transactions
			on Speech and Audio Processing, 2(2).  */
			double PriorsTmp = 1. / N + PRIORS[minIndex];

			vec MuTmp = 1. / PriorsTmp * (PRIORS[minIndex] * getMU(minIndex) + P / N);

			COMPONENTS[minIndex].sigma = PRIORS[minIndex] / PriorsTmp *
										 (getSIGMA(minIndex) + (getMU(minIndex) - MuTmp) * (getMU(minIndex) - MuTmp).t()) +
										 1. / (N * PriorsTmp) * (minSigma * eye(nVARS, nVARS) + (P - MuTmp) * (P - MuTmp).t());

			COMPONENTS[minIndex].mu = MuTmp;

			rowvec priors = PRIORS;
			priors[minIndex] = PriorsTmp;
			priors = priors / arma::norm(priors, 1);
			PRIORS = priors;
		}
	}


	void clear() {
		nSTATES = 1;
		PRIORS = ones<vec>(1);

		COMPONENTS.clear();
		COMPONENTS.resize(nSTATES);
	}


private:
	unsigned int	nVARS;
	unsigned int	nSTATES;
	gaussian_list_t COMPONENTS;
	rowvec			PRIORS;
};


//---------------------------------------------------------


static void error_callback(int error, const char* description){
	fprintf(stderr, "Error %d: %s\n", error, description);
}


//---------------------------------------------------------


int main(int argc, char **argv){

	//Setup GMM
	GMM_Model gmm(1,2);
	float minSigma = 2E2;
	float lambda = 250.0f;
	mat minSIGMA = eye(2,2) * minSigma;
	gaussian_list_t comps;
	matrix_list_t demos;
	// matrix_list_t repros;

	//Setup LQR
	// mat A(4,4), B(4,2);
	// float dt = 0.01f;
	// int iFactor = -8;
	// mat R = eye(2,2) * pow(10.0f,iFactor);
	// std::vector<mat> Q;
	// mat A1d; A1d << 0 << 1 << endr << 0 << 0 << endr;
	// mat B1d; B1d << 0 << endr << 1 << endr;
	// A = kron(A1d, eye(2,2)); //See Eq. (5.1.1) in doc/TechnicalReport.pdf
	// B = kron(B1d, eye(2,2)); //See Eq. (5.1.1) in doc/TechnicalReport.pdf


	//--------------- Setup of the rendering ---------------

	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 1200;
	window_size.win_height = 600;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;

	//Setup GUI
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		exit(1);

	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Demo Online GMM", window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	// Setup ImGui binding
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);
	ImVec4 clear_color = ImColor(255, 255, 255);


	//--------------- Main loop ---------------

	ImVector<ImVec2> points;
	bool adding_line = false;
	bool dispGMM = false;
	int nbPts = 0;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		// Start of rendering
		ImGui_ImplGlfwGL2_NewFrame();

		//Control panel GUI
		ImGui::SetNextWindowPos(ImVec2(2,2));
		// ImGui::SetNextWindowSize(ImVec2(350,140));
		ImGui::SetNextWindowSize(ImVec2(300,116));

		ImGui::Begin("Control Panel", NULL,
			ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
			ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings);

		if (ImGui::Button("Clear")){
			demos.clear();
			// repros.clear();
			gmm.clear();
			dispGMM = false;
		}

		ImGui::Text("Left-click to collect demonstrations");
		ImGui::Text("nbDemos: %d, nbPoints: %d, nbStates: %d", (int)demos.size(), (int)points.size(), (int)gmm.getNumSTATES());
		ImGui::SliderFloat("minSigma", &minSigma, 1E1, 2E3);
		ImGui::SliderFloat("lambda", &lambda, 10.0f, 500.0f);
		// if (ImGui::SliderInt("rFactor", &iFactor, -9, -6)){
		// 	R = eye(2,2) * pow(10.0f,iFactor);
		// }

		ImGui::End();


		//Get data
		ImVec2 mouse_pos_in_canvas(
			ImGui::GetIO().MousePos.x *
				(float) window_size.fb_width / (float) window_size.win_width,
			(window_size.win_height - ImGui::GetIO().MousePos.y) *
				(float) window_size.fb_height / (float) window_size.win_height
		);

		if (!ImGui::GetIO().WantCaptureMouse) { //Is outside gui?
			if (!adding_line && ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1)) { //Button pushed
				adding_line = true;
				if (!dispGMM){
					colvec p(2);
					p(0) = mouse_pos_in_canvas.x;
					p(1) = mouse_pos_in_canvas.y;
					comps.push_back(gaussian_t(p, minSIGMA));
					gmm.setCOMPONENTS(comps);
					comps.clear();
					dispGMM = true;
				}
			}

			if (adding_line){ //Trajectory recording
				points.push_back(mouse_pos_in_canvas);
				nbPts++;
				vec p(2);
				p(0) = mouse_pos_in_canvas.x;
				p(1) = mouse_pos_in_canvas.y;
				gmm.online_EMDP(nbPts, p, lambda, minSigma);

				if (!ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)) { //Button released
					adding_line = false;
					//Add demonstration
					mat demo(2, points.size());
					for (int t = 0; t < (int) points.size(); t++) {
						demo(0, t) = points[t].x;
						demo(1, t) = points[t].y;
					}
					demos.push_back(demo);

					//Compute sequence of states
					mat h(gmm.getNumSTATES(), points.size());
					for (int i = 0; i < (int) gmm.getNumSTATES(); i++) {
						h.row(i) = trans(mvn::getPDFValue(gmm.getCOMPONENTS(i).mu, gmm.getCOMPONENTS(i).sigma, demo));
					}
					//uword imax[points.size()];
					//for (int t=0; t<points.size(); t++){
					//	vec vTmp = h.col(t);
					//	vTmp.max(imax[t]);
					//}
					//
					// //LQR
					// vec vTmp(4,fill::zeros);
					// mat QTmp(4,4,fill::zeros);
					// mat Target(4,points.size(),fill::zeros);
					// for (int t=0; t<points.size(); t++){
					// 	QTmp.submat(0,0,1,1) = inv(gmm.getSIGMA(imax[t]));
					// 	Q.push_back(QTmp);
					// 	vTmp.subvec(0,1) = gmm.getMU(imax[t]);
					// 	Target.col(t) = vTmp;
					// }
					//
					// matrix_list_t gains = lqr::evaluate_gains_infinite_horizon(A, B, R, Q, Target);
					//
					// //Retrieve data
					// mat rData(2,points.size());
					// vec u(2);
					// vec X = join_cols(demo.col(0), zeros<vec>(2));
					// for (int t=0; t<points.size(); t++){
					// 	rData.col(t) = X.rows(0,1);
					// 	u = gains[t] * (Target.col(t) - X);
					// 	X += (A * X + B * u) * dt;
					// }
					// repros.push_back(rData);

					//Clean up
					points.clear();
					// Q.clear();
				}
			}
		}

		//GUI rendering
		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		//Rendering
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		glOrtho( 0, window_size.fb_width, 0,  window_size.fb_height, -1., 1.);
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();
		glPushMatrix();

		//Draw current demo
		glColor3f(0.0f, 0.0f, 0.0f);
		glBegin(GL_LINE_STRIP);
		for (int t=0; t<(int)points.size(); t++){
			glVertex2f(points[t].x, points[t].y);
		}
		glEnd();

		//Draw demos
		glColor3f(0.3f, 0.3f, 0.3f);
		for (int n=0; n<(int)demos.size(); n++){
			glBegin(GL_LINE_STRIP);
			for (int t = 0; t < (int) demos[n].n_cols; t++){
				glVertex2f(demos[n](0, t), demos[n](1, t));
			}
			glEnd();
		}

		//Draw repros
		// glColor3f(0.0f, 0.8f, 0.0f);
		// for (int n=0; n<(int)repros.size(); n++){
		// 	glBegin(GL_LINE_STRIP);
		// 	for (int t = 0; t < (int) repros[n].n_cols; t++){
		// 		glVertex2f(repros[n](0, t), repros[n](1, t));
		// 	}
		// 	glEnd();
		// }

		//Draw Gaussians
		if (dispGMM){
			vec d(2);
			mat V(2,2), R(2,2), pts(2,30);
			mat pts0(2,30);
			pts0 = join_cols(cos(linspace<rowvec>(0, 2 * datum::pi, 30)),
							 sin(linspace<rowvec>(0, 2 * datum::pi, 30)));

			glColor3f(0.8f, 0.0f, 0.0f);
			for (int i=0; i<(int)gmm.getNumSTATES(); i++){
				eig_sym(d, V, gmm.getSIGMA(i));
				R = V * sqrt(diagmat(d));
				pts = R * pts0;
				glBegin(GL_LINE_STRIP);
				for (int t=0; t<(int)pts.n_cols; t++){
					glVertex2f((pts(0,t)+gmm.getMU(i)(0)), (pts(1,t)+gmm.getMU(i)(1)));
				}
				glEnd();
				glBegin(GL_POINTS);
				for (int t=0; t<(int)pts.n_cols; t++){
					glVertex2f(gmm.getMU(i)(0), gmm.getMU(i)(1));
				}
				glEnd();
		  }
		}

		glPopMatrix();
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;
	}

	//Cleanup
	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();
	return 0;
}
