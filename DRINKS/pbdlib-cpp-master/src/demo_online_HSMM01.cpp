/*
 * demo_online_hsmm01.cpp
 *
 * Online HSMM learning.
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
 * Authors: Sylvain Calinon, Ioannis Havoutis
 */

#include <stdio.h>
#include <gfx2.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl2.h>
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


void get_state_seq(uword state_seq[], mat pred) {
	for (int t=0; t<pred.n_cols; t++){
		vec vTmp = pred.col(t);
		vTmp.max(state_seq[t]);
	}
}


//---------------------------------------------------------


class HSMM
{
public:
	// Constructor
	HSMM(unsigned int _nSTATES, unsigned int _nVARS) {
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
	//   Kulis, B. and Jordan, M. I. (2012). Revisiting k-means: New algorithms
	//   via bayesian nonparametrics. In Proc. Intl Conf. on Machine Learning
	//   (ICML)
	//
	// Input:
	//   N: Number of points processed
	//   P: current point being added to GMM
	//   lambda: splitting distance
	//   nimSigma: minimum covariance for regularization
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


	//----------------------------------------------------------------------------
	// Predict forward variable without predicted observations (implementation
	// for real-time, no checks on sizes done)
	//----------------------------------------------------------------------------
	void predict_forward_variable(mat& _AlphaPred) {
		mat ALPHA = Pd;
		ALPHA.each_col() %= PRIORS.t(); // % is the element wise product

		vec S = zeros<vec>(nSTATES);

		for (unsigned int i = 0; i < _AlphaPred.n_cols; i++)
		{
			// Alpha variable
			if (i > 0)
				updateALPHA(ALPHA, S);

			// Update S
			updateS(S, ALPHA);

			// Update alpha
			_AlphaPred.col(i) = sum(ALPHA, 1);
		}
	}


	void clear() {
		hsmm_transition.zeros();
		hsmm_transition_ticks.zeros();
		hsmm_priors.clear();
		hsmm_priors_ticks.clear();

		nSTATES = 1;
		PRIORS = ones<vec>(1);

		COMPONENTS.clear();
		COMPONENTS.resize(nSTATES);
	}


	void predict_forward_variable_deterministic(mat& _AlphaPred, int startState) {
		rowvec tempPriors = PRIORS;

		rowvec fixed_priors(nSTATES, fill::zeros);
		fixed_priors(startState) = 1.0;

		PRIORS = fixed_priors;

		initialize_forward_calculation();
		predict_forward_variable(_AlphaPred);

		PRIORS = tempPriors;
	}


	void predict_forward_variable_stochastic(mat& _AlphaPred) {
		rowvec tempPriors = PRIORS;
		PRIORS = hsmm_priors;

		initialize_forward_calculation();

		int nbSt = 0;
		int currTime = 0;
		rowvec iList(1, fill::zeros);
		int nbData = _AlphaPred.n_cols;
		int nStates = nSTATES;
		mat h = zeros(nStates, nbData);
		rowvec h1, h2;

		int nbD = this->Pd.n_cols;

		uword Iminmax;
		while (currTime < nbData) {
			if (nbSt==0){
				vec tmp = PRIORS.t() % randu(nStates);
				tmp.max(Iminmax);
				iList(0) = Iminmax;
				h1 = ones<rowvec>(nbData);
			}else{
				h1 = join_rows( join_rows(
						zeros<rowvec>(currTime), cumsum(this->Pd.row(iList(iList.n_elem-2))) ),
						ones<rowvec>( std::max( (nbData-currTime-nbD),0) ));
				currTime = currTime + round( DurationCOMPONENTS[(unsigned int)iList(iList.n_elem-2)].mu(0,0) );
			}
			h2 = join_rows( join_rows(
					ones<rowvec>(currTime),
					ones<rowvec>( this->Pd.row(iList(iList.n_elem-1)).n_elem )
					- cumsum(this->Pd.row(iList(iList.n_elem-1))) ),
					zeros<rowvec>( std::max( (nbData-currTime-nbD),0) ));

			h.row(iList(iList.n_elem-1)) = h.row(iList(iList.n_elem-1))
											+ min( join_cols(h1.cols(0,nbData-1),h2.cols(0,nbData-1)) );
			vec tmp= TransitionMatrix.row(iList(iList.n_elem-1)).t() % randu(nStates);
			tmp.max(Iminmax);
			iList.resize(iList.n_elem+1);
			iList(iList.n_elem-1) = Iminmax;

			nbSt = nbSt+1;
		}
		h = h / repmat(sum(h,0) ,nStates,1);
		_AlphaPred = h;
		PRIORS = tempPriors;
	}


	void integrate_demonstration(mat demo) {
		//Compute sequence of states
		const int nPoints = demo.n_cols;
		mat h(nSTATES, nPoints);
		for (int i=0; i<(int)nSTATES; i++){
			h.row(i) = trans(mvn::getPDFValue(getCOMPONENTS(i).mu, getCOMPONENTS(i).sigma, demo));
		}
		uword* imax = new uword[nPoints];
		get_state_seq(imax, h);

		hsmm_transition.resize(nSTATES, nSTATES);
		hsmm_transition_ticks.resize(nSTATES, nSTATES);
		hsmm_priors.resize(nSTATES);
		hsmm_priors_ticks.resize(nSTATES);

		//update hsmm priors for stochastic sampling
		hsmm_priors_ticks(imax[0]) += 1;
		hsmm_priors =  hsmm_priors_ticks / accu(hsmm_priors_ticks);

		//update duration statistics
		hsmm_duration_stats.resize(nSTATES);

		unsigned int s0 = imax[0], currStateDuration = 1;
		for (int t = 0; t < nPoints; t++) {
			if ( (imax[t] != s0) || (t+1 == nPoints) ) {
				if (t + 1 == nPoints)
					currStateDuration += 1;

				hsmm_duration_stats[s0](currStateDuration);
				currStateDuration = 0.0;
				hsmm_transition(s0, imax[t]) += 1.0;
				hsmm_transition_ticks(s0, imax[t]) += 1.0;
				s0 = imax[t];
			}
			currStateDuration += 1;
		}

		delete[] imax;

		// normalize the transition matrix
		for (int i = 0; i < hsmm_transition.n_rows ; i++){
			double row_sum = accu(hsmm_transition.row(i));
			if (row_sum > 1){
				double row_sum_ticks = accu(hsmm_transition_ticks.row(i));
				hsmm_transition.row(i) = hsmm_transition_ticks.row(i) / row_sum_ticks;
			}
		}

		TransitionMatrix = hsmm_transition;

		// Set hsmmd durations:
		mat _SIGMA = zeros(1,1);
		colvec _MU = zeros(1,1);
		gaussian_list_t hsmm_components;

		for (unsigned int i=0; i<nSTATES; i++) {
			_MU(0,0) = hsmm_duration_stats[i].mean();//state_duration_means(i);
			_SIGMA(0,0) = std::max( hsmm_duration_stats[i].var(), 1.0);
			hsmm_components.push_back(gaussian_t(_MU, _SIGMA));
		}

		DurationCOMPONENTS = hsmm_components;
	}


private:
	void initialize_forward_calculation() {
		// Calculate PD size:
		PdSize = 0;
		for (unsigned int i = 0;i<nSTATES;i++)
		{
			if (PdSize < accu(DurationCOMPONENTS[i].mu))
				PdSize = accu(DurationCOMPONENTS[i].mu);
		}

		PdSize *= 1.2; // Enlarge the Size of Pd for 'accuracy'

		Pd.zeros(nSTATES, PdSize);
	

		// Duration input vector
		mat dur = linspace(1,PdSize, PdSize);
		// Pre-Calculate Pd
		for (unsigned int i = 0;i<nSTATES;i++)
		{
			// Note that we need to transpose twice....
			// getPDFValue accepts a rowvec of values, but returns a colvec....
			Pd.row(i) = mvn::getPDFValue(DurationCOMPONENTS[i].mu,
										 DurationCOMPONENTS[i].sigma,
										 dur.t()
						).t();
		}
	}


	// Equation (12): ALPHA MATRIX without observation
	void updateALPHA(mat &ALPHA, const colvec& S) const {
		mat Atmp1 = Pd.cols(0, PdSize - 2);
		Atmp1.each_col() %= S;
	
		mat Atmp2 = ALPHA.cols(1, PdSize - 1);

		ALPHA.cols(0, PdSize - 2) = Atmp2  + Atmp1;
		ALPHA.col(PdSize - 1) = S % Pd.col(PdSize - 1);
	}


	// Equation (6): Update S
	void updateS(colvec &S, const mat& ALPHA) const {
		S = TransitionMatrix.t() * ALPHA.col(0);
	}


private:
	unsigned int nVARS;
	unsigned int nSTATES;

	gaussian_list_t COMPONENTS;
	gaussian_list_t DurationCOMPONENTS;
	rowvec          PRIORS;

	mat TransitionMatrix;

	// Variables for state duration
	mat  Pd;     // Matrix with precomputed probability values for the duration;
	unsigned int PdSize; // Maximum maximum duration step

	// For demonstration integration
	mat    hsmm_transition;
	mat    hsmm_transition_ticks;
	rowvec hsmm_priors;
	rowvec hsmm_priors_ticks;

	std::vector <running_stat<double> > hsmm_duration_stats;
};


//---------------------------------------------------------


static void error_callback(int error, const char* description){
	fprintf(stderr, "Error %d: %s\n", error, description);
}


//---------------------------------------------------------


// mat run_lqr(const mat& A, const mat& B, float dt, const mat& R, int traj_length,
// 			const colvec& start_state, uword state_seq[], const HSMM& model) {
//
// 	matrix_list_t Q;
// 	vec vTmp(4, fill::zeros);
// 	mat QTmp(4, 4, fill::zeros);
// 	mat Target(4, traj_length, fill::zeros);
// 	for (int t = 0; t < traj_length; ++t) {
// 		QTmp.submat(0, 0, 1, 1) = inv(model.getSIGMA(state_seq[t]));
// 		Q.push_back(QTmp);
// 		vTmp.subvec(0, 1) = model.getMU(state_seq[t]);
// 		Target.col(t) = vTmp;
// 	}
//
// 	matrix_list_t gains = lqr::evaluate_gains_infinite_horizon(A, B, R, Q, Target);
//
// 	//Retrieve data
// 	mat rData(2, traj_length);
// 	vec u(2);
// 	vec X = join_cols(start_state, zeros<vec>(2)); // second part sets the velocities to zero
// 	//	vec X = start_state; // for vel
// 	for (int t = 0; t < traj_length; ++t) {
// 		rData.col(t) = X.rows(0, 1);
// 		u = gains[t] * (Target.col(t) - X);
// 		X += (A * X + B * u) * dt;
// 	}
//
// 	return rData;
// }


//---------------------------------------------------------


int main(int argc, char **argv){

	//int nVars = 4;
	int nVars = 2; //position only
	HSMM hsmm(1,nVars);
	float minSigma = 2E2;
	float lambda = 250.0f;
	mat minSIGMA = eye(nVars,nVars) * minSigma;
	gaussian_list_t comps;
	matrix_list_t demos;
	// matrix_list_t repros;
	// matrix_list_t repros_hsmm;
	// matrix_list_t repros_sampling;

	//Setup hsmm
	int init_nStates = 1; // we need to initialize the states to something

	//Setup LQR
	// mat A(4,4), B(4,2);
	// float dt = 0.01f;
	//
	// float iFactor = -8;
	// mat R = eye(2,2) * pow(10.0f,iFactor);
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
		"Demo Online HSMM", window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	// Setup ImGui
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);	
	ImVec4 clear_color = ImColor(255, 255, 255);


	//--------------- Main loop ---------------

	ImVector<ImVec2> points;
	ImVector<ImVec2> pointsVel;
	bool adding_line = false;
	bool dispGMM = false;
	int nbPts = 0;

	ImVec2 mouse_pos_in_canvas;

	running_stat<float> xi;
	float counter = 2;
	while (!glfwWindowShouldClose(window)){
		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		if (window_result == gfx2::INVALID_SIZE)
			continue;

		// Start of rendering
		ImGui_ImplGlfwGL2_NewFrame();

		//Control panel GUI
		// static bool hsmm_sampling = false;

		ImGui::SetNextWindowPos(ImVec2(2,2));
		// ImGui::SetNextWindowSize(ImVec2(hsmm_sampling ? 400 : 300,
		// 								hsmm_sampling ? 250 : 180));
		ImGui::SetNextWindowSize(ImVec2(300, 116));

		ImGui::Begin("Control Panel", NULL,
				ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|
				ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings);

		if (ImGui::Button("Clear")){
			demos.clear();
			// repros.clear();
			// repros_sampling.clear();
			// repros_hsmm.clear();
			hsmm.clear();
			dispGMM = false;
		}

		ImGui::Text("Left-click to collect demonstrations");
		ImGui::Text("nbDemos: %i, nbPoints: %i, nbStates: %i", (int)demos.size(), (int)points.size(), (int)hsmm.getNumSTATES());
		ImGui::SliderFloat("minSigma", &minSigma, 1E1, 2E3);
		ImGui::SliderFloat("lambda", &lambda, 10.0f, 500.0f);
		// if (ImGui::SliderFloat("rFactor", &iFactor, -9.0f, -1.0f,  "%.1f" )){
		// 	R = eye(2,2) * pow(10.0f,iFactor);
		// }

		// static bool hsmm_stochastic_sampling = true;
		// ImGui::Checkbox("Stochastic sampling", &hsmm_stochastic_sampling);
		//
		// static int hsmm_sampling_start_state = 0;
		// static int hsmm_sampling_traj_length = 10;
		// ImGui::Checkbox("Run hsmm sampling", &hsmm_sampling);
		//
		// if (hsmm_sampling) {
		// 	ImGui::SliderInt("Start state", &hsmm_sampling_start_state, 0, hsmm.getNumSTATES() - 1);
		// 	ImGui::SliderInt("Trajectory length", &hsmm_sampling_traj_length, 2, 1000);
		// 	if (ImGui::Button("Sample") && (hsmm.getNumSTATES()!=1)) {
		// 		mat pred(hsmm.getNumSTATES(), hsmm_sampling_traj_length, fill::zeros);
		// 		if (!hsmm_stochastic_sampling) {
		// 			hsmm.predict_forward_variable_deterministic(pred, hsmm_sampling_start_state);
		// 		}else{
		// 			hsmm.predict_forward_variable_stochastic(pred);
		// 		}
		// 		uword state_seq[pred.n_cols];
		// 		get_state_seq(state_seq, pred);
		// 		mat hsmm_rData = run_lqr(A, B, dt,
		// 				R, hsmm_sampling_traj_length, hsmm.getMU(state_seq[0]), state_seq,
		// 				hsmm);
		// 		repros_sampling.push_back(hsmm_rData);
		// 	}
		// 	ImGui::SameLine();
		// 	if (ImGui::Button("Clear Samples")) {
		// 		repros_sampling.clear();
		// 	}
		// }

		ImGui::End();

		//Get data
		ImVec2 mouse_pos_in_canvas_curr(
			ImGui::GetIO().MousePos.x *
				(float) window_size.fb_width / (float) window_size.win_width,
			(window_size.win_height - ImGui::GetIO().MousePos.y) *
				(float) window_size.fb_height / (float) window_size.win_height
		);

		double alpha = 0.2;
		mouse_pos_in_canvas.x = mouse_pos_in_canvas.x +
				alpha*(mouse_pos_in_canvas_curr.x - mouse_pos_in_canvas.x);
		mouse_pos_in_canvas.y = mouse_pos_in_canvas.y +
				alpha*(mouse_pos_in_canvas_curr.y - mouse_pos_in_canvas.y);

		if (!ImGui::GetIO().WantCaptureMouse) { //Is outside gui?
			if (!adding_line && ImGui::IsMouseClicked(GLFW_MOUSE_BUTTON_1)){ //Button pushed
				adding_line = true;
				if (!dispGMM){
					colvec p(2);
					p(0) = mouse_pos_in_canvas.x;
					p(1) = mouse_pos_in_canvas.y;
					comps.push_back(gaussian_t(p, minSIGMA));
					hsmm.setCOMPONENTS(comps);
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

				hsmm.online_EMDP(nbPts, p, lambda, minSigma);

				if (!ImGui::IsMouseDown(GLFW_MOUSE_BUTTON_1)){ //Button released
					adding_line = false;
					//Add demonstration
					mat demo = mat(nVars,points.size());
					for (int t=0; t<(int)points.size(); t++){
						demo(0,t) = points[t].x;
						demo(1,t) = points[t].y;
					}
					demos.push_back(demo);

					//Compute sequence of states
					mat h(hsmm.getNumSTATES(), points.size());
					for (int i=0; i<(int)hsmm.getNumSTATES(); i++){
						h.row(i) = trans(
							mvn::getPDFValue(hsmm.getCOMPONENTS(i).mu,
											 hsmm.getCOMPONENTS(i).sigma,
					        				 demo)
						);
					}
					//uword imax[points.size()];
					//get_state_seq(imax, h);
					//
					// //LQR
					// mat rData = run_lqr(A, B, dt,
					// 		R, points.size(), demo.col(0), imax,
					// 		hsmm);
					// repros.push_back(rData);
					//
					// //.HSMM testing part
					// hsmm.integrate_demonstration(demo);
					// mat pred(hsmm.getNumSTATES(), points.size(), fill::zeros);
					//
					// hsmm.predict_forward_variable_deterministic(pred, imax[0]);
					//
					// uword state_seq[pred.n_cols];
					// get_state_seq(state_seq, pred);
					//
					// //LQR
					// mat hsmm_rData = run_lqr(A, B, dt,
					// 		R, /*Q,*/ points.size(), hsmm.getMU(state_seq[0]), state_seq,
					// 		hsmm);
					// repros_hsmm.push_back(hsmm_rData);
					// //.End of HSMM section//

					//Clean up
					points.clear();
					pointsVel.clear();
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
			for (int t=0; t<(int)demos[n].n_cols; t++){
				glVertex2f(demos[n](0,t), demos[n](1,t));
			}
			glEnd();
		}

		//Draw repros sampling
		// glColor3f(0.0f, 0.0f, 0.9f);
		// for (int n=0; n<(int)repros_sampling.size(); n++){
		// 	glBegin(GL_LINE_STRIP);
		// 	for (int t=0; t<(int)repros_sampling[n].n_cols; t++){
		// 		glVertex2f(repros_sampling[n](0,t), repros_sampling[n](1,t));
		// 	}
		// 	glEnd();
		// }

		//Draw Gaussians
		if (dispGMM){
			vec d(2);
			mat V(2,2), R(2,2), pts(2,30);
			mat pts0(2,30);
			pts0 = join_cols(cos(linspace<rowvec>(0,2*datum::pi,30)), sin(linspace<rowvec>(0,2*datum::pi,30)));
			glColor3f(0.8f, 0.0f, 0.0f);
			for (int i=0; i<(int)hsmm.getNumSTATES(); i++){
				eig_sym(d, V, hsmm.getSIGMA(i).submat(0,0,1,1));
				R = V * sqrt(diagmat(d)); 
				pts = R * pts0;
				glBegin(GL_LINE_STRIP);
				for (int t=0; t<(int)pts.n_cols; t++){
					glVertex2f((pts(0,t)+hsmm.getMU(i)(0)), (pts(1,t)+hsmm.getMU(i)(1)));
				}
				glEnd();
				glBegin(GL_POINTS);
				for (int t=0; t<(int)pts.n_cols; t++){
					glVertex2f(hsmm.getMU(i)(0), hsmm.getMU(i)(1));
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
