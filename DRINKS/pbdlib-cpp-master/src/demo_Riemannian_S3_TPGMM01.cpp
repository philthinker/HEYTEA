/*
 * demo_Riemannian_S3_TPGMM01.cpp
 *
 * TPGMM on S3 (unit quaternion) with two frames.
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
 * Author: Andras Kupcsik
 */

#include <stdio.h>
#include <vector>
#include <armadillo>

#include <gfx2.h>
#include <gfx_ui.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw_gl2.h>

#include <lqr.h>

using namespace arma;


static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//-----------------------------------------------

arma::mat QuatMatrix(arma::mat q) {
	arma::mat Q;
	Q = {
		{ q(0),-q(1),-q(2),-q(3) },
		{ q(1), q(0),-q(3), q(2) },
		{ q(2), q(3), q(0),-q(1) },
		{ q(3),-q(2), q(1), q(0) }
	};

	return Q;
}

//-----------------------------------------------

mat QuatToRotMat(vec4 q) {
	float w = q(0);
	float x = q(1);
	float y = q(2);
	float z = q(3);
	mat RotMat(3, 3);
	RotMat << 1 - 2 * y * y - 2 * z * z << 2 * x * y - 2 * z * w << 2 * x * z + 2 * y * w << endr
		   << 2 * x * y + 2 * z * w << 1 - 2 * x * x - 2 * z * z << 2 * y * z - 2 * x * w << endr
		   << 2 * x * z - 2 * y * w << 2 * y * z + 2 * x * w << 1 - 2 * x * x - 2 * y * y << endr;
	return RotMat;
}

//-----------------------------------------------

arma::mat acoslog(arma::mat x) {
	arma::mat acosx(1, x.size());

	for (int n = 0; n <= x.size() - 1; n++) {
		if (x(0, n) >= 1.0)
			x(0, n) = 1.0;
		if (x(0, n) <= -1.0)
			x(0, n) = -1.0;
		if (x(0, n) < 0) {
			acosx(0, n) = acos(x(0, n)) - M_PI;
		} else
			acosx(0, n) = acos(x(0, n));
	}

	return acosx;
}

//-----------------------------------------------

arma::mat expfct(arma::mat u) {
	arma::mat normv = sqrt(pow(u.row(0), 2) + pow(u.row(1), 2) + pow(u.row(2), 2));
	arma::mat Exp(4, u.n_cols);

	Exp.row(0) = cos(normv);
	Exp.row(1) = u.row(0) % sin(normv) / normv;
	Exp.row(2) = u.row(1) % sin(normv) / normv;
	Exp.row(3) = u.row(2) % sin(normv) / normv;

	return Exp;
}

//-----------------------------------------------

arma::mat logfct(arma::mat x) {
	arma::mat fullone;
	fullone.ones(size(x.row(0)));
	arma::mat scale(1, x.size() / 4);
	scale = acoslog(x.row(0)) / sqrt(fullone - pow(x.row(0), 2));

	arma::mat Log(3, x.size() / 4);
	Log.row(0) = x.row(1) % scale;
	Log.row(1) = x.row(2) % scale;
	Log.row(2) = x.row(3) % scale;

	return Log;
}

//-----------------------------------------------

arma::mat expmap(arma::mat u, arma::vec mu) {
	arma::mat x = QuatMatrix(mu) * expfct(u);
	return x;
}

//-----------------------------------------------

arma::mat logmap(arma::mat x, arma::vec mu) {
	arma::mat pole;
	arma::mat Q(4, 4, fill::ones);

	pole = {1,0,0,0};

	if (norm(mu - trans(pole)) < 1E-6) {
		Q = {
			{ 1,0,0,0 },
			{ 0,1,0,0 },
			{ 0,0,1,0 },
			{ 0,0,0,1 }
		};
	} else {
		Q = QuatMatrix(mu);
	}

	arma::mat u;
	u = logfct(trans(Q) * x);

	return u;
}

//-----------------------------------------------

arma::mat transp(vec g, vec h) {
	mat E;
	E << 0.0 << 0.0 << 0.0 << endr << 1.0 << 0.0 << 0.0 << endr << 0.0 << 1.0 << 0.0 << endr << 0.0 << 0.0 << 1.0;
	colvec tmpVec = zeros(4, 1);
	tmpVec.subvec(0, 2) = logmap(h, g);
	vec vm = QuatMatrix(g) * tmpVec;
	double mn = norm(vm, 2);
	mat Ac;
	if (mn < 1E-10) {
		Ac = eye(3, 3);
	}
	colvec uv = vm / mn;
	mat Rpar = eye(4, 4) - sin(mn) * (g * uv.t()) - (1 - cos(mn)) * (uv * uv.t());
	Ac = E.t() * QuatMatrix(h).t() * Rpar * QuatMatrix(g) * E;
	return Ac;
}

//-----------------------------------------------

arma::vec gaussPDF(mat Data, colvec Mu, mat Sigma) {

	int nbVar = Data.n_rows;
	int nbData = Data.n_cols;
	Data = Data.t() - repmat(Mu.t(), nbData, 1);

	vec prob = sum((Data * inv(Sigma)) % Data, 1);

	prob = exp(-0.5 * prob) / sqrt(pow((2 * datum::pi), nbVar) * det(Sigma) + DBL_MIN);

	return prob;
}

//-----------------------------------------------

void DrawFrame(ImGuiWindow& win, float lineWidth, mat A, vec center) {
	win.DrawList->PathClear();

	vec3 coordinatePos = zeros(3);
	ImVec2 cent, pt;

	cent.x = center(0);
	cent.y = center(1);

	for (int j = 0; j < 3; j++) {

		coordinatePos = A.col(j);

		pt.x = cent.x + coordinatePos(0);
		pt.y = cent.y - coordinatePos(1);

		if (j == 0)
			win.DrawList->AddLine(cent, pt, 0xffff0000, lineWidth);

		if (j == 1)
			win.DrawList->AddLine(cent, pt, 0xff00ff00, lineWidth);

		if (j == 2)
			win.DrawList->AddLine(cent, pt, 0xff0000ff, lineWidth);
	}
}


//-----------------------------------------------

int main(int argc, char **argv) {

	// Take 4k screens into account (framebuffer size != window size)
	gfx2::window_size_t window_size;
	window_size.win_width = 780;
	window_size.win_height = 540;
	window_size.fb_width = -1;	// Will be known later
	window_size.fb_height = -1;

	//Setup GUI
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		exit(1);

	GLFWwindow* window = gfx2::create_window_at_optimal_size(
		"Riemannian quaternion TP-GMM", window_size.win_width, window_size.win_height
	);

	glfwMakeContextCurrent(window);

	glEnable (GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Setup ImGui binding
	ImGui::CreateContext();
	ImGui_ImplGlfwGL2_Init(window, true);

	// white background
	ImVec4 clear_color = ImColor(255, 255, 255);
	ImVector<ImVec2> points, velocities;
	//vector<vec> pointsV, velocitiesV;
	float filterAlphaPos = 0.5;

	// Load Fonts
	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontDefault();
	ui::init(20.);

	arma_rng::set_seed_random();

	int nbStates = 5;
	int nbStates_prev = 5;
	int nbVar = 3;
	int nbSamples = 80;
	int nbVarMan = 4;
	int nbIterEM = 30;
	int nbIter = 10; //Number of iteration for the Gauss Newton algorithm
	double params_diagRegFact = 1E-4;
	int nbFrames = 2;
	vec e0 = { 0, 1.0, 0, 0.0 }; // center on manifold

	// Load manifold data, first row is time, 2:4 is data in fame 1, 5:7 is data in frame 2
	mat demo(2 * nbVarMan, 2 * nbSamples);
	demo.load("./data/data_quat_tpgmm_XU.txt", raw_ascii);

	mat demoX = demo; // data = [X, [U; 0]]
	mat demoU = demoX.submat(0, nbSamples, 5, 2 * nbSamples - 1);
	demoX = demoX.submat(0, 0, 7, nbSamples - 1);

	// cout << "demoX ncols/nrows: " << demoX.n_cols << ", " << demoX.n_rows << endl;
	// cout << "demoU ncols/nrows: " << demoU.n_cols << ", " << demoU.n_rows << endl;

	mat A1;
	A1 = eye(3, 3);
	mat A2;
	A2 << 1.0 << 0.0 << 0.0 << endr << 0.0 << 1.0 << 0.0 << endr << 0.0 << 0.0 << 1.0;
	colvec b1;
	b1 << 0.6399 << endr << 0.4254 << endr << 0.0 << endr << -0.6399;
	colvec b2;
	b2 << 0.3816 << endr << 0.8636 << endr << 0.0 << endr << 0.3295;

	mat MuManProduct;
	cube MuMan2;

	// Drawing
	bool init = true;
	vec x_centers = { 150, 300, 450, 600 };
	vec y_centers = { 375, 250, 125 };
	bool recompute = true;

	while (!glfwWindowShouldClose(window)) {

		//========== Recomputing tpgmm model with the latest task parameters ==========
		vec tmpVec;
		mat tmpCov;
		vec tmpEigenValues;
		mat tmpEigenVectors;
		vec Priors(nbStates);
		cube Sigma(nbVar * 2, nbVar * 2, nbStates);
		mat Mu(nbVar * 2, nbStates);

		// ++++++++++++++++++++++ RANDOM INIT ++++++++++++++++++
		/*			arma_rng::set_seed_random();  // set the seed to a random value

		 Priors = ones(nbStates);
		 Priors /= nbStates;

		 // Random init the means and covariances
		 //cout << "init menas and covs" << endl;
		 // 5 x nbStates

		 for (int i = 0; i < nbStates; i++) { //for some reason the matrix randn doesn't work => doing it with vectors
		 for (int j = 0; j < (1 + 2 * (nbVar - 1)); j++) {
		 Mu(j, i) = randn();
		 }
		 }
		 for (int i = 0; i < nbStates; i++) {
		 Sigma.slice(i) = eye(1 + 2 * (nbVar - 1), 1 + 2 * (nbVar - 1)) * 0.5; //%Covariance in the tangent plane at point MuMan
		 }
		 */

		// +++++++++++++++++++++++ KBINS INIT +++++++++++++++++++++++
		if (init || (nbStates != nbStates_prev)) {
			// cout << "Recomputing the model... " << endl;
			int nbData = 20;

			vec tSep = round(linspace<vec>(0, nbData, nbStates + 1));

			for (int i = 0; i < nbStates; i++) {
				rowvec id;

				for (int n = 0; n < 4; n++) { // data was made out of 4 demos
					vec vecDummy = round(n * nbData + linspace<vec>(tSep(i), tSep(i + 1) - 1, tSep(i + 1) - tSep(i)));

					id = join_horiz(id, vecDummy.t());
				}

				Priors(i) = id.n_elem;

				mat demoUColID = zeros(2 * nbVar, id.n_cols);

				for (int j = 0; j < id.n_elem; j++) {

					demoUColID.col(j) = demoU.col((unsigned int) id(j));
				}

				Mu.col(i) = mean(demoUColID, 1);
				Sigma.slice(i) = cov(demoUColID.t()) + eye(2 * nbVar, 2 * nbVar) * 1E-4;
			}
			Priors = Priors / sum(Priors);

			// +++++++++++++++++++++++++ INIT END +++++++++++++++++++++++++++++++++

			mat MuMan = join_vert(expmap(Mu.rows(0, 2), e0), expmap(Mu.rows(3, 5), e0));
			Mu = zeros(nbVar * 2, nbStates);

			cube u(nbVar * 2, nbSamples, nbStates); //uTmp in matlab code

			for (int nb = 0; nb < nbIterEM; nb++) {

				//E-step
				mat L = zeros(nbStates, nbSamples);
				mat xcTmp;

				for (int i = 0; i < nbStates; i++) {

					xcTmp = join_vert(logmap(demoX.rows(0, 3), MuMan.submat(0, i, 3, i)),
									  logmap(demoX.rows(4, 7), MuMan.submat(4, i, 7, i)));

					L.row(i) = Priors(i) * (gaussPDF(xcTmp, Mu.col(i), Sigma.slice(i)).t());

				}

				rowvec Lsum = sum(L, 0) + 1E-308;

				mat GAMMA = L / repmat(Lsum, nbStates, 1);

				colvec GammaSum = sum(GAMMA, 1);
				mat GAMMA2 = GAMMA / repmat(GammaSum, 1, nbSamples);

				//M-step
				for (int i = 0; i < nbStates; i++) {
					//Update Priors
					Priors(i) = sum(GAMMA.row(i)) / (nbSamples);

					//Update MuMan
					for (int n = 0; n < nbIter; n++) {
						u.slice(i) = join_vert(logmap(demoX.rows(0, 3), MuMan.submat(0, i, 3, i)),
											   logmap(demoX.rows(4, 7), MuMan.submat(4, i, 7, i)));

						MuMan.col(i) = join_vert(expmap(u.slice(i).rows(0, 2) * GAMMA2.row(i).t(), MuMan.submat(0, i, 3, i)),
								expmap(u.slice(i).rows(3, 5) * GAMMA2.row(i).t(), MuMan.submat(4, i, 7, i)));
					}
					//Update Sigma
					Sigma.slice(i) = u.slice(i) * diagmat(GAMMA2.row(i)) * u.slice(i).t() +
									 eye(2 * nbVar, 2 * nbVar) * params_diagRegFact;
				}

			}

			// for (int i = 0; i < nbStates; i++) {
			// 	cout << "============= Component #" << i << " ===========" << endl;
			// 	cout << "Prior: " << Priors(i) << endl;
			// 	cout << "MuMan:" << endl;
			// 	cout << MuMan.col(i).t() << endl;
			// 	cout << "Sigma: " << endl;
			// 	cout << Sigma.slice(i) << endl << endl;
			// }

			//Reformatting as a tensor GMM
			mat MuManOld = MuMan;
			cube SigmaOld = Sigma;
			MuMan2 = zeros(nbVarMan, nbFrames, nbStates);
			field<cube> Sigma2(nbStates);
			for (int i = 0; i < nbStates; i++) {
				cube dummyCube(nbVar, nbVar, nbFrames);
				Sigma2.at(i) = dummyCube;
			}

			for (int i = 0; i < nbStates; i++) {
				for (int m = 0; m < nbFrames; m++) {

					// dummy way to construct id and idMan, works for 2 frames
					uvec id;
					uvec idMan;
					if (m == 0) {
						id = {0, 1, 2};
					} else {
						id = {3, 4, 5};
					}

					if (m == 0) {
						idMan = {0, 1, 2, 3};
					} else {
						idMan = {4, 5, 6, 7};
					}

					for (int ii = 0; ii < idMan.size(); ii++) {
						MuMan2.slice(i).col(m)(ii) = MuManOld.col(i)(idMan(ii));
					}

					Sigma2.at(i).slice(m) = SigmaOld.slice(i).submat(id, id);

				}
			}

			// Transforming gaussians with respective frames
			for (int i = 0; i < nbStates; i++) {

				vec uTmp = A1 * logmap(MuMan2.slice(i).col(0).rows(0, 3), e0);
				MuMan2.slice(i).col(0).rows(0, 3) = expmap(uTmp, b1);
				mat Ac1 = transp(b1, MuMan2.slice(i).col(0).rows(0, 3));
				Sigma2.at(i).slice(0).submat(0, 0, 2, 2) = Ac1 * A1 * Sigma2.at(i).slice(0).submat(0, 0, 2, 2) * A1.t() * Ac1.t();

				uTmp = A2 * logmap(MuMan2.slice(i).col(1).rows(0, 3), e0);
				MuMan2.slice(i).col(1).rows(0, 3) = expmap(uTmp, b2);
				mat Ac2 = transp(b2, MuMan2.slice(i).col(1).rows(0, 3));
				Sigma2.at(i).slice(1).submat(0, 0, 2, 2) = Ac2 * A2 * Sigma2.at(i).slice(1).submat(0, 0, 2, 2) * A2.t() * Ac2.t();

			}

			//Gaussian product
			MuManProduct = zeros(nbVarMan, nbStates);
			cube SigmaProduct(nbVar, nbVar, nbStates);

			for (int i = 0; i < nbStates; i++) {
				mat componentsMu = zeros(nbVarMan, nbVar); // current means of the components
				cube U0 = zeros(nbVar, nbVar, nbStates);  //current covariances of the components

				vec tmpVec;
				mat tmpCov;
				vec tmpEigenValues;
				mat tmpEigenVectors;

				componentsMu.col(0) = MuMan2.slice(i).col(0).rows(0, 3);
				tmpCov = Sigma2.at(i).slice(0).submat(0, 0, 2, 2);
				eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);
				U0.slice(0) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

				componentsMu.col(1) = MuMan2.slice(i).col(1).rows(0, 3);
				tmpCov = Sigma2.at(i).slice(1).submat(0, 0, 2, 2);
				eig_sym(tmpEigenValues, tmpEigenVectors, tmpCov);
				U0.slice(1) = tmpEigenVectors * diagmat(pow(tmpEigenValues, 0.5));

				vec MuMan; // starting point on the manifold
				MuMan << 0.0 << endr << 0.0 << endr << 1.0 << endr << 0.0;
				MuMan /= norm(MuMan, 2);
				mat Sigma = zeros(nbVar, nbVar);

				mat MuTmp = zeros(nbVar, nbVar); //nbVar (on tangent space) x components
				cube SigmaTmp = zeros(nbVar, nbVar, nbStates); // nbVar (on tangent space) x nbVar (on tangent space) x components

				for (int n = 0; n < 10; n++) { // compute the Gaussian product
					// nbVar = 3 for S3 sphere tangent space
					colvec Mu = zeros(nbVar, 1);
					mat SigmaSum = zeros(nbVar, nbVar);

					for (int j = 0; j < 2; j++) { // we have two frames
						mat Ac = transp(componentsMu.col(j), MuMan);
						mat U1 = Ac * U0.slice(j);
						SigmaTmp.slice(j) = U1 * U1.t();
						//Tracking component for Gaussian i
						SigmaSum += inv(SigmaTmp.slice(j));
						MuTmp.col(j) = logmap(componentsMu.col(j), MuMan);
						Mu += inv(SigmaTmp.slice(j)) * MuTmp.col(j);
					}

					Sigma = inv(SigmaSum);
					//Gradient computation
					Mu = Sigma * Mu;

					MuMan = expmap(Mu, MuMan);
				}

				MuManProduct.col(i) = MuMan;
				SigmaProduct.slice(i) = Sigma;

				// cout << "================ Gaussian Product component #" << i << " ========================" << endl;
				// cout << "MuMan: " << MuMan.t() << endl;
				// cout << "Sigma: " << endl << Sigma << endl;
			}
		}

		glfwPollEvents();

		// Handling of the resizing of the window
		gfx2::window_result_t window_result =
			gfx2::handle_window_resizing(window, &window_size);

		ImGui_ImplGlfwGL2_NewFrame();

		if (init)
			ImGui::SetNextWindowPos(ImVec2(450, 430));

		ui::begin("demo");
		ImGuiWindow *win = ImGui::GetCurrentWindow();
		ui::end();

		if (init) { // Init the control panel
			ImGui::SetNextWindowPos(ImVec2(2, 2));
			ImGui::SetNextWindowCollapsed(false); //true);
		}

		nbStates_prev = nbStates;
		ImGui::SetNextWindowSize(ImVec2(400, 54));
		ImGui::Begin("Parameters", NULL, ImGuiWindowFlags_NoResize |
					 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove);
		ImGui::SliderInt("Nb states", &nbStates, 2, 6);

		ImGui::End();
		ImVec4 clip_rect2(0, 0, window_size.win_width, window_size.win_height);

		//=================== Plotting component frames ===================
		float plot_width = (window_size.win_width - 100) / 4;
		float plot_height = (window_size.win_height - 100) / 2;

		for (int i = 0; i < nbStates_prev; i++) { // nbStates_prev because we first want to recompute the model

			vec center({
				50 + plot_width * (i % 4) + plot_width / 2,
				80 + plot_height * (i / 4) + plot_height / 2,
			});

			mat Rot;
			for (int j = 0; j < 3; j++) {
				float linewidth;

				if (j == 2) {
					// frame product
					Rot = QuatToRotMat(MuManProduct.col(i));
					linewidth = 2.5;
				} else {
					// frame 1 or 2
					Rot = QuatToRotMat(MuMan2.slice(i).col(j).rows(0, 3));
					linewidth = 1.0;
				}

				DrawFrame(*win, linewidth, Rot * std::min(plot_width, plot_height) / 2, center);

			}
			std::string s = std::to_string(i);

			win->DrawList->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 1.5f,
								   ImVec2(center(0), center(1) - std::min(plot_width, plot_height) / 2),
								   ImColor(0, 0, 0, 255), s.c_str(),
								   NULL, 0.0f, &clip_rect2);
		}

		////=================== GUI rendering ===================
		glViewport(0, 0, window_size.fb_width, window_size.fb_height);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear (GL_COLOR_BUFFER_BIT);
		ImGui::Render();
		ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());

		glPopMatrix();
		glfwSwapBuffers(window);

		// Keyboard input
		if (ImGui::IsKeyPressed(GLFW_KEY_ESCAPE))
			break;

		init = false;
	}

	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}
