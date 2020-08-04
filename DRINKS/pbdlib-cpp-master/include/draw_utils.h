/*
 * draw_utils.h
 *
 *  Created on: 8 Mar 2016
 *      Author: ihavoutis
 */

#ifndef RLI_PBDLIB_GUI_SANDBOX_SRC_DRAW_UTILS_H_
#define RLI_PBDLIB_GUI_SANDBOX_SRC_DRAW_UTILS_H_

#include <stdio.h>

#include "armadillo"
#include <pbdlib/gmm.h>
#include <pbdlib/tpdpgmm.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include "imgui_impl_glfw.h"
#include <pbdlib/gmr.h>
//#include <pbdlib/lqr.h>
#include <pbdlib/taskparameters.h>
//#include "pbdlib/trajMPC.h"

#include "misc_utils.h"
#include <deque>

using namespace std;
using namespace pbdlib;
using namespace arma;

//plotting color
mat colorM = {{0.f,0.6211f,0.8164f},
		{0.9883f,0.4531f,0.f}, // orange
		{0.7422f,0.8555f,0.2227f},
		{0.1211f,0.5391f,0.4375f},
		{0.9961f,0.8789f,0.1016f}};
vec3 colBlack = {0.0, 0.0, 0.0};
vec3 colRed = {1.0, 0.0, 0.0};
vec3 colGreen = {0.0, 1.0, 0.0};
vec3 colBRGreen = {0.0, 0.26, 0.14};
vec3 colBlue = {0.0, 0.0, 1.0};
vec3 colGray = {0.3f, 0.3f, 0.3f};
vec3 colLightGray = {0.6f, 0.6f, 0.6f};
vec3 colLightBlue = {0.53, 0.8, 0.98};
vec3 colLightGreen = {0.56, 0.93, 0.56};

vector<vec3> colorCode;

void drawStrip(mat points, vec3 color, float linewidth){
	glLineWidth(linewidth);
	glColor3f(color[0], color[1], color[2]);
	glBegin(GL_LINE_STRIP);
	for (int t=0; t<(int)points.n_cols; t++){
		glVertex2f(points(0,t), points(1,t));
	}
	glEnd();
}

void drawStrip(vector<mat> points, vec3 color, float linewidth){
	for (int n=0; n<(int)points.size(); n++){
		drawStrip(points[n],color,linewidth);
	}
}

void drawStrip(ImVector<ImVec2> &points, vec3 color, float linewidth){
	glLineWidth(linewidth);
	glColor3f(color[0], color[1], color[2]);
	glBegin(GL_LINE_STRIP);
	for (int t=0; t<(int)points.size(); t++){
		glVertex2f(points[t].x, points[t].y);
	}
	glEnd();
}

void drawStrip(deque<vec> &points, int query_pt, vec offset, vec3 color, float linewidth){
	glLineWidth(linewidth);
	glColor3f(color[0], color[1], color[2]);
	glBegin(GL_LINE_STRIP);
	for (int t=query_pt-1; t<(int)points.size(); t++){
		glVertex2f(points[t](0) + offset(0), points[t](1) + offset(1));
	}
	glEnd();
}

void drawStrip(deque<vec> &points, int query_pt, vec3 color, float linewidth){
	glLineWidth(linewidth);
	glColor3f(color[0], color[1], color[2]);
	glBegin(GL_LINE_STRIP);
	for (int t=query_pt-1; t<(int)points.size(); t++){
		glVertex2f(points[t](0), points[t](1));
	}
	glEnd();
}
void drawStrip(vector<vec> &points, vec3 color, float linewidth){
	glLineWidth(linewidth);
	glColor3f(color[0], color[1], color[2]);
	glBegin(GL_LINE_STRIP);
	for (int t=0; t<(int)points.size(); t++){
		glVertex2f(points[t](0), points[t](1));
	}
	glEnd();
}

void drawProdGaussians(TPDPGMM &tpgmm, vec color){
	vec d(2);
	mat V(2,2), R(2,2), pts(2,30);
	mat pts0(2,30);
	pts0 = join_cols(cos(linspace<rowvec>(0,2*PI,30)), sin(linspace<rowvec>(0,2*PI,30)));
	glColor3f(color(0), color(1), color(2));
	for (int i=0; i<(int)tpgmm.getNumSTATES(); i++){
		eig_sym(d, V, tpgmm.getProdGMM()->getSIGMA(i).rows(0,1).cols(0,1));
		R = V * sqrt(diagmat(d));
		pts = R * pts0;
		glBegin(GL_LINE_STRIP);
		for (int t=0; t<(int)pts.n_cols; t++){
			glVertex2f((pts(0,t)+tpgmm.getProdGMM()->getMU(i)(0)),
					(pts(1,t)+tpgmm.getProdGMM()->getMU(i)(1)));
		}
		glEnd();
		glBegin(GL_POINTS);
			glVertex2f(tpgmm.getProdGMM()->getMU(i)(0),
					tpgmm.getProdGMM()->getMU(i)(1));
		glEnd();
	}
}

void drawGaussians(TPDPGMM &tpgmm, Ref refFrame[], mat colorM){
	vec d(2);
	mat V(2,2), R(2,2), pts(2,30);
	mat pts0(2,30);
	pts0 = join_cols(cos(linspace<rowvec>(0,2*PI,30)), sin(linspace<rowvec>(0,2*PI,30)));
	glLineWidth(2.0f);
	for (int i=0; i<(int)tpgmm.getNumSTATES(); i++) {
		for (int k = 0; k < (int) tpgmm.getNumFRAMES(); k++) {

			eig_sym(d, V, tpgmm.getGMMS(k).getSIGMA(i).rows(0,1).cols(0,1));
			R = refFrame[k].A*(V * sqrt(diagmat(d)));
			pts = R * pts0;

			glColor3f(colorM(k+1,0), colorM(k+1,1), colorM(k+1,2));
			glBegin(GL_LINE_STRIP);

			vec refMU = refFrame[k].b + refFrame[k].A*tpgmm.getGMMS(k).getMU(i).rows(0,1);

			for (int t = 0; t < (int) pts.n_cols; t++) {
				glVertex2f((pts(0, t) + refMU(0)), (pts(1, t) +refMU(1)));
			}
			glEnd();
			glBegin(GL_POINTS);
				glVertex2f(refMU(0),refMU(1));
			glEnd();
		}
	}
}

void drawGaussians(TPDPGMM &tpgmm, Ref refFrame[], vector<vec3> colorM){
	vec d(2);
	mat V(2,2), R(2,2), pts(2,30);
	mat pts0(2,30);
	pts0 = join_cols(cos(linspace<rowvec>(0,2*PI,30)), sin(linspace<rowvec>(0,2*PI,30)));
	glLineWidth(2.0f);
	for (int i=0; i<(int)tpgmm.getNumSTATES(); i++) {
		for (int k = 0; k < (int) tpgmm.getNumFRAMES(); k++) {

			eig_sym(d, V, tpgmm.getGMMS(k).getSIGMA(i).rows(0,1).cols(0,1));
			R = refFrame[k].A*(V * sqrt(diagmat(d)));
			pts = R * pts0;

			glColor3f(colorM[k](0), colorM[k](1), colorM[k](2));
			glBegin(GL_LINE_STRIP);

			vec refMU = refFrame[k].b + refFrame[k].A*tpgmm.getGMMS(k).getMU(i).rows(0,1);

			for (int t = 0; t < (int) pts.n_cols; t++) {
				glVertex2f((pts(0, t) + refMU(0)), (pts(1, t) +refMU(1)));
			}
			glEnd();
			glBegin(GL_POINTS);
				glVertex2f(refMU(0),refMU(1));
			glEnd();
		}
	}
}

void drawGaussians(TPGMM &tpgmm, Ref refFrame[], mat colorM){
	vec d(2);
	mat V(2,2), R(2,2), pts(2,30);
	mat pts0(2,30);
	pts0 = join_cols(cos(linspace<rowvec>(0,2*PI,30)), sin(linspace<rowvec>(0,2*PI,30)));
	glLineWidth(2.0f);
	for (int i=0; i<(int)tpgmm.getNumSTATES(); i++) {
		for (int k = 0; k < (int) tpgmm.getNumFRAMES(); k++) {

			eig_sym(d, V, tpgmm.getGMMS(k).getSIGMA(i).rows(0,1).cols(0,1));
			R = refFrame[k].A*(V * sqrt(diagmat(d)));
			pts = R * pts0;

//			glColor3f(colorM(0), colorM(1), colorM(2));
			if (k==0){glColor3f(1.0, 0.0, 0.0);}
			if (k==1){glColor3f(0.0, 0.0, 1.0);}
			glBegin(GL_LINE_STRIP);

			vec refMU = refFrame[k].b + refFrame[k].A*tpgmm.getGMMS(k).getMU(i).rows(0,1);

			for (int t = 0; t < (int) pts.n_cols; t++) {
				glVertex2f((pts(0, t) + refMU(0)), (pts(1, t) +refMU(1)));
			}
			glEnd();

			glBegin(GL_POINTS);
				glVertex2f(refMU(0),refMU(1));
			glEnd();
		}
	}
}


void drawGaussians(vector<GaussianDistribution> &vGD, vec color){
	glLineWidth(1.0f);
	vec d(2);
	mat V(2,2), R(2,2), pts(2,30);
	mat pts0(2,30);
	pts0 = join_cols(cos(linspace<rowvec>(0,2*PI,30)), sin(linspace<rowvec>(0,2*PI,30)));
	glColor3f(color(0), color(1), color(2));
	for (int i=0; i<(int)vGD.size(); i++){
		eig_sym(d, V, vGD[i].getSIGMA().rows(0,1).cols(0,1));
		R = V * sqrt(diagmat(d));
		pts = R * pts0;
		glBegin(GL_LINE_STRIP);
		for (int t=0; t<(int)pts.n_cols; t++){
			glVertex2f((pts(0,t)+vGD[i].getMU()(0)),
					(pts(1,t)+vGD[i].getMU()(1)));
		}
		glEnd();
		glBegin(GL_POINTS);
			glVertex2f(vGD[i].getMU()(0),
					vGD[i].getMU()(1));
		glEnd();
	}
}

void drawGaussian(GaussianDistribution &vGD, vec color){
	glLineWidth(1.0f);
	vec d(2);
	mat V(2,2), R(2,2), pts(2,30);
	mat pts0(2,30);
	pts0 = join_cols(cos(linspace<rowvec>(0,2*PI,30)), sin(linspace<rowvec>(0,2*PI,30)));
	glColor3f(color(0), color(1), color(2));
		eig_sym(d, V, vGD.getSIGMA().rows(0,1).cols(0,1));
		R = V * sqrt(diagmat(d));
		pts = R * pts0;
		glBegin(GL_LINE_STRIP);
		for (int t=0; t<(int)pts.n_cols; t++){
			glVertex2f((pts(0,t)+vGD.getMU()(0)),
					(pts(1,t)+vGD.getMU()(1)));
		}
		glEnd();
		glBegin(GL_POINTS);
			glVertex2f(vGD.getMU()(0),
					vGD.getMU()(1));
		glEnd();
}

//void drawSquare(int pos_x, int pos_y, int square_edge, float lineWidth = 5.0,
//		vec col = vec({0.8f, 0.2f, 0.2f})){
//	glLineWidth(lineWidth);
//	glBegin( GL_QUADS );
////	glColor3f( 0.8f, 0.2f, 0.2f );
//	glColor3f(col(0), col(1), col(2));
//	int dot_w = square_edge/2;
//	glVertex2f( pos_x - dot_w, pos_y - dot_w);
//	glVertex2f( pos_x - dot_w, pos_y + dot_w);
//	glVertex2f( pos_x + dot_w, pos_y + dot_w);
//	glVertex2f( pos_x + dot_w, pos_y - dot_w);
//	glEnd();
//}

void drawSquare(int pos_x, int pos_y, int square_edge, float lineWidth = 5.0,
		vec col = vec({0.8f, 0.2f, 0.2f}), int filled = 1){
	glLineWidth(lineWidth);
	if (filled)
		glBegin( GL_QUADS );
	else
		glBegin(GL_LINE_STRIP);
//	glColor3f( 0.8f, 0.2f, 0.2f );
	glColor3f(col(0), col(1), col(2));
	int dot_w = square_edge/2;
	glVertex2f( pos_x - dot_w, pos_y - dot_w);
	glVertex2f( pos_x - dot_w, pos_y + dot_w);
	glVertex2f( pos_x + dot_w, pos_y + dot_w);
	glVertex2f( pos_x + dot_w, pos_y - dot_w);
	glVertex2f( pos_x - dot_w, pos_y - dot_w);
	glEnd();
}

#endif /* RLI_PBDLIB_GUI_SANDBOX_SRC_DRAW_UTILS_H_ */
