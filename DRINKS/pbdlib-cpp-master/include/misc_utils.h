/*
 * draw_utils.h
 *
 *  Created on: 8 Mar 2016
 *      Author: ihavoutis
 */
 
#ifndef RLI_PBDLIB_GUI_SANDBOX_SRC_MISC_UTILS_H_
#define RLI_PBDLIB_GUI_SANDBOX_SRC_MISC_UTILS_H_

#include "armadillo"

using namespace std;
using namespace arma;

class Ref {
public:
	vec p1;
	vec p2;

	mat A;
	vec b;
	double theta;

	vec delta;

	void updateRef() {
		b = p1;
//		A = mat(2,2);
//		vec delta = p2-p1;
		delta = p2-p1;
		delta = delta/arma::norm(delta);
		A(0,0) = delta(0);
		A(1,0) = delta(1);
		A(0,1) = -delta(1);
		A(1,1) = delta(0);
	}

	Ref(){
		p1 = zeros(2);
		p2 = zeros(2);
		A = zeros(2,2);
		delta = zeros(2);
		theta = 0.0;
	}
};


class Remap {
public:
	Remap(float input_min, float input_max,
			float output_min, float output_max)
:_input_min(input_min), _input_max(input_max),
 _output_min(output_min), _output_max(output_max){};

private:
	float _input_min, _input_max, _output_min, _output_max;

public:
	float map( float input ) {
		return _output_min + (input - _input_min) *
				((_output_max - _output_min) / (_input_max - _input_min));
	}

	float mapInv( float input ) {
		return _input_min + (input - _output_min) *
						((_input_max - _input_min) / (_output_max - _output_min));
	}
};

#endif /* RLI_PBDLIB_GUI_SANDBOX_SRC_MISC_UTILS_H_ */
