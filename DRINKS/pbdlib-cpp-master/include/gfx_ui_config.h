#pragma once

// conversion to and from armadillo
#include "armadillo"
#define UI_TRANS2D_CLASS_EXTRA \
Trans2d(const arma::vec& p, const arma::mat& rot) { x=ImVec2(rot(0,0),rot(1,0));  x=ImVec2(rot(0,1),rot(1,1)); \
        pos.x=p[0];  pos.y=p[1]; } \
        operator arma::mat() const { arma::mat f; \
        f.col(0) = arma::vec({x.x, x.y, 0.0}); \
        f.col(1) = arma::vec({y.x, y.y, 0.0}); \
        f.col(2) = arma::vec({pos.x, pos.y, 1.}); return f; } \
        Trans2d(const arma::mat& f) \
        { \
            x.x=f(0,0);  x.y=f(1,0); \
            y.x=f(0,1);  y.y=f(1,1); \
            pos.x=f(0,2);  pos.y=f(1,2); \
        }
