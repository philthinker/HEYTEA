//MILK

#include <Eigen/Dense>

double cosInterp(double x0, double xg, double t);
std::array<double,16> vectorP2arrayCarte(std::vector<double> vectorIn,std::array<double,16> carteIn);
Eigen::Vector3d quatSubtraction(Eigen::Quaterniond q1, Eigen::Quaterniond q2);
Eigen::Matrix<double,6,6> vectorK2Matrix6(std::vector<double> k);