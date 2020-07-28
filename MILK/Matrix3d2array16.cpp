//Matrix3d2array16

#include <array>
#include <Eigen/Core>

std::array<double,16> Matrix3d2array16(Eigen::Matrix3d matrixIn){
    std::array<double,16> arrayOut;
    arrayOut.fill(0.0);
    for (short int i = 0; i < 3; i++)
    {
        for (short int j = 0; j < 3; j++)
        {
            arrayOut[j+i*4] = matrixIn.coeff(i,j);
        }
        arrayOut[i*4+3] = 0.0;
    }
    arrayOut[15] = 1.0;
    return arrayOut;
}