//FW_testBench

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <cmath>
#include <fstream>

#include <Eigen/Dense>

#include <franka/robot.h>
#include <franka/exception.h>
#include <franka/model.h>

#include "LACTIC/LACTIC.h"
#include "MILK/MILK.h"

int main(int argc, char** argv){
    std::string K_file(argv[1]);
    std::vector<std::vector<double>> Ks = readCSV(K_file);
    unsigned int N = Ks.size();
    unsigned int counter = 0;
    for (unsigned int i = 0; i < N; i++)
    {
        Eigen::Matrix<double,6,6> K = vectorK2Matrix6(Ks[i]);
        if (counter >= 100)
        {
            std::cout << K;
            std::cout << std::endl;
            counter = 0;
        }
        counter++;
    }
}