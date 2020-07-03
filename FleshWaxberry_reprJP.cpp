//FleshWaxberry_reprJP
//  Reproduce the joint position trjectory from a .csv file
//  S-spline interpolation
//
//  Haopeng Hu
//  2020.07.03
//
//  argv[0] <fci-ip> fileInName fileOutName speed

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <franka/robot.h>
#include <franka/exception.h>
#include <franka/model.h>

#include "LACTIC/LACTIC.h"

int main(int argc, char** argv){
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <fci-ip> fileInName fileOutName speed" << std::endl;
        return -1;
    }
    
    return 0;
}
