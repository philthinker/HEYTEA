//FleshWaxberry_zeroGravity
//  Generate zero gravity response
//  
// Haopeng Hu
// 2020.07.03
//
// argv[0] <fci-ip> fileOutName

#include <iostream>
#include <fstream>
#include <string>

#include <franka/robot.h>
#include <franka/exception.h>
#include <franka/model.h>

#include "LACTIC/LACTIC.h"

int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <fci-ip> fileOutName" << std::endl;
        return -1;
    }
    
}