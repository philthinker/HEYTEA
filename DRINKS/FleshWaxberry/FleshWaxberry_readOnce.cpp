//FleshWaxberry_readOnce
//  Read what you want once.
//
//  Haopeng Hu
//  2020.07.04
//
//  argv[0] <fci-ip> fileOutName

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <franka/robot.h>
#include <franka/exception.h>
#include <franka/model.h>

#include "LACTIC/LACTIC.h"

int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <fci-ip> fileOutName" << std::endl;
        return -1;
    }
    // File out name
    std::string fileOutName(argv[2]);
    // Data to be read
    std::string fileOutName_JP(fileOutName);    // Deep copy
    std::fstream fileOut_JP(fileOutName_JP.append("_JP.csv"),std::ios::out);
    std::string fileOutName_OTEE(fileOutName);
    std::fstream fileOut_OTEE(fileOutName_OTEE.append("_OTEE.csv"),std::ios::out);
    try
    {
        // Init. robot with some param.
        franka::Robot robot(argv[1]);
        // This time we play the robot in its teaching mode.
        // Start reading
        franka::RobotState state = robot.readOnce();
        for (unsigned int i = 0; i < 7; i++)
        {
            fileOut_JP << state.q[i] << ',';
        }
            fileOut_JP << std::endl;
        for (unsigned int i = 0; i < 16; i++)
        {
            fileOut_OTEE << state.O_T_EE[i] << ',';
        }
        fileOut_OTEE << std::endl;
    }
    catch(const franka::Exception& e)
    {
        std::cerr << e.what() << '\n';
        fileOut_JP.close();
        fileOut_OTEE.close();
        return -1;
    }
    fileOut_JP.close();
    fileOut_OTEE.close();
    return 0;
}