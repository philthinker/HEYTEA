//FleshWaxberry_lead
//  Lead Franka Emika to assembly
//
//  Haopeng Hu
//  2020.07.03
//
//  argv[0] <fci-ip> fileOutName fps timeout

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <franka/robot.h>
#include <franka/exception.h>
#include <franka/model.h>

#include "LACTIC/LACTIC.h"

int main(int argc, char** argv){
    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << " <fci-ip> fileOutName fps timeout" << std::endl;
        return -1;
    }
    // File out name
    std::string fileOutName(argv[2]);
    // Data to be read
    std::string fileOutName_JP(fileOutName);    // Deep copy
    std::fstream fileOut_JP(fileOutName_JP.append("_JP.csv"),std::ios::out);
    std::string fileOutName_OTEE(fileOutName);
    std::fstream fileOut_OTEE(fileOutName_OTEE.append("_OTEE.csv"),std::ios::out);
    // fps in [1,1000]
    unsigned int fps = std::floor(getDataFromInput(argv[3],1,1000));
    // timeout
    double timeout = getDataFromInput(argv[4],0.0,300);
    try
    {
        // Init. robot with some param.
        franka::Robot robot(argv[1]);
        // This time we play the robot in its teaching mode.
        // Do NOT set its impedance param.
        std::cout << "File: " << fileOutName << "_DATA.csv" << std::endl
            << "FPS: " << fps << " data/s" << std::endl
            << "Timeout: " << timeout << " s" << std::endl
            <<  "Press Enter to continue ..." << std::endl;
        std::cin.ignore();
        // Start reading
        unsigned int count = 1;
        double timer = 0.0;
        robot.read([&](const franka::RobotState& state) -> bool{
            if (count >= floor(1000/fps))
            {
                // Read the specified data
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
                // Reset counter
                count = 1;
            }else
            {
                count++;
            }
            timer++;
            return timer <= timeout * 1000; 
        });
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