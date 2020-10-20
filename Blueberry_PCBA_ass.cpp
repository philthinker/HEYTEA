//Blueberry_PCBA_ass
//  Move to the given Cartesian position while read something
//  Only position motion is required. No orientation motion.
//  For better performance, please move the robot to its inital position in advance.
//
//  Haopeng Hu
//  2020.10.19
//
// argv[0] <fci-ip> fileInName fps fileOutName
// fileInName: N x 3

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>

#include <franka/robot.h>
#include <franka/exception.h>

#include "LACTIC/LACTIC.h"
#include "MILK/MILK.h"

int main(int argc, char** argv){
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] <<" <fci-ip> fps fileOutName" << std::endl;
        return -1;
    }
    // Destination
    std::array<double,3> posi_goal;
    // fps [1,100]
    unsigned int fps = std::floor(getDataFromInput(argv[3],1,100));
    // File out
    std::string fileOutName(argv[4]);
    std::ofstream fileOut(fileOutName.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl;
    std::cout << "Data will be stored in file: " << fileOutName << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    std::cout << "Start moving ..." << std::endl;
    // Init. robot
    franka::Robot robot(argv[1]);
    robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 60.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 60.0, 25.0, 25.0, 25.0}});
    try{
        // Intermediate variables
        double timer = 0.0;
        std::array<double,16> carte_init;
        unsigned int fpsCounter = 1;
    /**
     * @Assembling phase: CartesianPose motion generation
     * No orientation motion
     */
    double pcbDepth = 0.015;
    timer = 0.0;
    double scaler = 0.2;
    robot.control([&timer,&carte_init,pcbDepth,fps,&fpsCounter,&fileOut,scaler](const franka::RobotState& state,franka::Duration period)
    -> franka::CartesianPose{
        // S-Spline Cartesian motion generation
        timer += period.toSec();
        if (timer == 0.0)
        {
            carte_init = state.O_T_EE_d;
        }
        franka::CartesianPose carte_c(carte_init);
        carte_c.O_T_EE[14] = cosInterp(carte_init[14],carte_init[14]-pcbDepth,timer*scaler);
        // Log
        if (fpsCounter >= 1000/fps)
        {
            fpsCounter = 1;
            for (unsigned int i = 0; i < 16; i++)
            {
                fileOut << state.O_T_EE[i] << ',';
            }
            fileOut << std::endl;
        }else
        {
            fpsCounter++;
        }
        // Terminal condition
        if (timer*scaler >= 1.0)
        {
            return franka::MotionFinished(carte_c);
        }
        return carte_c;
    });
    std::cout << "Assembling phase finished" << std::endl;
    }
    catch(const franka::Exception& e){
        std::cerr << e.what() <<'\n';
        fileOut.close();
        return -1;
    }
    fileOut.close();
    return 0;
}