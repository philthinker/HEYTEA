//Blueberry_Posi
//  Move to the given Cartesian position while read something
//  Only position motion is required. No orientation motion.
//
//  Haopeng Hu
//  2020.10.18
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
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] <<" <fci-ip> fileInName fps fileOutName" << std::endl;
        return -1;
    }
    // Destination
    std::array<double,16> carte_goal;
        std::string fileInName(argv[2]);
        std::vector<std::vector<double>> dataIn = readCSV(fileInName);
        for (unsigned int i = 0; i < 16; i++)
        {
            carte_goal[i] = dataIn[0][i];
        }
    // fps [1,100]
    std::string FPS(argv[4]);
    unsigned int fps = std::floor(getDataFromInput(argv[4],1,100));
    // File out
    std::string fileOutName(argv[3]);
    std::ofstream fileOut(fileOutName.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "The robot will move to Cartesian position:" << std::endl;
    for (unsigned i = 12; i < 15; i++)
    {
        std::cout << carte_goal[i] << ',';
    }
    std::cout << "Data will be stored in file: " << fileOutName << std::endl;
    std::cout << std::endl << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    // Joint pose motion
    try{
    double timer = 0.0;
    std::array<double,16> carte_init;
    unsigned int counter = 1;
    robot.control([&timer,&carte_init,carte_goal,fps,&counter,&fileOut](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose{
        // S-spline interpolation
        timer += period.toSec();
        if (timer == 0.0)
        {
            carte_init = state.O_T_EE;
        }
        franka::CartesianPose carte_c(carte_init);
        for (unsigned int i = 12; i < 15; i++)
        {
            carte_c.O_T_EE[i] = cosInterp(carte_init[i],carte_goal[i],timer*0.4);
        }
        // Read data to file out
        if (counter >= 1000/fps)
        {
            for (unsigned int i = 0; i < 7; i++)
            {
                fileOut << state.q[i] << ',';
            }
            fileOut << std::endl;
            counter = 1;
        }else
        {
            counter++;
        }
        if (timer*0.4 >= 1)
        {
            return franka::MotionFinished(carte_c);
        }
        return carte_c;
    });
    }
    catch(const franka::Exception& e){
        std::cerr << e.what() <<'\n';
        fileOut.close();
        return -1;
    }
    fileOut.close();
    return 0;
}