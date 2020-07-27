//FleshWaxberry_OneshotP
//  Move to the given Cartesian position
//
//  Haopeng Hu
//  2020/07/25
//
// argv[0] <fci-ip> fileInName

#include <iostream>
#include <string>
#include <vector>
#include <array>

#include <franka/robot.h>
#include <franka/exception.h>

#include "LACTIC/LACTIC.h"
#include "MILK/MILK.h"

int main(int argc, char** argv){
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] <<" <fci-ip> fileInName" << std::endl;
        return -1;
    }
    std::array<double,16> carte_goal = {{0.707433,-0.70678,0.000806757,0,
                                        -0.706781,-0.707433,4.44759e-05,0,
                                        0.000539291,-0.000601664,-1,0,
                                        0.306881,-9.92382e-05,0.540234,1,}};
    if(argc >= 3){
        std::string fileInName(argv[2]);
        std::vector<std::vector<double>> dataIn = readCSV(fileInName);
        for (unsigned int i = 12; i < 15; i++)
        {
            carte_goal[i] = dataIn[0][i];
        }
    }
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "The robot will move to Cartesian position:" << std::endl;
    for (unsigned i = 12; i < 15; i++)
    {
        std::cout << carte_goal[i] << ',';
    }
    std::cout << std::endl << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    try{
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
    double timer = 0.0;
    std::array<double,16> carte_init;
    robot.control([&timer,&carte_init,carte_goal](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose{
        // S-spline interpolation
        timer += period.toSec();
        if (timer == 0.0)
        {
            carte_init = state.O_T_EE_d;
        }
        franka::CartesianPose carte_c(carte_init);
        for (unsigned int i = 12; i < 15; i++)
        {
            carte_c.O_T_EE[i] = cosInterp(carte_init[i],carte_goal[i],timer*0.4);
        }
        if (timer*0.4 >= 1)
        {
            return franka::MotionFinished(carte_c);
        }
        return carte_c;
    });
    }
    catch(const franka::Exception& e){
        std::cout << e.what() << '\n';
    }
}