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
#include <cmath>

#include <franka/robot.h>
#include <franka/exception.h>
#include <franka/model.h>

#include "LACTIC/LACTIC.h"
#include "MILK/MILK.h"

int main(int argc, char** argv){
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <fci-ip> fileInName fileOutName speed" << std::endl;
        return -1;
    }
    // File in
    std::string fileInName(argv[2]);
    std::vector<std::vector<double>> JPIn = readCSV(fileInName);
    // speed & fps
    double speed = getDataFromInput(argv[4],0.1,1);
    // File out
    std::string fileOutName(argv[3]);
    std::string fileOutName_JP(fileOutName);
    std::string fileOutName_OTEE(fileOutName);
    std::ofstream fileOut_JP(fileOutName_JP.append("_JP.csv"),std::ios::out);
    std::ofstream fileOut_OTEE(fileOutName_OTEE.append("_OTEE.csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "Joint poses data are from " << fileInName << ".csv" << std::endl
        << "Data output file: " << fileOutName << "_DATA.csv" << std::endl
        << "Speed: " << speed << std::endl
        << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    try
    {
        // Init. robot
        franka::Robot robot(argv[1]);
        // Set default behavior
        robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
        robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
        robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
        // Robot control
        for (unsigned int j = 0; j < JPIn.size(); j++)
        {
            // Follow the joint poses one by one
            try
            {
                // Robot control
                double timer = 0.0;
                std::array<double,7> init_JP;
                std::array<double,7> goal_JP;
                for (unsigned int i = 0; i < 7; i++)
                {
                    goal_JP[i] = JPIn[j][i];
                }
                robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions{
                    // Reflex impedance controller and joint position motion generator
                    timer += period.toSec();
                    if (timer == 0.0)
                    {
                        init_JP = state.q_d;
                    }
                    franka::JointPositions JPc(init_JP);    // The initial state must be the current state
                    for (unsigned int i = 0; i < 7; i++)
                    {
                        // S-spline interpolation
                        JPc.q[i] = cosInterp(init_JP[i],goal_JP[i],timer*speed);
                    }
                    if (timer*speed >= 1)
                    {
                        franka::MotionFinished(JPc);
                    }
                    return JPc;
                });
                // Read what we wanna track
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
                robot.automaticErrorRecovery();
            }
            std::cout << "Joint pose " << j << " finished!" << std::endl;
        }
    }
    catch(const franka::Exception& e)
    {
        std::cerr << e.what() << '\n';
        fileOut_JP.close();
        fileOut_OTEE.close();
        return -1;
    }
    std::cout << "Motion finished!" << std::endl;
    fileOut_JP.close();
    fileOut_OTEE.close();
    return 0;
}
