//FleshWaxberry_pre_p
//  Franka's impedance control for pre-assembly phase
//  We assume that you HAVE moved the robot to the inital pose.
//
//  Haopeng Hu
//  2020.07.25
//  All rights reserved
//
//  Usage: argv[0] <fci-ip> fileIn1 fileOut

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
    if(argc<4){
        std::cerr << "Usage: " << argv[0] << " <fci-ip> fileIn1 fileOut" << std::endl;
        return -1;
    }
    // Read what we need
    std::string carte_pose_file(argv[2]);
    //std::string pose_covar_file(argv[3]);
    std::vector<std::vector<double>> carte_p = readCSV(carte_pose_file);
    //std::vector<std::vector<double>> pose_covar = readCSV(pose_covar_file);
    // Prepare the output
    std::string pose_out_file(argv[3]);
    std::ofstream pose_out(pose_out_file.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "Log data will be stored in file: " << pose_out_file << std::endl
        << "Press Enter to continue. Good Luck!" << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    // Set default param
    robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    franka::Model model = robot.loadModel();
    std::cout << "Robot is ready to move!" << std::endl;
    unsigned int counter = 1;
    try
    {
        // Init. the intermediate vaiable here
        unsigned int N = carte_p.size();
        std::cout << N << " data are read." << std::endl;
        //unsigned int counter = 1;
        double time = 0.0;
        std::array<double,16> init_pose;
        std::array<double,16> goal_pose;
        // Start robot controller
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose{
                // Cartesian motion generator
                time += period.toSec();
                if (time == 0.0)
                {
                    // The first must be the init. desired pose
                    init_pose = state.O_T_EE_d;
                    goal_pose = state.O_T_EE_d;
                }
                else
                {
                    goal_pose = vectorP2arrayCarte(carte_p[counter],init_pose);
                }
                if (counter == N)
                {
                    return franka::MotionFinished(goal_pose);
                }
                counter++;
                return goal_pose;
            });
    }
    catch(const franka::Exception& e)
    {
        std::cerr << e.what() << '\n';
        pose_out.close();
        std::cout << "counter: " << counter << std::endl;
        return -1;
    }
    pose_out.close();
    return 0;
}