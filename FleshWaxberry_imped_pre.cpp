//FleshWaxberry_imped_pre
//  Aggressive and fast impedance controller for pre-assembly phase
//  We assume that you HAVE moved the robot to the inital pose.
//
//  Haopeng Hu
//  2020.07.14
//  All rights reserved
//
//  Usage: argv[0] <fci-ip> fileIn1 fileIn2 fileOut

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
    if(argc<5){
        std::cerr << "Usage: " << argv[0] << " <fci-ip> fileIn1 fileIn2 fileOut" << std::endl;
        return -1;
    }
    // Read what we need
    std::string carte_pose_file(argv[2]);
    std::string pose_covar_file(argv[3]);
    std::vector<std::vector<double>> carte_pose = readCSV(carte_pose_file);
    std::vector<std::vector<double>> pose_covar = readCSV(pose_covar_file);
    // Prepare the output
    std::string pose_out_file(argv[4]);
    std::ofstream pose_out(pose_out_file.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "Log data will be stored in file: " << pose_out_file << std::endl
        << "Press Enter to continue. Good Luck!" << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    // Set default param
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    franka::Model model = robot.loadModel();
    try
    {
        // Control param.
        std::array<double,7> kGains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
        std::array<double,7> dGains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
        // Init. the intermediate vaiable here
        unsigned int N = carte_pose.size();
        unsigned int counter = 1;
        double time = 0.0;
        std::array<double,16> goal_pose;
        // Start robot controller
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                // Torque controller
                // Dynamics compensation
                std::array<double,7> coriolis = model.coriolis(state);
                std::array<double,42> jacobin = model.zeroJacobian(franka::Frame::kEndEffector,state);
                // Admittance control
                std::array<double,7> tau_act;
                for (unsigned int i = 0; i < 7; i++)
                {
                    tau_act[i] = kGains[i]*(state.q_d[i] - state.q[i]) - dGains[i]*state.dq[i] + coriolis[i];
                }
                return tau_act;
            },
            [&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose{
                // Cartesian motion generator
                time += period.toSec();
                if (time == 0.0)
                {
                    goal_pose = state.O_T_EE_d;
                }
                else
                {
                    goal_pose = vector2array16(carte_pose[counter]);
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
        return -1;
    }
    pose_out.close();
    return 0;
}