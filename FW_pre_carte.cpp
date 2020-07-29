//FW_pre_carte
//  Direct Cartesian motion generator for pre phase
//  We assume that you HAVE moved the robot to the inital pose.
//
//  Haopeng Hu
//  2020.07.28
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
    std::vector<std::vector<double>> carte_pose = readCSV(carte_pose_file); // N x 16
    unsigned int N = carte_pose.size();
    // Prepare the output
    std::string pose_out_file(argv[3]);
    std::ofstream pose_out(pose_out_file.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << N << " data are read." << std::endl
        << "Log data will be stored in file: " << pose_out_file << std::endl
        << "Press Enter to continue. Good Luck!" << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    // Set default param
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}});
    franka::Model model = robot.loadModel();
    std::cout << "Robot is ready to move!" << std::endl;
    unsigned int counter = 0;   // file line counter
    try
    {
        // Init. the intermediate vaiable here
        //unsigned int counter = 0;
        unsigned int fps_counter = 0;
        unsigned int log_counter = 0;
        double time = 0.0;
        robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
        robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
        std::array<double,16> goal_pose;
        std::array<double,16> init_pose;
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose{
                // Cartesian motion plan
                // We assume that you have moved the robot to the initial pose
                time += period.toSec();
                if (time == 0.0)
                {
                    init_pose = state.O_T_EE;
                    goal_pose = state.O_T_EE;
                }
                franka::CartesianPose carte_cmd(init_pose);
                if (fps_counter >= 1)
                {
                   for (unsigned int i = 12; i < 15; i++)
                   {
                       carte_cmd.O_T_EE[i] = carte_pose[counter][i];
                   }
                    counter++;
                    fps_counter = 0;
                }
                fps_counter++;
                // Log data
                if (log_counter >= 10)
                {
                    for (unsigned int i = 0; i < 16; i++)
                    {
                        pose_out << state.O_T_EE[i] << ',';
                    }
                    pose_out << std::endl;
                    log_counter = 0;
                }
                log_counter++;
                // Terminal condition
                if (counter > N-1)
                {
                    return franka::MotionFinished(carte_cmd);
                }
                return carte_cmd;
            }
        );
        std::cout << "counter: " << counter << std::endl;
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