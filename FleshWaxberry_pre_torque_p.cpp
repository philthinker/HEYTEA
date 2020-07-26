//FleshWaxberry_pre_torque_p
//  Aggressive and fast impedance controller for pre-assembly phase
//  We assume that you HAVE moved the robot to the inital pose.
//
//  Haopeng Hu
//  2020.07.26
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
    std::vector<std::vector<double>> carte_pose = readCSV(carte_pose_file);
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
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    franka::Model model = robot.loadModel();
    std::cout << "Robot is ready to move!" << std::endl;
    try
    {
        // Impedance control param.
        Eigen::MatrixXd K(6,6); // Stiffness
        Eigen::MatrixXd D(6,6); // Damping
        K.setZero();
        K.topLeftCorner(3,3) << 150 * Eigen::MatrixXd::Identity(3,3);   // x y z stiffness
        D.setZero();
        D.topLeftCorner(3,3) << std::sqrt(150) * Eigen::MatrixXd::Identity(3,3);   // x y z damping
        //std::array<double,7> kGains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
        //std::array<double,7> dGains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
        // Init. the intermediate vaiable here
        unsigned int N = carte_pose.size();
        std::cout << N << " data are read." << std::endl;
        unsigned int counter = 0;
        double time = 0.0;
        std::array<double,16> goal_pose_array;    // carte_pose
        // Start robot controller
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                time += period.toSec();
                if (time == 0.0)
                {
                    goal_pose_array = state.O_T_EE;
                }else
                {
                    goal_pose_array = vector2array16(carte_pose[counter]);
                    counter++;
                }
                // Torque controller
                // Dynamics compensation
                std::array<double,7> coriolis_array = model.coriolis(state);
                std::array<double,42> jacobin_array = model.zeroJacobian(franka::Frame::kEndEffector,state);
                // Convert array to Eigen
                Eigen::Map<const Eigen::Matrix<double,6,7>> jacobin(jacobin_array.data());
                Eigen::Map<const Eigen::Matrix<double,7,1>> coriolis(coriolis_array.data());
                Eigen::Map<const Eigen::Matrix<double,7,1>> curr_pose(state.O_T_EE.data());
                Eigen::Map<const Eigen::Matrix<double,7,1>> curr_qd(state.dq.data());
                Eigen::Map<Eigen::Matrix<double,4,4>> goal_pose(goal_pose_array.data());
                // Impedance control
                Eigen::VectorXd tau_act(7);
                Eigen::VectorXd error_p(7);
                error_p.setZero();
                error_p.head(3) << goal_pose - curr_pose;
                tau_act << jacobin.transpose() * (K * error_p - D * (jacobin * curr_qd)) + coriolis;
                std::array<double,7> tau_act_array;
                Eigen::VectorXd::Map(&tau_act_array[0],7) = tau_act;
                franka::Torques tau_c(tau_act_array);
                if (counter >= N)
                {
                    franka::MotionFinished(tau_c);
                }
                return tau_c;
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