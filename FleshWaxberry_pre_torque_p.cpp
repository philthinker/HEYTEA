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
    unsigned int counter = 0;
    try
    {
        // Impedance control param.
        Eigen::Matrix<double,6,6> K; // Stiffness
        Eigen::Matrix<double,6,6> D; // Damping
        K.setZero();
        K.topLeftCorner(3,3) << 20 * Eigen::MatrixXd::Identity(3,3);   // x y z stiffness
        D.setZero();
        D.topLeftCorner(3,3) << 10 * Eigen::MatrixXd::Identity(3,3);   // x y z damping
        //std::array<double,7> kGains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
        //std::array<double,7> dGains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};
        // Init. the intermediate vaiable here
        unsigned int N = carte_pose.size();
        std::cout << N << " data are read." << std::endl;
        //unsigned int counter = 0;
        double time = 0.0;
        std::array<double,16> goal_pose_array;    // carte_pose
        std::array<double,16> init_pose_array;
        unsigned int fps_counter = 0;
        // Start robot controller
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                time += period.toSec();
                if (time == 0.0)
                {
                    //Eigen::Map<const Eigen::Matrix<double,4,4>> goal_pose(state.O_T_EE.data());
                    goal_pose_array = state.O_T_EE;
                }else if(fps_counter >= 10)
                {
                    //goal_pose_array = vectorP2arrayCarte(carte_pose[counter],init_pose_array);
                    //goal_pose_array = vector2array16(carte_pose[counter]);
                    goal_pose_array = state.O_T_EE;
                    for (unsigned int i = 0; i < 16; i++)
                    {
                        goal_pose_array[i] = carte_pose[counter][i];
                    }
                    counter++;
                    fps_counter = 0;
                }
                fps_counter++;
                // Torque controller
                // Dynamics compensation
                std::array<double,7> coriolis_array = model.coriolis(state);
                std::array<double,42> jacobin_array = model.zeroJacobian(franka::Frame::kEndEffector,state);
                // Convert array to Eigen
                Eigen::Map<const Eigen::Matrix<double,6,7>> jacobin(jacobin_array.data());
                Eigen::Map<const Eigen::Matrix<double,7,1>> coriolis(coriolis_array.data());
                Eigen::Map<const Eigen::Matrix<double,4,4>> curr_pose(state.O_T_EE.data());
                Eigen::Map<const Eigen::Matrix<double,7,1>> curr_qd(state.dq.data());
                Eigen::Map<Eigen::Matrix<double,4,4>> goal_pose(goal_pose_array.data());
                // Impedance control signal
                Eigen::VectorXd tau_act(7);
                Eigen::Matrix<double,6,1> error_p;
                error_p.setZero();
                // Position error
                error_p.head(3) << goal_pose.topRightCorner(3,1) - curr_pose.topRightCorner(3,1);
                // Orientation error
                // Control law
                tau_act << jacobin.transpose() * (K * error_p - D * (jacobin * curr_qd)) + coriolis;
                std::array<double,7> tau_act_array{};
                Eigen::VectorXd::Map(&tau_act_array[0],7) = tau_act;
                franka::Torques tau_c(tau_act_array);
                if (counter > N)
                {
                    return franka::MotionFinished(tau_c);
                }
                return tau_c;
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