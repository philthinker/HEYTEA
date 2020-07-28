//FW_pre_torque
//  Aggressive and fast impedance controller for pre-assembly phase
//  We assume that you HAVE moved the robot to the inital pose.
//
//  Haopeng Hu
//  2020.07.27
//  All rights reserved
//
//  Usage: argv[0] <fci-ip> fileIn1 fileIn2 fileOut K D

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
    if(argc<7){
        std::cerr << "Usage: " << argv[0] << " <fci-ip> fileIn1 fileIn2 fileOut K D" << std::endl;
        return -1;
    }
    // Read what we need
    std::string carte_pose_file(argv[2]);
    std::string carte_quat_file(argv[3]);
    std::vector<std::vector<double>> carte_pose = readCSV(carte_pose_file); // N x 3
    std::vector<std::vector<double>> carte_quat = readCSV(carte_quat_file); // N x 4
    // Prepare the output
    std::string pose_out_file(argv[4]);
    std::ofstream pose_out(pose_out_file.append(".csv"),std::ios::out);
    // Stiffness and damping
    double stiffness = getDataFromInput(argv[5],10.0,500.0);
    double damping = getDataFromInput(argv[6],10.0,500.0);
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
    unsigned int counter = 0;   // file line counter
    try
    {
        // Impedance control param.
        Eigen::Matrix<double,6,6> K; // Stiffness
        Eigen::Matrix<double,6,6> D; // Damping
        K.setZero();
        K.topLeftCorner(3,3) << stiffness * Eigen::MatrixXd::Identity(3,3);   // x y z stiffness
        K.bottomRightCorner(3,3) << std::sqrt(stiffness) * Eigen::MatrixXd::Identity(3,3); // quat stiffness
        D.setZero();
        D.topLeftCorner(3,3) << damping * Eigen::MatrixXd::Identity(3,3);   // x y z damping
        D.bottomRightCorner(3,3) << std::sqrt(damping) * Eigen::MatrixXd::Identity(3,3);    // quat damping
        // Init. the intermediate vaiable here
        unsigned int N = carte_pose.size();
        std::cout << N << " data are read." << std::endl;
        //unsigned int counter = 0;
        // Init. the goals with current state for safety
        franka::RobotState curr_state = robot.readOnce();
        Eigen::Affine3d goal_trans(Eigen::Matrix4d::Map(curr_state.O_T_EE.data()));
        Eigen::Vector3d goal_posi(goal_trans.translation());
        Eigen::Quaterniond goal_quat(goal_trans.linear());
        unsigned int fps_counter = 0;
        double time = 0.0;
        // Start robot controller
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                // We asume that you have move the robot to the 1st given pose
                // Pure torque controller
                // Dynamics compensation
                std::array<double,7> coriolis_array = model.coriolis(state);
                std::array<double,42> jacobin_array = model.zeroJacobian(franka::Frame::kEndEffector,state);
                // Convert array to Eigen
                Eigen::Map<const Eigen::Matrix<double,6,7>> jacobin(jacobin_array.data());      // Spatial Jacobian
                Eigen::Map<const Eigen::Matrix<double,7,1>> coriolis(coriolis_array.data());    // Coriolis
                Eigen::Affine3d curr_trans(Eigen::Matrix4d::Map(state.O_T_EE.data()));          // Current Carte pose
                Eigen::Vector3d curr_posi(curr_trans.translation());                            // Current Carte position
                Eigen::Quaterniond curr_quat(curr_trans.linear());                              // Current quaternion
                Eigen::Map<const Eigen::Matrix<double,7,1>> curr_dq(state.dq.data());           // Current joint vel.
                // The goals
                if(fps_counter >= 1+counter/20)
                {
                    for (unsigned int i = 0; i < 3; i++)
                    {
                        goal_posi[i] = carte_pose[counter][i];
                    }
                    // Note that Eigen::Quaternion::coeffs is (x,y,z,w)
                    goal_quat.w() = carte_quat[counter][0];
                    goal_quat.x() = carte_quat[counter][1];
                    goal_quat.y() = carte_quat[counter][2];
                    goal_quat.z() = carte_quat[counter][3];
                    counter++;
                    fps_counter = 0;
                }
                fps_counter++;
                // Error
                Eigen::Matrix<double,6,1> error_pose;
                error_pose.setZero();
                // Position error
                error_pose.head(3) << goal_posi - curr_posi;
                // Orientation error
                // Double cover issue
                if (goal_quat.coeffs().dot(curr_quat.coeffs()) < 0.0)
                {
                    curr_quat.coeffs() = -curr_quat.coeffs();
                }
                // Quaternion difference
                Eigen::Quaterniond error_quat(goal_quat.conjugate()*curr_quat);
                error_pose.tail(3) << error_quat.x(),error_quat.y(),error_quat.z();
                error_pose.tail(3) << -curr_trans.linear() * error_pose.tail(3);
                // Control law
                // Impedance control signal
                Eigen::VectorXd tau_act(7);
                tau_act << jacobin.transpose() * (K * error_pose - D * (jacobin * curr_dq)) + coriolis;
                std::array<double,7> tau_act_array{};
                Eigen::VectorXd::Map(&tau_act_array[0],7) = tau_act;
                franka::Torques tau_c(tau_act_array);
                // Write the tau_act_array for test
                /*
                if(fps_counter >= 1+counter/10-1)
                {
                    for (short int i = 0; i < 7; i++)
                    {
                        pose_out << tau_act_array[i] << ',';
                    }
                    pose_out << std::endl;
                }
                */
                // Terminal condition
                if (counter > N-1)
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