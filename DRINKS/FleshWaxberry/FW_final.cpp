//FW_final
//  Aggressive and fast impedance controller for pre-assembly phase
//  We assume that you HAVE moved the robot to the inital pose.
//  There is a joint space compensation after the torque control loop
//
//  Haopeng Hu
//  2020.09.12
//  All rights reserved
//
//  Usage: argv[0] <fci-ip> fileIn1 fileIn2 fileIn3 fileIn4 fileOut

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
    if(argc<8){
        std::cerr << "Usage: " << argv[0] << " <fci-ip> carte_pose carte_quat K JP_pre JP_cis fileOutName" << std::endl;
        // carte_pose: N x 3
        // carte_quat: N x 4
        // K: N x 6 >= 0
        //  JP_pre: 2 x 7: q_init, q_pre
        //  JP_cis: 2 x 7: q_cis, q_goal
        return -1;
    }
    // Read what we need
    std::string carte_pose_file(argv[2]);
    std::string carte_quat_file(argv[3]);
    std::string K_file(argv[4]);
    std::string JP_pre_file(argv[5]);
    std::string JP_cis_file(argv[6]);
    std::vector<std::vector<double>> carte_pose = readCSV(carte_pose_file); // N x 3
    std::vector<std::vector<double>> carte_quat = readCSV(carte_quat_file); // N x 4
    std::vector<std::vector<double>> Ks = readCSV(K_file);                  // N x 6
    std::vector<std::vector<double>> JP_pre = readCSV(JP_pre_file);         // 2 x 7
    std::vector<std::vector<double>> JP_cis = readCSV(JP_cis_file);         // 2 x 7
    unsigned int N = carte_pose.size();
    // Prepare the output
    std::string log_out_file(argv[7]);
    std::ofstream log_out(log_out_file.append(".csv"),std::ios::out);
    // Stiffness and damping
    double stiffness = 30;
    double damping = 20;
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << N << " data are read." << std::endl
        << "Log data will be stored in file: " << log_out_file << std::endl
        << "Press Enter to continue. Good Luck!" << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    franka::Model model = robot.loadModel();
    std::cout << "Robot is ready to move!" << std::endl;
    // Set default param for pre-assembly phase
    std::cout << "Pre-assembly phase" << std::endl;
    // Note that it is assumed no collision occurrs during this phase
    robot.setCollisionBehavior(
            {{15.0, 15.0, 12.0, 10.0, 8.0, 8.0, 8.0}}, {{15.0, 15.0, 12.0, 10.0, 8.0, 8.0, 8.0}},
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}});
    unsigned int counter = 0;   // file line counter
    try
    {
        // Impedance control param. initialization
        Eigen::Matrix<double,6,6> K; // Stiffness
        Eigen::Matrix<double,6,6> D; // Damping
        K = vectorK2Matrix6(Ks[counter]);
        D.setZero();
        D.topLeftCorner(3,3) << damping * Eigen::MatrixXd::Identity(3,3);   // x y z damping
        D.bottomRightCorner(3,3) << std::sqrt(damping) * Eigen::MatrixXd::Identity(3,3);    // quat damping
        // Init. the goals with current state for safety
        franka::RobotState curr_state = robot.readOnce();
        Eigen::Affine3d goal_trans(Eigen::Matrix4d::Map(curr_state.O_T_EE.data()));
        Eigen::Vector3d goal_posi(goal_trans.translation());
        Eigen::Quaterniond goal_quat(goal_trans.linear());
        // Init. the intermediate vaiable here
        //unsigned int counter = 0;
        unsigned int fps_counter = 0;
        unsigned int log_counter = 0;
        // Start robot controller
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                // We asume that you have move the robot to the 1st given pose
                // Pure torque controller
                // Dynamics compensation
                std::array<double,7> coriolis_array = model.coriolis(state);
                std::array<double,42> jacobin_array = model.zeroJacobian(franka::Frame::kEndEffector,state);
                //std::array<double,7> gravity_array = model.gravity(state);
                // Convert array to Eigen
                Eigen::Map<const Eigen::Matrix<double,6,7>> jacobin(jacobin_array.data());      // Spatial Jacobian
                Eigen::Map<const Eigen::Matrix<double,7,1>> coriolis(coriolis_array.data());    // Coriolis
                //Eigen::Map<const Eigen::Matrix<double,7,1>> gravity(gravity_array.data());
                Eigen::Affine3d curr_trans(Eigen::Matrix4d::Map(state.O_T_EE.data()));          // Current Carte pose
                Eigen::Vector3d curr_posi(curr_trans.translation());                            // Current Carte position
                Eigen::Quaterniond curr_quat(curr_trans.linear());                              // Current quaternion
                Eigen::Map<const Eigen::Matrix<double,7,1>> curr_dq(state.dq.data());           // Current joint vel.
                // The goals
                if(fps_counter >= 5)
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
                    K = vectorK2Matrix6(Ks[counter]);
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
                error_pose.tail(3) << quatSubtraction(goal_quat,curr_quat);
                // Control law
                // Impedance control signal
                Eigen::VectorXd tau_act(7);
                //alpha = 1.0 + alp_counter/N;
                tau_act << jacobin.transpose() * (K * error_pose - D * (jacobin * curr_dq) ) + coriolis;
                std::array<double,7> tau_act_array{};
                Eigen::VectorXd::Map(&tau_act_array[0],7) = tau_act;
                franka::Torques tau_c(tau_act_array);
                // Write the tau_act_array for test
                if(log_counter >= 100)
                {
                    log_counter = 0;
                }
                log_counter++;
                // Terminal condition
                if (counter > N-1)
                {
                    counter = N-1;
                        // Final control loop is done
                        tau_c.tau_J = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
                        return franka::MotionFinished(tau_c);
                }
                return tau_c;
            }
        );
        // Joint motion compensation for pre-assembly phase
        robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
        robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
        double time = 0.0;
        std::array<double,7> goal_q = vector2array7(JP_pre[1]);
        std::array<double,7> init_q;
        std::array<double,7> curr_q;
        std::cout << "Joint position compensation" << std::endl;
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions{
            // Joint motion with given goal joint position
            time += period.toSec();
            if(time == 0.0)
            {
                // The init goal must be the current q.
                init_q = state.q;
                curr_q = state.q_d;
            }else
            {
                // S-spline interpolation
                for(unsigned int i = 0; i < 7; i++)
                {
                    curr_q[i] = cosInterp(init_q[i],goal_q[i],time*0.2);
                }
            }
            franka::JointPositions q_c = curr_q;
            if(time*0.2 >= 1)
            {
                return franka::MotionFinished(q_c);
            }
            return q_c;
        });
        // Cis-assembly phase
        // Set default param.
        std::cout << "Cis-assembly phase" << std::endl;
        robot.setCollisionBehavior(
            {{15.0, 15.0, 12.0, 10.0, 8.0, 8.0, 8.0}}, {{15.0, 15.0, 12.0, 10.0, 8.0, 8.0, 8.0}},
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
            {{15.0, 15.0, 20.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 20.0, 15.0, 15.0, 15.0}});
        stiffness = 2000;
        damping = 100;
        counter = 0;   // step counter now
        double total_length = 0.020;    // 20mm
        double d_length = 0.001;        // 2mm
        N = std::floor(total_length/d_length);
        // Impedance control param. initialization
        K.setZero();
        K.topLeftCorner(3,3) << stiffness * Eigen::MatrixXd::Identity(3,3);   // x y z stiffness
        K.topLeftCorner(2,2) << std::sqrt(stiffness/2) * Eigen::MatrixXd::Identity(2,2);  // x y stiffness
        K.bottomRightCorner(3,3) << std::sqrt(stiffness) * Eigen::MatrixXd::Identity(3,3); // quat stiffness
        D.setZero();
        D.topLeftCorner(3,3) << damping * Eigen::MatrixXd::Identity(3,3);   // x y z damping
        D.bottomRightCorner(3,3) << std::sqrt(damping) * Eigen::MatrixXd::Identity(3,3);    // quat damping
        // Init. the goals with current state for safety
        curr_state = robot.readOnce();
        goal_trans = Eigen::Matrix4d::Map(curr_state.O_T_EE.data());
        goal_posi = goal_trans.translation();
        goal_quat = goal_trans.linear();
        // Init. the intermediate vaiable here
        fps_counter = 0;
        log_counter = 0;
        // Start robot controller
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                // We asume that you have move the robot to the 1st given pose
                // Pure torque controller
                // Dynamics compensation
                std::array<double,7> coriolis_array = model.coriolis(state);
                std::array<double,42> jacobin_array = model.zeroJacobian(franka::Frame::kEndEffector,state);
                //std::array<double,7> gravity_array = model.gravity(state);
                // Convert array to Eigen
                Eigen::Map<const Eigen::Matrix<double,6,7>> jacobin(jacobin_array.data());      // Spatial Jacobian
                Eigen::Map<const Eigen::Matrix<double,7,1>> coriolis(coriolis_array.data());    // Coriolis
                //Eigen::Map<const Eigen::Matrix<double,7,1>> gravity(gravity_array.data());
                Eigen::Affine3d curr_trans(Eigen::Matrix4d::Map(state.O_T_EE.data()));          // Current Carte pose
                Eigen::Vector3d curr_posi(curr_trans.translation());                            // Current Carte position
                Eigen::Quaterniond curr_quat(curr_trans.linear());                              // Current quaternion
                Eigen::Map<const Eigen::Matrix<double,7,1>> curr_dq(state.dq.data());           // Current joint vel.
                // The goals
                if(fps_counter >= 100)
                {
                    goal_posi[2] -= d_length; // Move downward
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
                error_pose.tail(3) << quatSubtraction(goal_quat,curr_quat);
                // Control law
                // Impedance control signal
                Eigen::VectorXd tau_act(7);
                //alpha = 1.0 + alp_counter/N;
                tau_act << jacobin.transpose() * (K * error_pose - D * (jacobin * curr_dq) ) + coriolis;
                std::array<double,7> tau_act_array{};
                Eigen::VectorXd::Map(&tau_act_array[0],7) = tau_act;
                franka::Torques tau_c(tau_act_array);
                // Write the tau_act_array for test
                if(log_counter >= 100)
                {
                    log_counter = 0;
                }
                log_counter++;
                // Terminal condition
                if (counter > N-1)
                {
                    counter = N-1;
                        // Final control loop is done
                        tau_c.tau_J = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
                        return franka::MotionFinished(tau_c);
                }
                return tau_c;
            }
        );
        // Joint motion compensation for cis-assembly phase

        robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
        robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
        robot.setCollisionBehavior(
            {{15.0, 15.0, 12.0, 10.0, 8.0, 8.0, 8.0}}, {{15.0, 15.0, 12.0, 10.0, 8.0, 8.0, 8.0}},
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
            {{15.0, 15.0, 40.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 40.0, 15.0, 15.0, 15.0}});
        time = 0.0;
        goal_q = vector2array7(JP_cis[1]);
        std::cout << "Joint position compensation" << std::endl;
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions{
            // Joint motion with given goal joint position
            time += period.toSec();
            if(time == 0.0)
            {
                // The init goal must be the current q.
                init_q = state.q;
                curr_q = state.q_d;
            }else
            {
                // S-spline interpolation
                for(unsigned int i = 0; i < 7; i++)
                {
                    curr_q[i] = cosInterp(init_q[i],goal_q[i],time*0.2);
                }
            }
            franka::JointPositions q_c = curr_q;
            if(time*0.2 >= 1)
            {
                return franka::MotionFinished(q_c);
            }
            return q_c;
        });
    }
    catch(const franka::Exception& e)
    {
        std::cerr << e.what() << '\n';
        log_out.close();
        std::cout << "counter: " << counter << std::endl;
        return -1;
    }
    std::cout << "Finished assembly, counter = " << counter << std::endl;
    log_out.close();
    return 0;
}