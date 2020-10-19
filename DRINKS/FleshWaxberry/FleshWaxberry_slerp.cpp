//FleshWaxberry_slerp
//  Move to the given Orientation
//
//  Haopeng Hu
//  2020.07.29
//
// argv[0] <fci-ip> fileInName

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <cmath>

#include <Eigen/Dense>

#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>

#include "LACTIC/LACTIC.h"
#include "MILK/MILK.h"

int main(int argc, char** argv){
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] <<" <fci-ip> fileIn" << std::endl;
        return -1;
    }
    // Destination
    std::array<double,4> quat_goal_array;
    std::string fileInName(argv[2]);
    std::vector<std::vector<double>> dataIn = readCSV(fileInName);
    for (unsigned int i = 0; i < 4; i++)
    {
        quat_goal_array[i] = dataIn[0][i];
    }
    // To Eigen data (orientation)
    Eigen::Quaterniond quat_goal;
    quat_goal.w() = quat_goal_array[0];
    quat_goal.x() = quat_goal_array[1];
    quat_goal.y() = quat_goal_array[2];
    quat_goal.z() = quat_goal_array[3];
    // File out
    //std::string fileOutName(argv[3]);
    //std::ofstream fileOut(fileOutName.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "The robot will move to orientation:" << std::endl;
    for (unsigned i = 0; i < 4; i++)
    {
        std::cout << quat_goal_array[i] << ',';
    }
    //std::cout << "Data will be stored in file: " << fileOutName << std::endl;
    std::cout << std::endl << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    franka::Model model = robot.loadModel();
    //robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    //robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}});
    std::cout << "Robot is ready to move" << std::endl;
    // Controller param.
    double stiffness = 30000;
    double damping = 200;
    Eigen::MatrixXd K(6,6); K.setZero();
    Eigen::MatrixXd D(6,6); D.setZero();
    K.topLeftCorner(3,3) << stiffness * Eigen::MatrixXd::Identity(3,3);
    K.bottomRightCorner(3,3) << std::sqrt(stiffness) * Eigen::MatrixXd::Identity(3,3);
    D.topLeftCorner(3,3) << damping * Eigen::MatrixXd::Identity(3,3);
    D.bottomRightCorner(3,3) << std::sqrt(damping) * Eigen::MatrixXd::Identity(3,3);
    try{
        // Init. state
        franka::RobotState init_state = robot.readOnce();
        Eigen::Affine3d trans_init(Eigen::Matrix4d::Map(init_state.O_T_EE.data()));
        Eigen::Quaterniond quat_init(trans_init.linear());
        Eigen::Quaterniond quat_d(quat_init);
        std::array<double,16> init_pose;
        double timer = 0.0;
        double scalar = 1;
        unsigned int counter = 0;
        unsigned int fps_counter = 0;
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                // Impedance controller
                counter++;
                // Read the state
                Eigen::Affine3d trans_t(Eigen::Matrix4d::Map(state.O_T_EE.data()));
                Eigen::Quaterniond quat_t(trans_t.linear());
                Eigen::Map<const Eigen::Matrix<double,7,1>> dq_t(state.dq.data());
                std::array<double,7> coriolis_array = model.coriolis(state);
                std::array<double,42> jacobin_array = model.zeroJacobian(franka::Frame::kEndEffector,state);
                std::array<double,7> gravity_array = model.gravity(state);
                Eigen::Map<const Eigen::Matrix<double,6,7>> jacobin(jacobin_array.data());      // Spatial Jacobian
                Eigen::Map<const Eigen::Matrix<double,7,1>> coriolis(coriolis_array.data());    // Coriolis
                Eigen::Map<const Eigen::Matrix<double,7,1>> gravity(gravity_array.data());      // Gravity
                // SLERP
                if(counter >= 10){
                    counter = 0;
                    timer += 0.002;
                    if(timer > 1.0){
                        timer = 1.0;
                    }
                    quat_d = quat_init.slerp(timer,quat_goal);
                }
                // Compute the error
                Eigen::VectorXd error(6); error.setZero();
                if(quat_d.coeffs().dot(quat_t.coeffs()) < 0.0){ // Double cover issue
                    quat_t.coeffs() = -quat_t.coeffs();
                }
                Eigen::Quaterniond error_quat(quat_t.inverse()*quat_d);
                error.tail(3) << error_quat.x(),error_quat.y(),error_quat.z();
                error.tail(3) << -trans_t.linear() * error.tail(3);
                // Control law
                Eigen::VectorXd tau_c(7); tau_c.setZero();
                tau_c << jacobin.transpose() * (K * error - D * (jacobin * dq_t)) + coriolis;
                std::array<double,7> tau_c_array;
                Eigen::VectorXd::Map(&tau_c_array[0],7) = tau_c;
                franka::Torques tau_c_franka(tau_c_array);
                if(timer >= 1.0){
                    double error_final = error.norm();
                    if(error_final <= 0.01){
                        return franka::MotionFinished(tau_c_franka);
                    }
                }
                return tau_c_franka;
            });
        franka::RobotState final_state = robot.readOnce();
        Eigen::Affine3d trans_final(Eigen::Matrix4d::Map(final_state.O_T_EE.data()));
        Eigen::Quaterniond quat_final(trans_final.linear());
        if(quat_final.w() < 0.0){
            quat_final.coeffs() = -quat_final.coeffs();
        }
        std::cout << "Final quaternion: " << std::endl << quat_final.coeffs();
        std::cout << std::endl;
    }
    catch(const franka::Exception& e){
        std::cerr << e.what() <<'\n';
        //fileOut.close();
        return -1;
    }
    //fileOut.close();
    return 0;
}
