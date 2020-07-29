//FleshWaxberry_slerp
//  Move to the given Orientation
//
//  Haopeng Hu
//  2020/07/28
//
// argv[0] <fci-ip> fileInName fileOutName

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
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] <<" <fci-ip> fileIn fileOut" << std::endl;
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
    std::string fileOutName(argv[3]);
    std::ofstream fileOut(fileOutName.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "The robot will move to orientation:" << std::endl;
    for (unsigned i = 0; i < 4; i++)
    {
        std::cout << quat_goal_array[i] << ',';
    }
    std::cout << "Data will be stored in file: " << fileOutName << std::endl;
    std::cout << std::endl << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    franka::Model model = robot.loadModel();
    robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}});
    std::cout << "Robot is ready to move" << std::endl;
    // Controller param.
    double stiffness = 300;
    double damping = 200;
    Eigen::MatrixXd K(6,6); K.setZero();
    Eigen::MatrixXd D(6,6); D.setZero();
    K.topLeftCorner(3,3) << stiffness * Eigen::MatrixXd::Identity(3,3);
    K.bottomRightCorner(3,3) << std::sqrt(stiffness) * Eigen::MatrixXd::Identity(3,3);
    D.topLeftCorner(3,3) << damping * Eigen::MatrixXd::Identity(3,3);
    D.bottomRightCorner(3,3) << std::sqrt(damping) * Eigen::MatrixXd::Identity(3,3);
    try{
        // Init. state
        Eigen::Quaterniond quat_init;
        std::array<double,16> init_pose;
        double timer = 0.0;
        double scalar = 0.1;
        unsigned int fps_counter = 0;
        robot.control(
            [&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
                // Impedance controller
                // Read the state
                Eigen::Affine3d trans_t(Eigen::Matrix4d::Map(state.O_T_EE.data()));
                Eigen::Affine3d trans_d(Eigen::Matrix4d::Map(state.O_T_EE_d.data()));
                Eigen::Quaterniond quat_t(trans_t.linear());
                Eigen::Quaterniond quat_d(trans_d.linear());
                Eigen::Map<const Eigen::Matrix<double,7,1>> dq_t(state.dq.data());
                std::array<double,7> coriolis_array = model.coriolis(state);
                std::array<double,42> jacobin_array = model.zeroJacobian(franka::Frame::kEndEffector,state);
                std::array<double,7> gravity_array = model.gravity(state);
                Eigen::Map<const Eigen::Matrix<double,6,7>> jacobin(jacobin_array.data());      // Spatial Jacobian
                Eigen::Map<const Eigen::Matrix<double,7,1>> coriolis(coriolis_array.data());    // Coriolis
                Eigen::Map<const Eigen::Matrix<double,7,1>> gravity(gravity_array.data());      // Gravity
                // Compute the error
                Eigen::VectorXd error(6); error.setZero();
                if(quat_d.coeffs().dot(quat_t.coeffs()) < 0.0){ // Double cover issue
                    quat_t.coeffs() = -quat_t.coeffs();
                }
                Eigen::Quaterniond error_quat(quat_d.inverse()*quat_t);
                error.tail(3) << error_quat.x(),error_quat.y(),error_quat.z();
                error.tail(3) << -trans_t.linear() * error.tail(3);
                // Control law
                Eigen::VectorXd tau_c(7); tau_c.setZero();
                tau_c << jacobin.transpose() * (K * error - D * (jacobin * dq_t)) + coriolis;
                std::array<double,7> tau_c_array;
                Eigen::VectorXd::Map(&tau_c_array[0],7) = tau_c;
                franka::Torques tau_c_franka(tau_c_array);
                return tau_c_franka;
            },
            [&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose{
            // SLERP for orientation
            timer += period.toSec();
            if (timer == 0.0)
            {
                init_pose = state.O_T_EE;
                Eigen::Affine3d init_trans(Eigen::Matrix4d::Map(init_pose.data()));
                quat_init = init_trans.linear();
            }
            Eigen::Quaterniond quat_cmd(quat_init.slerp(timer * scalar, quat_goal));
            Eigen::Matrix3d rotm_cmd(quat_cmd.toRotationMatrix());
            std::array<double,16> pose_cmd_array = Matrix3d2array16(rotm_cmd);
            pose_cmd_array[12] = init_pose[12];
            pose_cmd_array[13] = init_pose[13];
            pose_cmd_array[14] = init_pose[14];
            franka::CartesianPose pose_cmd_franka(pose_cmd_array);
            if(fps_counter >= 10){
                fps_counter = 0;
                for (unsigned int i = 0; i < 16; i++)
                {
                    fileOut << pose_cmd_franka.O_T_EE[i] << ',';
                }
                fileOut << std::endl;
            }
            fps_counter++;
            franka::CartesianPose tmp_pose_cmd(init_pose);
            if(timer*scalar >= 1){
                return franka::MotionFinished(pose_cmd_franka);
                //return franka::MotionFinished(tmp_pose_cmd);
            }
            return pose_cmd_franka;
            //return tmp_pose_cmd;
        });
    }
    catch(const franka::Exception& e){
        std::cerr << e.what() <<'\n';
        fileOut.close();
        return -1;
    }
    fileOut.close();
    return 0;
}
