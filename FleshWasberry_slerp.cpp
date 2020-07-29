//FleshWasberry_slerp
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
    robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}}, {{20.0, 20.0, 18.0, 15.0, 10.0, 10.0, 10.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}},
            {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}}, {{15.0, 15.0, 15.0, 15.0, 15.0, 15.0}});
    std::cout << "Robot is ready to move" << std::endl;
    try{
        // Init. state
        franka::RobotState init_state = robot.readOnce();
        Eigen::Affine3d init_trans(Eigen::Matrix4d::Map(init_state.O_T_EE.data()));
        Eigen::Vector3d posi_init(init_trans.translation());
        Eigen::Quaterniond quat_init(init_trans.linear());
        unsigned int fps_counter = 1;
        double timer = 0.0;
        double scalar = 0.01;
        robot.control(
            [&timer, &quat_init, &quat_goal, &fileOut, &posi_init, scalar](const franka::RobotState& state, 
                                                                        franka::Duration period) -> franka::CartesianPose{
            // SLERP for orientation
            timer += period.toSec();
            Eigen::Quaterniond quat_cmd(quat_init.slerp(timer * scalar, quat_goal));
            Eigen::Matrix3d rotm_cmd(quat_cmd.toRotationMatrix());
            std::array<double,16> pose_cmd_array = Matrix3d2array16(rotm_cmd);
            pose_cmd_array[12] = posi_init[0];
            pose_cmd_array[13] = posi_init[1];
            pose_cmd_array[14] = posi_init[2];
            franka::CartesianPose pose_cmd_franka(pose_cmd_array);
            if(timer*scalar >= 1){
                return franka::MotionFinished(pose_cmd_franka);
            }
            return pose_cmd_franka;
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
