//FleshWaxberry_OneshotCarte
//  Move to the given Cartesian pose while write something
//
//  Haopeng Hu
//  2020/07/28
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
    std::array<double,16> carte_goal;
    std::string fileInName(argv[2]);
    std::vector<std::vector<double>> dataIn = readCSV(fileInName);
    for (unsigned int i = 0; i < 16; i++)
    {
        carte_goal[i] = dataIn[0][i];
    }
    // To Eigen data (orientation)
    Eigen::Affine3d goal_trans(Eigen::Matrix4d::Map(carte_goal.data()));
    Eigen::Quaterniond goal_quat(goal_trans.linear());
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "The robot will move to Cartesian position:" << std::endl;
    for (unsigned i = 12; i < 15; i++)
    {
        std::cout << carte_goal[i] << ',';
    }
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
    try{
        // Init. state
        franka::RobotState init_state = robot.readOnce();
        Eigen::Affine3d init_trans(Eigen::Matrix4d::Map(init_state.O_T_EE.data()));
        Eigen::Quaterniond init_quat(init_trans.linear());
        std::array<double,16> carte_init;
        double timer = 0.0;
    robot.control(
        [&timer,&carte_init,&carte_goal](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose{
        // S-spline interpolation for position
        timer += period.toSec();
        if (timer == 0.0)
        {
            carte_init = state.O_T_EE;
        }
        franka::CartesianPose carte_c(carte_init);
        // S-spline position interpolation
        for (unsigned int i = 12; i < 15; i++)
        {
            carte_c.O_T_EE[i] = cosInterp(carte_init[i],carte_goal[i],timer*0.1);
        }
        if(timer*0.1 >= 1){
            return franka::MotionFinished(carte_c);
        }
        return carte_c;
        });
    timer = 0.0;
    robot.control(
        [&timer,&init_quat,&goal_quat,&carte_init](const franka::RobotState& state,franka::Duration period) -> franka::CartesianPose{
            // SLERP for orientation
            timer += period.toSec();
            if(timer == 0.0){
                carte_init = state.O_T_EE;
            }
            franka::CartesianPose carte_c(carte_init);
            // SLERP
            Eigen::Quaterniond cmd_quat(init_quat.slerp(timer*0.00001,goal_quat));
            std::array<double,16> cmd_trans_array = Matrix3d2array16(cmd_quat.toRotationMatrix());
            for (unsigned int i = 0; i < 12; i++)
            {
                carte_c.O_T_EE[i] = cmd_trans_array[i];
            }
            // Terminal condition
            if(timer*0.00001 >= 1){
                return franka::MotionFinished(carte_c);
            }
            return carte_c;
        }
    );
    }
    catch(const franka::Exception& e){
        std::cerr << e.what() <<'\n';
        return -1;
    }
    return 0;
}