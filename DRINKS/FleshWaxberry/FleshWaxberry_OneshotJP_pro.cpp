//FleshWaxberry_OneshotJP_pro
//  Move to the given  joint position while read something
//
//  Haopeng Hu
//  2020/07/07
//
// argv[0] <fci-ip> fileInName fileOutName fps

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>

#include <franka/robot.h>
#include <franka/exception.h>

#include "LACTIC/LACTIC.h"
#include "MILK/MILK.h"

int main(int argc, char** argv){
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] <<" <fci-ip> fileInName fileOutName fps" << std::endl;
        return -1;
    }
    // Destination
    std::array<double,7> q_goal;
        std::string fileInName(argv[2]);
        std::vector<std::vector<double>> dataIn = readCSV(fileInName);
        for (unsigned int i = 0; i < 7; i++)
        {
            q_goal[i] = dataIn[0][i];
        }
    // fps [1,100]
    std::string FPS(argv[4]);
    unsigned int fps = std::floor(getDataFromInput(argv[4],1,100));
    // File out
    std::string fileOutName(argv[3]);
    std::ofstream fileOut(fileOutName.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << "The robot will move to joint pose:" << std::endl;
    for (unsigned i = 0; i < 7; i++)
    {
        std::cout << q_goal[i] << ',';
    }
    std::cout << "Data will be stored in file: " << fileOutName << std::endl;
    std::cout << std::endl << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    // Init. robot
    franka::Robot robot(argv[1]);
    robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    // Joint pose motion
    try{
    double timer = 0.0;
    std::array<double,7> q_init;
    unsigned int counter = 1;
    robot.control([&timer,&q_init,q_goal,fps,&counter,&fileOut](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions{
        // S-spline interpolation
        timer += period.toSec();
        if (timer == 0.0)
        {
            q_init = state.q_d;
        }
        franka::JointPositions q_c(q_init);
        for (unsigned int i = 0; i < 7; i++)
        {
            q_c.q[i] = cosInterp(q_init[i],q_goal[i],timer*0.4);
        }
        // Read data to file out
        if (counter >= 1000/fps)
        {
            for (unsigned int i = 0; i < 7; i++)
            {
                fileOut << state.q[i] << ',';
            }
            fileOut << std::endl;
            counter = 1;
        }else
        {
            counter++;
        }
        if (timer*0.4 >= 1)
        {
            return franka::MotionFinished(q_c);
        }
        return q_c;
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