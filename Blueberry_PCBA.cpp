//Blueberry_PCBA
//  Move to the given Cartesian position while read something
//  Only position motion is required. No orientation motion.
//  For better performance, please move the robot to its inital position in advance.
//
//  Haopeng Hu
//  2020.10.18
//
// argv[0] <fci-ip> fileInName fps fileOutName
// fileInName: N x 3

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
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0] <<" <fci-ip> appPosiFile preJPFile fps fileOutName" << std::endl;
        return -1;
    }
    // Destination
    std::array<double,3> posi_goal;
    std::array<double,7> pre_JP;
    std::string appPosiFile(argv[2]);
    std::string preJPFile(argv[3]);
    std::vector<std::vector<double>> dataIn = readCSV(appPosiFile);
    std::vector<std::vector<double>> preJPIn = readCSV(preJPFile);
    unsigned int N = dataIn.size(); // Num. of route points
    for (unsigned int i = 0; i < 3; i++)
    {
        posi_goal[i] = dataIn[0][i];    // Initial position
    }
    for (unsigned int i = 0; i < 7; i++)
    {
        pre_JP[i] = preJPIn[0][i];
    }
    // fps [1,100]
    unsigned int fps = std::floor(getDataFromInput(argv[4],1,100));
    // File out
    std::string fileOutName(argv[5]);
    std::ofstream fileOut(fileOutName.append(".csv"),std::ios::out);
    // Ready
    std::cout << "Keep the user stop at hand!" << std::endl
        << N << " Cartesian positions are read." << std::endl
        << "The robot will move to Cartesian position:" << std::endl;
    for (unsigned i = 0; i < 3; i++)
    {
        std::cout << posi_goal[i] << ',';
    }
    std::cout << std::endl << "Data will be stored in file: " << fileOutName << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    std::cout << "Start moving ..." << std::endl;
    // Init. robot
    franka::Robot robot(argv[1]);
    robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
    try{
        /**
         * @CartesianPose motion generation
         * No orientation motion
         */
        // Intermediate variables
        double timer = 0.0;
        std::array<double,16> carte_init;
        unsigned int fpsCounter = 1;
        double scaler = 1.25;
        for (unsigned int n = 0; n < N; n++)
        {
            // Carteisn pose motion plan
            // Route points are pushed back one by one.
            timer = 0.0;
            fpsCounter = 1;
            for (unsigned int i = 0; i < 3; i++)
            {
                posi_goal[i] = dataIn[n][i];
            }
            robot.control([&timer,&carte_init,posi_goal,fps,&fpsCounter,&fileOut,scaler](const franka::RobotState& state, franka::Duration period)
                -> franka::CartesianPose{
                    // S-Spline Cartesian position interpolation
                    timer += period.toSec();
                    if (timer == 0.0)
                    {
                        carte_init = state.O_T_EE_d;
                    }
                    franka::CartesianPose carte_c(carte_init);
                    for (unsigned int i = 12; i < 15; i++)
                    {
                        carte_c.O_T_EE[i] = cosInterp(carte_init[i],posi_goal[i-12],timer*scaler);
                    }
                    // Log
                    if (fpsCounter >= 1000/fps)
                    {
                        fpsCounter = 1;
                        for (unsigned int i = 0; i < 16; i++)
                        {
                            fileOut << state.O_T_EE[i] << ',';
                        }
                        fileOut << std::endl;
                    }else
                    {
                        fpsCounter++;
                    }
                    // Terminal condition
                    if (timer*scaler >= 1)
                    {
                        return franka::MotionFinished(carte_c);
                    }
                    return carte_c;
                });
            std::cout << "Position: " << n << " finished." << std::endl;
        }
    /**
     * @Pre-Assembly pose compensation
     * JointPose motion generation
     */
    std::cout << "Pre-assembly JP compensation ..." << std::endl;
    timer = 0.0;
    franka::JointPositions JP_init(pre_JP);
    robot.control([&timer,&pre_JP,&JP_init,fps,&fpsCounter,&fileOut](const franka::RobotState& state,franka::Duration period)
    -> franka::JointPositions{
        // S-Spline joint motion generation
        timer += period.toSec();
        if (timer == 0.0)
        {
            JP_init.q = state.q_d;
        }
        franka::JointPositions JP_c(JP_init);
        for (unsigned int i = 0; i < 7; i++)
        {
            JP_c.q[i] = cosInterp(JP_init.q[i],pre_JP[i],timer*0.4);
        }
        // Log
        if (fpsCounter >= 1000/fps)
        {
            fpsCounter = 1;
            for (unsigned int i = 0; i < 16; i++)
            {
                fileOut << state.O_T_EE[i] << ',';
            }
            fileOut << std::endl;
        }else
        {
            fpsCounter++;
        }
        // Terminal condition
        if (timer*0.4 >= 1.0)
        {
            return franka::MotionFinished(JP_c);
        }
        return JP_c;
    });
    std::cout << "Pre-assembly joint motion finished" << std::endl;
    /**
     * @Assembling phase: CartesianPose motion generation
     * No orientation motion
     */
    robot.setCollisionBehavior(
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 60.0, 25.0, 25.0, 25.0}},
            {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 60.0, 25.0, 25.0, 25.0}});
    /**
     * @Assembling phase: CartesianPose motion generation
     * No orientation motion
     */
    double pcbDepth = 0.016;
    timer = 0.0;
    scaler = 0.2;
    robot.control([&timer,&carte_init,pcbDepth,fps,&fpsCounter,&fileOut,scaler](const franka::RobotState& state,franka::Duration period)
    -> franka::CartesianPose{
        // S-Spline Cartesian motion generation
        timer += period.toSec();
        if (timer == 0.0)
        {
            carte_init = state.O_T_EE_d;
        }
        franka::CartesianPose carte_c_ass(carte_init);
        carte_c_ass.O_T_EE[14] = cosInterp(carte_init[14],carte_init[14]-pcbDepth,timer*scaler);
        // Log
        if (fpsCounter >= 1000/fps)
        {
            fpsCounter = 1;
            for (unsigned int i = 0; i < 16; i++)
            {
                fileOut << state.O_T_EE[i] << ',';
            }
            fileOut << std::endl;
        }else
        {
            fpsCounter++;
        }
        // Terminal condition
        if (timer*scaler >= 1.0)
        {
            return franka::MotionFinished(carte_c_ass);
        }
        return carte_c_ass;
    });
    std::cout << "Assembling phase finished" << std::endl;
    }
    catch(const franka::Exception& e){
        std::cerr << e.what() <<'\n';
        fileOut.close();
        return -1;
    }
    fileOut.close();
    return 0;
}