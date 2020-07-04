//FleshWaxberry_zeroGravity
//  Generate zero gravity response
//  
// Haopeng Hu
// 2020.07.03
//
// argv[0] <fci-ip> fileOutName fps

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <franka/robot.h>
#include <franka/exception.h>
#include <franka/model.h>

#include "LACTIC/LACTIC.h"

int main(int argc, char** argv){
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <fci-ip> fileOutName fps" << std::endl;
        return -1;
    }
    // File out
    std::string fileOutName(argv[2]);
    std::string fileOutName_JP(fileOutName);
    std::string fileOutName_OTEE(fileOutName);
    std::string fileOutName_EETK(fileOutName);
    std::ofstream fileOut_JP(fileOutName_JP.append("_JP.csv"),std::ios::out);
    std::ofstream fileOut_OTEE(fileOutName_OTEE.append("_OTEE.csv"),std::ios::out);
    std::ofstream fileOut_EETK(fileOutName_EETK.append("_EETK.csv",std::ios::out));
    // fps [1 1000]
    unsigned int fps = std::floor(getDataFromInput(argv[3],1,1000));
    // Ready
    std::cout << "Zero gravity robot control." << std::endl
        << "Data recorded will be saved in file: " << fileOutName << "_DATA.csv " << fps << "/s" << std::endl
        << "Make sure the robot is BLUE. Press Enter to continue..." << std::endl;
    std::cin.ignore();
    try
    {
        franka::Robot robot(argv[1]);
        // Set default param.
        robot.setCartesianImpedance({{3000,3000,3000,300,300,300}});
        robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
        robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});
        // Torque control
        unsigned int count = 1;
        robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::Torques{
            // Zero torque
            std::array<double,7> torque = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
            if (count >= std::floor(1000/fps))
            {
                // Read state
                for (unsigned int i = 0; i < 7; i++)
                {
                    fileOut_JP << state.q[i] << ',';
                }
                fileOut_JP << std::endl;
                for (unsigned int i = 0; i < 16; i++)
                {
                    fileOut_OTEE << state.O_T_EE[i] << ',';
                    fileOut_EETK << state.EE_T_K[i] << ',';
                }
                fileOut_OTEE << std::endl;
                fileOut_EETK << std::endl;
                count = 1;
            }else
            {
                count++;
            }
            return torque;            
        });
    }
    catch(const franka::Exception& e)
    {
        std::cerr << e.what() << '\n';
        fileOut_JP.close();
        fileOut_OTEE.close();
        fileOut_EETK.close();
        return -1;
    }
    catch(const std::exception& e)
    {
        fileOut_JP.close();
        fileOut_OTEE.close();
        fileOut_EETK.close();
        return 0;
    }
    fileOut_JP.close();
    fileOut_OTEE.close();
    fileOut_EETK.close();
    return 0;
}