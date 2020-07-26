cmake_minimum_required(VERSION 3.4)

# Project name
project(HEYTEA)

# Set standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
list(APPEND CMAKE_PREFIX_PATH ~/Github/YOGURT/libfranka/build)
find_package(Eigen3 REQUIRED)
find_package(Franka REQUIRED)

# Add libraries
add_subdirectory(LACTIC)
add_subdirectory(MILK)

# Set variable
set(DRINKS 
    FleshWaxberry_main
    #FleshWaxberry_lead
    #FleshWaxberry_reprJP
    FleshWaxberry_reprJP_K
    #FleshWaxberry_zeroGravity
    FleshWaxberry_readOnce
    #FleshWaxberry_OneshotJP
    #FleshWaxberry_OneshotJP_pro
    #FleshWaxberry_OneshotCarte_pro
    FleshWaxberry_imped_pre
    FleshWaxberry_imped_pre_p
    FleshWaxberry_pre_p
    FleshWaxberry_OneshotP
)

# Add executable files and link libraries
foreach(drink ${DRINKS})
    add_executable(${drink} ${drink}.cpp)
    target_link_libraries(${drink} LACTIC MILK Franka::Franka Eigen3::Eigen)
endforeach()