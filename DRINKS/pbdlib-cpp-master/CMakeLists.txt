cmake_minimum_required(VERSION 2.8)

project(pbdlib_cpp)

set(CMAKE_MAJOR_VERSION 1)
set(CMAKE_MINOR_VERSION 0)
set(CMAKE_PATCH_VERSION 0)


# Dependencies
find_package(OpenGL REQUIRED)

if(NOT WIN32)
	find_package(GLEW REQUIRED)
	find_package(Armadillo 5.4 REQUIRED)
else()
	set(ARMADILLO_DIR "ARMADILLO_DIR-NOTFOUND" CACHE PATH "Path to Armadillo")
	set(GLEW_DIR "GLEW_DIR-NOTFOUND" CACHE PATH "Path to GLEW")
	set(GLFW_DIR "GLFW_DIR-NOTFOUND" CACHE PATH "Path to GLFW")

	if (NOT ${ARMADILLO_DIR} STREQUAL "ARMADILLO_DIR-NOTFOUND")
		set(ARMADILLO_INCLUDE_DIR "${ARMADILLO_DIR}/include")
		set(BLAS_LIB "${ARMADILLO_DIR}/examples/lib_win64/blas_win64_MT.lib")
		set(LAPACK_LIB "${ARMADILLO_DIR}/examples/lib_win64/lapack_win64_MT.lib")

		FILE(COPY "${ARMADILLO_DIR}/examples/lib_win64/blas_win64_MT.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
		FILE(COPY "${ARMADILLO_DIR}/examples/lib_win64/lapack_win64_MT.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
	else()
		message(FATAL_ERROR "Please indicate the path to the following dependencies: Armadillo, GLEW and GLFW")
	endif()

	if (NOT ${GLEW_DIR} STREQUAL "GLEW_DIR-NOTFOUND")
		set(GLEW_INCLUDE_DIR "${GLEW_DIR}/include")
		set(GLEW_LIBRARIES "${GLEW_DIR}/lib/Release/x64/glew32s.lib")
	else()
		message(FATAL_ERROR "Please indicate the path to the following dependencies: Armadillo, GLEW and GLFW")
	endif()

	if (NOT ${GLFW_DIR} STREQUAL "GLFW_DIR-NOTFOUND")
		set(GLFW_INCLUDE_DIR "${GLFW_DIR}/include")
		set(GLFW_LIB "${GLFW_DIR}/lib-vc2015/glfw3.lib")
	else()
		message(FATAL_ERROR "Please indicate the path to the following dependencies: Armadillo, GLEW and GLFW")
	endif()
endif()


if(APPLE)
	set(ARMADILLO_LIBRARIES
		${ARMADILLO_LIBRARIES}
		lapack
		blas
	)

	include_directories(
		/usr/local/include
	)

	if(EXISTS /usr/local/Frameworks)
		link_directories(
			/usr/local/Frameworks
		)
	endif()

	link_directories(
		/usr/local/lib
	)

	set(GLFWLIB_SEARCH_PATH /usr/local/lib)
	find_library(GLFW_LIB NAMES glfw3 glfw PATHS ${GLFWLIB_SEARCH_PATH})

	# Hack to include Mac frameworks
	set(GLFW_LIB
		${GLFW_LIB}
		"-framework CoreFoundation"
		"-framework OpenGL" 
		"-framework Cocoa"
		"-framework AppKit"
		"-framework CoreVideo"
		"-framework QTKit"
		"-framework IOKit"
		"-framework AGL"
		"-framework GLUT"
	)

	# OpenGL is deprecated since macOS 10.14, we need this to silence the warnings
	add_definitions(-DGL_SILENCE_DEPRECATION)

elseif(UNIX)

	set(GLFWLIB_SEARCH_PATH /usr/lib/x86_64-linux-gnu/ /usr/local/lib/)
	find_library(GLFW_LIB glfw PATHS ${GLFWLIB_SEARCH_PATH})

endif()


# Check compiler compatibility
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
	message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


# Copy data/ to build so that the examples can load from that location
FILE(COPY ${PROJECT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)


# Setup the include directories
include_directories(
	${PROJECT_SOURCE_DIR}/include/
	${PROJECT_SOURCE_DIR}/include/arpack-arma/include
	${ARMADILLO_INCLUDE_DIR}
	${GLEW_INCLUDE_DIR}
)


if(WIN32)
	include_directories(
		${GLFW_INCLUDE_DIR}
	)
endif()


# Setup the libraries to link
if(WIN32)
	set(ARMADILLO_LIBRARIES
		${ARMADILLO_LIBRARIES}
		${BLAS_LIB}
		${LAPACK_LIB}
	)
endif()

set(LIBRARIES
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
	${GLFW_LIB}
	${ARMADILLO_LIBRARIES}
)


# Additional flags needed for Visual Studio
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:MSVCRT")
endif()


# Declare the support files needed for OpenGL2 demos
set(GL2_SUPPORT_SOURCES
	${PROJECT_SOURCE_DIR}/src/utils/imgui_impl_glfw_gl2.cpp
	${PROJECT_SOURCE_DIR}/src/utils/imgui.cpp
	${PROJECT_SOURCE_DIR}/src/utils/imgui_draw.cpp
	${PROJECT_SOURCE_DIR}/src/utils/gfx_ui.cpp
)


# Declare the support files needed for OpenGL2 demos (using gfx.h)
set(GFX_SUPPORT_SOURCES
	${GL2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/utils/gfx.cpp
)


# Declare the support files needed for OpenGL2 (using gfx2.h) demos
set(GFX2_SUPPORT_SOURCES
	${GL2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/utils/gfx2.cpp
)


# Declare the support files needed for OpenGL3 (using GLSL & gfx3.h) demos
set(GFX3_SUPPORT_SOURCES
	${PROJECT_SOURCE_DIR}/src/utils/imgui_impl_glfw_gl3.cpp
	${PROJECT_SOURCE_DIR}/src/utils/imgui.cpp
	${PROJECT_SOURCE_DIR}/src/utils/imgui_draw.cpp
	${PROJECT_SOURCE_DIR}/src/utils/gfx_ui.cpp
	${PROJECT_SOURCE_DIR}/src/utils/gfx3.cpp
)



# Executables
add_executable(demo_ergodicControl_2D01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_ergodicControl_2D01.cpp
	${PROJECT_SOURCE_DIR}/src/utils/mpc_utils.cpp
)
target_link_libraries(demo_ergodicControl_2D01
	${LIBRARIES}
)

add_executable(demo_ergodicControl_nD01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_ergodicControl_nD01.cpp
	${PROJECT_SOURCE_DIR}/src/utils/mpc_utils.cpp
	${PROJECT_SOURCE_DIR}/src/utils/ImGuizmo.cpp
)
target_link_libraries(demo_ergodicControl_nD01
	${LIBRARIES}
)

add_executable(demo_GMR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_GMR01.cpp
)
target_link_libraries(demo_GMR01
	${LIBRARIES}
)

add_executable(demo_GPR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_GPR01.cpp
)
target_link_libraries(demo_GPR01
	${LIBRARIES}
)

add_executable(demo_HSMM_batchLQR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_HSMM_batchLQR01.cpp
)
target_link_libraries(demo_HSMM_batchLQR01
	${LIBRARIES}
)

add_executable(demo_LWR_batch01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_LWR_batch01.cpp
)
target_link_libraries(demo_LWR_batch01
	${LIBRARIES}
)

add_executable(demo_LWR_iterative01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_LWR_iterative01.cpp
)
target_link_libraries(demo_LWR_iterative01
	${LIBRARIES}
)

add_executable(demo_MPC_batch01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_MPC_batch01.cpp
	${PROJECT_SOURCE_DIR}/src/utils/mpc_utils.cpp
	${PROJECT_SOURCE_DIR}/src/utils/gl2ps.c
)
target_link_libraries(demo_MPC_batch01
	${LIBRARIES}
)

add_executable(demo_MPC_iterative01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_MPC_iterative01.cpp
	${PROJECT_SOURCE_DIR}/src/utils/mpc_utils.cpp
)
target_link_libraries(demo_MPC_iterative01
	${LIBRARIES}
)

add_executable(demo_MPC_semitied01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_MPC_semitied01.cpp
	${PROJECT_SOURCE_DIR}/src/utils/mpc_utils.cpp
	${PROJECT_SOURCE_DIR}/src/utils/gl2ps.c
)
target_link_libraries(demo_MPC_semitied01
	${LIBRARIES}
)

add_executable(demo_MPC_velocity01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_MPC_velocity01.cpp
	${PROJECT_SOURCE_DIR}/src/utils/mpc_utils.cpp
	${PROJECT_SOURCE_DIR}/src/utils/gl2ps.c
)
target_link_libraries(demo_MPC_velocity01
	${LIBRARIES}
)

add_executable(demo_online_GMM01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_online_GMM01.cpp
)
target_link_libraries(demo_online_GMM01
	${LIBRARIES}
)

add_executable(demo_online_HSMM01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_online_HSMM01.cpp
)
target_link_libraries(demo_online_HSMM01
	${LIBRARIES}
)

add_executable(demo_proMP01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_proMP01.cpp
)
target_link_libraries(demo_proMP01
	${LIBRARIES}
)

add_executable(demo_Riemannian_SPD_GMR_Mandel01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_SPD_GMR_Mandel01.cpp
)
target_link_libraries(demo_Riemannian_SPD_GMR_Mandel01
	${LIBRARIES}
)

add_executable(demo_Riemannian_SPD_GMR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_SPD_GMR01.cpp
)
target_link_libraries(demo_Riemannian_SPD_GMR01
	${LIBRARIES}
)

add_executable(demo_Riemannian_SPD_interp02
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_SPD_interp02.cpp
)
target_link_libraries(demo_Riemannian_SPD_interp02
	${LIBRARIES}
)

add_executable(demo_Riemannian_pose_batchLQR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_pose_batchLQR01.cpp
)
target_link_libraries(demo_Riemannian_pose_batchLQR01
	${LIBRARIES}
)

add_executable(demo_Riemannian_pose_GMM01
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_pose_GMM01.cpp
)
target_link_libraries(demo_Riemannian_pose_GMM01
	${ARMADILLO_LIBRARIES}
)

add_executable(demo_Riemannian_pose_infHorLQR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_pose_infHorLQR01.cpp
)
target_link_libraries(demo_Riemannian_pose_infHorLQR01
	${LIBRARIES}
)

add_executable(demo_Riemannian_pose_TPGMM01
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_pose_TPGMM01.cpp
)
target_link_libraries(demo_Riemannian_pose_TPGMM01
	${ARMADILLO_LIBRARIES}
)

add_executable(demo_Riemannian_S3_infHorLQR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_S3_infHorLQR01.cpp
)
target_link_libraries(demo_Riemannian_S3_infHorLQR01
	${LIBRARIES}
)

add_executable(demo_Riemannian_S3_TPGMM01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_S3_TPGMM01.cpp
)
target_link_libraries(demo_Riemannian_S3_TPGMM01
	${LIBRARIES}
)

add_executable(demo_Riemannian_S2_GMM01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_S2_GMM01.cpp
)
target_link_libraries(demo_Riemannian_S2_GMM01
	${LIBRARIES}
)

add_executable(demo_Riemannian_S2_infHorLQR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_S2_infHorLQR01.cpp
)
target_link_libraries(demo_Riemannian_S2_infHorLQR01
	${LIBRARIES}
)

add_executable(demo_Riemannian_S2_product01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_S2_product01.cpp
)
target_link_libraries(demo_Riemannian_S2_product01
	${LIBRARIES}
)

add_executable(demo_Riemannian_S2_TPGMM01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_Riemannian_S2_TPGMM01.cpp
)
target_link_libraries(demo_Riemannian_S2_TPGMM01
	${LIBRARIES}
)

add_executable(demo_TPMPC01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_TPMPC01.cpp
)
target_link_libraries(demo_TPMPC01
	${LIBRARIES}
)

add_executable(demo_TPGMMProduct01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_TPGMMProduct01.cpp
)
target_link_libraries(demo_TPGMMProduct01
	${LIBRARIES}
)

add_executable(demo_TPGMR01
	${GFX2_SUPPORT_SOURCES}
	${PROJECT_SOURCE_DIR}/src/demo_TPGMR01.cpp
)
target_link_libraries(demo_TPGMR01
	${LIBRARIES}
)
