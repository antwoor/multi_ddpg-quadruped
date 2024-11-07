cmake_minimum_required(VERSION 3.15)
project(unitree_legged_sdk)

include_directories(include)

link_directories(lib)

add_compile_options(-std=c++11)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(amd64)|(AMD64)")
  set(EXTRA_LIBS -pthread libunitree_legged_sdk_amd64.so lcm)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64)|(ARM64)|(aarch64)|(AARCH64)")
  set(EXTRA_LIBS -pthread libunitree_legged_sdk_arm64.so lcm)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm32)|(ARM32)|(aarch32)|(AARCH32)")
  set(EXTRA_LIBS -pthread libunitree_legged_sdk_arm32.so lcm)
else()
  message(FATAL_ERROR "Processor architecture not recognized: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CMAKE_CXX_FLAGS "-O3")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third-party/pybind11 ${CMAKE_CURRENT_SOURCE_DIR}/third-party/pybind11/build)
pybind11_add_module(robot_interface_a1 python_interface.cpp)
target_link_libraries(robot_interface_a1 PRIVATE ${EXTRA_LIBS})

add_executable(example_walk examples/example_walk.cpp)
target_link_libraries(example_walk PRIVATE ${EXTRA_LIBS})

add_executable(example_torque examples/example_torque.cpp)
target_link_libraries(example_torque PRIVATE ${EXTRA_LIBS})
