cmake_minimum_required (VERSION 3.1)

project(pimatch)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -Wall -mfpu=neon")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -mfpu=neon")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mfpu=neon")

# The version number.
set(pimatch_VERSION_MAJOR 0)
set(pimatch_VERSION_MINOR 1)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
 
find_package(GTest REQUIRED)

include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  "${GTEST_INCLUDE_DIRS}"
  )
 
add_library(${PROJECT_NAME} SHARED
  src/HammingMatch.cpp
  src/HammingTree.cpp
  )

add_library(TestUtil SHARED
  test/TestUtil.cpp
  )

add_executable(HammingMatchTest
  test/HammingMatchTest.cpp
  )
target_link_libraries(HammingMatchTest
  ${PROJECT_NAME}
  TestUtil
  ${GTEST_BOTH_LIBRARIES})

add_executable(HammingTreeTest
  test/HammingTreeTest.cpp
  )
target_link_libraries(HammingTreeTest
  ${PROJECT_NAME}
  TestUtil
  ${GTEST_BOTH_LIBRARIES})

add_executable(Benchmark
  demo/Benchmark.cpp)

target_link_libraries(Benchmark
  ${PROJECT_NAME})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

if (${OpenCV_FOUND})
  include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS})

  add_executable(HammingTreeDemo
    demo/Demo.cpp
    )

  target_link_libraries(HammingTreeDemo
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${PROJECT_SOURCE_DIR}/demo/DBoW2/lib/libDBoW2.so)

endif (${OpenCV_FOUND})
