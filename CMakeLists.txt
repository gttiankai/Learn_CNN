cmake_minimum_required(VERSION 3.20)
project(WinogradeConv)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SOURCE_CODE *.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O0")

add_executable(WinogradeConv ${SOURCE_CODE})
