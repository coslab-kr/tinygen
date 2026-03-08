cmake_minimum_required(VERSION 3.22) 

project(TinyGen)

add_compile_options(-std=c++17 -fno-exceptions -fno-rtti)

add_subdirectory(compiler)
