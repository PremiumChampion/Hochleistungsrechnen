add_executable(milestone02 main.cpp src/field.cpp)


target_include_directories(milestone02 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/includes")


target_link_libraries(milestone02 PRIVATE lib SDL3::SDL3)
