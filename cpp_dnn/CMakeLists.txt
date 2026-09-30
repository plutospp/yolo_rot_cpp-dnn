cmake_minimum_required(VERSION 3.21)
project(yolo_rot_dnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 5 REQUIRED)

add_executable(yolo_rot_dnn
    src/main.cpp
    src/detector.cpp
)

target_include_directories(yolo_rot_dnn PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_rot_dnn PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(yolo_rot_dnn PRIVATE /W4 /utf-8)
endif()

if(WIN32)
    if(OpenCV_LIB_PATH)
        set(_libdir "${OpenCV_LIB_PATH}")
    else()
        set(_libdir "${OpenCV_DIR}")
    endif()
    get_filename_component(_bin "${_libdir}/../bin" ABSOLUTE)
    file(GLOB _dlls "${_bin}/*.dll")
    list(FILTER _dlls EXCLUDE REGEX "d\\.dll$")
    if(_dlls)
        add_custom_command(TARGET yolo_rot_dnn POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_dlls} $<TARGET_FILE_DIR:yolo_rot_dnn>)
    else()
        message(WARNING "no OpenCV DLLs in ${_bin}; add it to PATH")
    endif()
endif()
