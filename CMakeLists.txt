cmake_minimum_required(VERSION 3.5)

project(SuperResolution VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets LinguistTools)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets LinguistTools)

set(3LIB_DIR /Users/pmlinelone/Code/CCP3lib/3lib/installed)
list(APPEND CMAKE_PREFIX_PATH ${3LIB_DIR}/opencv_4.9.0)
list(APPEND CMAKE_PREFIX_PATH ${3LIB_DIR}/libtorch)

find_package(OpenCV REQUIRED)
set(OpenCV_INSTALL_PATH ${3LIB_DIR}/opencv_4.9.0)
set(OpenCV_DIR ${OpenCV_INSTALL_PATH}/lib/cmake/opencv4)
include_directories(${OpenCV_INSTALL_PATH}/include/opencv4)

find_package(Torch REQUIRED)
set(Torch_INSTALL_PATH ${3LIB_DIR}/libtorch)
set(Torch_DIR ${Torch_INSTALL_PATH}/share/cmake/Torch)
include_directories(${Torch_INSTALL_PATH}/include)
include_directories(${Torch_INSTALL_PATH}/include/Aten)
include_directories(${Torch_INSTALL_PATH}/include/c10)
include_directories(${Torch_INSTALL_PATH}/include/caffe2)
include_directories(${Torch_INSTALL_PATH}/include/torch)
include_directories(${Torch_INSTALL_PATH}/include/torch/csrc)
# add_executable(ResolutionModel ResolutionModel.cpp ResolutionModel.h)
# target_link_libraries("${TORCH_LIBRARIES}")
set_property(TARGET PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TS_FILES SuperResolution_zh_CN.ts)

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
        ResolutionModel.h
        ResolutionModel.cpp
        ${TS_FILES}
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(SuperResolution
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        res.qrc
        Image.h
        Image.cpp
        # ResolutionModel.h
        # ResolutionModel.cpp
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET SuperResolution APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation

    qt_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
else()
    if(ANDROID)
        add_library(SuperResolution SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(SuperResolution
            ${PROJECT_SOURCES}
        )
    endif()

    qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
endif()

target_link_libraries(SuperResolution PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(SuperResolution PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(SuperResolution PRIVATE "${OpenCV_LIBS}")

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.SuperResolution)
endif()
set_target_properties(SuperResolution PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS SuperResolution
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(SuperResolution)
endif()
