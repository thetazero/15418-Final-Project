set(BINARY ${CMAKE_PROJECT_NAME})

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

set(LIB_SOURCES
    board.h board.cpp
    engine_board.h engine_board.cpp
    timing.h
)

set(SOURCES ${SOURCES})

add_executable(${BINARY}_run ${SOURCES})

add_library(${BINARY}_lib STATIC ${LIB_SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${BINARY}_run PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${BINARY}_lib PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(FATAL_ERROR "OpenMP not found")
endif()


