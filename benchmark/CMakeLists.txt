set(BINARY ${CMAKE_PROJECT_NAME}_bench)

set(BENCHMARK_SOURCES
        main.cpp
)

set(SOURCES ${BENCHMARK_SOURCES})

add_executable(${BINARY} ${BENCHMARK_SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${BINARY} PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(FATAL_ERROR "OpenMP not found")
endif()

target_link_libraries(${BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib)