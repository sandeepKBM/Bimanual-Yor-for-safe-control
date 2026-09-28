# Built as an add_subdirectory() of a mujoco_mpc (MJPC) checkout, which provides
# the libmjpc / threadpool / mujoco::mujoco / absl targets. See ../README note in
# build_mjpc.sh. SIMULATION ONLY.
add_executable(mjpc_bimanual_server mjpc_server.cc bimanual_cup_task.cc bimanual_cup_task.h)
target_link_libraries(mjpc_bimanual_server libmjpc threadpool mujoco::mujoco absl::strings Threads::Threads)
target_include_directories(mjpc_bimanual_server PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(mjpc_bimanual_server PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc_bimanual_server PRIVATE ${MJPC_LINK_OPTIONS})
