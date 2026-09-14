set(CMAKE_FOLDER ${CMAKE_FOLDER}/euler)

set(ONEFLOW_EULER_BACKEND_DIR
  ${CMAKE_SOURCE_DIR}/ports/kunshan/oneflow_1d_hip
)

add_executable(oneflow_euler_backend_cpu_test
  euler_backend_contract_test.cpp
  ${ONEFLOW_EULER_BACKEND_DIR}/OneDEuler.cpp
  ${ONEFLOW_EULER_BACKEND_DIR}/OneDEulerBackend.cpp
  ${ONEFLOW_EULER_BACKEND_DIR}/OneDWeno5.cpp
)

target_include_directories(oneflow_euler_backend_cpu_test
  PRIVATE
    ${ONEFLOW_EULER_BACKEND_DIR}
)

target_compile_options(oneflow_euler_backend_cpu_test
  PRIVATE
    -ffp-contract=off
)

target_link_libraries(oneflow_euler_backend_cpu_test
  PRIVATE
    GTest::gtest_main
)

gtest_discover_tests(oneflow_euler_backend_cpu_test
  DISCOVERY_MODE PRE_TEST
)

add_executable(oneflow_flux_backend_bridge_test
  flux_backend_bridge_test.cpp
  ${ONEFLOW_EULER_BACKEND_DIR}/OneDEuler.cpp
  ${ONEFLOW_EULER_BACKEND_DIR}/OneDEulerBackend.cpp
  ${ONEFLOW_EULER_BACKEND_DIR}/OneDWeno5.cpp
  ${CMAKE_SOURCE_DIR}/codes/accel/src/CpuFluxBackend.cpp
)

target_include_directories(oneflow_flux_backend_bridge_test
  PRIVATE
    ${ONEFLOW_EULER_BACKEND_DIR}
    ${CMAKE_SOURCE_DIR}/codes/accel/include
    ${CMAKE_SOURCE_DIR}/codes/project/include
    ${CMAKE_SOURCE_DIR}/codes/basic/include
)

target_compile_options(oneflow_flux_backend_bridge_test
  PRIVATE
    -ffp-contract=off
)

target_link_libraries(oneflow_flux_backend_bridge_test
  PRIVATE
)

# Bridge test is a standalone main(), not a gtest
# Run manually: ./bin/oneflow_flux_backend_bridge_test


add_executable(oneflow_euler_domain_contract_test
  ${CMAKE_SOURCE_DIR}/tests/euler_domain_contract_test.cpp)
target_include_directories(oneflow_euler_domain_contract_test PRIVATE
  ${CMAKE_SOURCE_DIR}/codes/accel/include
  ${CMAKE_SOURCE_DIR}/codes/basic/include
  ${CMAKE_SOURCE_DIR}/codes/project/include)
target_link_libraries(oneflow_euler_domain_contract_test PRIVATE GTest::gtest_main)
gtest_discover_tests(oneflow_euler_domain_contract_test DISCOVERY_MODE PRE_TEST)

add_executable(oneflow_euler_domain_state_registry_test
  ${CMAKE_SOURCE_DIR}/tests/euler_domain_state_registry_test.cpp
  ${CMAKE_SOURCE_DIR}/codes/accel/src/EulerDomainStateRegistry.cpp)
target_include_directories(oneflow_euler_domain_state_registry_test PRIVATE
  ${CMAKE_SOURCE_DIR}/codes/accel/include
  ${CMAKE_SOURCE_DIR}/codes/basic/include
  ${CMAKE_SOURCE_DIR}/codes/project/include)
target_link_libraries(oneflow_euler_domain_state_registry_test PRIVATE GTest::gtest_main)
gtest_discover_tests(oneflow_euler_domain_state_registry_test DISCOVERY_MODE PRE_TEST)

add_executable(oneflow_euler_domain_state_lifecycle_test
  ${CMAKE_SOURCE_DIR}/tests/euler_domain_state_lifecycle_test.cpp
  ${CMAKE_SOURCE_DIR}/codes/accel/src/EulerDomainStateRegistry.cpp
  ${CMAKE_SOURCE_DIR}/codes/accel/src/EulerDomainStateLifecycle.cpp)
target_include_directories(oneflow_euler_domain_state_lifecycle_test PRIVATE
  ${CMAKE_SOURCE_DIR}/codes/accel/include
  ${CMAKE_SOURCE_DIR}/codes/basic/include
  ${CMAKE_SOURCE_DIR}/codes/project/include)
target_link_libraries(oneflow_euler_domain_state_lifecycle_test PRIVATE GTest::gtest_main)
gtest_discover_tests(oneflow_euler_domain_state_lifecycle_test DISCOVERY_MODE PRE_TEST)

add_executable(oneflow_euler_cpu_adapter_test
  ${CMAKE_SOURCE_DIR}/tests/euler_cpu_adapter_test.cpp
  ${CMAKE_SOURCE_DIR}/codes/accel/src/EulerCpuAdapter.cpp
  ${CMAKE_SOURCE_DIR}/codes/accel/src/CpuFluxBackend.cpp)
target_include_directories(oneflow_euler_cpu_adapter_test PRIVATE
  ${CMAKE_SOURCE_DIR}/codes/accel/include
  ${CMAKE_SOURCE_DIR}/codes/basic/include
  ${CMAKE_SOURCE_DIR}/codes/project/include)
target_link_libraries(oneflow_euler_cpu_adapter_test PRIVATE GTest::gtest_main)
gtest_discover_tests(oneflow_euler_cpu_adapter_test DISCOVERY_MODE PRE_TEST)
