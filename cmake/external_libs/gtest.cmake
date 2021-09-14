set(gtest_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set(gtest_CFLAGS "-D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

if(ENABLE_GITEE)
    set(GTEST_URL "https://gitee.com/mirrors/googletest/repository/archive/release-1.8.1.tar.gz")
    set(GTEST_MD5 "711b149c9a74e4602235bcc3f8d4b60f")
else()
    set(GTEST_URL "https://github.com/google/googletest/archive/release-1.8.1.tar.gz")
    set(GTEST_MD5 "2e6fbeb6a91310a16efe181886c59596")
endif()

akg_add_pkg(gtest
        VER 1.8.1
        LIBS gtest
        URL ${GTEST_URL}
        MD5 ${GTEST_MD5}
	CMAKE_OPTION -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF
	-DCMAKE_MACOSX_RPATH=TRUE -Dgtest_disable_pthreads=ON)
include_directories(${gtest_INC})
add_library(akg::gtest ALIAS gtest::gtest)

file(COPY ${gtest_LIBPATH}/libgtest.a DESTINATION ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
file(COPY ${gtest_LIBPATH}/libgtest_main.a DESTINATION ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
