cmake_minimum_required(VERSION 3.18)
project(BamEdge2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(BAM_WINDOWS TRUE)
    message(STATUS "Target platform: Windows")
else()
    set(BAM_WINDOWS FALSE)
    message(STATUS "Target platform: Linux")
endif()

# ---------------------------------------------------------------------------
# Paths — override with -D flags on the cmake command line
# ---------------------------------------------------------------------------

get_filename_component(MODELS_DIR "${MODELS_DIR}" ABSOLUTE)

set(MODEL_A_PT "${MODELS_DIR}/bam_model_a.pt")
set(MODEL_B_PT "${MODELS_DIR}/bam_model_b.pt")

message(STATUS "NUKE_ROOT            = ${NUKE_ROOT}")
message(STATUS "LIBTORCH_INCLUDE_DIR = ${LIBTORCH_INCLUDE_DIR}")
message(STATUS "MODEL_A_PT           = ${MODEL_A_PT}")
message(STATUS "MODEL_B_PT           = ${MODEL_B_PT}")

# Normalize path separators so cmake_install.cmake never embeds backslashes
# that CMake later misreads as escape sequences (CMP0010 warnings on Windows).
if(NOT "${LIBTORCH_INCLUDE_DIR}" STREQUAL "")
    file(TO_CMAKE_PATH "${LIBTORCH_INCLUDE_DIR}" LIBTORCH_INCLUDE_DIR)
endif()

if("${LIBTORCH_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR
        "LIBTORCH_INCLUDE_DIR is not set.\n"
        "Download the LibTorch zip matching Nuke's PyTorch version from https://pytorch.org/get-started/locally/\n"
        "(headers only — runtime libs come from NUKE_ROOT)\n"
        "Rerun cmake with -DLIBTORCH_INCLUDE_DIR=/path/to/libtorch")
endif()

foreach(_pt IN ITEMS "${MODEL_A_PT}" "${MODEL_B_PT}")
    if(NOT EXISTS "${_pt}")
        message(FATAL_ERROR
            "Model file not found: ${_pt}\n"
            "Download models first.")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# LibTorch headers
# ---------------------------------------------------------------------------

set(TORCH_INCLUDE_DIRS
    "${LIBTORCH_INCLUDE_DIR}/include"
    "${LIBTORCH_INCLUDE_DIR}/include/torch/csrc/api/include"
)

# _GLIBCXX_USE_CXX11_ABI must match how LibTorch was compiled (Linux only).
if(NOT MSVC)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
else()
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
endif()

# ---------------------------------------------------------------------------
# Model embedding
#
# Linux  : objcopy --input-target binary → ELF64 .o
# Windows: NASM    incbin              → PE/COFF Win64 .o
#
# Both produce the same extern "C" symbols in BamEdge2.cpp:
#   _binary_bam_model_a_pt_start / _end
#   _binary_bam_model_b_pt_start / _end
# ---------------------------------------------------------------------------

if(BAM_WINDOWS)

    find_program(NASM_EXECUTABLE nasm
        HINTS "$ENV{LOCALAPPDATA}/bin/NASM"
        DOC "NASM assembler — needed for Windows model embedding (override with -DNASM_EXECUTABLE=...)")
    if(NOT NASM_EXECUTABLE)
        message(FATAL_ERROR
            "NASM not found.\n"
            "Install from https://www.nasm.us or pass -DNASM_EXECUTABLE=C:/path/to/nasm.exe")
    endif()

    function(embed_model_nasm PT_FILE SYM_BASE OUT_OBJ)
        set(ASM_FILE "${CMAKE_BINARY_DIR}/${SYM_BASE}.asm")
        set(OBJ_FILE "${CMAKE_BINARY_DIR}/${SYM_BASE}.o")
        set(PT_COPY  "${CMAKE_BINARY_DIR}/${SYM_BASE}.pt")

        file(WRITE "${ASM_FILE}"
"section .data\n"
"global _binary_${SYM_BASE}_pt_start\n"
"global _binary_${SYM_BASE}_pt_end\n"
"_binary_${SYM_BASE}_pt_start:\n"
"incbin \"${SYM_BASE}.pt\"\n"
"_binary_${SYM_BASE}_pt_end:\n"
        )

        add_custom_command(
            OUTPUT  "${OBJ_FILE}"
            COMMAND ${CMAKE_COMMAND} -E copy "${PT_FILE}" "${PT_COPY}"
            COMMAND "${NASM_EXECUTABLE}" -f win64 "${ASM_FILE}" -o "${OBJ_FILE}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            DEPENDS "${PT_FILE}"
            COMMENT "Embedding ${SYM_BASE} (NASM win64)"
        )
        set(${OUT_OBJ} "${OBJ_FILE}" PARENT_SCOPE)
    endfunction()

    embed_model_nasm("${MODEL_A_PT}" "bam_model_a" MODEL_A_OBJ)
    embed_model_nasm("${MODEL_B_PT}" "bam_model_b" MODEL_B_OBJ)

else()

    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/bam_model_a.o"
        COMMAND ${CMAKE_COMMAND} -E copy "${MODEL_A_PT}" "${CMAKE_BINARY_DIR}/bam_model_a.pt"
        COMMAND objcopy
                --input-target  binary
                --output-target elf64-x86-64
                --binary-architecture i386:x86-64
                bam_model_a.pt
                bam_model_a.o
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        DEPENDS "${MODEL_A_PT}"
        COMMENT "Embedding bamEdge2A model (${MODEL_A_PT})"
    )

    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/bam_model_b.o"
        COMMAND ${CMAKE_COMMAND} -E copy "${MODEL_B_PT}" "${CMAKE_BINARY_DIR}/bam_model_b.pt"
        COMMAND objcopy
                --input-target  binary
                --output-target elf64-x86-64
                --binary-architecture i386:x86-64
                bam_model_b.pt
                bam_model_b.o
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        DEPENDS "${MODEL_B_PT}"
        COMMENT "Embedding bamEdge2B model (${MODEL_B_PT})"
    )

    set(MODEL_A_OBJ "${CMAKE_BINARY_DIR}/bam_model_a.o")
    set(MODEL_B_OBJ "${CMAKE_BINARY_DIR}/bam_model_b.o")

endif()

# ---------------------------------------------------------------------------
# Torch link targets — taken directly from NUKE_ROOT.
#
# Nuke ships a complete LibTorch runtime (CPU + CUDA) on both platforms.
# Linux : link the .so files directly; Nuke sets LD_LIBRARY_PATH to its own
#         directory at startup so the dynamic linker finds them at runtime.
# Windows: Nuke does not ship .lib import libraries; generate them from the
#          DLLs via dumpbin → .def → lib.exe.  The delay-load hook in
#          BamEdge2.cpp reuses already-loaded DLLs (GetModuleHandleW check),
#          falling back to copying them from NUKE_ROOT into install/lib/.
# ---------------------------------------------------------------------------

if(BAM_WINDOWS)

    set(BAM_LIB_DIR "${CMAKE_BINARY_DIR}/bamlibs")
    file(MAKE_DIRECTORY "${BAM_LIB_DIR}")

    # c10.dll exports data symbols; the standard c10.lib marks them with the
    # DATA keyword, which makes MSVC refuse /DELAYLOAD.  Work around this by
    # generating custom import libs from dumpbin exports — listing every
    # symbol as a plain function (no DATA keyword) — so delay-load compiles.
    # The .def files use the ORIGINAL DLL names (c10.dll / torch_cpu.dll) so
    # that torch_cuda.dll's own import for torch_cpu.dll resolves to the
    # single already-loaded module and does not trigger a second
    # static-initializer run ("Key already registered" crash).
    get_filename_component(_msvc_bin "${CMAKE_LINKER}" DIRECTORY)
    find_program(LIB_EXECUTABLE NAMES lib
        HINTS "${_msvc_bin}" REQUIRED NO_DEFAULT_PATH
        DOC "MSVC lib.exe — creates import libraries")
    find_program(DUMPBIN_EXECUTABLE NAMES dumpbin
        HINTS "${_msvc_bin}" REQUIRED NO_DEFAULT_PATH
        DOC "MSVC dumpbin.exe — lists DLL exports")
    find_program(PYTHON_EXECUTABLE NAMES python3 python py
        REQUIRED DOC "Python 3 interpreter")

    set(BAM_TORCH_LIB "${BAM_LIB_DIR}/bam_torch_cpu.lib")
    set(BAM_C10_LIB   "${BAM_LIB_DIR}/bam_c10.lib")

    # make_delayload_lib(DLL_FILE DLL_NAME LIB_OUT)
    # Creates a custom import lib that references DLL_NAME.dll without DATA
    # attributes, enabling /DELAYLOAD on DLLs with exported data symbols.
    function(make_delayload_lib DLL_FILE DLL_NAME LIB_OUT)
        set(_def "${BAM_LIB_DIR}/${DLL_NAME}.def")
        add_custom_command(
            OUTPUT "${LIB_OUT}"
            COMMAND "${PYTHON_EXECUTABLE}"
                    "${DUMPBIN_EXECUTABLE}" "${DLL_FILE}"
                    "${DLL_NAME}" "${_def}"
            COMMAND "${LIB_EXECUTABLE}"
                    /def:"${_def}" /out:"${LIB_OUT}" /machine:x64 /nologo
            DEPENDS "${DLL_FILE}"
            COMMENT "Creating delay-load import lib for ${DLL_NAME}.dll (MSVC)"
        )
    endfunction()

    make_delayload_lib("${NUKE_ROOT}/c10.dll"       "c10"       "${BAM_C10_LIB}")
    make_delayload_lib("${NUKE_ROOT}/torch_cpu.dll" "torch_cpu" "${BAM_TORCH_LIB}")

    set(_bam_lib_targets "${BAM_TORCH_LIB}" "${BAM_C10_LIB}")
    set(TORCH_LINK_LIBS  "${BAM_TORCH_LIB}" "${BAM_C10_LIB}")
    set(BAM_DELAYLOAD_OPTIONS /DELAYLOAD:torch_cpu.dll /DELAYLOAD:c10.dll)

    if(EXISTS "${NUKE_ROOT}/torch_cuda.dll")
        message(STATUS "CUDA LibTorch found in Nuke (Windows) — enabling CUDA support")
        add_compile_definitions(BAM_WITH_CUDA)

        set(BAM_TORCH_CUDA_LIB "${BAM_LIB_DIR}/bam_torch_cuda.lib")
        set(BAM_C10_CUDA_LIB   "${BAM_LIB_DIR}/bam_c10_cuda.lib")

        make_delayload_lib("${NUKE_ROOT}/c10_cuda.dll"   "c10_cuda"   "${BAM_C10_CUDA_LIB}")
        make_delayload_lib("${NUKE_ROOT}/torch_cuda.dll" "torch_cuda" "${BAM_TORCH_CUDA_LIB}")

        list(APPEND _bam_lib_targets "${BAM_TORCH_CUDA_LIB}" "${BAM_C10_CUDA_LIB}")
        list(APPEND TORCH_LINK_LIBS  "${BAM_TORCH_CUDA_LIB}" "${BAM_C10_CUDA_LIB}")
        list(APPEND BAM_DELAYLOAD_OPTIONS /DELAYLOAD:torch_cuda.dll /DELAYLOAD:c10_cuda.dll)
    else()
        message(STATUS "No torch_cuda.dll in Nuke (Windows) — CPU-only build")
    endif()

    add_custom_target(bam_torch_libs DEPENDS ${_bam_lib_targets})

else()

    set(TORCH_LINK_LIBS
        "${NUKE_ROOT}/libtorch_cpu.so"
        "${NUKE_ROOT}/libc10.so"
    )

    if(EXISTS "${NUKE_ROOT}/libtorch_cuda.so")
        message(STATUS "CUDA LibTorch found in Nuke — enabling CUDA support")
        add_compile_definitions(BAM_WITH_CUDA)
        list(APPEND TORCH_LINK_LIBS
            "${NUKE_ROOT}/libtorch_cuda.so"
            "${NUKE_ROOT}/libc10_cuda.so"
        )
    else()
        message(STATUS "No libtorch_cuda.so in NUKE_ROOT — CPU-only build")
    endif()

endif()

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------

add_library(BamEdge2 MODULE
    BamEdge2.cpp
    "${MODEL_A_OBJ}"
    "${MODEL_B_OBJ}"
)

if(BAM_WINDOWS)
    add_dependencies(BamEdge2 bam_torch_libs)
endif()

target_include_directories(BamEdge2 PRIVATE
    "${NUKE_ROOT}/include"
    ${TORCH_INCLUDE_DIRS}
)

if(BAM_WINDOWS)
    # Embed the official libtorch lib path so the plugin can load torch_cuda.dll
    # from there (with all bundled CUDA deps) as a fallback to Nuke's build.
    target_compile_definitions(BamEdge2 PRIVATE
        "BAM_LIBTORCH_LIB_DIR=\"${LIBTORCH_INCLUDE_DIR}/lib\"")
endif()

if(BAM_WINDOWS)
    set(BAM_DDIMAGE_LIB "${NUKE_ROOT}/DDImage.lib")
    target_link_options(BamEdge2 PRIVATE ${BAM_DELAYLOAD_OPTIONS})
    target_link_libraries(BamEdge2 PRIVATE delayimp.lib)

    target_link_libraries(BamEdge2 PRIVATE
        "${BAM_DDIMAGE_LIB}"
        ${TORCH_LINK_LIBS}
    )
else()
    target_link_libraries(BamEdge2 PRIVATE
        "${NUKE_ROOT}/libDDImage.so"
        ${TORCH_LINK_LIBS}
    )
endif()

# Nuke plugins must not have the 'lib' prefix
set_target_properties(BamEdge2 PROPERTIES PREFIX "")

if(BAM_WINDOWS)
    set_target_properties(BamEdge2 PROPERTIES SUFFIX ".dll")
else()
    set_target_properties(BamEdge2 PROPERTIES SUFFIX ".so")
    set_target_properties(BamEdge2 PROPERTIES
        INSTALL_RPATH         "${NUKE_ROOT}"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
endif()

# ---------------------------------------------------------------------------
# Install: put plugin into install directory.
#
# Torch DLLs are not bundled — Nuke pre-loads them before any plugin is
# touched, so the delay-load hook's GetModuleHandleW check always finds them.
# ---------------------------------------------------------------------------

set(INSTALL_DIR "${CMAKE_BINARY_DIR}/bamEdge2")

install(TARGETS BamEdge2 LIBRARY DESTINATION "${INSTALL_DIR}")