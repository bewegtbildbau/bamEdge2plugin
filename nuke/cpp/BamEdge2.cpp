/**
 * BamEdge2.cpp
 * Nuke NDK plugin — LaMa-based edge fixing with embedded TorchScript models.
 *
 * Input:  0  img   Premultiplied RGBA.  Alpha defines the edge fixing region.
 *
 * Pipeline per frame:
 *   1. Unpremult RGB   (straight = premult / alpha where alpha > 0)
 *   2. Binarise alpha  (> 0 → 1.0)
 *   3. Erode binary mask by 'Edge Size' pixels  → eroded
 *   4. Model mask      = binary * (1 - eroded)   [thin edge band]
 *   5. Run LaMa inference on straight RGB + model mask  (background thread)
 *   6. Transition mask = boxBlur(model mask, Blend Transition)
 *   7. Composite       = original_straight*(1-t) + fixes*t
 *   8. Premult output  = composite * alpha
 *
 * Output channels:
 *   rgba                — premultiplied result
 *   bE2_model_mask      — edge-band mask sent to the model  (all 4 RGBA identical)
 *   bE2_transition_mask — (blurred) compositing mask        (all 4 RGBA identical)
 *
 * Models are embedded at link time; no external .pt files are needed at runtime.
 *
 * Async rendering:
 *   engine() returns the original passthrough immediately while the background
 *   thread runs inference.  asapUpdate() then triggers a re-render that shows
 *   the actual model result.  This prevents Nuke from freezing / showing the
 *   "force quit" dialog during the (potentially long) model forward pass.
 */

// ---------------------------------------------------------------------------
// Windows: delay-load hook — reuses torch DLLs already loaded by Nuke if
// present (same PyTorch version), falling back to lib\ next to this plugin.
// ---------------------------------------------------------------------------
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <delayimp.h>
#include <string>

// VEH to capture exception details if c10.dll's static initializers fail.
static DWORD     g_c10VehCode  = 0;
static ULONG_PTR g_c10VehFault = 0;
static ULONG_PTR g_c10VehRip   = 0;
static DWORD     g_c10VehRW    = 0;

static LONG CALLBACK bamC10VEH(PEXCEPTION_POINTERS ep) {
    if (!g_c10VehCode) {
        g_c10VehCode = ep->ExceptionRecord->ExceptionCode;
        g_c10VehRip  = reinterpret_cast<ULONG_PTR>(ep->ExceptionRecord->ExceptionAddress);
        if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
                ep->ExceptionRecord->NumberParameters >= 2) {
            g_c10VehRW    = static_cast<DWORD>(ep->ExceptionRecord->ExceptionInformation[0]);
            g_c10VehFault = static_cast<ULONG_PTR>(ep->ExceptionRecord->ExceptionInformation[1]);
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static FARPROC WINAPI bamDelayLoadHook(unsigned dliNotify, PDelayLoadInfo pdli)
{
    if (dliNotify != dliNotePreLoadLibrary)
        return nullptr;

    std::string name(pdli->szDll);
    if (name != "torch_cpu.dll" && name != "c10.dll")
        return nullptr;

    wchar_t selfPath[MAX_PATH] = {};
    HMODULE hSelf = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&bamDelayLoadHook), &hSelf);
    GetModuleFileNameW(hSelf, selfPath, MAX_PATH);

    wchar_t* sep = wcsrchr(selfPath, L'\\');
    if (!sep) return nullptr;
    sep[1] = L'\0';

    // Register lib\ so bam_torch_cpu.dll's import-table deps resolve from there.
    wchar_t libDir[MAX_PATH] = {};
    _snwprintf_s(libDir, MAX_PATH, _TRUNCATE, L"%slib", selfPath);
    AddDllDirectory(libDir);

    // Prepend lib\ to PATH so c10.dll's internal LoadLibrary() calls (e.g.
    // libiomp5md.dll, fbgemm.dll) find our copies.  Nuke ships CUDA 12.8, which
    // is the same version we build against, so the CUDA DLLs (cudart64_12.dll
    // etc.) are already in Nuke's process — no need to pre-load them here.
    {
        static bool s_pathPatched = false;
        if (!s_pathPatched) {
            s_pathPatched = true;
            wchar_t curPath[32768] = {};
            GetEnvironmentVariableW(L"PATH", curPath, 32768);
            wchar_t check[MAX_PATH] = {};
            _snwprintf_s(check, MAX_PATH, _TRUNCATE, L"%slib", selfPath);
            if (!wcsstr(curPath, check)) {
                wchar_t newPath[32768] = {};
                _snwprintf_s(newPath, 32768, _TRUNCATE, L"%slib;%s", selfPath, curPath);
                SetEnvironmentVariableW(L"PATH", newPath);
                fprintf(stderr, "[BamEdge2] prepended lib\\ to PATH\n");
                fflush(stderr);
            }
        }
    }

    // torch_cpu.dll imports c10.dll.  Pre-load c10.dll from lib\ first so
    // torch_cpu.dll's import resolves to our copy.  If Nuke already loaded a
    // compatible c10.dll the GetModuleHandle check finds it and reuses it —
    // same PyTorch 2.7.1 version, so no double-init.
    if (name == "torch_cpu.dll") {
        HMODULE hExisting = GetModuleHandleW(L"c10.dll");
        if (hExisting) {
            wchar_t existPath[MAX_PATH] = {};
            GetModuleFileNameW(hExisting, existPath, MAX_PATH);
            fprintf(stderr, "[BamEdge2] c10.dll already in process: %ls\n", existPath);
        } else {
            wchar_t c10Path[MAX_PATH] = {};
            _snwprintf_s(c10Path, MAX_PATH, _TRUNCATE, L"%slib\\c10.dll", selfPath);

            g_c10VehCode = 0; g_c10VehFault = 0; g_c10VehRip = 0; g_c10VehRW = 0;
            PVOID hVEH = AddVectoredExceptionHandler(1, bamC10VEH);

            HMODULE hC10 = LoadLibraryExW(c10Path, nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                LOAD_LIBRARY_SEARCH_SYSTEM32     |
                LOAD_LIBRARY_SEARCH_USER_DIRS);
            DWORD c10Err = GetLastError();

            if (hVEH) RemoveVectoredExceptionHandler(hVEH);

            if (!hC10) {
                if (g_c10VehCode == EXCEPTION_ACCESS_VIOLATION)
                    fprintf(stderr,
                            "[BamEdge2] c10.dll FAILED err=%lu AV %s addr=0x%llX rip=0x%llX\n",
                            c10Err,
                            g_c10VehRW == 0 ? "READ" : "WRITE",
                            static_cast<unsigned long long>(g_c10VehFault),
                            static_cast<unsigned long long>(g_c10VehRip));
                else
                    fprintf(stderr, "[BamEdge2] c10.dll FAILED err=%lu VEH=0x%08X\n",
                            c10Err, g_c10VehCode);
            } else {
                fprintf(stderr, "[BamEdge2] c10.dll OK\n");
            }
            fflush(stderr);
        }
    }

    wchar_t wideName[256] = {};
    MultiByteToWideChar(CP_ACP, 0, pdli->szDll, -1, wideName, 256);

    // Reuse Nuke's already-loaded copy if present (same PyTorch version).
    HMODULE hExisting = GetModuleHandleW(wideName);
    if (hExisting) {
        fprintf(stderr, "[BamEdge2] %s already in process — reusing\n", pdli->szDll);
        fflush(stderr);
        return reinterpret_cast<FARPROC>(hExisting);
    }

    // Fallback: load from lib\ next to this plugin.
    wchar_t fullPath[MAX_PATH] = {};
    _snwprintf_s(fullPath, MAX_PATH, _TRUNCATE, L"%slib\\%s", selfPath, wideName);

    HMODULE hLib = LoadLibraryExW(fullPath, nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
        LOAD_LIBRARY_SEARCH_SYSTEM32     |
        LOAD_LIBRARY_SEARCH_USER_DIRS);

    if (!hLib)
        fprintf(stderr, "[BamEdge2] delay-load failed for %s (error %lu)\n",
                pdli->szDll, GetLastError());

    return reinterpret_cast<FARPROC>(hLib);
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = bamDelayLoadHook;

// Try to load torch_cuda.dll from the same directory as torch_cpu.dll.
// Attempts restricted search first (picks up dirs Python's import torch added
// via os.add_dll_directory), then falls back to plain LoadLibraryW (PATH).
// Safe to call multiple times — does nothing if already in the process.
static void bamTryLoadTorchCuda(const char* tag)
{
    if (GetModuleHandleW(L"torch_cuda.dll")) {
        fprintf(stderr, "[BamEdge2] %s: torch_cuda.dll already in process\n", tag);
        fflush(stderr);
        return;
    }

#ifdef BAM_LIBTORCH_LIB_DIR
    // Try the official libtorch lib directory first — it bundles all CUDA
    // dependencies (cudart64_12, cublas, etc.) alongside torch_cuda.dll so
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR is sufficient to resolve everything.
    {
        wchar_t ltDir[MAX_PATH] = {};
        MultiByteToWideChar(CP_UTF8, 0, BAM_LIBTORCH_LIB_DIR, -1, ltDir, MAX_PATH);
        wchar_t ltPath[MAX_PATH];
        _snwprintf_s(ltPath, MAX_PATH, _TRUNCATE, L"%s/torch_cuda.dll", ltDir);
        fprintf(stderr, "[BamEdge2] %s: trying libtorch: %ls\n", tag, ltPath); fflush(stderr);
        HMODULE h = LoadLibraryExW(ltPath, nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
            LOAD_LIBRARY_SEARCH_SYSTEM32     |
            LOAD_LIBRARY_SEARCH_USER_DIRS);
        if (h) {
            fprintf(stderr, "[BamEdge2] %s: torch_cuda.dll OK (libtorch)\n", tag);
            fflush(stderr);
            return;
        }
        fprintf(stderr, "[BamEdge2] %s: libtorch FAIL err=%lu\n", tag, GetLastError());
        fflush(stderr);
    }
#endif

    // Fallback: same directory as the already-loaded torch_cpu.dll (Nuke's dir).
    wchar_t cpuPath[MAX_PATH] = {};
    HMODULE hCpu = GetModuleHandleW(L"torch_cpu.dll");
    if (!hCpu || !GetModuleFileNameW(hCpu, cpuPath, MAX_PATH)) {
        fprintf(stderr, "[BamEdge2] %s: torch_cpu.dll not found in process\n", tag);
        fflush(stderr);
        return;
    }
    wchar_t* sep = wcsrchr(cpuPath, L'\\');
    if (!sep) return;
    sep[1] = L'\0';
    wchar_t cudaPath[MAX_PATH];
    _snwprintf_s(cudaPath, MAX_PATH, _TRUNCATE, L"%storch_cuda.dll", cpuPath);
    fprintf(stderr, "[BamEdge2] %s: trying Nuke dir: %ls\n", tag, cudaPath); fflush(stderr);

    HMODULE h = LoadLibraryExW(cudaPath, nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
        LOAD_LIBRARY_SEARCH_SYSTEM32     |
        LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (!h)
        h = LoadLibraryW(cudaPath);

    if (h)
        fprintf(stderr, "[BamEdge2] %s: torch_cuda.dll OK (Nuke dir)\n", tag);
    else
        fprintf(stderr, "[BamEdge2] %s: torch_cuda.dll FAIL err=%lu\n", tag, GetLastError());
    fflush(stderr);
}

#endif  // _WIN32

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Channel.h"
#include "DDImage/Op.h"
#include "DDImage/ImagePlane.h"

#include <torch/script.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <thread>
#include <vector>

using namespace DD::Image;

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------

static std::string detectGpuName()
{
#ifdef _WIN32
    // Use NVML (part of the NVIDIA driver) — no CUDA toolkit headers needed.
    HMODULE hNvml = LoadLibraryW(L"nvml.dll");
    if (!hNvml) return "None";

    typedef int (*PFN_nvmlInit)();
    typedef int (*PFN_nvmlDeviceGetHandleByIndex)(unsigned int, void**);
    typedef int (*PFN_nvmlDeviceGetName)(void*, char*, unsigned int);
    typedef int (*PFN_nvmlShutdown)();

    auto pfnInit    = (PFN_nvmlInit)                   GetProcAddress(hNvml, "nvmlInit_v2");
    auto pfnGetHnd  = (PFN_nvmlDeviceGetHandleByIndex) GetProcAddress(hNvml, "nvmlDeviceGetHandleByIndex");
    auto pfnGetName = (PFN_nvmlDeviceGetName)          GetProcAddress(hNvml, "nvmlDeviceGetName");
    auto pfnShut    = (PFN_nvmlShutdown)               GetProcAddress(hNvml, "nvmlShutdown");

    std::string result = "None";
    if (pfnInit && pfnGetHnd && pfnGetName && pfnInit() == 0) {
        void* dev = nullptr;
        if (pfnGetHnd(0, &dev) == 0) {
            char name[256] = {};
            if (pfnGetName(dev, name, sizeof(name)) == 0)
                result = std::string(name);
        }
        if (pfnShut) pfnShut();
    }
    FreeLibrary(hNvml);
    return result;
#else
    return "None";
#endif
}

static const std::string& gpuInfoText()
{
    static const std::string s = std::string("Local GPU:  ") + detectGpuName();
    return s;
}

// Flush LibTorch's CUDA caching allocator so GPU memory is returned to the
// driver immediately.  Called on node destruction and on CPU fallback.
// Uses GetProcAddress to avoid pulling in cuda_runtime_api.h headers.
static void tryEmptyGpuCache()
{
#ifdef _WIN32
    HMODULE h = GetModuleHandleW(L"torch_cuda.dll");
    if (!h) return;
    typedef void (*PFN)();
    // MSVC-mangled name of c10::cuda::CUDACachingAllocator::emptyCache()
    auto pfn = reinterpret_cast<PFN>(
        GetProcAddress(h, "?emptyCache@CUDACachingAllocator@cuda@c10@@YAXXZ"));
    if (pfn) pfn();
#endif
}

// ---------------------------------------------------------------------------
// Embedded model data — linked in as raw ELF sections by objcopy
// ---------------------------------------------------------------------------

extern "C" {
    extern const unsigned char _binary_bam_model_a_pt_start[];
    extern const unsigned char _binary_bam_model_a_pt_end[];
    extern const unsigned char _binary_bam_model_b_pt_start[];
    extern const unsigned char _binary_bam_model_b_pt_end[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const int PAD_MODULO = 32;

static const char* const MODEL_ITEMS[]  = { "bamEdge2A", "bamEdge2B", nullptr };

// ---------------------------------------------------------------------------
// Seekable istream wrapper over a raw byte range.
// ---------------------------------------------------------------------------

struct RawBuf : std::streambuf {
    RawBuf(const char* begin, const char* end)
        : _begin(begin), _end(end)
    {
        setg(const_cast<char*>(begin),
             const_cast<char*>(begin),
             const_cast<char*>(end));
    }

protected:
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override
    {
        if (!(which & std::ios_base::in))
            return pos_type(off_type(-1));
        char* np = const_cast<char*>(_begin) + static_cast<std::ptrdiff_t>(pos);
        if (np < _begin || np > _end)
            return pos_type(off_type(-1));
        setg(const_cast<char*>(_begin), np, const_cast<char*>(_end));
        return pos;
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which) override
    {
        if (!(which & std::ios_base::in))
            return pos_type(off_type(-1));
        char* np;
        if      (dir == std::ios_base::beg) np = const_cast<char*>(_begin) + off;
        else if (dir == std::ios_base::cur) np = gptr() + off;
        else                                np = const_cast<char*>(_end)   + off;
        if (np < _begin || np > _end)
            return pos_type(off_type(-1));
        setg(const_cast<char*>(_begin), np, const_cast<char*>(_end));
        return pos_type(np - _begin);
    }

private:
    const char* _begin;
    const char* _end;
};

// ---------------------------------------------------------------------------
// sRGB transfer functions.
// Nuke operates in scene-linear; LaMa was trained on sRGB images.
// Encode to sRGB before the forward pass, decode afterwards.
// Values are clamped to [0, 1] — LaMa has no concept of HDR.
// ---------------------------------------------------------------------------

static inline float linearToSrgb(float v)
{
    v = std::clamp(v, 0.0f, 1.0f);
    return (v <= 0.0031308f) ? (12.92f * v)
                              : (1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f);
}

static inline float srgbToLinear(float v)
{
    v = std::clamp(v, 0.0f, 1.0f);
    return (v <= 0.04045f) ? (v / 12.92f)
                            : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

// ---------------------------------------------------------------------------
// Morphological erosion — binary float mask, separable box, clamp-to-edge border.
// ---------------------------------------------------------------------------

static void erodeBoxBinary(const float* src, float* dst,
                           int W, int H, int PW, int radius)
{
    if (radius <= 0) {
        for (int y = 0; y < H; ++y)
            std::memcpy(dst + y * PW, src + y * PW, W * sizeof(float));
        return;
    }

    std::vector<float> tmp(W * H, 0.0f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bool ok = true;
            for (int dx = -radius; dx <= radius && ok; ++dx) {
                int nx = std::clamp(x + dx, 0, W - 1);
                if (src[y * PW + nx] < 0.5f)
                    ok = false;
            }
            tmp[y * W + x] = ok ? 1.0f : 0.0f;
        }
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bool ok = true;
            for (int dy = -radius; dy <= radius && ok; ++dy) {
                int ny = std::clamp(y + dy, 0, H - 1);
                if (tmp[ny * W + x] < 0.5f)
                    ok = false;
            }
            dst[y * PW + x] = ok ? 1.0f : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Separable box blur — clamp-to-edge padding
// ---------------------------------------------------------------------------

static void blurBox(const float* src, float* dst,
                    int W, int H, int PW, int radius)
{
    if (radius <= 0) {
        for (int y = 0; y < H; ++y)
            std::memcpy(dst + y * PW, src + y * PW, W * sizeof(float));
        return;
    }

    const float norm = 1.0f / static_cast<float>(2 * radius + 1);
    std::vector<float> tmp(W * H, 0.0f);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int dx = -radius; dx <= radius; ++dx)
                s += src[y * PW + std::clamp(x + dx, 0, W - 1)];
            tmp[y * W + x] = s * norm;
        }
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy)
                s += tmp[std::clamp(y + dy, 0, H - 1) * W + x];
            dst[y * PW + x] = s * norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Custom output layers
// ---------------------------------------------------------------------------

struct MaskChans {
    Channel model[4];
    Channel trans[4];
    MaskChans() {
        model[0] = getChannel("bE2_model_mask.red");
        model[1] = getChannel("bE2_model_mask.green");
        model[2] = getChannel("bE2_model_mask.blue");
        model[3] = getChannel("bE2_model_mask.alpha");
        trans[0] = getChannel("bE2_transition_mask.red");
        trans[1] = getChannel("bE2_transition_mask.green");
        trans[2] = getChannel("bE2_transition_mask.blue");
        trans[3] = getChannel("bE2_transition_mask.alpha");
    }
};
static MaskChans s_mask;

// ---------------------------------------------------------------------------
// Plugin class
// ---------------------------------------------------------------------------

class BamEdge2 : public Iop
{
public:
    static const char* const CLASS;
    static const Op::Description d;

    explicit BamEdge2(Node* node)
        : Iop(node)
        , _modelIdx(0)
        , _deviceIdx(0)
        , _useGpuIfAvailable(true)
        , _edgeSize(5.0)
        , _blendSize(0.0)
        , _loadedModelIdx(-1)
        , _loadedDeviceIdx(-1)
        , _moduleLoaded(false)
        , _state(0)
        , _fetchStarted(false)
        , _generation(0)
        , _cachedFrame(-1)
        , _cachedModelIdx(-1)
        , _cachedDeviceIdx(-1)
        , _cachedEdgeSize(-1.0)
        , _cachedBlendSize(-1.0)
        , _cachedBBoxX(0), _cachedBBoxY(0), _cachedBBoxR(0), _cachedBBoxT(0)
        , _cachedInputHash(0)
        , _fW(0), _fH(0), _fBotY(0), _fLeftX(0)
        , _PW(0), _PH(0)
    {
    }

    ~BamEdge2() override
    {
        ++_generation;  // signal any in-flight background thread to abort
        if (_loadedDeviceIdx == 1) {
            _module.reset();
            tryEmptyGpuCache();
        }
    }

    const char* Class()       const override { return CLASS; }
    const char* displayName() const override { return "BamEdge2"; }
    const char* node_help()   const override
    {
        return
            "LaMa-based edge fixing (bamEdge2).\n\n"
            "Input (img): premultiplied RGBA.\n\n"
            "Edge Size: defines the binary alpha mask for the edge fixing. "
            "Blend Transition: blurs the mask to soften the boundary "
            "between the result and the original image.\n\n"
            "Extra outputs: bE2_model_mask, bE2_transition_mask\n\n"
            "The viewer shows the original image while inference runs in the "
            "background and updates automatically when done.";
    }

    int minimum_inputs() const override { return 1; }
    int maximum_inputs() const override { return 2; }
    const char* input_label(int n, char*) const override
    {
        if (n == 0) return "img";
        if (n == 1) return "mask";
        return "";
    }

    void knobs(Knob_Callback f) override
    {
        Text_knob(f, gpuInfoText().c_str());

        Bool_knob(f, &_useGpuIfAvailable, "use_gpu", "Use GPU if available");
        SetFlags(f, Knob::STARTLINE);
        Tooltip(f, "When checked, runs on the detected CUDA GPU.\n"
                   "Falls back to CPU automatically if CUDA is unavailable.");

        Enumeration_knob(f, &_modelIdx,  MODEL_ITEMS,  "model",      "Model");
        SetFlags(f, Knob::STARTLINE);
        Tooltip(f, "bamEdge2A\nbamEdge2B");

        Double_knob(f, &_edgeSize,  "edge_size",  "Edge Size");
        SetFlags(f, Knob::STARTLINE);
        SetRange(f, 0, 20);
        Tooltip(f, "Size of the edge to fix.\n"
                   "Has no effect when a mask input is connected.");

        Double_knob(f, &_blendSize, "blend_size", "Blend Transition");
        SetFlags(f, Knob::STARTLINE);
        SetRange(f, 0, 20);
        Tooltip(f, "Blur radius for the compositing mask.\n"
                   "0 = hard cut; higher = softer feathered transition.\n"
                   "Has no effect when a mask input is connected.");

        Tab_knob(f, "Info");
        Text_knob(f,
            "<style>a { color: #7ab8d4; }</style>"
            "<p><b>BamEdge2</b></p>"

            "<p>A fast AI helper for edge fixes.<br>"
            "It is designed to help clean up edge artifacts quickly.<br>"
            "It is <b>not</b> a solution to fix all edge problems.</p>"

            "<p><b>Models</b><br>"
            "bamEdge2A will most likely give the best result<br>"
            "bamEdge2B gives you an alternative result</p>"

            "<p><b>Inputs</b><br>"
            "img: premultiplied RGBA<br>"
            "mask (optional): when connected used for the prediction</p>"

            "<p><b>Output</b><br>"
            "It outputs additional layers:<br>"
            "- the region it predicts<br>"
            "- a transition mask</p>"

            "<p><b>Frame-based processing</b><br>"
            "The model processes each frame independently with no temporal awareness.<br>"
            "While inference runs in the background the viewer shows the original "
            "image and updates automatically when the result is ready.</p>"

            "<p><b>About the model</b><br>"
            "Fine-tuned version of a <a href=\"https://github.com/advimman/lama\">LaMa</a> model<br>"
            "with augmented training data focused on edge regions.</p>"

            "<hr>"
            "<p>"
            "Erik Schneider 2026<br>"
            "<a href=\"https://bewegtbildbau.de/\">bewegtbildbau.de</a><br>"
            "<a href=\"mailto:erik@bewegtbildbau.de\">erik@bewegtbildbau.de</a>"
            "</p>"
        );
    }

    void _validate(bool for_real) override
    {
        _deviceIdx = _useGpuIfAvailable ? 1 : 0;

        if (input(0)) {
            input(0)->validate(for_real);
            copy_info();
            if (input(1))
                input(1)->validate(for_real);
        } else {
            set_out_channels(ChannelSetInit(0));
            return;
        }

        const Box& bbox  = info_.box();
        const int  frame = for_real
                               ? static_cast<int>(outputContext().frame())
                               : _cachedFrame;

        uint64_t inputHash;
        {
            DD::Image::Hash h;
            h.append(input(0)->hash().value());
            if (input(1)) h.append(input(1)->hash().value());
            inputHash = h.value();
        }

        const bool needsRestart =
            (frame        != _cachedFrame      ) ||
            (_modelIdx    != _cachedModelIdx   ) ||
            (_deviceIdx   != _cachedDeviceIdx  ) ||
            (_edgeSize    != _cachedEdgeSize   ) ||
            (_blendSize   != _cachedBlendSize  ) ||
            (bbox.x()     != _cachedBBoxX      ) ||
            (bbox.y()     != _cachedBBoxY      ) ||
            (bbox.r()     != _cachedBBoxR      ) ||
            (bbox.t()     != _cachedBBoxT      ) ||
            (inputHash    != _cachedInputHash   );

        if (needsRestart) {
            _cachedFrame      = frame;
            _cachedModelIdx   = _modelIdx;
            _cachedDeviceIdx  = _deviceIdx;
            _cachedEdgeSize   = _edgeSize;
            _cachedBlendSize  = _blendSize;
            _cachedBBoxX = bbox.x(); _cachedBBoxY = bbox.y();
            _cachedBBoxR = bbox.r(); _cachedBBoxT = bbox.t();
            _cachedInputHash  = inputHash;
            ++_generation;
            _state.store(0, std::memory_order_release);
            _fetchStarted.store(false, std::memory_order_release);
        }

        ChannelSet out = Mask_RGBA;
        for (int i = 0; i < 4; ++i) {
            out += s_mask.model[i];
            out += s_mask.trans[i];
        }
        info_.turn_on(out);
        set_out_channels(out);

        if (_state.load(std::memory_order_relaxed) == 1)
            warning("BamEdge2: computing in background...");
    }

    void _request(int /*x*/, int /*y*/, int /*r*/, int /*t*/,
                  ChannelMask /*channels*/, int count) override
    {
        input(0)->request(info_.x(), info_.y(),
                          info_.r(), info_.t(), Mask_RGBA, count);
        if (input(1))
            input(1)->request(info_.x(), info_.y(),
                              info_.r(), info_.t(), Mask_Alpha, count);
    }

    void engine(int y, int x, int r, ChannelMask channels, Row& row) override;

    void append(DD::Image::Hash& hash) override
    {
        Iop::append(hash);
    }

private:
    int    _modelIdx;
    int    _deviceIdx;   // derived each _validate(): 0=cpu, 1=cuda
    bool   _useGpuIfAvailable;
    double _edgeSize;
    double _blendSize;

    // Model — heap-allocated so torch is never touched at node construction time.
    // shared_ptr so runInference can keep a local reference during the forward
    // pass even if another thread calls _module.reset() concurrently.
    std::shared_ptr<torch::jit::script::Module>   _module;
    int                                           _loadedModelIdx;
    int                                           _loadedDeviceIdx;
    bool                                          _moduleLoaded;

    // Async state machine.
    //   0 = IDLE      — no data fetched yet for this frame
    //   1 = COMPUTING — fetchAndPrepare() done; background thread is running
    //   2 = DONE      — inference complete; _fR/G/B hold the model result
    std::atomic<int>  _state;
    // CAS flag: exactly one engine() thread calls fetchAndPrepare().
    // All others (and the winner) return zeros immediately so Nuke's render
    // pool stays free for the fetchPlane() call inside fetchAndPrepare().
    std::atomic<bool> _fetchStarted;
    std::atomic<int>  _generation;   // incremented on every invalidation

    // Cache key — what parameters were in effect when we last computed.
    int      _cachedFrame;
    int      _cachedModelIdx;
    int      _cachedDeviceIdx;
    double   _cachedEdgeSize;
    double   _cachedBlendSize;
    int      _cachedBBoxX, _cachedBBoxY, _cachedBBoxR, _cachedBBoxT;
    uint64_t _cachedInputHash;  // combined hash of both inputs' pixel content

    // Full-frame output buffers (W*H flat, stride=W, row 0 = bottom y).
    // While state==COMPUTING these hold the premultiplied passthrough.
    // When state==DONE they hold the premultiplied model result.
    int _fW, _fH, _fBotY, _fLeftX;
    std::vector<float> _fR, _fG, _fB, _fA;
    std::vector<float> _fModelMask, _fTransMask;

    // Padded intermediate data shared with the background thread.
    int _PW, _PH;
    std::vector<float> _prepRStr, _prepGStr, _prepBStr;   // straight RGB
    std::vector<float> _prepAlpha;
    std::vector<float> _prepModelMask, _prepTransMask;

    // Called from engine() on the Nuke thread — fast path.
    void fetchAndPrepare();

    // Called from the background thread — slow (torch forward pass).
    void runInference(int gen);

    bool ensureModule();

    static Op* build(Node* node) { return new BamEdge2(node); }
};


// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const char* const BamEdge2::CLASS = "BamEdge2";

const Op::Description BamEdge2::d(
    BamEdge2::CLASS,
    "Filter/BamEdge2",
    BamEdge2::build
);


// ---------------------------------------------------------------------------
// Model loading (lazy, cached per model index + device)
// ---------------------------------------------------------------------------

bool BamEdge2::ensureModule()
{
    fprintf(stderr, "[BamEdge2] ensureModule: entry\n"); fflush(stderr);

    // No mutex needed: ensureModule() is only called from runInference(),
    // and only one runInference() thread runs at a time (guaranteed by the
    // _fetchStarted CAS in engine()).

    if (_module && _loadedModelIdx == _modelIdx && _loadedDeviceIdx == _deviceIdx) {
        fprintf(stderr, "[BamEdge2] ensureModule: already loaded\n"); fflush(stderr);
        return true;
    }

    // TorchScript models register C10 operator keys in global process-lifetime
    // registries on first load.  Calling jit::load a second time on the same
    // model data crashes with "Key already registered with the same priority".
    // So: only reload when the model selection changed; a device-only change
    // just moves the already-loaded module to the new device.
    const bool needLoad = !_module || (_loadedModelIdx != _modelIdx);

    if (needLoad) {
        fprintf(stderr, "[BamEdge2] ensureModule: resetting module\n"); fflush(stderr);
        _module.reset();
        _moduleLoaded = false;
    } else {
        fprintf(stderr, "[BamEdge2] ensureModule: device change — moving to %s\n",
                _deviceIdx == 1 ? "CUDA" : "CPU"); fflush(stderr);
    }

    try {
        if (needLoad) {
#ifdef _WIN32
            // Best-effort pre-load of torch_cuda.dll before jit::load.
            // torch_cpu.dll's getCUDAHooks() uses std::call_once — if it fires
            // before torch_cuda.dll registers its hooks, CUDA is permanently
            // unavailable for that load.  Loading CUDA deps can fail early
            // (before Python's import torch has added CUDA dirs), so we also
            // retry below just before module_.to(CUDA).
            bamTryLoadTorchCuda("pre-jit");
#endif
            fprintf(stderr, "[BamEdge2] ensureModule: picking model symbols, modelIdx=%d\n",
                    _modelIdx); fflush(stderr);
            const unsigned char* start;
            const unsigned char* end;
            if (_modelIdx == 0) {
                start = _binary_bam_model_a_pt_start;
                end   = _binary_bam_model_a_pt_end;
            } else {
                start = _binary_bam_model_b_pt_start;
                end   = _binary_bam_model_b_pt_end;
            }
            fprintf(stderr, "[BamEdge2] ensureModule: start=%p end=%p\n",
                    (void*)start, (void*)end); fflush(stderr);

            const std::size_t sz = static_cast<std::size_t>(end - start);
            fprintf(stderr, "[BamEdge2] Loading embedded model %s (%zu MB)\n",
                    MODEL_ITEMS[_modelIdx], sz / (1024 * 1024)); fflush(stderr);

            fprintf(stderr, "[BamEdge2] ensureModule: constructing RawBuf\n"); fflush(stderr);
            RawBuf   buf(reinterpret_cast<const char*>(start),
                         reinterpret_cast<const char*>(end));
            std::istream stream(&buf);

            fprintf(stderr, "[BamEdge2] ensureModule: calling torch::jit::load\n"); fflush(stderr);
            _module = std::make_shared<torch::jit::script::Module>(
                          torch::jit::load(stream));
            fprintf(stderr, "[BamEdge2] ensureModule: torch::jit::load done\n"); fflush(stderr);
            _module->eval();
        }

        // Move to the requested device.  Track the ACTUAL device we land on
        // (may fall back to CPU if the driver is absent or too old).
        int actualDeviceIdx = _deviceIdx;
        if (_deviceIdx == 1) {
#ifdef _WIN32
            // Retry loading torch_cuda.dll — by now Python's import torch may have
            // called os.add_dll_directory() for CUDA paths, making deps findable.
            bamTryLoadTorchCuda("pre-to-cuda");
#endif
            try {
                _module->to(torch::kCUDA);
            } catch (const c10::Error& e) {
                fprintf(stderr, "[BamEdge2] ensureModule: CUDA unavailable (%s), running on CPU\n",
                        e.what()); fflush(stderr);
                warning("BamEdge2: CUDA not available — running on CPU.");
                _module->to(torch::kCPU);
                tryEmptyGpuCache();
                actualDeviceIdx = 0;
            }
        } else {
            _module->to(torch::kCPU);
        }

        _loadedModelIdx  = _modelIdx;
        _loadedDeviceIdx = actualDeviceIdx;   // actual device, not just requested
        _moduleLoaded    = true;
        fprintf(stderr, "[BamEdge2] ensureModule: model ready\n"); fflush(stderr);
        return true;
    }
    catch (const c10::Error& e) {
        fprintf(stderr, "[BamEdge2] ensureModule: c10::Error: %s\n", e.what()); fflush(stderr);
        error("BamEdge2: failed to load model: %s", e.what());
        return false;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "[BamEdge2] ensureModule: std::exception: %s\n", e.what()); fflush(stderr);
        error("BamEdge2: failed to load model: %s", e.what());
        return false;
    }
}


// ---------------------------------------------------------------------------
// fetchAndPrepare — runs on the Nuke render thread, must complete quickly.
//
// Fetches the input ImagePlane, unpremults, computes the masks, and fills
// _fR/G/B/A with the premultiplied passthrough so engine() can return
// something useful while the background thread is still running.
// ---------------------------------------------------------------------------

void BamEdge2::fetchAndPrepare()
{
    const Box& bbox = info_.box();
    const int W = bbox.w();
    const int H = bbox.h();
    _fW     = W;
    _fH     = H;
    _fBotY  = bbox.y();
    _fLeftX = bbox.x();

    _PW = W + (PAD_MODULO - W % PAD_MODULO) % PAD_MODULO;
    _PH = H + (PAD_MODULO - H % PAD_MODULO) % PAD_MODULO;

    const int edgeSize  = std::max(0, static_cast<int>(std::round(_edgeSize)));
    const int blendSize = std::max(0, static_cast<int>(std::round(_blendSize)));

    ImagePlane inPlane(bbox, false, Mask_RGBA, 4);
    input(0)->fetchPlane(inPlane);

    const float*  imgData = inPlane.readable();
    const int64_t imgCS   = inPlane.chanStride();
    const int rIdx = inPlane.chanNo(Chan_Red);
    const int gIdx = inPlane.chanNo(Chan_Green);
    const int bIdx = inPlane.chanNo(Chan_Blue);
    const int aIdx = inPlane.chanNo(Chan_Alpha);

    // Padded working buffers (stride = PW)
    _prepAlpha.assign(_PH * _PW, 0.0f);
    _prepRStr .assign(_PH * _PW, 0.0f);
    _prepGStr .assign(_PH * _PW, 0.0f);
    _prepBStr .assign(_PH * _PW, 0.0f);

    auto copyIn = [&](int ci, std::vector<float>& buf) {
        if (ci < 0) return;
        for (int row = 0; row < H; ++row)
            std::memcpy(buf.data() + row * _PW,
                        imgData + int64_t(ci) * imgCS + row * W,
                        W * sizeof(float));
    };
    copyIn(aIdx, _prepAlpha);
    copyIn(rIdx, _prepRStr);
    copyIn(gIdx, _prepGStr);
    copyIn(bIdx, _prepBStr);

    // Unpremult
    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col) {
            const int   idx = row * _PW + col;
            const float a   = _prepAlpha[idx];
            if (a > 0.0f) { _prepRStr[idx] /= a; _prepGStr[idx] /= a; _prepBStr[idx] /= a; }
            else           { _prepRStr[idx] = _prepGStr[idx] = _prepBStr[idx] = 0.0f; }
        }

    // Binary mask
    std::vector<float> binary(_PH * _PW, 0.0f);
    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col)
            binary[row * _PW + col] = (_prepAlpha[row * _PW + col] > 0.0f) ? 1.0f : 0.0f;

    _prepModelMask.assign(_PH * _PW, 0.0f);
    _prepTransMask.assign(_PH * _PW, 0.0f);

    if (input(1)) {
        // External mask path
        ImagePlane maskPlane(bbox, false, Mask_Alpha, 1);
        input(1)->fetchPlane(maskPlane);
        const float*  maskData = maskPlane.readable();
        const int64_t maskCS   = maskPlane.chanStride();
        const int     mIdx     = maskPlane.chanNo(Chan_Alpha);

        for (int row = 0; row < H; ++row)
            for (int col = 0; col < W; ++col) {
                const int idx = row * _PW + col;
                const float mv = (mIdx >= 0)
                    ? maskData[int64_t(mIdx) * maskCS + row * W + col]
                    : 0.0f;
                _prepModelMask[idx] = (mv > 0.0f) ? 1.0f : 0.0f;
            }
        for (int row = 0; row < H; ++row)
            for (int col = 0; col < W; ++col)
                _prepTransMask[row * _PW + col] = _prepModelMask[row * _PW + col];
    } else {
        // Auto edge mask path
        std::vector<float> eroded(_PH * _PW, 0.0f);
        erodeBoxBinary(binary.data(), eroded.data(), W, H, _PW, edgeSize);
        for (int row = 0; row < H; ++row)
            for (int col = 0; col < W; ++col) {
                const int idx = row * _PW + col;
                _prepModelMask[idx] = binary[idx] * (1.0f - eroded[idx]);
            }

        const int transErode = std::max(0, edgeSize - blendSize);
        std::vector<float> transEroded(_PH * _PW, 0.0f);
        std::vector<float> transEdge  (_PH * _PW, 0.0f);
        erodeBoxBinary(binary.data(), transEroded.data(), W, H, _PW, transErode);
        for (int row = 0; row < H; ++row)
            for (int col = 0; col < W; ++col) {
                const int idx = row * _PW + col;
                transEdge[idx] = binary[idx] * (1.0f - transEroded[idx]);
            }

        std::vector<float> preBl(_PH * _PW, 0.0f);
        for (int row = 0; row < H; ++row)
            for (int col = 0; col < W; ++col) {
                const int idx = row * _PW + col;
                preBl[idx] = transEdge[idx] + (1.0f - binary[idx]);
            }
        blurBox(preBl.data(), _prepTransMask.data(), W, H, _PW, blendSize);
        for (int row = 0; row < H; ++row)
            for (int col = 0; col < W; ++col) {
                const int idx = row * _PW + col;
                _prepTransMask[idx] = std::clamp(
                    _prepTransMask[idx] - (1.0f - binary[idx]), 0.0f, 1.0f);
            }
    }

    // Flat mask caches (stride = W, no padding)
    _fModelMask.resize(W * H);
    _fTransMask.resize(W * H);
    for (int row = 0; row < H; ++row) {
        std::memcpy(_fModelMask.data() + row * W, _prepModelMask.data() + row * _PW, W * sizeof(float));
        std::memcpy(_fTransMask.data() + row * W, _prepTransMask.data() + row * _PW, W * sizeof(float));
    }

    // Flat alpha (stride = W, no padding)
    _fA.resize(W * H);
    for (int row = 0; row < H; ++row)
        std::memcpy(_fA.data() + row * W, _prepAlpha.data() + row * _PW, W * sizeof(float));

    // Fill _fR/G/B with the premultiplied passthrough.
    // This is what engine() serves while the background thread is running.
    _fR.assign(W * H, 0.0f);
    _fG.assign(W * H, 0.0f);
    _fB.assign(W * H, 0.0f);
    if (rIdx >= 0)
        for (int row = 0; row < H; ++row)
            std::memcpy(_fR.data() + row * W,
                        imgData + int64_t(rIdx) * imgCS + row * W, W * sizeof(float));
    if (gIdx >= 0)
        for (int row = 0; row < H; ++row)
            std::memcpy(_fG.data() + row * W,
                        imgData + int64_t(gIdx) * imgCS + row * W, W * sizeof(float));
    if (bIdx >= 0)
        for (int row = 0; row < H; ++row)
            std::memcpy(_fB.data() + row * W,
                        imgData + int64_t(bIdx) * imgCS + row * W, W * sizeof(float));
}


// ---------------------------------------------------------------------------
// runInference — runs on a detached background thread.
//
// Checks _generation before and after the forward pass so that stale results
// (from a superseded frame or changed parameters) are silently discarded.
// When successful, overwrites _fR/G/B with the composited model result and
// calls asapUpdate() to schedule a re-render.
// ---------------------------------------------------------------------------

void BamEdge2::runInference(int gen)
{
    fprintf(stderr, "[BamEdge2] runInference start gen=%d\n", gen); fflush(stderr);

    if (_generation.load(std::memory_order_acquire) != gen) {
        fprintf(stderr, "[BamEdge2] runInference: stale gen, aborting\n"); fflush(stderr);
        return;
    }

    fprintf(stderr, "[BamEdge2] runInference: calling ensureModule\n"); fflush(stderr);
    if (!ensureModule()) {
        fprintf(stderr, "[BamEdge2] runInference: model load failed\n"); fflush(stderr);
        _state.store(2, std::memory_order_release);
        return;
    }
    fprintf(stderr, "[BamEdge2] runInference: ensureModule done\n"); fflush(stderr);

    if (_generation.load(std::memory_order_acquire) != gen) {
        fprintf(stderr, "[BamEdge2] runInference: stale gen after load, aborting\n"); fflush(stderr);
        return;
    }

    const int W  = _fW,  H  = _fH;
    const int PW = _PW,  PH = _PH;
    fprintf(stderr, "[BamEdge2] runInference: forward pass %dx%d pad %dx%d on %s\n",
            W, H, PW, PH, _loadedDeviceIdx == 1 ? "CUDA" : "CPU"); fflush(stderr);

    fprintf(stderr, "[BamEdge2] runInference: building imgBuf\n"); fflush(stderr);
    std::vector<float> imgBuf(3 * PH * PW, 0.0f);
    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col) {
            const int pIdx = row * PW + col;
            imgBuf[0 * PH * PW + pIdx] = linearToSrgb(_prepRStr[pIdx]);
            imgBuf[1 * PH * PW + pIdx] = linearToSrgb(_prepGStr[pIdx]);
            imgBuf[2 * PH * PW + pIdx] = linearToSrgb(_prepBStr[pIdx]);
        }

    // Capture a local shared_ptr so the module stays alive even if another
    // runInference thread calls _module.reset() while we're in forward().
    std::shared_ptr<torch::jit::script::Module> module = _module;
    if (!module) {
        fprintf(stderr, "[BamEdge2] runInference: module null after ensureModule — aborting\n"); fflush(stderr);
        return;
    }

    // Use _loadedDeviceIdx (the actual device after any CPU fallback),
    // NOT _deviceIdx (the requested device).
    fprintf(stderr, "[BamEdge2] runInference: creating tensors\n"); fflush(stderr);
    torch::Tensor result;
    try {
        const torch::Device device = (_loadedDeviceIdx == 1) ? torch::kCUDA : torch::kCPU;
        auto imgTensor  = torch::from_blob(imgBuf.data(), {1, 3, PH, PW}).to(device);
        auto maskTensor = torch::from_blob(_prepModelMask.data(), {1, 1, PH, PW}).clone().to(device);

        fprintf(stderr, "[BamEdge2] runInference: running forward\n"); fflush(stderr);
        {
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> inputs = { imgTensor, maskTensor };
            result = module->forward(inputs).toTensor().to(torch::kCPU).contiguous();
        }
    } catch (const c10::Error& e) {
        fprintf(stderr, "[BamEdge2] runInference: forward error: %s\n", e.what()); fflush(stderr);
        _state.store(2, std::memory_order_release);
        return;
    } catch (const std::exception& e) {
        fprintf(stderr, "[BamEdge2] runInference: forward std::exception: %s\n", e.what()); fflush(stderr);
        _state.store(2, std::memory_order_release);
        return;
    }
    fprintf(stderr, "[BamEdge2] runInference: forward done\n"); fflush(stderr);
    const float* resData = result.data_ptr<float>();

    if (_generation.load(std::memory_order_acquire) != gen) {
        fprintf(stderr, "[BamEdge2] runInference: stale gen after forward, discarding\n"); fflush(stderr);
        return;
    }

    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col) {
            const int   pIdx = row * PW + col;
            const int   fIdx = row * W  + col;
            const float t    = _prepTransMask[pIdx], ot = 1.0f - t;
            const float a    = _prepAlpha[pIdx];
            const float r_lin = srgbToLinear(resData[0 * PH * PW + pIdx]);
            const float g_lin = srgbToLinear(resData[1 * PH * PW + pIdx]);
            const float b_lin = srgbToLinear(resData[2 * PH * PW + pIdx]);
            _fR[fIdx] = (_prepRStr[pIdx] * ot + r_lin * t) * a;
            _fG[fIdx] = (_prepGStr[pIdx] * ot + g_lin * t) * a;
            _fB[fIdx] = (_prepBStr[pIdx] * ot + b_lin * t) * a;
        }

    fprintf(stderr, "[BamEdge2] runInference: composite done, storing DONE\n"); fflush(stderr);
    _state.store(2, std::memory_order_release);
    fprintf(stderr, "[BamEdge2] runInference: finished\n"); fflush(stderr);
}


// ---------------------------------------------------------------------------
// Per-row engine
// ---------------------------------------------------------------------------

void BamEdge2::engine(int y, int x, int r, ChannelMask channels, Row& row)
{
    const int myGen = _generation.load(std::memory_order_acquire);

    // One thread wins the CAS, fetches the input and kicks off the background
    // inference thread.  All other threads fall through to the wait loop below.
    if (_state.load(std::memory_order_acquire) == 0) {
        bool expected = false;
        if (_fetchStarted.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            fetchAndPrepare();
            _state.store(1, std::memory_order_release);
            std::thread([this, myGen]() { runInference(myGen); }).detach();
        }
    }

    // Block until inference completes (state == 2).
    // If the frame is invalidated while we wait (user scrubs / changes params),
    // _generation increments and we bail out so render threads are not stuck.
    while (_state.load(std::memory_order_acquire) < 2) {
        if (_generation.load(std::memory_order_acquire) != myGen) {
            const int npix = r - x;
            if (npix > 0) {
                foreach (z, channels)
                    std::memset(row.writable(z) + x, 0, npix * sizeof(float));
            }
            return;
        }
        std::this_thread::yield();
    }

    // State == 2 (DONE): serve the premultiplied model result.
    const int numPix   = r - x;
    const int cacheRow = y - _fBotY;
    const int colOff   = x - _fLeftX;

    if (cacheRow < 0 || cacheRow >= _fH || colOff < 0 || numPix <= 0) {
        if (numPix > 0) {
            foreach (z, channels)
                std::memset(row.writable(z) + x, 0, numPix * sizeof(float));
        }
        return;
    }

    const int cacheOff = cacheRow * _fW + colOff;

    foreach (z, channels) {
        float*       dst = row.writable(z) + x;
        const float* src = nullptr;

        if      (z == Chan_Red)   src = _fR.data() + cacheOff;
        else if (z == Chan_Green) src = _fG.data() + cacheOff;
        else if (z == Chan_Blue)  src = _fB.data() + cacheOff;
        else if (z == Chan_Alpha) src = _fA.data() + cacheOff;
        else {
            for (int i = 0; i < 4; ++i) {
                if (z == s_mask.model[i]) { src = _fModelMask.data() + cacheOff; break; }
                if (z == s_mask.trans[i]) { src = _fTransMask.data() + cacheOff; break; }
            }
        }

        if (src)
            std::memcpy(dst, src, numPix * sizeof(float));
        else
            std::memset(dst, 0, numPix * sizeof(float));
    }
}
