// Lightweight crash/minidump helper. Install early in process to capture
// unhandled crashes and write diagnostics to disk. Minimal, header-only.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <processthreadsapi.h>
#include <timeapi.h>
#include <io.h>
#include <fcntl.h>
// For symbol resolution
#include <dbghelp.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#else
#include <signal.h>
#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#include <unistd.h>
#endif
#endif

namespace fftfree {

static std::atomic<bool> g_crash_handler_installed{false};

inline std::string timestamp_string() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return ss.str();
}

#if defined(_WIN32)

// Attempt to write a minidump via dbghelp.dll when an unhandled exception occurs.
inline void write_minidump(EXCEPTION_POINTERS* exinfo) {
    // Dynamic load to avoid link dependency in CI/builds that don't provide dbghelp.
    HMODULE hDbg = LoadLibraryA("dbghelp.dll");
    if (!hDbg) return;
    using MiniDumpWriteDump_t = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, DWORD, void*, void*, void*);
    auto fn = reinterpret_cast<MiniDumpWriteDump_t>(GetProcAddress(hDbg, "MiniDumpWriteDump"));
    if (!fn) { FreeLibrary(hDbg); return; }

    DWORD pid = GetCurrentProcessId();
    HANDLE proc = GetCurrentProcess();

    std::string fname = "fftfree-crash-" + timestamp_string() + ".dmp";
    HANDLE fh = CreateFileA(fname.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) { FreeLibrary(hDbg); return; }

    // MINIDUMP_TYPE 0 => MiniDumpNormal
    const DWORD dumpType = 0;

    struct MINIDUMP_EXCEPTION_INFORMATION_WRAPPER {
        DWORD ThreadId;
        EXCEPTION_POINTERS* ExInfo;
        BOOL ClientPointers;
    } mei;
    mei.ThreadId = GetCurrentThreadId();
    mei.ExInfo = exinfo;
    mei.ClientPointers = FALSE;

    // Call the function
    fn(proc, pid, fh, dumpType, &mei, nullptr, nullptr);

    CloseHandle(fh);
    // Notify user
    std::fprintf(stderr, "Wrote minidump: %s\n", fname.c_str());
    FreeLibrary(hDbg);
}

// Write a human-readable stack trace (best-effort) to stderr and a .log file.
inline void write_text_backtrace(EXCEPTION_POINTERS* /*exinfo*/) {
    const int kMaxFrames = 62;
    void* frames[kMaxFrames];
    USHORT captured = CaptureStackBackTrace(0, kMaxFrames, frames, nullptr);

    // Try to resolve symbols via dbghelp
    HMODULE hDbg = LoadLibraryA("dbghelp.dll");
    HANDLE proc = GetCurrentProcess();
    if (hDbg) {
        using SymInitialize_t = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
        using SymFromAddr_t = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
        using SymSetOptions_t = DWORD(WINAPI*)(DWORD);

        auto pSymInitialize = reinterpret_cast<SymInitialize_t>(GetProcAddress(hDbg, "SymInitialize"));
        auto pSymFromAddr = reinterpret_cast<SymFromAddr_t>(GetProcAddress(hDbg, "SymFromAddr"));
        auto pSymSetOptions = reinterpret_cast<SymSetOptions_t>(GetProcAddress(hDbg, "SymSetOptions"));

        if (pSymSetOptions) {
            // Request undecorated names
            pSymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
        }
        if (pSymInitialize) {
            pSymInitialize(proc, nullptr, TRUE);
        }

        std::string fname = "fftfree-crash-" + timestamp_string() + ".log";
        FILE* f = std::fopen(fname.c_str(), "w");
        if (f) {
            std::fprintf(f, "CaptureStackBackTrace frames=%hu\n", captured);
        }

        for (USHORT i = 0; i < captured; ++i) {
            DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
            char symbol_buf[sizeof(SYMBOL_INFO) + 1024];
            PSYMBOL_INFO pSym = reinterpret_cast<PSYMBOL_INFO>(symbol_buf);
            std::memset(pSym, 0, sizeof(symbol_buf));
            pSym->SizeOfStruct = sizeof(SYMBOL_INFO);
            pSym->MaxNameLen = 1024;
            DWORD64 displacement = 0;
            const char* name = "<unknown>";
            if (pSymFromAddr && pSymFromAddr(proc, addr, &displacement, pSym)) {
                name = pSym->Name;
            }
            std::fprintf(stderr, "#%02u %p %s +0x%llx\n", (unsigned)i, (void*)addr, name, (unsigned long long)displacement);
            if (f) std::fprintf(f, "#%02u %p %s +0x%llx\n", (unsigned)i, (void*)addr, name, (unsigned long long)displacement);
        }
        if (f) {
            std::fprintf(f, "Wrote text backtrace to %s\n", fname.c_str());
            std::fclose(f);
        }

        if (pSymInitialize) {
            // SymCleanup is optional; try to call if available
            auto pSymCleanup = reinterpret_cast<BOOL(WINAPI*)(HANDLE)>(GetProcAddress(hDbg, "SymCleanup"));
            if (pSymCleanup) pSymCleanup(proc);
        }
        FreeLibrary(hDbg);
    } else {
        // Fallback: just print addresses
        std::fprintf(stderr, "Stack frames (addresses): captured=%hu\n", captured);
        for (USHORT i = 0; i < captured; ++i) {
            std::fprintf(stderr, "#%02u %p\n", (unsigned)i, frames[i]);
        }
    }
}

LONG WINAPI vectored_exception_handler(EXCEPTION_POINTERS* exinfo) {
    // Try to produce an immediate, human-readable trace on stderr.
    write_text_backtrace(exinfo);
    // Also write a minidump for post-mortem analysis
    write_minidump(exinfo);
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void install_crash_handler_impl() {
    if (g_crash_handler_installed.exchange(true)) return;
    // Install vectored handler so we get called for crashes
    AddVectoredExceptionHandler(1, reinterpret_cast<PVECTORED_EXCEPTION_HANDLER>(vectored_exception_handler));
}

#else

inline void write_backtrace_to_file(int signo, void* context) {
    std::string fname = "fftfree-crash-" + timestamp_string() + ".log";
    FILE* f = std::fopen(fname.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "Received signal %d\n", signo);
#if defined(__linux__) || defined(__APPLE__)
    void* bt[64];
    int n = backtrace(bt, static_cast<int>(std::size(bt)));
    backtrace_symbols_fd(bt, n, fileno(f));
#endif
    std::fprintf(f, "Wrote backtrace to %s\n", fname.c_str());
    std::fclose(f);
    // Also write to stderr
    std::fprintf(stderr, "Wrote crash log: %s\n", fname.c_str());
}

inline void signal_handler(int signo, siginfo_t* si, void* context) {
    (void)si;
    write_backtrace_to_file(signo, context);
    // Re-raise default to allow usual crash behavior (core, abort)
    signal(signo, SIG_DFL);
    raise(signo);
}

inline void install_crash_handler_impl() {
    if (g_crash_handler_installed.exchange(true)) return;
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
}

#endif

// Public installer. Safe to call multiple times.
inline void install_crash_handler() {
    // Honor environment opt-out
    const char* env = std::getenv("FFTFREE_DISABLE_CRASH_HANDLER");
    if (env && std::strcmp(env, "1") == 0) return;
    install_crash_handler_impl();
}

} // namespace fftfree
