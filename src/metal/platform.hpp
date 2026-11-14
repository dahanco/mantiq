// src/metal/platform.hpp - the small POSIX surface the engine uses: file opens, positional reads, anonymous mappings,
// page locking and two memory queries.  macOS and Windows spell these differently, so they are gathered here once and
// the engine calls the same names on both (a cpu build of mntq-metal beyond Apple Silicon).
#pragma once

#include <cstddef>
#include <cstdint>

// The name this build's messages carry: mntq-metal for the Metal build, mntq-cpu for the cpu one (-DMNTQ_ENABLE_CPU)
#ifndef MNTQ_EXE_NAME
#define MNTQ_EXE_NAME "mntq-metal"
#endif

#if defined(_WIN32)

// windows.h's min/max macros would eat std::min / std::max / numeric_limits<...>::max() all over the engine
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <fcntl.h>
#include <io.h>
#include <vector>
#include <windows.h>

#else

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif

#endif

namespace mntq::metal::plat {

// ---------------------------------------------------------------- files

#if defined(_WIN32)

inline int file_open_ro(const char* path) { return ::_open(path, _O_RDONLY | _O_BINARY); }
inline int file_close(int fd) { return ::_close(fd); }
inline int file_nocache(int) { return 0; }   // macOS' F_NOCACHE; ReadFile never fills the page cache's place here

// ReadFile with its own offset: it does not move the handle's file pointer, so the parallel reader's workers can share
// one fd - the guarantee POSIX pread gives.
inline int64_t file_pread(int fd, void* dst, size_t bytes, uint64_t offset) {
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED ov{};
    ov.Offset = (DWORD) (offset & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) (offset >> 32);
    DWORD got = 0;
    if (!ReadFile(h, dst, (DWORD) bytes, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return (int64_t) got;
        errno = EIO;
        return -1;
    }
    return (int64_t) got;
}

#else

inline int file_open_ro(const char* path) { return ::open(path, O_RDONLY); }
inline int file_close(int fd) { return ::close(fd); }
inline int file_nocache(int fd) {
#if defined(__APPLE__)
    return ::fcntl(fd, F_NOCACHE, 1);   // the pool is the cache; the page cache would hold a second copy
#else
    (void) fd;
    return 0;
#endif
}
inline int64_t file_pread(int fd, void* dst, size_t bytes, uint64_t offset) {
    return (int64_t) ::pread(fd, dst, bytes, (off_t) offset);
}

#endif

// ---------------------------------------------------------------- memory

#if defined(_WIN32)

inline void* map_anon(size_t bytes) {
    return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
inline void unmap(void* p, size_t) { if (p) VirtualFree(p, 0, MEM_RELEASE); }
inline bool lock(void* p, size_t bytes) { return VirtualLock(p, bytes) != 0; }
inline size_t page_size() {
    SYSTEM_INFO si{};
    GetNativeSystemInfo(&si);
    return (size_t) si.dwPageSize;
}

inline size_t free_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return SIZE_MAX;
    return (size_t) ms.ullAvailPhys;
}

// the machine's RAM, what the expert pool's budget is sized from: free_memory() counts only what is free right now,
// and Windows reclaims its standby/cache pages for memory a program writes, so the pool is not limited by it.
inline size_t total_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return SIZE_MAX;
    return (size_t) ms.ullTotalPhys;
}

// The performance cores' COUNT, as the Apple branch answers (hw.perflevel0.physicalcpu): the OS's EfficiencyClass
// tells a hybrid chip's cores apart - the higher class is the faster core (i7-12700K: class 1 -> 8 P-cores, class 0
// -> 4 E-cores).  GetActiveProcessorCount returns the LOGICAL processors instead (20 here: every core's SMT sibling
// and every E-core), and a graph bound by the memory bus runs SLOWER with each thread past the P-cores (measured
// 4.2x: 20 threads 2.1 tok/s vs 4.2x that on the P-cores).  -1: the caller falls back on hardware_concurrency().
inline int physical_cores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return -1;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len))
        return -1;
    int best = -1, n = 0;
    for (const char* p = buf.data(); p < buf.data() + len;) {
        const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
        if (e->Relationship == RelationProcessorCore) {
            const int cls = (int) e->Processor.EfficiencyClass;
            if (cls > best) { best = cls; n = 1; }
            else if (cls == best) ++n;
        }
        p += e->Size;
    }
    return n > 0 ? n : -1;
}

#else

inline void* map_anon(size_t bytes) {
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
inline void unmap(void* p, size_t bytes) { if (p) ::munmap(p, bytes); }
inline bool lock(void* p, size_t bytes) { return ::mlock(p, bytes) == 0; }
inline size_t page_size() { return (size_t) ::getpagesize(); }

#if defined(__APPLE__)

// what the Mac could give this process now: free, inactive, purgeable and speculative pages (the dense weights and the
// state are already allocated, so they are not in it)
inline size_t free_memory() {
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (::host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &n) != KERN_SUCCESS) return SIZE_MAX;
    const uint64_t pages = (uint64_t) vs.free_count + vs.inactive_count + vs.purgeable_count + vs.speculative_count;
    return (size_t) (pages * (uint64_t) ::getpagesize());
}

inline size_t total_memory() {
    int64_t mem = 0;
    size_t sz = sizeof mem;
    if (::sysctlbyname("hw.memsize", &mem, &sz, nullptr, 0) != 0 || mem <= 0) return SIZE_MAX;
    return (size_t) mem;
}

inline int physical_cores() {
    int perf = 0;
    size_t sz = sizeof perf;
    if (::sysctlbyname("hw.perflevel0.physicalcpu", &perf, &sz, nullptr, 0) != 0 || perf <= 0) return -1;
    return perf;
}

#else

inline size_t free_memory() {
    struct sysinfo si{};
    if (::sysinfo(&si) != 0) return SIZE_MAX;
    return (size_t) si.freeram * (size_t) si.mem_unit;
}

inline size_t total_memory() {
    struct sysinfo si{};
    if (::sysinfo(&si) != 0) return SIZE_MAX;
    return (size_t) si.totalram * (size_t) si.mem_unit;
}

inline int physical_cores() { return -1; }   // -1: the caller falls back on hardware_concurrency()

#endif

#endif

}  // namespace mntq::metal::plat
