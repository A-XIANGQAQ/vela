// SPDX-License-Identifier: GPL-3.0-or-later WITH AdditionRef-Vela-Qualcomm-QNN-linking-exception
// Copyright (C) 2026 the Vela author
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with this program (the file LICENSE at the top
// of the Vela repository). If not, see <https://www.gnu.org/licenses/>.
//
// Additional permission under GNU GPL version 3 section 7
//
// If you modify this Program, or any covered work, by linking or combining it with the Qualcomm AI Engine Direct (QNN)
// runtime libraries or the Qualcomm AI Engine Direct SDK header files (or a modified version of them), containing parts
// covered by the terms of the Qualcomm AI Stack License, the licensors of this Program grant you additional permission to
// convey the resulting work. This includes loading those libraries at run time (for example with dlopen()).
//
// fsr4npud: native aarch64 NPU helper daemon for the FSR 4 networks (Hexagon HTP). Runs as vela-npu.service (systemd Type=notify).
// One process serves every graph, so all of them share one cDSP fastrpc session (the cDSP has only 7). It listens on one abstract
// AF_UNIX SOCK_SEQPACKET socket per graph (protocol: fsr4npu_proto.h). Clients pass dma-buf fds (or ask the daemon to allocate one
// shared buffer), the daemon registers them with QNN as QNN_HTP_MEM_SHARED_BUFFER (the zero-copy path) and runs QnnGraph_execute per
// frame request, one at a time.
// Graph life cycle: every socket exists from the start. HELLO and ALLOC answer from the sizes in the graph file (no context needed), so
// connecting never waits for a load. The first client starts QnnContext_createFromBinary on the loader thread; the client's mem handles
// are registered and its kick is read when the load ends, so until then it sees a slow execute and its frames run on the GPU network. A
// graph without a client stays loaded for --graph-idle-ms, at most --max-loaded stay loaded (least recently used goes first), and a
// graph with a client is never unloaded. A bad or missing file fails only its own socket and is tried again after --graph-retry-ms.
// Features: --alloc auto (rpcmem on the kernel system dma-heap, else a Vulkan-exported buffer); a CDSP power vote held only
// while a client is connected; a soft and a hard execute timeout (a slow execute drops that client, only a stuck one restarts
// the daemon); ping-pong input and output slots (ALLOC F_PINGPONG / F_PINGPONG_OUT; kick bit 32 / 33, RUN v[0] bit 0 / 1);
// clean SIGTERM; --allow-gid; a listener health check.
// --backend stub: no QNN, no fastrpc; copies the input tensor into every output (exercises the full IPC/fence/dma-buf path
// without root). A stub graph file is one line: stub BYTES NOUT [EXEC_DELAY_US] [LOAD_MS]. Allocation in stub mode uses the daemon's
// own Turnip device (exportable dma-buf), in qnn mode rpcmem; --alloc memfd is for IPC tests on a machine without dma-bufs.
// Usage: fsr4npud --graphs a,b,c --graph-dir DIR [--socket-prefix fsr4npud_] [--max-loaded 3] [--graph-idle-ms 120000] [--graph-retry-ms 10000]
//                 [--load-hard-timeout-ms 20000]            (or one graph, loaded at the start: --name N --ctx lite_x.bin)
//                 [--backend stub|qnn] [--alloc auto|rpcmem|vk|memfd] [--exec-timeout-ms 2000]
//                 [--exec-hard-timeout-ms 10000] [--power-grace-ms 2000] [--power-corner turbo] [--stub-bytes 8294400] [--stub-nout 1]
//                 [--stub-delay-us 0] [--stub-load-ms 0] [--max-clients 4] [--allow-uid U]... [--exit-after-clients K] [--max-seconds S] [--qnn-log L]
//                 [--power-live PATH|off] [--power-stats-jobs N] [--graph-set auto|released|low]
// Governor: the daemon steps the core corner by the measured slack between the start corner (--power-corner, default turbo) and
// svs_plus (see "R4 governor" below). FSR4NPUD_POWER_GOV=0 or the control-file key gov=0: the fixed vote of package 15.
// Control-file keys: gov=0|1 govmin=<corner >= svs_plus> govdown=<us> govup=<us> govq=<1..50> govrails=core|both govfloor=<us, 0 = auto>.
// PACKAGE 16: the core power corner follows the frame chain's slack (governor, on by default; FSR4NPUD_POWER_GOV=0 turns it off).
// For tests the power vote can be changed while a client runs, through a control file (env FSR4NPUD_POWER_LIVE or
// --power-live <absolute path>; off by default). Keys: core= bus= coremin= busmin= (voltage corners, the
// --power-corner names), sleep=0|1, sleeplat=<us>, dcvs=0|1, mode=perf|duty|updown|uponly|saver|saver_aggr, hmx=<corner>|default|off,
// hmxperf=high|low. Missing keys = the start values (--power-corner, sleep disabled, latency 40, no HMX vote = package 15).
// The file is stat()ed at most every 200 ms; a change re-sends the vote while voted ON and logs "power_vote LIVE ...". A bad
// file is ignored (one "power_live BAD" line) and the settings stay. Every --power-stats-jobs executes (and at each change) a
// "power_win" line gives the execute times of that window next to the settings it ran with. See NPUPWR/NOTES.md.
#define _GNU_SOURCE 1
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpMem.h"
extern "C" {
#include "fsr4npu_proto.h"
}
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
using namespace std;

static double now_ms() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static double T0;
static void logf(const char* fmt, ...) { va_list a; va_start(a, fmt); fprintf(stderr, "[%9.3f] ", (now_ms() - T0) / 1e3); vfprintf(stderr, fmt, a); fputc('\n', stderr); va_end(a); }
static uint64_t fnv64(const uint8_t* p, size_t n) { uint64_t h = 1469598103934665603ULL; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; } return h; }
static void sdNotify(const char* msg) {   // systemd Type=notify, no libsystemd dependency
  const char* e = getenv("NOTIFY_SOCKET"); if (!e || !*e || (e[0] != '/' && e[0] != '@')) return;
  int s = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0); if (s < 0) return;
  sockaddr_un a{}; a.sun_family = AF_UNIX; size_t n = strlen(e); if (n >= sizeof a.sun_path) { close(s); return; }
  memcpy(a.sun_path, e, n); if (a.sun_path[0] == '@') a.sun_path[0] = 0;
  if (sendto(s, msg, strlen(msg), MSG_NOSIGNAL, (sockaddr*)&a, (socklen_t)(offsetof(sockaddr_un, sun_path) + n)) < 0) {}
  close(s); }
static volatile sig_atomic_t gStop = 0;
static void onSig(int) { gStop = 1; }

// ---------------- options
static string optName = FSR4NPU_DEFAULT_NAME, optBackend = "stub", optCtx;
static uint64_t optStubBytes = 8294400; static int optStubNout = 1, optStubDelayUs = 0, optStubLoadMs = 0, optMaxClients = 4, optExitAfter = 0, optMaxSec = 0, optQnnLog = 0;
static vector<uid_t> allowUid; static vector<gid_t> allowGid;
// 3 loaded graphs: a game uses the Performance and Ultra Performance graphs of one output size and may pass through a second size.
static string optGraphs, optGraphDir, optSocketPrefix = "fsr4npud_"; static int optMaxLoaded = 3, optGraphIdleMs = 120000, optGraphRetryMs = 10000, optLoadHardMs = 20000;
// --alloc vk|rpcmem. Without /dev/dma_heap, rpcmem is FASTRPC_IOCTL_ALLOC_DMA_BUFF: the dma-buf's release
// (fastrpc_release -> fastrpc_buf_free) dereferences buf->fl->cctx, and buf->fl is freed with the fastrpc session -> a client
// or GPU import outliving the daemon's session is a use-after-free (seen as a kernel Oops). vk = a Turnip-exported
// dma-buf (msm GEM), mapped into the DSP by QNN's SHARED_BUFFER registration: its lifetime no longer depends on fastrpc.
// Default auto = rpcmem when /dev/dma_heap/system is openable (libcdsprpc then allocates from the kernel system dma-heap;
// in testing: deterministic output, 0 of 119 reruns differed, no Oops in 10 teardowns), verified by the dma-buf exporter name; else vk.
static string optAlloc = "auto"; static bool gAllocVk = false, gAllocReady = true, gMemfd = false; static string gAllocName = "?";
// Hung-HTP watchdog. A QnnGraph_execute blocked in the fastrpc ioctl cannot be cancelled in-process (a graph can wedge the
// DSP session), so on timeout the daemon tells the in-flight client (M_ERROR E_TIMEOUT / done eventfd 2+11), logs, and
// _exits 75; process exit closes the fastrpc fd so the kernel tears the DSP session down, and systemd (Restart=on-failure)
// starts a fresh daemon + QNN context. Clients see E_TIMEOUT/E_DISCONNECTED and fall back.
// Two stages. Soft (--exec-timeout-ms): the in-flight client is told E_TIMEOUT at once (falls back to the GPU); if the
// execute then returns, only that client is dropped (its late result can no longer be matched) and the daemon keeps its
// DSP session (no teardown, no restart). Hard (--exec-hard-timeout-ms): the execute never returned -> _exit 75.
// 3 soft timeouts within 120 s = a sick session -> ordered exit 75 (fresh session via systemd).
static int optExecTimeoutMs = 2000, optExecHardMs = 10000, gExitCode = 0;
static std::atomic<uint64_t> gExecId{0}, gExecTimedOutId{0}; static double gSoftT[3] = {-1e18, -1e18, -1e18}; static int gSoftN = 0;
// Power vote: the --power-corner voltage corner (default turbo 0x80) + DSP sleep disabled only while >= 1 client is connected;
// released --power-grace-ms after the last client left (a layer reconnect on swapchain re-creation does not flap the vote).
static int optPowerGraceMs = 2000; static bool gVoted = false; static double gIdleSince = 0; static uint32_t gPcid = 0;
static QnnHtpDevice_PerfInfrastructure_t* gPerf = nullptr;
// --power-corner (or env FSR4NPUD_POWER_CORNER; the option wins): the core+bus voltage corner of the ON vote, one of max (0xA0, the vote
// of packages <= 14), turbo_plus, turbo, nom_plus, nom, svs_plus, svs, svs2 or dcvs; dcvs = no corner vote (DCVS_VOLTAGE_CORNER_DISABLE),
// DCVS on, mode ADJUST_UP_DOWN (the header has no BALANCED); sleep disabled + latency 40 as for every corner.
// Default turbo, not max: RP6 (SM8550) DRG 1080p FSR 4 +6.5 % fps (HTP 8.92 -> 9.66 ms, GPU keeps a higher clock, less SoC heat), Manor Lords 720p no loss.
static string optPowerCorner = "turbo"; static uint32_t gCornerVal = DCVS_VOLTAGE_VCORNER_TURBO; static bool gCornerDcvs = false;
static bool powerCornerParse(const string& n) {
  static const struct { const char* n; uint32_t v; } T[] = {{"max", DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER}, {"turbo_plus", DCVS_VOLTAGE_VCORNER_TURBO_PLUS},
    {"turbo", DCVS_VOLTAGE_VCORNER_TURBO}, {"nom_plus", DCVS_VOLTAGE_VCORNER_NOM_PLUS}, {"nom", DCVS_VOLTAGE_VCORNER_NOM}, {"svs_plus", DCVS_VOLTAGE_VCORNER_SVS_PLUS},
    {"svs", DCVS_VOLTAGE_VCORNER_SVS}, {"svs2", DCVS_VOLTAGE_VCORNER_SVS2}, {"dcvs", DCVS_VOLTAGE_CORNER_DISABLE}};
  for (auto& t : T) if (n == t.n) { optPowerCorner = n; gCornerVal = t.v; gCornerDcvs = n == "dcvs"; return true; } return false; }

// ---------------- NPUPWR: live power settings (main thread only: the control file is read, and the vote sent, between two executes)
static const struct { const char* n; uint32_t v; } kCorner[] = {{"max", DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER}, {"turbo_plus", DCVS_VOLTAGE_VCORNER_TURBO_PLUS},
  {"turbo", DCVS_VOLTAGE_VCORNER_TURBO}, {"nom_plus", DCVS_VOLTAGE_VCORNER_NOM_PLUS}, {"nom", DCVS_VOLTAGE_VCORNER_NOM}, {"svs_plus", DCVS_VOLTAGE_VCORNER_SVS_PLUS},
  {"svs", DCVS_VOLTAGE_VCORNER_SVS}, {"svs2", DCVS_VOLTAGE_VCORNER_SVS2}, {"dcvs", DCVS_VOLTAGE_CORNER_DISABLE}};
// HMX_V2 takes the "expanded" corners (QnnHtpPerfInfrastructure_ExpVoltageCorner_t). The --power-corner names map to them as HAP does
// (svs2 = LOW_SVS, svs_plus = SVS_L1, nom_plus = NOM_L1, turbo_plus = TUR_L1); the expanded names themselves are accepted too.
static const struct { const char* n; uint32_t v; } kExpCorner[] = {{"max", DCVS_EXP_VCORNER_MAX}, {"turbo_plus", DCVS_EXP_VCORNER_TUR_L1}, {"turbo", DCVS_EXP_VCORNER_TUR},
  {"nom_plus", DCVS_EXP_VCORNER_NOM_L1}, {"nom", DCVS_EXP_VCORNER_NOM}, {"svs_plus", DCVS_EXP_VCORNER_SVS_L1}, {"svs", DCVS_EXP_VCORNER_SVS}, {"svs2", DCVS_EXP_VCORNER_LOW_SVS},
  {"min", DCVS_EXP_VCORNER_MIN}, {"low_svs_d2", DCVS_EXP_VCORNER_LOW_SVS_D2}, {"low_svs_d1", DCVS_EXP_VCORNER_LOW_SVS_D1}, {"low_svs", DCVS_EXP_VCORNER_LOW_SVS},
  {"low_svs_l0", DCVS_EXP_VCORNER_LOW_SVS_L0}, {"svs_l0", DCVS_EXP_VCORNER_SVS_L0}, {"svs_l1", DCVS_EXP_VCORNER_SVS_L1}, {"svs_l2", DCVS_EXP_VCORNER_SVS_L2},
  {"nom_l1", DCVS_EXP_VCORNER_NOM_L1}, {"nom_l2", DCVS_EXP_VCORNER_NOM_L2}, {"tur", DCVS_EXP_VCORNER_TUR}, {"tur_l1", DCVS_EXP_VCORNER_TUR_L1},
  {"tur_l2", DCVS_EXP_VCORNER_TUR_L2}, {"tur_l3", DCVS_EXP_VCORNER_TUR_L3}};
static const struct { const char* n; uint32_t v; } kMode[] = {{"perf", QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE}, {"duty", QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_DUTY_CYCLE_MODE},
  {"updown", QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN}, {"uponly", QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_ONLY_UP},
  {"saver", QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE}, {"saver_aggr", QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_AGGRESSIVE_MODE}};
enum { HMX_OFF = -1, HMX_DEFAULT = -2 };   // off: no HMX_V2 config at all (package 15); default: hmxPickDefault 1 (HMX scales with the DCVS corner)
struct PowerSet { uint32_t core, bus, coreMin, busMin; int sleep; uint32_t sleepLat; int dcvs; uint32_t mode; int hmx; int hmxLow;
  int gov; uint32_t govMin; int govDown, govUp, govQ, govRails, govFloor; };   // R4 governor: on, lowest corner, margins (us), gap percentile, 0 core / 1 core+bus, floor us (0 = auto)
static PowerSet gPS, gPS0; static bool gPSFile = false;   // gPS0 = the start values; gPSFile: gPS came from the control file
static unsigned gPSGen = 0; static unsigned long gPSRc = 0; static bool gPSRcKnown = false; static bool gHmxVoted = false;
static string optPowerLive = ""; static int optPowerStatsJobs = -1;   // package 16: no control file by default; statistics: -1 = 600 with a control file, else off
static const char* cName(uint32_t v) { for (auto& t : kCorner) if (t.v == v) return t.n; return "?"; }
static const char* eName(int v) { if (v == HMX_OFF) return "off"; if (v == HMX_DEFAULT) return "default"; for (auto& t : kExpCorner) if ((int)t.v == v) return t.n; return "?"; }
static const char* mName(uint32_t v) { for (auto& t : kMode) if (t.v == v) return t.n; return "?"; }
// R4 governor start values (env FSR4NPUD_POWER_GOV, _GOV_DOWN_US, _GOV_UP_US, _GOV_Q, _GOV_RAILS, _GOV_FLOOR_US, _GOV_MIN); the rules: "R4 governor" below
static int optGov = 1, optGovDownUs = 500, optGovUpUs = 150, optGovQ = 25, optGovRails = 0, optGovFloorUs = 0; static uint32_t optGovMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
static PowerSet powerDefaults() {   // exactly the package-15 ON vote for the --power-corner value
  PowerSet s; s.core = s.bus = s.coreMin = s.busMin = gCornerVal; s.sleep = 0; s.sleepLat = 40; s.dcvs = gCornerDcvs ? 1 : 0;
  s.mode = gCornerDcvs ? QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN : QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE; s.hmx = HMX_OFF; s.hmxLow = 0;
  s.gov = (optGov && !gCornerDcvs) ? 1 : 0; s.govMin = optGovMin; s.govDown = optGovDownUs; s.govUp = optGovUpUs; s.govQ = optGovQ; s.govRails = optGovRails; s.govFloor = optGovFloorUs; return s; }
static bool psEq(const PowerSet& a, const PowerSet& b) { return a.core == b.core && a.bus == b.bus && a.coreMin == b.coreMin && a.busMin == b.busMin && a.sleep == b.sleep
  && a.sleepLat == b.sleepLat && a.dcvs == b.dcvs && a.mode == b.mode && a.hmx == b.hmx && a.hmxLow == b.hmxLow
  && a.gov == b.gov && (!a.gov || (a.govMin == b.govMin && a.govDown == b.govDown && a.govUp == b.govUp && a.govQ == b.govQ && a.govRails == b.govRails && a.govFloor == b.govFloor)); }
static string psDesc(const PowerSet& s) {
  char b[480]; char cm[40] = "", bm[40] = "";
  if (s.coreMin != s.core) snprintf(cm, sizeof cm, " (min %s 0x%X)", cName(s.coreMin), s.coreMin);
  if (s.busMin != s.bus) snprintf(bm, sizeof bm, " (min %s 0x%X)", cName(s.busMin), s.busMin);
  snprintf(b, sizeof b, "core %s 0x%X%s bus %s 0x%X%s sleep %d sleeplat %u dcvs %d mode %s hmx %s%s", cName(s.core), s.core, cm, cName(s.bus), s.bus, bm, s.sleep, s.sleepLat, s.dcvs,
           mName(s.mode), eName(s.hmx), s.hmx >= 0 ? (s.hmxLow ? " hmxperf low" : " hmxperf high") : "");
  if (s.gov) { size_t L = strlen(b); snprintf(b + L, sizeof b - L, " gov 1 (min %s down %d up %d q %d rails %s floor %d)", cName(s.govMin), s.govDown, s.govUp, s.govQ, s.govRails ? "both" : "core", s.govFloor); }
  return b; }
// Parses the control file text into o (missing keys = start values). Returns "" when valid, else the reason.
static string psParse(const string& txt, PowerSet& o) {
  PowerSet s = gPS0; bool seen[17] = {}; static const char* K[] = {"core", "bus", "coremin", "busmin", "sleep", "sleeplat", "dcvs", "mode", "hmx", "hmxperf",
    "gov", "govmin", "govdown", "govup", "govq", "govrails", "govfloor"};
  size_t i = 0, n = txt.size();
  while (i < n) {
    if (txt[i] == '#') { while (i < n && txt[i] != '\n') i++; continue; }
    if (isspace((unsigned char)txt[i])) { i++; continue; }
    size_t j = i; while (j < n && !isspace((unsigned char)txt[j]) && txt[j] != '#') j++;
    string tok = txt.substr(i, j - i); i = j; size_t eq = tok.find('=');
    if (eq == string::npos || eq == 0) return "token '" + tok.substr(0, 40) + "' is not key=value";
    string k = tok.substr(0, eq), v = tok.substr(eq + 1); int ki = -1; for (int q = 0; q < 17; q++) if (k == K[q]) ki = q;
    if (ki < 0) return "unknown key '" + k.substr(0, 40) + "'";
    if (seen[ki]) return "key '" + k + "' given twice"; seen[ki] = true;
    if (ki <= 3) { bool ok = false; uint32_t cv = 0; for (auto& t : kCorner) if (v == t.n) { ok = true; cv = t.v; } if (!ok) return k + "=" + v.substr(0, 40) + ": not a corner (max turbo_plus turbo nom_plus nom svs_plus svs svs2 dcvs)";
      if (ki == 0) s.core = cv; else if (ki == 1) s.bus = cv; else if (ki == 2) s.coreMin = cv; else s.busMin = cv; }
    else if (ki == 4 || ki == 6) { if (v != "0" && v != "1") return k + "=" + v.substr(0, 40) + ": must be 0 or 1"; (ki == 4 ? s.sleep : s.dcvs) = v[0] - '0'; }
    else if (ki == 5) { if (v.empty() || v.size() > 5 || v.find_first_not_of("0123456789") != string::npos) return "sleeplat=" + v.substr(0, 40) + ": must be 1..65535 (us)";
      unsigned long u = strtoul(v.c_str(), nullptr, 10); if (u < 1 || u > 65535) return "sleeplat=" + v + ": must be 1..65535 (us)"; s.sleepLat = (uint32_t)u; }
    else if (ki == 7) { bool ok = false; for (auto& t : kMode) if (v == t.n) { ok = true; s.mode = t.v; } if (!ok) return "mode=" + v.substr(0, 40) + ": must be perf, duty, updown, uponly, saver or saver_aggr"; }
    else if (ki == 8) { if (v == "off") s.hmx = HMX_OFF; else if (v == "default") s.hmx = HMX_DEFAULT;
      else { bool ok = false; for (auto& t : kExpCorner) if (v == t.n) { ok = true; s.hmx = (int)t.v; } if (!ok) return "hmx=" + v.substr(0, 40) + ": not a corner name, default or off"; } }
    else if (ki == 9) { if (v != "high" && v != "low") return "hmxperf=" + v.substr(0, 40) + ": must be high or low"; s.hmxLow = v == "low"; }
    else if (ki == 10) { if (v != "0" && v != "1") return "gov=" + v.substr(0, 40) + ": must be 0 or 1"; s.gov = v[0] - '0'; }
    else if (ki == 11) { bool ok = false; for (auto& t : kCorner) if (v == t.n && t.v >= DCVS_VOLTAGE_VCORNER_SVS_PLUS && t.v != DCVS_VOLTAGE_CORNER_DISABLE) { ok = true; s.govMin = t.v; }
      if (!ok) return "govmin=" + v.substr(0, 40) + ": must be svs_plus, nom, nom_plus, turbo, turbo_plus or max (never below svs_plus)"; }
    else if (ki == 15) { if (v != "core" && v != "both") return "govrails=" + v.substr(0, 40) + ": must be core or both"; s.govRails = v == "both"; }
    else { if (v.empty() || v.size() > 5 || v.find_first_not_of("0123456789") != string::npos) return k + "=" + v.substr(0, 40) + ": must be a number";
      const int u = atoi(v.c_str()); if (ki == 14) { if (u < 1 || u > 50) return "govq=" + v + ": must be 1..50"; s.govQ = u; }
      else if (u > 50000) return k + "=" + v + ": must be 0..50000 (us)"; else if (ki == 12) s.govDown = u; else if (ki == 13) s.govUp = u; else s.govFloor = u; }
  }
  // follow-ups of the given keys: a min not given = its target; DCVS on when a rail has no corner vote (dcvs), unless dcvs= says so
  const uint32_t DIS = DCVS_VOLTAGE_CORNER_DISABLE;
  if (!seen[2]) s.coreMin = s.core; if (!seen[3]) s.busMin = s.bus;
  if ((s.core == DIS) != (s.coreMin == DIS)) return "core and coremin: dcvs (no corner vote) must be on both or neither";
  if ((s.bus == DIS) != (s.busMin == DIS)) return "bus and busmin: dcvs (no corner vote) must be on both or neither";
  if (s.core != DIS && s.coreMin > s.core) return string("coremin ") + cName(s.coreMin) + " is above core " + cName(s.core);
  if (s.bus != DIS && s.busMin > s.bus) return string("busmin ") + cName(s.busMin) + " is above bus " + cName(s.bus);
  const bool anyDis = s.core == DIS || s.bus == DIS;
  if (!seen[6]) s.dcvs = (anyDis || gPS0.dcvs) ? 1 : 0; else if (anyDis && !s.dcvs) return "dcvs=0 with a rail at dcvs (no corner vote): nothing would set its clock";
  if (!seen[7]) s.mode = s.dcvs ? QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN : QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  if (seen[9] && s.hmx < 0) return "hmxperf needs hmx=<corner>";
  if (s.gov && (s.core == DIS || s.dcvs)) { if (seen[10]) return "gov=1 needs a core corner and dcvs=0 (the governor pins the corner itself)"; s.gov = 0; }   // gov from the start values: off for such a file
  o = s; return ""; }
// ---------------- R4 governor (AUDIT/REPORT3 A1, AUDIT/REPORT4 part 2): the core corner follows the slack. Main thread only.
// On by default (package 16); off with env FSR4NPUD_POWER_GOV=0 or the control-file key gov=0.
// Signal: the idle gap of each job, gap(N) = kick(N+1) read - done(N) written, both on this daemon's clock (onKick / onRun).
// In the layer's exact mode the GPU runs WAIT HTP(N) -> K2(N) -> pre(N+1) -> E1 -> kick(N+1), so
//   gap = (GPU chain K2 + pre, the "floor") + (how long the result lay unused before the GPU asked for it, the "slack").
// A gap at the floor = the GPU waited for the NPU (a lower corner makes the frame longer); a gap above it = free time.
// The daemon cannot see the floor (it is GPU time), so it learns it:
//   - any gap ever seen is an upper bound (ring of the last 20 windows' p10: "low gap");
//   - a step whose exec change shows up in the kick period (down: period grows, up: period shrinks) proves a stall; the gap of
//     the stalled window IS the floor ("learned", kept 120 s);
//   - until then a prior: FSR4NPUD_POWER_GOV_FLOOR_PCT (40) % of the exec time at the top corner, or govfloor=<us>.
// Rules per window (>= 20 jobs and >= 500 ms): slack = gap p<govq> - floor.
//   down one corner: slack - predicted exec growth >= govdown us in 4 windows in a row; the window after the step is checked:
//                    kick period grew by more than max(0.3 ms, half the exec growth) -> back up at once, floor learned, hold-off doubles (2..60 s);
//   up one corner:   floor learned and slack < govup us; or floor not learned and the gap sits at its recent minimum below the
//                    level last seen without a stall -> an "up probe" (period shrinks = it was a stall; else come back down later).
// Limits: never below svs_plus (govmin=), never above the start corner (--power-corner, the package default) or the file's core=.
// One client only (two clients, a pause > 2 s or a new client: top corner, start again). One "power_gov STEP" line per step.
static int optGovLog = 0, optGovFloorPct = 40, optGovWinJobs = 20, optGovWinMs = 500, optGovDownWins = 4;   // the other governor options are defined above powerDefaults()
static const uint32_t kGovLadder[] = {DCVS_VOLTAGE_VCORNER_SVS_PLUS, DCVS_VOLTAGE_VCORNER_NOM, DCVS_VOLTAGE_VCORNER_NOM_PLUS, DCVS_VOLTAGE_VCORNER_TURBO,
  DCVS_VOLTAGE_VCORNER_TURBO_PLUS, DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER};
enum { GOV_NL = 6, GOV_RING = 20 };
// exec time at ladder[i-1] / exec time at ladder[i]: a guess used only until corner i-1 was visited (then the measured time).
// Measured on RP6 with core = bus: max -> turbo 1.08, turbo -> nom 1.11..1.19 (TRIED 2026-11-23); nom -> svs_plus is not measured.
static const double kGovRatio[GOV_NL] = {0, 1.20, 1.06, 1.06, 1.04, 1.04};
static int govIdx(uint32_t corner) { for (int i = 0; i < GOV_NL; i++) if (kGovLadder[i] == corner) return i; return -1; }
struct Gov {
  bool active = false; int lo = 0, hi = 0, k = 0, client = -1, skip = 0, okWins = 0, lowN = 0, lowI = 0; double lastKick = 0, lastDone = 0, winT0 = 0;
  vector<double> vG, vP, vH; double H[GOV_NL] = {}, Ht[GOV_NL] = {}, low[GOV_RING] = {};
  double floorEx = 0, floorExT = -1e18;                       // the floor learned at a proven stall
  double verG[GOV_NL] = {}, verT[GOV_NL] = {};                 // per corner: a gap level (window p10) seen without a stall (and when)
  double holdUntil = 0, backoffMs = 2000, lastUpT = -1e18;    // after a failed step down
  double upProbeAt = 0, upBackoffMs = 4000;                   // after an up probe that found no stall
  int probe = 0; double pP = 0, pPm = 0, pH = 0, pG = 0, pGlo = 0;               // -1 / +1: the last step is checked in the next window against these
  unsigned nStep = 0, nWin = 0; };
static Gov gGov;
static void govReset() {   // new settings, a new ON vote, a new client: top corner, nothing learned
  Gov n; n.nStep = gGov.nStep; n.nWin = gGov.nWin; gGov = n;
  if (!gPS.gov || gPS.core == DCVS_VOLTAGE_CORNER_DISABLE) return;
  const uint32_t top = (gPS0.core != DCVS_VOLTAGE_CORNER_DISABLE && gPS0.core < gPS.core) ? gPS0.core : gPS.core;   // never above the start corner
  gGov.hi = govIdx(top); if (gGov.hi < 0) return;   // the top is below svs_plus: nothing to steer
  gGov.lo = govIdx(gPS.govMin); if (gGov.lo < 0) gGov.lo = 0; if (gGov.lo > gGov.hi) gGov.lo = gGov.hi;
  gGov.active = true; gGov.k = gGov.hi; }
// The vote actually sent: the control-file settings, with the core corner (rails both: and a higher bus corner) replaced by the governor's.
static PowerSet psEff() {
  PowerSet e = gPS; if (!gGov.active) return e;
  e.core = e.coreMin = kGovLadder[gGov.k];
  if (gPS.govRails && e.bus != DCVS_VOLTAGE_CORNER_DISABLE && e.bus > e.core) { e.bus = e.core; if (e.busMin > e.bus) e.busMin = e.bus; }
  return e; }
static void govLogCfg(const char* src);
// stub backend only: the stub's execute delay scales with the effective core corner, so the governor can be tested off-device.
// FSR4NPUD_STUB_CORNER_PCT="turbo=100,nom_plus=106,..." replaces entries of the table (percent of --stub-delay-us).
static int gStubPct[GOV_NL + 2] = {135, 112, 106, 100, 96, 92, 160, 200};   // ladder order, then svs, svs2
static int stubCornerPct() { const uint32_t c = psEff().core; const int i = govIdx(c); if (i >= 0) return gStubPct[i];
  return c == DCVS_VOLTAGE_VCORNER_SVS ? gStubPct[GOV_NL] : c == DCVS_VOLTAGE_VCORNER_SVS2 ? gStubPct[GOV_NL + 1] : 100; }
// ---------------- NPUPWR: per-window execute statistics with the settings they ran under
static vector<double> gWinEx, gWinSvc; static double gWinT0 = 0; static unsigned gWinN = 0;
static double wall_s() { timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static void winFlush(const char* why) {
  if (gWinEx.empty()) { gWinT0 = now_ms(); return; }
  vector<double> e = gWinEx, v = gWinSvc; sort(e.begin(), e.end()); sort(v.begin(), v.end()); double sum = 0; for (double x : e) sum += x; const double dur = (now_ms() - gWinT0) / 1e3;
  char rc[32]; if (gPSRcKnown) snprintf(rc, sizeof rc, "0x%lx", gPSRc); else snprintf(rc, sizeof rc, "n/a");
  logf("power_win %u gen %u jobs %zu over %.2f s (%.1f /s) exec mean %.3f p50 %.3f p90 %.3f max %.3f ms service p50 %.3f p90 %.3f ms | %s | %s rc %s | end %s | wall %.3f",
       ++gWinN, gPSGen, e.size(), dur, dur > 0 ? e.size() / dur : 0.0, sum / e.size(), e[e.size() / 2], e[e.size() * 9 / 10], e.back(), v.empty() ? 0.0 : v[v.size() / 2], v.empty() ? 0.0 : v[v.size() * 9 / 10],
       psDesc(psEff()).c_str(), gPSFile ? "file" : "start values", rc, why, wall_s());
  gWinEx.clear(); gWinSvc.clear(); gWinT0 = now_ms(); }
static void winAdd(double ex, double svc) {
  if (optPowerStatsJobs <= 0) return; if (gWinEx.empty() && gWinT0 == 0) gWinT0 = now_ms();
  gWinEx.push_back(ex); gWinSvc.push_back(svc); if ((int)gWinEx.size() >= optPowerStatsJobs) winFlush("window full"); }
static std::atomic<double> gExecStart{0}; static std::atomic<int> gExecSock{-1}, gExecDone{-1}; static std::atomic<uint64_t> gExecSeq{0};

// ---------------- graphs (both backends)
struct Graph {
  string sock, path, name = "stub"; int ls = -1; bool keep = false;   // keep: --name mode, loaded at the start and never unloaded
  enum State { IDLE, LOADING, LOADED, UNLOADING, FAILED } st = IDLE; bool infoReady = false, reload = false, evictRetried = false; double retryAt = 0, lastUse = 0;
  uint32_t nt = 0; uint64_t bytes[FSR4NPU_MAXT] = {}; int stubDelayUs = -1; int pins = 0;   // pins: connected clients
  // valid while LOADED; the loader thread owns them during LOADING and UNLOADING
  Qnn_GraphHandle_t G = nullptr; Qnn_ContextHandle_t CTX = nullptr; QnnSystemContext_Handle_t sc = nullptr; vector<uint8_t> bin; Qnn_Tensor_t T[FSR4NPU_MAXT];
  // ANF graph family (one context holding graphs <name>_p0 .. _p<ng-1>): ng > 1; the phase of a frame picks GP[phase] / TP[phase]. ng == 1: unused.
  int ng = 1; Qnn_GraphHandle_t GP[64] = {}; Qnn_Tensor_t TP[64][2];
};
static vector<Graph*> graphs; static bool isQnn = false;
static bool optLoadConc = false;   // experiment: contexts load without gQnnMu

// ---------------- QNN state
// The QNN documents do not say that QnnContext_createFromBinary or QnnContext_free may run beside QnnGraph_execute (only QnnBackend_create is called
// thread-safe), so every call into the backend holds gQnnMu. A frame that meets a load waits for it and goes to the GPU network by the slow-execute rule.
static QNN_INTERFACE_VER_TYPE Q{}; static QNN_SYSTEM_INTERFACE_VER_TYPE S{}; static Qnn_BackendHandle_t BE = nullptr; static Qnn_DeviceHandle_t DV = nullptr; static std::mutex gQnnMu;
typedef void (*RInit)(); typedef void (*RDeinit)(); typedef void* (*RAlloc)(int, uint32_t, int); typedef int (*RFd)(void*); typedef void (*RFree)(void*);
typedef int (*RegDma)(int, uint32_t);
static RAlloc rAlloc = nullptr; static RFd rToFd = nullptr; static RFree rFree = nullptr; static RDeinit rDeinit = nullptr; static RegDma rRegDma = nullptr;

static int dtypeBytes(Qnn_DataType_t d) { unsigned x = (unsigned)d & 0xff; int bits = ((x >> 4) & 0xf) * 10 + (x & 0xf); return bits >= 8 ? bits / 8 : 1; }
static uint64_t tensorBytes(const Qnn_Tensor_t& t) { uint64_t n = dtypeBytes(t.v1.dataType); for (uint32_t k = 0; k < t.v1.rank; k++) n *= t.v1.dimensions[k]; return n; }

#define QK(x) do { Qnn_ErrorHandle_t e_ = (x); if (e_ != QNN_SUCCESS) { logf("FAIL %s = 0x%lx", #x, (unsigned long)e_); return false; } } while (0)
static bool qnnOpen() {   // backend, device, power id, rpcmem: the fastrpc session of this process; no graph
  double t0 = now_ms();
  void* hs = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL); void* hb = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL); void* hr = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (!hs || !hb || !hr) { logf("FAIL dlopen %s", dlerror()); return false; }
  typedef Qnn_ErrorHandle_t (*SysProv)(const QnnSystemInterface_t***, uint32_t*); typedef Qnn_ErrorHandle_t (*Prov)(const QnnInterface_t***, uint32_t*);
  const QnnSystemInterface_t** sp; const QnnInterface_t** ip; uint32_t ns = 0, ni = 0;
  QK(((SysProv)dlsym(hs, "QnnSystemInterface_getProviders"))(&sp, &ns)); QK(((Prov)dlsym(hb, "QnnInterface_getProviders"))(&ip, &ni));
  S = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME; Q = ip[0]->QNN_INTERFACE_VER_NAME;
  Qnn_LogHandle_t lgh = nullptr;
  if (optQnnLog > 0) Q.logCreate([](const char* fmt, QnnLog_Level_t lv, uint64_t, va_list a) { fprintf(stderr, "QNNLOG%d ", (int)lv); vfprintf(stderr, fmt, a); size_t L = strlen(fmt); if (!L || fmt[L - 1] != '\n') fputc('\n', stderr); }, (QnnLog_Level_t)optQnnLog, &lgh);
  QK(Q.backendCreate(lgh, nullptr, &BE)); Q.deviceCreate(nullptr, nullptr, &DV);
  { QnnDevice_Infrastructure_t inf = nullptr;   // create the power config id only; the vote follows the clients (powerVote)
    if (Q.deviceGetInfrastructure(&inf) == QNN_SUCCESS && inf) { gPerf = &((QnnHtpDevice_Infrastructure_t*)inf)->perfInfra; QK(gPerf->createPowerConfigId(0, 0, &gPcid));
      logf("power: config id %u, no vote until a client connects", gPcid); } else logf("power: no HTP perf infrastructure, no votes"); }
  ((RInit)dlsym(hr, "rpcmem_init"))(); rAlloc = (RAlloc)dlsym(hr, "rpcmem_alloc"); rToFd = (RFd)dlsym(hr, "rpcmem_to_fd"); rFree = (RFree)dlsym(hr, "rpcmem_free");
  rDeinit = (RDeinit)dlsym(hr, "rpcmem_deinit"); rRegDma = (RegDma)dlsym(hr, "remote_register_dma_handle");
  logf("backend open %.1f ms; rpcmem alloc %p to_fd %p free %p remote_register_dma_handle %p", now_ms() - t0, (void*)rAlloc, (void*)rToFd, (void*)rFree, (void*)rRegDma);
  return rAlloc && rToFd;
}
// What a graph file says, with no context and no DSP. sc stays alive with T[]: their dimensions point into it.
struct Parsed { int ng = 1; string names[64]; Qnn_Tensor_t TP[64][2]; vector<uint8_t> bin; QnnSystemContext_Handle_t sc = nullptr; Qnn_Tensor_t T[FSR4NPU_MAXT]; uint32_t nt = 0; uint64_t bytes[FSR4NPU_MAXT] = {}; string name; int stubDelayUs = -1, stubLoadMs = 0; };
static bool parseStub(const Graph* g, Parsed& o) {
  unsigned long long b = optStubBytes; int nout = optStubNout; o.stubLoadMs = optStubLoadMs; o.name = g->name;
  if (!g->path.empty()) { FILE* f = fopen(g->path.c_str(), "r"); if (!f) { logf("FAIL cannot open graph file %s", g->path.c_str()); return false; }
    int n = fscanf(f, "stub %llu %d %d %d", &b, &nout, &o.stubDelayUs, &o.stubLoadMs); fclose(f); if (n < 2) { logf("FAIL %s is not a stub graph file", g->path.c_str()); return false; } }
  if (nout < 1 || nout > FSR4NPU_MAXT - 1) return false;
  o.nt = 1 + nout; for (uint32_t k = 0; k < o.nt; k++) o.bytes[k] = b; return true; }
static bool parseGraph(const Graph* g, Parsed& o) {
  if (!isQnn) return parseStub(g, o);
  FILE* f = fopen(g->path.c_str(), "rb"); if (!f) { logf("FAIL cannot open context %s", g->path.c_str()); return false; }
  fseek(f, 0, SEEK_END); size_t bsz = ftell(f); fseek(f, 0, SEEK_SET); o.bin.resize(bsz);
  if (fread(o.bin.data(), 1, bsz, f) != bsz) { fclose(f); logf("FAIL cannot read context %s", g->path.c_str()); return false; } fclose(f);
  QK(S.systemContextCreate(&o.sc)); const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bis = 0;
  QK(S.systemContextGetBinaryInfo(o.sc, o.bin.data(), bsz, &bi, &bis));
  const QnnSystemContext_GraphInfo_t* gi = bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1 ? bi->contextBinaryInfoV1.graphs : bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2 ? bi->contextBinaryInfoV2.graphs : bi->contextBinaryInfoV3.graphs;
  const char* gname; const Qnn_Tensor_t *gin, *gout; uint32_t nOut;
  if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { gname = gi->graphInfoV1.graphName; gin = gi->graphInfoV1.graphInputs; gout = gi->graphInfoV1.graphOutputs; nOut = gi->graphInfoV1.numGraphOutputs; }
  else if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { gname = gi->graphInfoV2.graphName; gin = gi->graphInfoV2.graphInputs; gout = gi->graphInfoV2.graphOutputs; nOut = gi->graphInfoV2.numGraphOutputs; }
  else { gname = gi->graphInfoV3.graphName; gin = gi->graphInfoV3.graphInputs; gout = gi->graphInfoV3.graphOutputs; nOut = gi->graphInfoV3.numGraphOutputs; }
  if (nOut < 1 || nOut > FSR4NPU_MAXT - 1) { logf("FAIL graph has %u outputs", nOut); return false; }
  o.T[0] = gin[0];
  if (nOut == 2) { int iu = strcmp(gout[0].v1.name, "p3u_o") == 0 ? 0 : 1; o.T[1] = gout[iu]; o.T[2] = gout[1 - iu]; }   // with two outputs, the one named p3u_o goes first
  else for (uint32_t k = 0; k < nOut; k++) o.T[1 + k] = gout[k];
  o.nt = 1 + nOut; o.name = gname; for (uint32_t k = 0; k < o.nt; k++) o.bytes[k] = tensorBytes(o.T[k]);
  // ANF family: more than one graph in the context, each named <family>_p<phase> with 1 input and 1 output of the same sizes
  const uint32_t ngr = bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1 ? bi->contextBinaryInfoV1.numGraphs : bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2 ? bi->contextBinaryInfoV2.numGraphs : bi->contextBinaryInfoV3.numGraphs;
  if (ngr > 1) {
    if (ngr > 64 || nOut != 1) { logf("FAIL context has %u graphs (a family needs 2..64 graphs with 1 output)", ngr); return false; }
    for (uint32_t k = 0; k < ngr; k++) {
      const QnnSystemContext_GraphInfo_t* gk = gi + k; const char* nm; const Qnn_Tensor_t *ki, *ko; uint32_t no;
      if (gk->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { nm = gk->graphInfoV1.graphName; ki = gk->graphInfoV1.graphInputs; ko = gk->graphInfoV1.graphOutputs; no = gk->graphInfoV1.numGraphOutputs; }
      else if (gk->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { nm = gk->graphInfoV2.graphName; ki = gk->graphInfoV2.graphInputs; ko = gk->graphInfoV2.graphOutputs; no = gk->graphInfoV2.numGraphOutputs; }
      else { nm = gk->graphInfoV3.graphName; ki = gk->graphInfoV3.graphInputs; ko = gk->graphInfoV3.graphOutputs; no = gk->graphInfoV3.numGraphOutputs; }
      const char* us = strrchr(nm, '_'); int pk = (us && us[1] == 'p' && isdigit((unsigned char)us[2])) ? atoi(us + 2) : -1;
      if (pk < 0 || pk >= (int)ngr || !o.names[pk].empty() || no != 1 || tensorBytes(ki[0]) != o.bytes[0] || tensorBytes(ko[0]) != o.bytes[1]) { logf("FAIL graph %s is not a phase of a family of %u (name _p<k>, k < count, 1 output, same sizes)", nm, ngr); return false; }
      o.names[pk] = nm; o.TP[pk][0] = ki[0]; o.TP[pk][1] = ko[0];
    }
    o.ng = (int)ngr; o.name = o.names[0]; o.T[0] = o.TP[0][0]; o.T[1] = o.TP[0][1];
  }
  return true;
}
static void parsedFree(Parsed& o) { if (o.sc) S.systemContextFree(o.sc); o.sc = nullptr; vector<uint8_t>().swap(o.bin); }

// The ON vote from gPS (NPUPWR). With the start values (no control file) the DCVS_V3 struct is field for field the package-15 one.
// Caller holds gQnnMu.
static Qnn_ErrorHandle_t powerSendOn() {
  QnnHtpPerfInfrastructure_PowerConfig_t d; memset(&d, 0, sizeof d); d.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3; auto& v = d.dcvsV3Config;
  const PowerSet E = psEff();   // R4 governor: gPS with the governor's core corner; governor off: E = gPS
  v.contextId = gPcid; v.setDcvsEnable = 1; v.setSleepLatency = 1; v.setSleepDisable = 1; v.setBusParams = 1; v.setCoreParams = 1;
  v.dcvsEnable = E.dcvs ? 1 : 0; v.powerMode = (QnnHtpPerfInfrastructure_PowerMode_t)E.mode;
  v.sleepLatency = E.sleepLat; v.sleepDisable = E.sleep ? 0 : 1;   // start values: latency 40, sleep disabled (packages <= 15)
  v.busVoltageCornerMin = (QnnHtpPerfInfrastructure_VoltageCorner_t)E.busMin; v.busVoltageCornerTarget = v.busVoltageCornerMax = (QnnHtpPerfInfrastructure_VoltageCorner_t)E.bus;
  v.coreVoltageCornerMin = (QnnHtpPerfInfrastructure_VoltageCorner_t)E.coreMin; v.coreVoltageCornerTarget = v.coreVoltageCornerMax = (QnnHtpPerfInfrastructure_VoltageCorner_t)E.core;
  const QnnHtpPerfInfrastructure_PowerConfig_t* c1[] = {&d, nullptr}; return gPerf->setPowerConfig(gPcid, c1); }
// HMX_V2 in its own call after the DCVS_V3 one, so a refusal (the docs: v75 and later only; SM8550 is v73) never touches the main vote.
// hmx=off after an explicit vote sends hmxPickDefault 1 (the docs: the client must revert it itself). Caller holds gQnnMu. Returns rc, or 0 if nothing sent.
static Qnn_ErrorHandle_t powerSendHmx(bool forceDefault, bool* sent) {
  *sent = false; const bool pickDefault = forceDefault || gPS.hmx == HMX_DEFAULT || gPS.hmx == HMX_OFF;
  if (pickDefault && !gHmxVoted && (forceDefault || gPS.hmx == HMX_OFF)) return QNN_SUCCESS;   // nothing explicit to undo
  QnnHtpPerfInfrastructure_PowerConfig_t h; memset(&h, 0, sizeof h); h.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2; auto& x = h.hmxV2Config;
  x.hmxPickDefault = pickDefault ? 1 : 0; x.hmxPerfMode = gPS.hmxLow && !pickDefault ? QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_LOW : QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;
  const QnnHtpPerfInfrastructure_ExpVoltageCorner_t ec = pickDefault ? DCVS_EXP_VCORNER_DISABLE : (QnnHtpPerfInfrastructure_ExpVoltageCorner_t)gPS.hmx;
  x.hmxVoltageCornerMin = x.hmxVoltageCornerTarget = x.hmxVoltageCornerMax = ec;
  const QnnHtpPerfInfrastructure_PowerConfig_t* c1[] = {&h, nullptr}; Qnn_ErrorHandle_t e = gPerf->setPowerConfig(gPcid, c1); *sent = true;
  if (e == QNN_SUCCESS) gHmxVoted = !pickDefault; return e; }
static void powerVote(bool on, const char* why) {
  if (!gPerf || on == gVoted) return;
  std::lock_guard<std::mutex> ql(gQnnMu);
  Qnn_ErrorHandle_t e, eh = QNN_SUCCESS; bool hs = false;
  if (on) { govReset(); e = powerSendOn(); if (e == QNN_SUCCESS) eh = powerSendHmx(false, &hs); gPSRc = (unsigned long)(e != QNN_SUCCESS ? e : eh); gPSRcKnown = true; }
  else {
    if (gHmxVoted) eh = powerSendHmx(true, &hs);   // back to the default HMX vote while the HVX vote still stands
    QnnHtpPerfInfrastructure_PowerConfig_t d; memset(&d, 0, sizeof d); d.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3; auto& v = d.dcvsV3Config;
    v.contextId = gPcid; v.setDcvsEnable = 1; v.setSleepLatency = 1; v.setSleepDisable = 1; v.setBusParams = 1; v.setCoreParams = 1;
    v.dcvsEnable = 1; v.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE; v.sleepLatency = 1000; v.sleepDisable = 0;   // no corner vote: DSP DCVS decides
    v.busVoltageCornerMin = v.busVoltageCornerTarget = v.busVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
    v.coreVoltageCornerMin = v.coreVoltageCornerTarget = v.coreVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
    const QnnHtpPerfInfrastructure_PowerConfig_t* c1[] = {&d, nullptr}; e = gPerf->setPowerConfig(gPcid, c1);
    if (e != QNN_SUCCESS) {   // release refused: drop the whole vote client instead
      Qnn_ErrorHandle_t e2 = gPerf->destroyPowerConfigId(gPcid); uint32_t np = 0; Qnn_ErrorHandle_t e3 = gPerf->createPowerConfigId(0, 0, &np); if (e3 == QNN_SUCCESS) gPcid = np;
      logf("power_vote release rc 0x%lx -> destroyPowerConfigId rc 0x%lx, new id rc 0x%lx", (unsigned long)e, (unsigned long)e2, (unsigned long)e3); e = e2; gHmxVoted = false; } }
  if (e == QNN_SUCCESS) gVoted = on;
  char hx[48] = ""; if (hs) snprintf(hx, sizeof hx, ", hmx %s rc 0x%lx", on ? eName(gPS.hmx) : "default", (unsigned long)eh);
  if (on && !gPSFile && psEq(gPS, gPS0)) {   // start values: the package-15 line, word for word
    char onTxt[128]; if (gCornerDcvs) snprintf(onTxt, sizeof onTxt, "ON corner dcvs (no corner vote 0x%x), DCVS on, adjust up-down, sleep disabled", gCornerVal);
    else snprintf(onTxt, sizeof onTxt, "ON corner %s 0x%X, DCVS off, sleep disabled", optPowerCorner.c_str(), gCornerVal);
    logf("power_vote %s (%s) rc 0x%lx", onTxt, why, (unsigned long)e); }
  else if (on) logf("power_vote ON %s (control file, gen %u) (%s) rc 0x%lx%s", psDesc(gPS).c_str(), gPSGen, why, (unsigned long)e, hx);
  else logf("power_vote %s (%s) rc 0x%lx%s", "OFF released (DCVS on, power saver, sleep allowed)", why, (unsigned long)e, hx); }
// NPUPWR: new settings from the control file (or the start values when it went away): re-sent at once while voted ON.
static void powerLiveApply(const PowerSet& n, bool fromFile, const char* src) {
  winFlush("settings change");   // the window so far ran with the old settings
  gPS = n; gPSFile = fromFile; gPSGen++; govReset();
  if (isQnn && gPerf && gVoted) {
    Qnn_ErrorHandle_t e, eh = QNN_SUCCESS; bool hs = false; double t0 = now_ms();
    { std::lock_guard<std::mutex> ql(gQnnMu); e = powerSendOn(); if (e == QNN_SUCCESS) eh = powerSendHmx(false, &hs); }
    gPSRc = (unsigned long)(e != QNN_SUCCESS ? e : eh); gPSRcKnown = true;
    char hx[48] = ""; if (hs) snprintf(hx, sizeof hx, " hmx rc 0x%lx", (unsigned long)eh);
    logf("power_vote LIVE %s gen %u (%s) rc 0x%lx%s, %.2f ms | wall %.3f", psDesc(gPS).c_str(), gPSGen, src, (unsigned long)e, hx, now_ms() - t0, wall_s()); }
  else { gPSRcKnown = false;
    logf("power_vote LIVE %s gen %u (%s) rc n/a: %s | wall %.3f", psDesc(gPS).c_str(), gPSGen, src,
         !isQnn ? "stub backend, no QNN power infrastructure: not sent" : !gPerf ? "no HTP perf infrastructure: not sent" : "no client (not voted): sent with the next ON vote", wall_s()); }
  govLogCfg(src); }
// NPUPWR: the control file. A cheap lstat() at most every 200 ms; it is read only when inode, size, mtime or ctime changed.
static double gLiveNext = 0; static bool gLiveHad = false; static struct { ino_t ino; off_t size; long ms, mns, cs, cns; bool any; } gLiveSig{};
static void powerLiveCheck(bool force) {
  if (optPowerLive.empty()) return; const double tn = now_ms(); if (!force && tn < gLiveNext) return; gLiveNext = tn + 200;
  struct stat st; if (lstat(optPowerLive.c_str(), &st) != 0) {
    if (gLiveSig.any || force) { gLiveSig = {}; if (gLiveHad) { gLiveHad = false; if (gPSFile || !psEq(gPS, gPS0)) powerLiveApply(gPS0, false, "control file gone: start values"); }
      else if (force) logf("power_live: control file %s absent: start values (%s)", optPowerLive.c_str(), psDesc(gPS0).c_str()); }
    return; }
  if (gLiveSig.any && gLiveSig.ino == st.st_ino && gLiveSig.size == st.st_size && gLiveSig.ms == st.st_mtim.tv_sec && gLiveSig.mns == st.st_mtim.tv_nsec
      && gLiveSig.cs == st.st_ctim.tv_sec && gLiveSig.cns == st.st_ctim.tv_nsec) return;
  gLiveSig.ino = st.st_ino; gLiveSig.size = st.st_size; gLiveSig.ms = st.st_mtim.tv_sec; gLiveSig.mns = st.st_mtim.tv_nsec; gLiveSig.cs = st.st_ctim.tv_sec; gLiveSig.cns = st.st_ctim.tv_nsec; gLiveSig.any = true;
  string why;
  if (!S_ISREG(st.st_mode)) why = "not a regular file (a symlink or other is refused)";
  else if (st.st_uid != getuid() && st.st_uid != 0) why = "owner uid " + to_string((unsigned)st.st_uid) + " is neither the daemon's uid " + to_string((unsigned)getuid()) + " nor root";
  else if (st.st_size > 4096) why = "larger than 4096 bytes";
  string txt;
  if (why.empty()) { int fd = open(optPowerLive.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW); if (fd < 0) why = string("open: ") + strerror(errno);
    else { char b[4097]; ssize_t r = read(fd, b, 4096); close(fd); if (r < 0) why = string("read: ") + strerror(errno); else txt.assign(b, (size_t)r); } }
  PowerSet n; if (why.empty()) { why = psParse(txt, n); for (char& ch : why) if (ch == '\n' || ch == '\r') ch = ' '; }
  if (!why.empty()) { logf("power_live BAD %s: %s; ignored, settings kept: %s gen %u", optPowerLive.c_str(), why.c_str(), psDesc(gPS).c_str(), gPSGen); return; }
  gLiveHad = true;
  if (psEq(n, gPS) && gPSFile) { logf("power_live: %s re-read, same settings (gen %u)", optPowerLive.c_str(), gPSGen); return; }
  if (psEq(n, gPS)) { gPSFile = true; logf("power_live: %s read, same as the current settings (%s), nothing sent (gen %u)", optPowerLive.c_str(), psDesc(gPS).c_str(), gPSGen); return; }
  powerLiveApply(n, true, "control file"); }
// ---------------- R4 governor: the rules (see the comment at struct Gov)
static double govPct(vector<double> v, int q) { sort(v.begin(), v.end()); size_t i = v.size() * (size_t)q / 100; if (i >= v.size()) i = v.size() - 1; return v[i]; }
static void govLogCfg(const char* src) {
  static int was = -1;
  if (gPS.gov && !gGov.active) { logf("power_gov OFF (%s): gov=1 but the core corner %s has no corner vote or is below svs_plus: nothing to steer", src, cName(gPS.core)); was = 0; return; }
  if (!gGov.active) { if (was == 1) logf("power_gov OFF (%s): core back to %s 0x%X", src, cName(gPS.core), gPS.core); was = 0; return; }
  char fl[64]; if (gPS.govFloor > 0) snprintf(fl, sizeof fl, "%d us (govfloor)", gPS.govFloor); else snprintf(fl, sizeof fl, "%d %% of the exec time at the top", optGovFloorPct);
  logf("power_gov ON (%s): core between %s 0x%X and %s 0x%X, start at the top, rails %s | window >= %d jobs and >= %d ms | down: slack - predicted step >= %d us in %d windows, "
       "checked by the kick period | up: slack < %d us or an up probe | slack = gap p%d - floor; gap = kick(N+1) - done(N); floor: learned at a stall, else min(low gap, prior %s)",
       src, cName(kGovLadder[gGov.lo]), kGovLadder[gGov.lo], cName(kGovLadder[gGov.hi]), kGovLadder[gGov.hi], gPS.govRails ? "core+bus" : "core",
       optGovWinJobs, optGovWinMs, gPS.govDown, optGovDownWins, gPS.govUp, gPS.govQ, fl);
  was = 1; }
static void govClearWin() { gGov.vG.clear(); gGov.vP.clear(); gGov.vH.clear(); gGov.okWins = 0; }
// One step. Sends the vote (qnn, voted), one log line. A refused vote keeps the old corner and stops down steps for 60 s.
static bool govApply(int nk, const char* why) {
  Gov& g = gGov; const int ok_ = g.k; const uint32_t from = kGovLadder[g.k], to = kGovLadder[nk]; const char* dir = nk > g.k ? "up" : "down";
  winFlush("governor step");   // the power_win so far ran at the old corner
  g.k = nk; char rc[96]; bool ok = true;
  if (isQnn && gPerf && gVoted) { double t0 = now_ms(); Qnn_ErrorHandle_t e; { std::lock_guard<std::mutex> ql(gQnnMu); e = powerSendOn(); }
    gPSRc = (unsigned long)e; gPSRcKnown = true; ok = e == QNN_SUCCESS; snprintf(rc, sizeof rc, "rc 0x%lx, %.2f ms%s", (unsigned long)e, now_ms() - t0, ok ? "" : " REFUSED: corner kept, no down steps for 60 s"); }
  else snprintf(rc, sizeof rc, "rc n/a: %s", !isQnn ? "stub backend, not sent" : !gPerf ? "no HTP perf infrastructure" : "not voted, sent with the next ON vote");
  logf("power_gov STEP %u %s core %s 0x%X -> %s 0x%X%s | %s | %s | wall %.3f", ++g.nStep, dir, cName(from), from, cName(to), to, gPS.govRails ? " (bus follows)" : "", why, rc, wall_s());
  if (!ok) { g.k = ok_; g.holdUntil = now_ms() + 60000; g.probe = 0; }
  govClearWin(); g.skip = 2;   // the two jobs around the vote change are not measured
  return ok; }
static void govWindow(double now) {
  Gov& g = gGov; g.nWin++;
  double Pm = 0; for (double x : g.vP) Pm += x; Pm /= g.vP.size();
  const double P50 = govPct(g.vP, 50), H50 = govPct(g.vH, 50), G50 = govPct(g.vG, 50), Gq = govPct(g.vG, gPS.govQ), G10 = govPct(g.vG, 10); const size_t n = g.vG.size();
  g.vG.clear(); g.vP.clear(); g.vH.clear();
  g.H[g.k] = g.H[g.k] > 0 ? 0.5 * g.H[g.k] + 0.5 * H50 : H50; g.Ht[g.k] = now;
  g.low[g.lowI] = G10; g.lowI = (g.lowI + 1) % GOV_RING; if (g.lowN < GOV_RING) g.lowN++;
  double lowG = 1e18; for (int i = 0; i < g.lowN; i++) lowG = min(lowG, g.low[i]);
  const double up = gPS.govUp / 1000.0, down = gPS.govDown / 1000.0; char why[400];
  // 1. the check of the last step: did the kick period follow the exec time?
  if (g.probe) { const int dir = g.probe; g.probe = 0; const double dH = (H50 - g.pH) * -dir, dP = (P50 - g.pP) * -dir, dPm = (Pm - g.pPm) * -dir;   // all > 0 when the period followed
    // median: an uncapped game; mean: a capped game shows a stall as dropped frames, the median stays
    const double spread = dir < 0 ? G50 - G10 : g.pG - g.pGlo;   // of the window that would be the stalled one
    const bool tight = spread <= max(0.25, 0.5 * max(dP, 0.0));
    const bool stall = dH > 0.05 && tight && (dP > max(0.3, 0.5 * dH) || dPm > max(0.6, dH));
    if (dir < 0 && stall) {   // the step down made the frame longer: this window sits at the floor
      // uncapped game: the period grew by no more than the exec time and this window's gap is the floor. A frame limiter turns the
      // stall into dropped frames (the period jumps by a whole frame): then the floor lies within one step below the gap before the step.
      g.floorEx = dP <= 1.2 * dH ? min(G50, g.pG) : g.pG - 0.5 * dH; g.floorExT = now; g.backoffMs = now - g.lastUpT < 60000 ? min(g.backoffMs * 2, 60000.0) : 2000; g.lastUpT = now; g.holdUntil = now + g.backoffMs;
      snprintf(why, sizeof why, "step down undone: kick period p50 %.2f -> %.2f (mean %.2f -> %.2f) ms with exec %.2f -> %.2f ms; floor learned %.2f ms; next try in %.0f s", g.pP, P50, g.pPm, Pm, g.pH, H50, g.floorEx, g.backoffMs / 1e3);
      govApply(g.k + 1, why); return; }
    if (dir < 0) { g.verG[g.k] = G10; g.verT[g.k] = now; }   // accepted: this gap level is free of a stall
    if (dir > 0 && stall) { g.floorEx = g.pG; g.floorExT = now; g.upBackoffMs = 4000; }   // the lower corner was a stall: its gap was the floor
    if (dir > 0 && !stall) { if (g.k > 0) { g.verG[g.k - 1] = g.pGlo; g.verT[g.k - 1] = now; } g.upProbeAt = now + g.upBackoffMs; g.upBackoffMs = min(g.upBackoffMs * 2, 64000.0); } }
  // 2. the floor and the slack
  const bool learned = now - g.floorExT < 120000; const double Htop = g.H[g.hi] > 0 ? g.H[g.hi] : H50;
  double fl = learned ? g.floorEx : gPS.govFloor > 0 ? gPS.govFloor / 1000.0 : optGovFloorPct / 100.0 * Htop; const char* src = learned ? "learned" : gPS.govFloor > 0 ? "govfloor" : "prior";
  if (lowG < fl) { fl = lowG; src = "low gap"; if (learned) g.floorEx = lowG; }
  const double S = Gq - fl;
  double dHp = 0; if (g.k > g.lo) dHp = (g.H[g.k - 1] > 0 && now - g.Ht[g.k - 1] < 120000) ? max(0.0, g.H[g.k - 1] - H50) : H50 * (kGovRatio[g.k] - 1);
  if (optGovLog) logf("power_gov win %u core %s jobs %zu | period p50 %.2f mean %.2f exec p50 %.2f gap p50 %.2f p%d %.2f p10 %.2f low %.2f ms | floor %.2f (%s) slack %.2f next step +%.2f ms | ok %d",
                      g.nWin, cName(kGovLadder[g.k]), n, P50, Pm, H50, G50, gPS.govQ, Gq, G10, lowG, fl, src, S, dHp, g.okWins);
  // 3. up
  if (g.k < g.hi) {
    if (learned && S < up) { snprintf(why, sizeof why, "slack %.2f ms < %.2f (gap p%d %.2f - floor %.2f %s); period p50 %.2f exec p50 %.2f ms", S, up, gPS.govQ, Gq, fl, src, P50, H50);
      govApply(g.k + 1, why); return; }
    const bool seenFree = g.verT[g.k] > 0 && now - g.verT[g.k] < 30000 && Gq >= g.verG[g.k] - 0.2;   // this gap level was free of a stall not long ago
    if (!learned && Gq - lowG < up && !seenFree && now >= g.upProbeAt) {
      snprintf(why, sizeof why, "up probe: gap p%d %.2f ms at its low %.2f, floor not known; period p50 %.2f exec p50 %.2f ms", gPS.govQ, Gq, lowG, P50, H50);
      g.pP = P50; g.pPm = Pm; g.pH = H50; g.pG = G50; g.pGlo = G10; if (govApply(g.k + 1, why)) g.probe = 1; return; } }
  // 4. down
  if (g.k > g.lo && now >= g.holdUntil && S - dHp >= down) {
    if (++g.okWins >= optGovDownWins) {
      snprintf(why, sizeof why, "slack %.2f ms (gap p%d %.2f - floor %.2f %s) - step %.2f >= %.2f in %d windows; period p50 %.2f exec p50 %.2f ms", S, gPS.govQ, Gq, fl, src, dHp, down, optGovDownWins, P50, H50);
      g.pP = P50; g.pPm = Pm; g.pH = H50; g.pG = G50; g.pGlo = G10; if (govApply(g.k - 1, why)) g.probe = -1; } }
  else g.okWins = 0; }
// A job that is not part of a frame chain (RUN message): the governor goes back to the top and forgets its window.
static void govNotChain() {
  Gov& g = gGov; if (!g.active) return; g.client = -1; govClearWin(); g.probe = 0; g.lowN = g.lowI = 0; g.floorExT = -1e18;
  if (g.k != g.hi) govApply(g.hi, "a RUN message (not a frame chain): top corner"); }
// Called after every good execute of the eventfd path: tKick = when the daemon read the kick, tDone = after the done was written.
static void govJob(int clientId, int nClients, double tKick, double tDone, double exMs) {
  Gov& g = gGov; if (!g.active) return;
  if (nClients != 1 || clientId != g.client) {   // not one game's frame chain: top corner, start again
    const bool had = g.client >= 0; g.client = clientId; g.lastKick = tKick; g.lastDone = tDone; govClearWin(); g.probe = 0; g.lowN = g.lowI = 0; g.floorExT = -1e18;
    if (g.k != g.hi) govApply(g.hi, had ? "another client or two clients at once: top corner" : "first client: top corner"); return; }
  const double P = tKick - g.lastKick, G = tKick - g.lastDone; g.lastKick = tKick; g.lastDone = tDone;
  if (P > 2000) { govClearWin(); g.probe = 0; g.lowN = g.lowI = 0; g.floorExT = -1e18; if (g.k != g.hi) govApply(g.hi, "first job after a pause > 2 s: top corner"); return; }
  if (P > 100) { govClearWin(); g.probe = 0; return; }   // a menu or a hitch: not a frame cadence
  if (g.skip > 0) { g.skip--; return; }
  if (g.vG.empty()) g.winT0 = tKick;
  g.vG.push_back(G); g.vP.push_back(P); g.vH.push_back(exMs);
  if ((int)g.vG.size() < optGovWinJobs || tDone - g.winT0 < optGovWinMs) return;
  govWindow(tDone); }
// env start values of the governor; returns "" or the reason
static string govEnv() {
  auto num = [](const char* k, int lo, int hi, int* o) -> string { const char* v = getenv(k); if (!v || !*v) return ""; char* e = nullptr; long x = strtol(v, &e, 10);
    if (*e || x < lo || x > hi) return string(k) + " must be " + to_string(lo) + ".." + to_string(hi); *o = (int)x; return ""; };
  string r;
  if (!(r = num("FSR4NPUD_POWER_GOV", 0, 1, &optGov)).empty()) return r; if (!(r = num("FSR4NPUD_POWER_GOV_DOWN_US", 0, 20000, &optGovDownUs)).empty()) return r;
  if (!(r = num("FSR4NPUD_POWER_GOV_UP_US", 0, 20000, &optGovUpUs)).empty()) return r; if (!(r = num("FSR4NPUD_POWER_GOV_Q", 1, 50, &optGovQ)).empty()) return r;
  if (!(r = num("FSR4NPUD_POWER_GOV_FLOOR_US", 0, 50000, &optGovFloorUs)).empty()) return r; if (!(r = num("FSR4NPUD_POWER_GOV_FLOOR_PCT", 1, 200, &optGovFloorPct)).empty()) return r;
  if (!(r = num("FSR4NPUD_POWER_GOV_LOG", 0, 1, &optGovLog)).empty()) return r; if (!(r = num("FSR4NPUD_POWER_GOV_WIN_JOBS", 5, 1000, &optGovWinJobs)).empty()) return r;
  if (!(r = num("FSR4NPUD_POWER_GOV_WIN_MS", 50, 10000, &optGovWinMs)).empty()) return r; if (!(r = num("FSR4NPUD_POWER_GOV_DOWN_WINS", 1, 100, &optGovDownWins)).empty()) return r;
  if (const char* v = getenv("FSR4NPUD_POWER_GOV_RAILS")) { if (!strcmp(v, "core")) optGovRails = 0; else if (!strcmp(v, "both")) optGovRails = 1; else return "FSR4NPUD_POWER_GOV_RAILS must be core or both"; }
  if (const char* v = getenv("FSR4NPUD_POWER_GOV_MIN")) { bool ok = false; for (auto& t : kCorner) if (!strcmp(v, t.n) && govIdx(t.v) >= 0) { ok = true; optGovMin = t.v; } if (!ok) return "FSR4NPUD_POWER_GOV_MIN must be svs_plus, nom, nom_plus, turbo, turbo_plus or max"; }
  if (const char* v = getenv("FSR4NPUD_STUB_CORNER_PCT")) { string s = v; size_t a = 0; while (a < s.size()) { size_t b = s.find(',', a); if (b == string::npos) b = s.size(); string t = s.substr(a, b - a); a = b + 1; size_t q = t.find('=');
      if (q == string::npos) return "FSR4NPUD_STUB_CORNER_PCT: name=percent,..."; int p = atoi(t.c_str() + q + 1); string nm = t.substr(0, q); int ix = -1;
      for (auto& c : kCorner) if (nm == c.n) ix = govIdx(c.v) >= 0 ? govIdx(c.v) : c.v == DCVS_VOLTAGE_VCORNER_SVS ? GOV_NL : c.v == DCVS_VOLTAGE_VCORNER_SVS2 ? GOV_NL + 1 : -1;
      if (ix < 0 || p < 1 || p > 1000) return "FSR4NPUD_STUB_CORNER_PCT: bad entry " + t; gStubPct[ix] = p; } }
  return ""; }
static string dmabufExporter(int fd) {   // /proc/self/fdinfo exp_name of a dma-buf: system = kernel dma-heap, else the fastrpc exporter
  char p[64]; snprintf(p, sizeof p, "/proc/self/fdinfo/%d", fd); FILE* f = fopen(p, "r"); if (!f) return ""; char l[256]; string r;
  while (fgets(l, sizeof l, f)) if (!strncmp(l, "exp_name:", 9)) { char* q = l + 9; while (*q == ' ' || *q == '\t') q++; r = q; while (!r.empty() && (r.back() == '\n' || r.back() == ' ')) r.pop_back(); break; }
  fclose(f); return r; }

// ---------------- stub-mode allocator: the daemon's own Turnip device, exportable dma-buf (no root)
struct VkA { PFN_vkGetInstanceProcAddr gipa = nullptr; VkInstance inst = 0; VkPhysicalDevice pd = 0; VkDevice dev = 0; VkPhysicalDeviceMemoryProperties mp{};
  PFN_vkCreateBuffer createBuffer; PFN_vkDestroyBuffer destroyBuffer; PFN_vkGetBufferMemoryRequirements req; PFN_vkAllocateMemory alloc; PFN_vkFreeMemory freeMem; PFN_vkGetMemoryFdKHR getFd; bool ok = false; };
static VkA VA;
static bool vkInit() {
  if (VA.ok) return true;
  void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL); if (!h) { logf("vk: dlopen libvulkan.so.1 failed"); return false; }
  VA.gipa = (PFN_vkGetInstanceProcAddr)dlsym(h, "vkGetInstanceProcAddr");
  auto createInst = (PFN_vkCreateInstance)VA.gipa(nullptr, "vkCreateInstance");
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_1; ai.pApplicationName = "fsr4npud";
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ic.pApplicationInfo = &ai; if (createInst(&ic, nullptr, &VA.inst) != VK_SUCCESS) return false;
  auto enumPd = (PFN_vkEnumeratePhysicalDevices)VA.gipa(VA.inst, "vkEnumeratePhysicalDevices"); uint32_t n = 1; VkPhysicalDevice pds[1];
  if (enumPd(VA.inst, &n, pds) < 0 || n < 1) return false; VA.pd = pds[0];
  ((PFN_vkGetPhysicalDeviceMemoryProperties)VA.gipa(VA.inst, "vkGetPhysicalDeviceMemoryProperties"))(VA.pd, &VA.mp);
  const char* ext[] = {"VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf"}; float pr = 1.0f;
  VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueCount = 1; q.pQueuePriorities = &pr;
  VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &q; dc.enabledExtensionCount = 2; dc.ppEnabledExtensionNames = ext;
  if (((PFN_vkCreateDevice)VA.gipa(VA.inst, "vkCreateDevice"))(VA.pd, &dc, nullptr, &VA.dev) != VK_SUCCESS) return false;
  auto gdpa = (PFN_vkGetDeviceProcAddr)VA.gipa(VA.inst, "vkGetDeviceProcAddr");
  VA.createBuffer = (PFN_vkCreateBuffer)gdpa(VA.dev, "vkCreateBuffer"); VA.destroyBuffer = (PFN_vkDestroyBuffer)gdpa(VA.dev, "vkDestroyBuffer");
  VA.req = (PFN_vkGetBufferMemoryRequirements)gdpa(VA.dev, "vkGetBufferMemoryRequirements"); VA.alloc = (PFN_vkAllocateMemory)gdpa(VA.dev, "vkAllocateMemory");
  VA.freeMem = (PFN_vkFreeMemory)gdpa(VA.dev, "vkFreeMemory"); VA.getFd = (PFN_vkGetMemoryFdKHR)gdpa(VA.dev, "vkGetMemoryFdKHR");
  VA.ok = VA.getFd != nullptr; logf("vk allocator ready: %s", VA.ok ? "yes" : "no"); return VA.ok;
}
static int vkAllocFd(uint64_t size, VkDeviceMemory* memOut) {
  if (!vkInit()) return -1;
  VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bc.pNext = &eb; bc.size = size; bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer b; if (VA.createBuffer(VA.dev, &bc, nullptr, &b) != VK_SUCCESS) return -1;
  VkMemoryRequirements mr; VA.req(VA.dev, b, &mr); uint32_t mt = 99;
  for (uint32_t i = 0; i < VA.mp.memoryTypeCount; i++) if ((mr.memoryTypeBits >> i & 1) && (VA.mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
  VA.destroyBuffer(VA.dev, b, nullptr); if (mt == 99) return -1;
  VkExportMemoryAllocateInfo ex{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO}; ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
  VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.pNext = &ex; ma.allocationSize = mr.size; ma.memoryTypeIndex = mt;
  VkDeviceMemory m; if (VA.alloc(VA.dev, &ma, nullptr, &m) != VK_SUCCESS) return -1;
  VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR}; gi.memory = m; gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT; int fd = -1;
  if (VA.getFd(VA.dev, &gi, &fd) != VK_SUCCESS) { VA.freeMem(VA.dev, m, nullptr); return -1; }
  *memOut = m; return fd;
}

static bool cleanFd(int fd, uint64_t n) {   // zero the buffer, then clean+invalidate it to the point of coherency once, before any device reads it
  void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); if (m == MAP_FAILED) { logf("cleanFd mmap: %s", strerror(errno)); return false; }
  dma_buf_sync s0{DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW}; ioctl(fd, DMA_BUF_IOCTL_SYNC, &s0); memset(m, 0, n);
  for (uintptr_t x = (uintptr_t)m & ~63ul, e = (uintptr_t)m + n; x < e; x += 64) asm volatile("dc civac, %0" ::"r"(x) : "memory");
  asm volatile("dsb sy" ::: "memory"); dma_buf_sync s1{DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW}; ioctl(fd, DMA_BUF_IOCTL_SYNC, &s1); munmap(m, n); return true; }
// qnn: which allocator? rpcmem is decided at the start. The Vulkan path needs a context for its self-test: it is decided at the first load (allocVkTest), and ALLOC waits until then.
static void allocDecide() {
  gAllocVk = false; string mode = optAlloc;
  if (mode == "memfd") { gAllocName = "rpcmem/test"; logf("allocator: memfd test mode, no dma-buf check"); return; }
  if (mode == "auto") { int h = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (h >= 0) { close(h); mode = "rpcmem"; logf("allocator auto: /dev/dma_heap/system is openable -> rpcmem"); }
    else { logf("allocator auto: /dev/dma_heap/system not openable (%s) -> vk", strerror(errno)); mode = "vk"; } }
  if (mode == "rpcmem") {   // without the heap libcdsprpc silently falls back to FASTRPC_IOCTL_ALLOC_DMA_BUFF (the teardown Oops described at optAlloc): check
    void* b = rAlloc(25, 1, 65536); int fd = b ? rToFd(b) : -1; string ex = fd >= 0 ? dmabufExporter(fd) : string("alloc failed"); if (b) rFree(b);
    if (ex == "system") { gAllocName = "rpcmem/heap"; logf("allocator: rpcmem on the kernel system dma-heap (exporter %s)", ex.c_str()); return; }
    if (optAlloc == "rpcmem") { gAllocName = "rpcmem/fastrpc"; logf("allocator: rpcmem FORCED, exporter '%s' is NOT the system dma-heap (teardown can Oops on 7.2.6)", ex.c_str()); return; }
    logf("allocator auto: rpcmem exporter '%s' is not the system dma-heap -> vk", ex.c_str()); }
  gAllocReady = false; gAllocName = "vk?"; }
static void allocVkTest(Graph* g) {   // can QNN register a Turnip dma-buf as SHARED_BUFFER? decided once
  gAllocReady = true;
  uint64_t n = (g->bytes[0] + 4095) & ~4095ull; VkDeviceMemory vm = 0; int fd = vkAllocFd(n, &vm);
  if (fd < 0) { gAllocName = "rpcmem/fallback"; logf("allocator: vk export failed, using rpcmem (fastrpc-lifetime buffers)"); return; }
  Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT; QnnMemHtp_Descriptor_t hd; memset(&hd, 0, sizeof hd);
  md.memShape = {g->T[0].v1.rank, g->T[0].v1.dimensions, nullptr}; md.dataType = g->T[0].v1.dataType;
  hd.type = QNN_HTP_MEM_SHARED_BUFFER; hd.size = n; hd.sharedBufferConfig.fd = fd; hd.sharedBufferConfig.offset = 0; md.memType = QNN_MEM_TYPE_CUSTOM; md.customInfo = &hd;
  Qnn_MemHandle_t mh = nullptr; Qnn_ErrorHandle_t e; { std::lock_guard<std::mutex> ql(gQnnMu); e = Q.memRegister(g->CTX, &md, 1, &mh); if (e == QNN_SUCCESS) Q.memDeRegister(&mh, 1); }
  close(fd); VA.freeMem(VA.dev, vm, nullptr); gAllocVk = e == QNN_SUCCESS; gAllocName = gAllocVk ? "vk" : "rpcmem/fallback";
  logf("allocator: %s (vk SHARED_BUFFER self-test rc 0x%lx)", gAllocVk ? "vk dma-buf" : "rpcmem fallback", (unsigned long)e); }

// ---------------- clients
struct Client {
  int sock = -1, kick = -1, done = -1; uid_t uid = 0; pid_t pid = 0; int id = 0; Graph* g = nullptr; bool registered = false, quiet = false; uint32_t setupFlags = 0;
  int fds[FSR4NPU_MAXT]; uint64_t fdSize[FSR4NPU_MAXT]; void* map[FSR4NPU_MAXT]; int nfd = 0;
  bool ready = false; uint32_t nt = 0; fsr4npu_tdesc t[FSR4NPU_MAXT];
  Qnn_MemHandle_t mh[FSR4NPU_MAXT]; Qnn_Tensor_t qt[FSR4NPU_MAXT]; bool regDma[FSR4NPU_MAXT];
  bool alt = false; fsr4npu_tdesc tAlt{}; Qnn_MemHandle_t mhAlt = nullptr; Qnn_Tensor_t qtAlt{};   // ping-pong input slot 1 (F_PINGPONG)
  bool altOut = false; uint64_t offOut1 = 0; Qnn_MemHandle_t mhOut1[FSR4NPU_MAXT]{}; Qnn_Tensor_t qtOut1[FSR4NPU_MAXT]{};   // output slot 1 (F_PINGPONG_OUT)
  void* rpcBase = nullptr; VkDeviceMemory vkMem = 0;
  void* stMap = nullptr; uint64_t stLen = 0; vector<double> vInv; long stale = 0;   // cached read-only view of the buffer (input invalidation / diagnostics)
  long frames = 0, errors = 0; vector<double> vExec, vFence, vSvc; double tConn = 0; bool timedOut = false;
};
static vector<Client*> clients; static int clientSeq = 0, clientsDone = 0; static bool gExiting = false;
static void trimLoaded();

static string fdLink(int fd) { char p[64], l[160] = {0}; snprintf(p, sizeof p, "/proc/self/fd/%d", fd); if (readlink(p, l, sizeof l - 1) < 0) return ""; return l; }
static void stat3(const char* k, vector<double> v, string& out) { if (v.empty()) return; sort(v.begin(), v.end()); char b[160];
  snprintf(b, sizeof b, " %s p50 %.3f p90 %.3f max %.3f ms", k, v[v.size() / 2], v[v.size() * 9 / 10], v.back()); out += b; }

static void unreg(Client* c) {
  bool anyH = c->mhAlt != nullptr; for (uint32_t k = 0; k < FSR4NPU_MAXT; k++) anyH = anyH || c->mh[k] || c->mhOut1[k];
  std::unique_lock<std::mutex> ql(gQnnMu, std::defer_lock); if (isQnn && anyH) ql.lock();   // a client that never reached its context has nothing to deregister and does not wait for a load
  c->registered = false;
  for (uint32_t k = 0; k < FSR4NPU_MAXT; k++) {
    if (isQnn && c->mh[k]) { Qnn_ErrorHandle_t e = Q.memDeRegister(&c->mh[k], 1); if (e != QNN_SUCCESS) logf("client %d memDeRegister %u rc 0x%lx", c->id, k, (unsigned long)e); c->mh[k] = nullptr; }
  }
  if (isQnn && c->mhAlt) { Qnn_ErrorHandle_t e = Q.memDeRegister(&c->mhAlt, 1); if (e != QNN_SUCCESS) logf("client %d memDeRegister in1 rc 0x%lx", c->id, (unsigned long)e); c->mhAlt = nullptr; }
  for (uint32_t k = 0; k < FSR4NPU_MAXT; k++) if (isQnn && c->mhOut1[k]) { Qnn_ErrorHandle_t e = Q.memDeRegister(&c->mhOut1[k], 1); if (e != QNN_SUCCESS) logf("client %d memDeRegister out1 %u rc 0x%lx", c->id, k, (unsigned long)e); c->mhOut1[k] = nullptr; }
  c->alt = false; c->altOut = false;
  if (ql.owns_lock()) ql.unlock();
  if (c->stMap) { munmap(c->stMap, c->stLen); c->stMap = nullptr; c->stLen = 0; }
  for (int i = 0; i < c->nfd; i++) { if (c->map[i]) munmap(c->map[i], c->fdSize[i]); c->map[i] = nullptr; if (c->fds[i] >= 0 && !(i == 0 && c->rpcBase)) close(c->fds[i]); c->fds[i] = -1; }
  c->nfd = 0;
  if (c->rpcBase && rFree) { rFree(c->rpcBase); c->rpcBase = nullptr; }
  if (c->vkMem && VA.ok) { VA.freeMem(VA.dev, c->vkMem, nullptr); c->vkMem = 0; }
  if (c->kick >= 0) close(c->kick); if (c->done >= 0) close(c->done); c->kick = c->done = -1;
  c->ready = false;
}
static void dropClient(Client* c, const char* why) {
  string s; stat3("exec", c->vExec, s); stat3("fence_wait", c->vFence, s); stat3("service", c->vSvc, s); stat3("inval", c->vInv, s); if (c->stale) { char b[48]; snprintf(b, sizeof b, " stale_frames %ld", c->stale); s += b; }
  if (!c->quiet) logf("client %d of @%s (pid %d uid %u) gone: %s; frames %ld errors %ld over %.1f s;%s", c->id, c->g->sock.c_str(), (int)c->pid, (unsigned)c->uid, why, c->frames, c->errors, (now_ms() - c->tConn) / 1e3, s.c_str());
  Graph* g = c->g; unreg(c); if (c->sock >= 0) close(c->sock); clients.erase(find(clients.begin(), clients.end(), c)); delete c; clientsDone++;
  g->pins--; g->lastUse = now_ms();   // after unreg: every mem handle of a context is gone before the context can be freed
  if (clients.empty()) gIdleSince = now_ms();
  if (!gExiting) trimLoaded();
}
static void msgInit(fsr4npu_msg& m, uint32_t type) { memset(&m, 0, sizeof m); m.magic = FSR4NPU_MAGIC; m.version = FSR4NPU_VERSION; m.type = type; }
static bool sendMsg(Client* c, const fsr4npu_msg& m, const int* fds = nullptr, int nfd = 0) {
  iovec io{(void*)&m, sizeof m}; char cm[CMSG_SPACE(sizeof(int) * FSR4NPU_MAXFD)]; memset(cm, 0, sizeof cm); msghdr mh{}; mh.msg_iov = &io; mh.msg_iovlen = 1;
  if (nfd > 0) { mh.msg_control = cm; mh.msg_controllen = CMSG_SPACE(sizeof(int) * nfd); cmsghdr* h = CMSG_FIRSTHDR(&mh); h->cmsg_level = SOL_SOCKET; h->cmsg_type = SCM_RIGHTS; h->cmsg_len = CMSG_LEN(sizeof(int) * nfd); memcpy(CMSG_DATA(h), fds, sizeof(int) * nfd); }
  return sendmsg(c->sock, &mh, MSG_NOSIGNAL | MSG_DONTWAIT) == (ssize_t)sizeof m;
}
static bool sendErr(Client* c, int st, uint64_t seq, const char* text) { fsr4npu_msg m; msgInit(m, M_ERROR); m.status = st; m.seq = seq; snprintf(m.text, sizeof m.text, "%s", text); if (!c->quiet) logf("client %d error %d: %s", c->id, st, text); c->errors++; return sendMsg(c, m); }

static int registerTensors(Client* c, uint32_t flags) {
  std::unique_lock<std::mutex> ql(gQnnMu, std::defer_lock); if (isQnn) ql.lock();
  for (uint32_t k = 0; k < c->nt; k++) {
    int i = (int)c->t[k].fd_idx;
    if (!isQnn) { if (!c->map[i]) { c->map[i] = mmap(nullptr, c->fdSize[i], PROT_READ | PROT_WRITE, MAP_SHARED, c->fds[i], 0); if (c->map[i] == MAP_FAILED) { c->map[i] = nullptr; logf("client %d mmap fd %d failed: %s", c->id, c->fds[i], strerror(errno)); return E_FD; } } continue; }
    if ((flags & F_REGDMA) && !c->regDma[i]) { int r = rRegDma ? rRegDma(c->fds[i], (uint32_t)c->fdSize[i]) : -1; logf("client %d remote_register_dma_handle fd %d size %llu rc %d", c->id, c->fds[i], (unsigned long long)c->fdSize[i], r); c->regDma[i] = true; }
    Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT; QnnMemHtp_Descriptor_t hd; memset(&hd, 0, sizeof hd);
    c->qt[k] = c->g->T[k]; md.memShape = {c->g->T[k].v1.rank, c->g->T[k].v1.dimensions, nullptr}; md.dataType = c->g->T[k].v1.dataType;
    hd.type = QNN_HTP_MEM_SHARED_BUFFER; hd.size = c->fdSize[i]; hd.sharedBufferConfig.fd = c->fds[i]; hd.sharedBufferConfig.offset = c->t[k].off;
    md.memType = QNN_MEM_TYPE_CUSTOM; md.customInfo = &hd; Qnn_MemHandle_t mh = nullptr;
    Qnn_ErrorHandle_t e = Q.memRegister(c->g->CTX, &md, 1, &mh);
    logf("client %d memRegister tensor %u SHARED_BUFFER fd %d size %llu offset %llu rc 0x%lx", c->id, k, c->fds[i], (unsigned long long)c->fdSize[i], (unsigned long long)c->t[k].off, (unsigned long)e);
    if (e != QNN_SUCCESS) return E_REGISTER;
    c->mh[k] = mh; c->qt[k].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; c->qt[k].v1.memHandle = mh;
  }
  if (c->alt && isQnn) {   // input slot 1 = graph input tensor 0 at tAlt.off of the same buffer
    int i = (int)c->tAlt.fd_idx; Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT; QnnMemHtp_Descriptor_t hd; memset(&hd, 0, sizeof hd);
    c->qtAlt = c->g->T[0]; md.memShape = {c->g->T[0].v1.rank, c->g->T[0].v1.dimensions, nullptr}; md.dataType = c->g->T[0].v1.dataType;
    hd.type = QNN_HTP_MEM_SHARED_BUFFER; hd.size = c->fdSize[i]; hd.sharedBufferConfig.fd = c->fds[i]; hd.sharedBufferConfig.offset = c->tAlt.off;
    md.memType = QNN_MEM_TYPE_CUSTOM; md.customInfo = &hd; Qnn_MemHandle_t mh = nullptr; Qnn_ErrorHandle_t e = Q.memRegister(c->g->CTX, &md, 1, &mh);
    logf("client %d memRegister input slot 1 SHARED_BUFFER fd %d offset %llu rc 0x%lx", c->id, c->fds[i], (unsigned long long)c->tAlt.off, (unsigned long)e);
    if (e != QNN_SUCCESS) return E_REGISTER;
    c->mhAlt = mh; c->qtAlt.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; c->qtAlt.v1.memHandle = mh;
  }
  if (c->altOut && isQnn) for (uint32_t k = 1; k < c->nt; k++) {   // output slot 1 = the graph outputs at offOut1 (+ offsets relative to output 0)
    int i = (int)c->t[k].fd_idx; Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT; QnnMemHtp_Descriptor_t hd; memset(&hd, 0, sizeof hd);
    c->qtOut1[k] = c->g->T[k]; md.memShape = {c->g->T[k].v1.rank, c->g->T[k].v1.dimensions, nullptr}; md.dataType = c->g->T[k].v1.dataType;
    hd.type = QNN_HTP_MEM_SHARED_BUFFER; hd.size = c->fdSize[i]; hd.sharedBufferConfig.fd = c->fds[i]; hd.sharedBufferConfig.offset = c->offOut1 + (c->t[k].off - c->t[1].off);
    md.memType = QNN_MEM_TYPE_CUSTOM; md.customInfo = &hd; Qnn_MemHandle_t mh = nullptr; Qnn_ErrorHandle_t e = Q.memRegister(c->g->CTX, &md, 1, &mh);
    logf("client %d memRegister output %u slot 1 SHARED_BUFFER fd %d offset %llu rc 0x%lx", c->id, k, c->fds[i], (unsigned long long)hd.sharedBufferConfig.offset, (unsigned long)e);
    if (e != QNN_SUCCESS) return E_REGISTER;
    c->mhOut1[k] = mh; c->qtOut1[k].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; c->qtOut1[k].v1.memHandle = mh;
  }
  return S_OK;
}
// Validate + adopt fds/tensors. bufFds: buffer fds owned by now by this function (closed on failure).
static int adopt(Client* c, const fsr4npu_msg& m, int* fds, int nbuf, int kick, int done) {
  int st = S_OK; char why[96] = "";
  if (m.ntensors != c->g->nt) { st = E_SIZE; snprintf(why, sizeof why, "ntensors %u, graph has %u", m.ntensors, c->g->nt); }
  for (int i = 0; i < nbuf && st == S_OK; i++) {
    string l = fdLink(fds[i]); if (!gMemfd && l.find("dmabuf") == string::npos) { st = E_FD; snprintf(why, sizeof why, "fd %d is not a dma-buf (%s)", i, l.c_str()); break; }
    off_t sz = lseek(fds[i], 0, SEEK_END); if (sz <= 0) { st = E_FD; snprintf(why, sizeof why, "fd %d size unknown", i); break; } c->fdSize[i] = (uint64_t)sz; }
  for (uint32_t k = 0; k < m.ntensors && st == S_OK; k++) {
    const fsr4npu_tdesc& t = m.t[k];
    if ((int)t.fd_idx >= nbuf) { st = E_PROTO; snprintf(why, sizeof why, "tensor %u fd_idx %u >= %d fds", k, t.fd_idx, nbuf); break; }
    if (t.bytes && t.bytes != c->g->bytes[k]) { st = E_SIZE; snprintf(why, sizeof why, "tensor %u bytes %llu != graph %llu", k, (unsigned long long)t.bytes, (unsigned long long)c->g->bytes[k]); break; }
    if (t.off % FSR4NPU_ALIGN) { st = E_SIZE; snprintf(why, sizeof why, "tensor %u offset %llu not %u-aligned", k, (unsigned long long)t.off, FSR4NPU_ALIGN); break; }
    if (t.off + c->g->bytes[k] > c->fdSize[t.fd_idx]) { st = E_SIZE; snprintf(why, sizeof why, "tensor %u off %llu + %llu > buffer %llu", k, (unsigned long long)t.off, (unsigned long long)c->g->bytes[k], (unsigned long long)c->fdSize[t.fd_idx]); break; }
    for (uint32_t j = 0; j < k; j++) if (m.t[j].fd_idx == t.fd_idx && t.off < m.t[j].off + c->g->bytes[j] && m.t[j].off < t.off + c->g->bytes[k]) { st = E_SIZE; snprintf(why, sizeof why, "tensors %u and %u overlap", j, k); }
  }
  if (st == S_OK && kick >= 0 && (fdLink(kick).find("eventfd") == string::npos || fdLink(done).find("eventfd") == string::npos)) { st = E_FD; snprintf(why, sizeof why, "kick/done are not eventfds"); }
  if (st != S_OK) { for (int i = 0; i < nbuf; i++) close(fds[i]); if (kick >= 0) close(kick); if (done >= 0) close(done); sendErr(c, st, 0, why); return st; }
  unreg(c);   // re-setup replaces the previous registration
  c->nfd = nbuf; for (int i = 0; i < nbuf; i++) { c->fds[i] = fds[i]; c->map[i] = nullptr; c->regDma[i] = false; }
  c->nt = m.ntensors; for (uint32_t k = 0; k < c->nt; k++) { c->t[k] = m.t[k]; c->t[k].bytes = c->g->bytes[k]; c->mh[k] = nullptr; }
  c->kick = kick; c->done = done;
  return S_OK;
}

static int dtHash = -1, dtRerun = 0, dtSync = 0;
static uint64_t xh(const void* p, size_t n) { const uint64_t* q = (const uint64_t*)p; uint64_t h = 0x9E3779B97F4A7C15ull; for (size_t i = 0; i < n / 8; i++) { h ^= q[i]; h *= 0x100000001B3ull; h ^= h >> 29; } return h; }
static void dsync(int fd, uint64_t f) { dma_buf_sync s{f}; if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) < 0) logf("dma-buf sync 0x%llx failed: %s", (unsigned long long)f, strerror(errno)); }
static void dtHashOut(Client* c, uint64_t* h) { const uint8_t* b = (const uint8_t*)c->map[0]; int fd = c->fds[0]; dsync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
  for (uint32_t k = 0; k < c->nt; k++) h[k] = xh(b + c->t[k].off, c->g->bytes[k]); dsync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ); }
// CPU-cache invalidation of the HTP input. The cDSP compute context banks are dma-coherent (device tree 'dma-coherent'), the GPU
// (Adreno) is not: the GPU writes the input to memory, the HTP's coherent read snoops the CPU caches, and a line the CPU
// holds from before the GPU's write (speculative fill through the kernel linear map; the pages are cacheable system-heap
// pages) wins over memory. dc civac by VA on our read-only mapping acts on the physical line, so every such copy is gone
// before QnnGraph_execute; lines filled after it come from memory (the GPU has finished writing). The CPU never writes the
// buffer after cleanFd, so civac never writes back data.
static int gStInv = -1, gStDiag = 0; static vector<uint8_t> gStA;
static void civac(const uint8_t* p, uint64_t n) { for (uintptr_t x = (uintptr_t)p & ~63ul, e = (uintptr_t)p + n; x < e; x += 64) asm volatile("dc civac, %0" ::"r"(x) : "memory"); asm volatile("dsb sy" ::: "memory"); }
static long gPhaseBad = 0;
static int execute(Client* c, double* exMs, int slot = 0, int phase = 0) {
  const Qnn_Tensor_t* qin = ((slot & 1) && c->alt) ? &c->qtAlt : &c->qt[0];   // ping-pong input (slot bit 0)
  const bool o1 = (slot & 2) && c->altOut; Qnn_Tensor_t* qout = o1 ? &c->qtOut1[1] : &c->qt[1];   // ping-pong output (slot bit 1)
  if (dtHash < 0) { dtHash = getenv("FSR4NPUD_HASH_OUTPUT") ? atoi(getenv("FSR4NPUD_HASH_OUTPUT")) : 0; dtRerun = getenv("FSR4NPUD_RERUN") ? atoi(getenv("FSR4NPUD_RERUN")) : 0;
    dtSync = getenv("FSR4NPUD_DMABUF_SYNC") ? atoi(getenv("FSR4NPUD_DMABUF_SYNC")) : 0; logf("debug options: output hash %d, rerun %d, dma-buf sync %d", dtHash, dtRerun, dtSync); }
  if (isQnn && dtHash && !c->map[0] && c->nfd == 1) { void* m = mmap(nullptr, c->fdSize[0], PROT_READ, MAP_SHARED, c->fds[0], 0); c->map[0] = m == MAP_FAILED ? nullptr : m; if (!c->map[0]) logf("mmap of the shared buffer failed: %s", strerror(errno)); }
  uint64_t hA[FSR4NPU_MAXT] = {}, hB[FSR4NPU_MAXT] = {}, hC[FSR4NPU_MAXT] = {}; bool dh = isQnn && dtHash && c->map[0];
  if (isQnn && (dtSync & 1)) for (int i = 0; i < c->nfd; i++) { dsync(c->fds[i], DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW); dsync(c->fds[i], DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW); }
  { static int dly = getenv("FSR4NPUD_DELAY_US") ? atoi(getenv("FSR4NPUD_DELAY_US")) : 0; if (isQnn && dly > 0) usleep(dly); }
  if (dh) dtHashOut(c, hA);
  if (gStInv < 0) { gStInv = getenv("FSR4NPUD_INPUT_INVALIDATE") ? atoi(getenv("FSR4NPUD_INPUT_INVALIDATE")) : 1; gStDiag = getenv("FSR4NPUD_INPUT_DIAG") ? atoi(getenv("FSR4NPUD_INPUT_DIAG")) : 0; logf("input cache invalidation %d, diagnostics %d", gStInv, gStDiag); }
  const uint64_t inOff = ((slot & 1) && c->alt) ? c->tAlt.off : c->t[0].off; const uint8_t* inP = (isQnn && c->stMap) ? (const uint8_t*)c->stMap + inOff : nullptr;
  if (inP && gStDiag) { gStA.resize(c->g->bytes[0]); memcpy(gStA.data(), inP, c->g->bytes[0]); }   // diagnostics: the coherent view = what the HTP reads (INV=0)
  if (inP && gStInv) { double i0 = now_ms(); civac(inP, c->g->bytes[0]); c->vInv.push_back(now_ms() - i0); }
  std::unique_lock<std::mutex> ql(gQnnMu, std::defer_lock); if (isQnn) ql.lock();   // before the clock: the timeouts count the execute, not a load beside it
  double t0 = now_ms(); int st = S_OK;
  uint64_t myId = ++gExecId; gExecSock = c->sock; gExecDone = c->done; gExecStart = t0;
  if (isQnn) { Qnn_GraphHandle_t GH = c->g->G; Qnn_Tensor_t pin, pout;
    if (c->g->ng > 1) {   // ANF family: the graph and the tensor ids of this frame's jitter phase
      if (phase < 0 || phase >= c->g->ng) { if (gPhaseBad++ % 100 == 0) logf("client %d phase %d >= %d graphs: phase 0 (%ld so far)", c->id, phase, c->g->ng, gPhaseBad); phase = 0; }
      GH = c->g->GP[phase]; pin = *qin; pin.v1.id = c->g->TP[phase][0].v1.id; pin.v1.name = c->g->TP[phase][0].v1.name; pout = *qout; pout.v1.id = c->g->TP[phase][1].v1.id; pout.v1.name = c->g->TP[phase][1].v1.name; qin = &pin; qout = &pout; }
    Qnn_ErrorHandle_t e = Q.graphExecute(GH, qin, 1, qout, c->nt - 1, nullptr, nullptr); if (e != QNN_SUCCESS) { logf("client %d graphExecute rc 0x%lx", c->id, (unsigned long)e); st = E_EXEC; }
    if (dtSync & 2) for (int i = 0; i < c->nfd; i++) { dsync(c->fds[i], DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW); dsync(c->fds[i], DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW); }
    if (dh) { dtHashOut(c, hB);
      { uint64_t h1[FSR4NPU_MAXT], h5[FSR4NPU_MAXT]; usleep(1000); dtHashOut(c, h1); usleep(4000); dtHashOut(c, h5);
        bool l1 = h1[1] != hB[1] || h1[2] != hB[2], l5 = h5[1] != h1[1] || h5[2] != h1[2]; if (l1 || l5) logf("late output change: client %d frame %ld late1 %d late5 %d", c->id, c->frames, (int)l1, (int)l5); }
      if (dtRerun) { Q.graphExecute(c->g->G, qin, 1, &c->qt[1], c->nt - 1, nullptr, nullptr); dtHashOut(c, hC); }
      logf("output hash: client %d frame %ld in %016llx in_after %016llx o0 %016llx o1 %016llx%s r_o0 %016llx r_o1 %016llx ex %.3f", c->id, c->frames, (unsigned long long)hA[0], (unsigned long long)hB[0],
           (unsigned long long)hB[1], (unsigned long long)hB[2], dtRerun ? ((hC[1] != hB[1] || hC[2] != hB[2]) ? " RERUN_DIFF" : " rerun_same") : "", (unsigned long long)hC[1], (unsigned long long)hC[2], now_ms() - t0); } }
  else {
    for (int i = 0; i < c->nfd; i++) { dma_buf_sync s{DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW}; ioctl(c->fds[i], DMA_BUF_IOCTL_SYNC, &s); }
    const uint8_t* in = (const uint8_t*)c->map[c->t[0].fd_idx] + (((slot & 1) && c->alt) ? c->tAlt.off : c->t[0].off);
    for (uint32_t k = 1; k < c->nt; k++) memcpy((uint8_t*)c->map[c->t[k].fd_idx] + (o1 ? c->offOut1 + (c->t[k].off - c->t[1].off) : c->t[k].off), in, min(c->g->bytes[0], c->g->bytes[k]));
    for (int i = 0; i < c->nfd; i++) { dma_buf_sync s{DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW}; ioctl(c->fds[i], DMA_BUF_IOCTL_SYNC, &s); }
    { const int dl = c->g->stubDelayUs >= 0 ? c->g->stubDelayUs : optStubDelayUs; if (dl > 0) { double e = now_ms() + dl / 1000.0 * stubCornerPct() / 100.0; while (now_ms() < e) {} } }   // R4: stub time by corner
  }
  gExecStart = 0; *exMs = now_ms() - t0; if (ql.owns_lock()) ql.unlock();
  if (inP && gStDiag) {   // diagnostics: memory (after civac) vs the view before execute, per 64-byte line
    civac(inP, c->g->bytes[0]); long nl = 0; uint64_t first[4] = {}; const uint8_t* A = gStA.data();
    for (uint64_t o = 0; o < c->g->bytes[0]; o += 64) if (memcmp(A + o, inP + o, std::min<uint64_t>(64, c->g->bytes[0] - o))) { if (nl < 4) first[nl] = o; nl++; }
    const uint64_t hv = xh(A, c->g->bytes[0]), hf = xh(inP, c->g->bytes[0]); civac(inP, c->g->bytes[0]);
    const uint64_t oOff = (slot & 2) && c->altOut ? c->offOut1 : c->t[1].off; const uint8_t* oP = (const uint8_t*)c->stMap + oOff; const uint64_t ho = xh(oP, c->g->bytes[1]); civac(oP, c->g->bytes[1]);
    logf("input view: client %d frame %ld slot %d in_view %016llx in_mem %016llx out %016llx%s", c->id, c->frames, slot, (unsigned long long)hv, (unsigned long long)hf, (unsigned long long)ho, nl ? " STALE" : "");
    if (nl) { c->stale++; logf("stale input: client %d frame %ld slot %d: %ld input lines of 64 B differ (view vs memory), first at +%llu +%llu +%llu +%llu (pixel %llu)", c->id, c->frames, slot, nl,
      (unsigned long long)first[0], (unsigned long long)first[1], (unsigned long long)first[2], (unsigned long long)first[3], (unsigned long long)(first[0] / 16)); } }
  if (gExecTimedOutId == myId) { c->timedOut = true; st = E_TIMEOUT; double tn = now_ms(); gSoftT[gSoftN++ % 3] = tn;
    logf("client %d execute returned after %.0f ms (> %d ms soft timeout): dropping this client, session kept", c->id, *exMs, optExecTimeoutMs);
    if (tn - gSoftT[0] < 120000 && tn - gSoftT[1] < 120000 && tn - gSoftT[2] < 120000) { logf("FAIL 3 slow executes within 120 s: ordered exit 75 for a fresh DSP session"); gExitCode = 75; gStop = 1; } }
  return st;
}

static void onRun(Client* c, const fsr4npu_msg& m, int* fds, int nfd, double tRecv) {
  fsr4npu_msg r; msgInit(r, M_DONE); r.seq = m.seq; double fenceMs = 0;
  if (!c->ready) { for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_STATE, m.seq, "RUN before SETUP/ALLOC"); return; }
  if (!c->registered && c->g->ng > 1) { for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_BUSY, m.seq, "family still loading"); return; }
  if ((m.flags & F_SYNCFD) && nfd >= 1) {
    double f0 = now_ms(); pollfd p{fds[0], POLLIN, 0}; int pr = poll(&p, 1, 1000); fenceMs = now_ms() - f0;
    for (int i = 0; i < nfd; i++) close(fds[i]);
    if (pr != 1) { sendErr(c, E_FENCE, m.seq, pr == 0 ? "sync_file not signalled within 1000 ms" : "sync_file poll error"); return; }
  } else for (int i = 0; i < nfd; i++) close(fds[i]);
  gExecSeq = m.seq; double ex = 0; int st = execute(c, &ex, (m.flags & F_PINGPONG) ? int(m.v[0] & 3) : 0, (m.flags & F_PHASE) ? int(m.v[1] & 0xff) : 0); if (c->timedOut) { dropClient(c, "execute exceeded the soft timeout (client was told E_TIMEOUT)"); return; }
  r.status = st; r.v[0] = (uint64_t)(ex * 1e6); r.v[1] = (uint64_t)(fenceMs * 1e6);
  if (!isQnn && st == S_OK && (m.flags & F_VERIFY)) { r.v[2] = fnv64((const uint8_t*)c->map[c->t[0].fd_idx] + c->t[0].off, c->g->bytes[0]); r.v[3] = fnv64((const uint8_t*)c->map[c->t[1].fd_idx] + c->t[1].off, c->g->bytes[1]); }
  if (st) c->errors++; else { c->frames++; c->vExec.push_back(ex); c->vFence.push_back(fenceMs); }
  c->vSvc.push_back(now_ms() - tRecv); if (!st) winAdd(ex, c->vSvc.back());
  if (!sendMsg(c, r)) logf("client %d DONE send failed (client gone?)", c->id);
  govNotChain();   // a RUN message is not the layer's frame chain: top corner (fsr4npu-check times stay those of the start corner)
}

static void handleMsg(Client* c) {
  double tRecv = now_ms(); fsr4npu_msg m; iovec io{&m, sizeof m}; char cm[CMSG_SPACE(sizeof(int) * FSR4NPU_MAXFD)];
  msghdr mh{}; mh.msg_iov = &io; mh.msg_iovlen = 1; mh.msg_control = cm; mh.msg_controllen = sizeof cm;
  ssize_t n = recvmsg(c->sock, &mh, MSG_CMSG_CLOEXEC | MSG_DONTWAIT);
  int fds[FSR4NPU_MAXFD]; int nfd = 0;
  for (cmsghdr* h = CMSG_FIRSTHDR(&mh); h; h = CMSG_NXTHDR(&mh, h)) if (h->cmsg_level == SOL_SOCKET && h->cmsg_type == SCM_RIGHTS) {
    int k = (int)((h->cmsg_len - CMSG_LEN(0)) / sizeof(int)); for (int i = 0; i < k; i++) { int fd; memcpy(&fd, CMSG_DATA(h) + i * sizeof(int), sizeof fd); if (nfd < FSR4NPU_MAXFD) fds[nfd++] = fd; else close(fd); } }
  if (n == 0 || (n < 0 && errno != EAGAIN)) { for (int i = 0; i < nfd; i++) close(fds[i]); dropClient(c, n == 0 ? "hung up" : strerror(errno)); return; }
  if (n < 0) return;
  if (n != (ssize_t)sizeof m || m.magic != FSR4NPU_MAGIC || m.version != FSR4NPU_VERSION || (mh.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) {
    for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_PROTO, 0, "bad packet (size/magic/version/truncated)"); dropClient(c, "protocol error"); return; }
  if (c->g->st == Graph::FAILED && (m.type == M_HELLO || m.type == M_SETUP || m.type == M_ALLOC)) {   // a bad graph answers at once: its client goes on without the NPU, the other sockets are not touched
    for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_SYS, m.seq, "graph file missing or bad"); dropClient(c, "graph not available"); clientsDone--; return; }
  switch (m.type) {
  case M_HELLO: { fsr4npu_msg r; msgInit(r, M_INFO); r.ntensors = c->g->nt; r.v[0] = isQnn ? B_QNN : B_STUB; for (uint32_t k = 0; k < c->g->nt; k++) r.t[k].bytes = c->g->bytes[k];
    snprintf(r.text, sizeof r.text, "%s", c->g->name.c_str()); for (int i = 0; i < nfd; i++) close(fds[i]); sendMsg(c, r); break; }
  case M_SETUP: {
    int kick = -1, done = -1, nbuf = nfd;
    if (m.flags & F_EFD) { if (nfd < 3) { for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_PROTO, 0, "F_EFD needs buffer fds + 2 eventfds"); break; } kick = fds[nfd - 2]; done = fds[nfd - 1]; nbuf = nfd - 2; }
    if (nbuf < 1) { for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_PROTO, 0, "SETUP without buffer fds"); break; }
    if (adopt(c, m, fds, nbuf, kick, done) != S_OK) break;
    c->setupFlags = m.flags; const bool now = c->g->st == Graph::LOADED;   // not loaded yet: registered when the load ends
    int st = now ? registerTensors(c, m.flags) : S_OK; if (st) { unreg(c); sendErr(c, st, 0, "register failed"); break; }
    c->registered = now; c->ready = true; fsr4npu_msg r; msgInit(r, M_SETUP_OK); r.ntensors = c->nt; for (uint32_t k = 0; k < c->nt; k++) r.t[k] = c->t[k]; sendMsg(c, r);
    logf("client %d SETUP ok: %d buffer fd(s), %u tensors, eventfd %s", c->id, nbuf, c->nt, kick >= 0 ? "yes" : "no"); break; }
  case M_ALLOC: {
    int kick = -1, done = -1; if (m.flags & F_EFD) { if (nfd != 2) { for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_PROTO, 0, "ALLOC F_EFD needs 2 eventfds"); break; } kick = fds[0]; done = fds[1]; }
    else for (int i = 0; i < nfd; i++) close(fds[i]);
    uint64_t off[FSR4NPU_MAXT], tot = 0; for (uint32_t k = 0; k < c->g->nt; k++) { off[k] = tot; tot += (c->g->bytes[k] + 4095) & ~4095ull; }
    uint64_t offAlt = 0; if (m.flags & F_PINGPONG) { offAlt = tot; tot += (c->g->bytes[0] + 4095) & ~4095ull; }   // input slot 1 after the outputs
    uint64_t offOut1 = 0; if (m.flags & F_PINGPONG_OUT) { offOut1 = tot; for (uint32_t k = 1; k < c->g->nt; k++) tot += (c->g->bytes[k] + 4095) & ~4095ull; }   // output slot 1 after input slot 1
    int bfd = -1; void* base = nullptr; VkDeviceMemory vm = 0;
    if (isQnn && !gAllocVk) { base = rAlloc(25, 1, (int)tot); bfd = base ? rToFd(base) : -1; }
    else if (gMemfd) { bfd = memfd_create("fsr4npu", MFD_CLOEXEC); if (bfd >= 0 && ftruncate(bfd, (off_t)tot) != 0) { close(bfd); bfd = -1; } }
    else bfd = vkAllocFd(tot, &vm);
    if (isQnn && bfd >= 0 && !(getenv("FSR4NPUD_NO_CLEAN") && atoi(getenv("FSR4NPUD_NO_CLEAN")))) { double c0 = now_ms(); cleanFd(bfd, tot); logf("client %d buffer zeroed + cleaned to PoC (%.2f ms)", c->id, now_ms() - c0); }
    if (bfd < 0) { if (kick >= 0) { close(kick); close(done); } sendErr(c, E_ALLOC, 0, isQnn ? "rpcmem_alloc failed" : "vk export alloc failed"); break; }
    // qnn: register rpcmem's own fd (rpcmem_free closes it). stub: the exported fd is ours.
    int own = bfd;
    fsr4npu_msg am; msgInit(am, M_ALLOC); am.ntensors = c->g->nt; for (uint32_t k = 0; k < c->g->nt; k++) { am.t[k].fd_idx = 0; am.t[k].off = off[k]; am.t[k].bytes = c->g->bytes[k]; }
    int one[1] = {own};
    if (base) one[0] = fcntl(bfd, F_DUPFD_CLOEXEC, 3);   // adopt() closes on failure: hand it a dup to validate, swap in bfd after
    if (adopt(c, am, one, 1, kick, done) != S_OK) { if (base) rFree(base); if (vm) VA.freeMem(VA.dev, vm, nullptr); break; }
    if (base) { close(c->fds[0]); c->fds[0] = bfd; }
    if (offAlt) { c->alt = true; c->tAlt.fd_idx = 0; c->tAlt.off = offAlt; c->tAlt.bytes = c->g->bytes[0]; }
    if (offOut1) { c->altOut = true; c->offOut1 = offOut1; }
    c->rpcBase = base; c->vkMem = vm;
    if (isQnn && !base && getenv("FSR4NPUD_REGISTER_ATTR")) { typedef int (*RegAttr)(int, uint32_t, uint32_t); static RegAttr ra = (RegAttr)dlsym(RTLD_DEFAULT, "remote_register_dma_handle_attr");
      uint32_t at = (uint32_t)atoi(getenv("FSR4NPUD_REGISTER_ATTR")); int r = ra ? ra(c->fds[0], (uint32_t)tot, at) : -999; logf("remote_register_dma_handle_attr fd %d len %llu attr %u rc %d", c->fds[0], (unsigned long long)tot, at, r); }
    c->setupFlags = 0; const bool now = c->g->st == Graph::LOADED;   // not loaded yet: registered when the load ends; the client has its buffer now
    int st = now ? registerTensors(c, 0) : S_OK; if (st) { unreg(c); sendErr(c, st, 0, "register failed"); break; }
    c->registered = now;
    if (isQnn) { void* m = mmap(nullptr, tot, PROT_READ, MAP_SHARED, c->fds[0], 0); if (m != MAP_FAILED) { c->stMap = m; c->stLen = tot; } else logf("client %d: mmap of the input view failed: %s (no input invalidation)", c->id, strerror(errno)); }
    c->ready = true; fsr4npu_msg r; msgInit(r, M_SETUP_OK); r.ntensors = c->nt; r.total = tot; for (uint32_t k = 0; k < c->nt; k++) r.t[k] = c->t[k]; r.v[4] = c->alt ? c->tAlt.off : 0; r.v[5] = c->altOut ? c->offOut1 : 0;
    int sfd[1] = {c->fds[0]}; sendMsg(c, r, sfd, 1);
    logf("client %d ALLOC ok: %s buffer %llu bytes, %u tensors, eventfd %s%s", c->id, base ? "rpcmem" : gMemfd ? "memfd" : "vk", (unsigned long long)tot, c->nt, kick >= 0 ? "yes" : "no", now ? "" : ", context still loading"); break; }
  case M_RUN: onRun(c, m, fds, nfd, tRecv); break;
  case M_BYE: for (int i = 0; i < nfd; i++) close(fds[i]); dropClient(c, "bye"); break;
  default: for (int i = 0; i < nfd; i++) close(fds[i]); sendErr(c, E_PROTO, m.seq, "unknown message type"); break;
  }
}
static void onKick(Client* c) {
  uint64_t v; if (read(c->kick, &v, 8) != 8) return; double tRecv = now_ms();
  uint64_t out; if (!c->ready) out = 2 + (uint64_t)(-E_STATE);
  else if (!c->registered) out = 2 + (uint64_t)(-E_BUSY);   // only reached by a family (ng > 1): not ready, the client runs its fallback
  else { double ex = 0; int st = execute(c, &ex, int((v >> 32) & 3), int((v >> 40) & 0xff)); if (c->timedOut) { dropClient(c, "execute exceeded the soft timeout (client was told E_TIMEOUT)"); return; }
    out = st ? 2 + (uint64_t)(-st) : 1; if (st) c->errors++; else { c->frames++; c->vExec.push_back(ex); } }
  c->vSvc.push_back(now_ms() - tRecv); if (c->ready && out == 1) winAdd(c->vExec.back(), c->vSvc.back());
  if (write(c->done, &out, 8) != 8) logf("client %d done eventfd write failed", c->id);
  if (c->ready && out == 1) govJob(c->id, (int)clients.size(), tRecv, now_ms(), c->vExec.back());
}

// ---------------- loader thread: file reads and context create/free, never on the frame path. The main thread owns the graph states; a job and its result pass through a locked queue.
enum JobType { J_INFO, J_LOAD, J_UNLOAD };
struct Job { JobType type; Graph* g; };
struct Result { Qnn_GraphHandle_t gp[64] = {}; JobType type = J_INFO; Graph* g = nullptr; bool ok = false, ctxFail = false; double ms = 0; Parsed p; Qnn_ContextHandle_t ctx = nullptr; Qnn_GraphHandle_t gh = nullptr; };
static std::mutex gJobMu; static std::condition_variable gJobCv; static deque<Job> gJobs; static deque<Result*> gResults; static bool gJobQuit = false; static int gWakeFd = -1;
static std::atomic<double> gLoadStart{0};
static void post(JobType t, Graph* g) { std::lock_guard<std::mutex> l(gJobMu); gJobs.push_back({t, g}); gJobCv.notify_one(); }
static bool contextLoad(Result* r) {
  Graph* g = r->g; if (!parseGraph(g, r->p)) return false;
  if (!isQnn) { if (r->p.stubLoadMs > 0) usleep(r->p.stubLoadMs * 1000); return true; }
  std::unique_lock<std::mutex> ql(gQnnMu, std::defer_lock); if (!optLoadConc) ql.lock(); gLoadStart = now_ms();   // the hung-load clock starts with the call into the DSP
  Qnn_ErrorHandle_t e = Q.contextCreateFromBinary(BE, DV, nullptr, r->p.bin.data(), r->p.bin.size(), &r->ctx, nullptr);
  if (e != QNN_SUCCESS) { logf("graph %s: contextCreateFromBinary rc 0x%lx", g->sock.c_str(), (unsigned long)e); gLoadStart = 0; r->ctx = nullptr; r->ctxFail = true; return false; }
  e = Q.graphRetrieve(r->ctx, r->p.name.c_str(), &r->gh);
  if (e != QNN_SUCCESS) { logf("graph %s: graphRetrieve rc 0x%lx", g->sock.c_str(), (unsigned long)e); Q.contextFree(r->ctx, nullptr); r->ctx = nullptr; gLoadStart = 0; return false; }
  for (int k = 1; r->p.ng > 1 && k < r->p.ng; k++) {   // ANF family: every phase graph of the context by name
    e = Q.graphRetrieve(r->ctx, r->p.names[k].c_str(), &r->gp[k]);
    if (e != QNN_SUCCESS) { logf("graph %s: graphRetrieve %s rc 0x%lx", g->sock.c_str(), r->p.names[k].c_str(), (unsigned long)e); Q.contextFree(r->ctx, nullptr); r->ctx = nullptr; gLoadStart = 0; return false; } }
  r->gp[0] = r->gh;
  gLoadStart = 0; return true; }
static void loaderMain() {
  for (;;) { Job j; { std::unique_lock<std::mutex> l(gJobMu); gJobCv.wait(l, [] { return gJobQuit || !gJobs.empty(); }); if (gJobQuit) return; j = gJobs.front(); gJobs.pop_front(); }
    Result* r = new Result(); r->type = j.type; r->g = j.g; double t0 = now_ms();
    if (j.type == J_INFO) { r->ok = parseGraph(j.g, r->p); if (isQnn) parsedFree(r->p); }
    else if (j.type == J_LOAD) { r->ok = contextLoad(r); if (!r->ok && isQnn) parsedFree(r->p); }
    else { if (isQnn) { { std::lock_guard<std::mutex> ql(gQnnMu); gLoadStart = now_ms(); Qnn_ErrorHandle_t e = Q.contextFree(j.g->CTX, nullptr); gLoadStart = 0; if (e != QNN_SUCCESS) logf("graph %s: contextFree rc 0x%lx", j.g->sock.c_str(), (unsigned long)e); }
        if (j.g->sc) S.systemContextFree(j.g->sc); j.g->sc = nullptr; vector<uint8_t>().swap(j.g->bin); } r->ok = true; }
    r->ms = now_ms() - t0; { std::lock_guard<std::mutex> l(gJobMu); gResults.push_back(r); } uint64_t v = 1; if (write(gWakeFd, &v, 8) != 8) {} } }

static string loadedList() { string s; for (Graph* g : graphs) if (g->st == Graph::LOADED) s += (s.empty() ? "" : " ") + g->sock; return s.empty() ? "none" : s; }
static string optGraphSet = "auto", gSetTxt = "released (no self-check)";
static void statusLine() { string st = "STATUS=serving " + to_string(graphs.size()) + " socket(s), alloc " + (isQnn ? gAllocName : string(gMemfd ? "memfd" : "vk")) + ", graphs: " + gSetTxt + ", loaded: " + loadedList(); sdNotify(st.c_str()); }
static int loadedCount() { int n = 0; for (Graph* g : graphs) if (g->st == Graph::LOADED || g->st == Graph::LOADING) n++; return n; }
static int optMaxGraphs = 0;   // 0 = off; the DSP session runs only about 52 graphs in total (measured), an ANF family counts its phase graphs
static int loadedGraphs() { int n = 0; for (Graph* g : graphs) if (g->st == Graph::LOADED || g->st == Graph::LOADING) n += g->ng > 0 ? g->ng : 1; return n; }
static Graph* lruIdle() { Graph* b = nullptr; for (Graph* g : graphs) if (g->st == Graph::LOADED && g->pins == 0 && !g->keep && (!b || g->lastUse < b->lastUse)) b = g; return b; }
static void startUnload(Graph* g, const char* why) { logf("graph %s: unloading (%s)", g->sock.c_str(), why); g->st = Graph::UNLOADING; post(J_UNLOAD, g); }
static void startLoad(Graph* g) {
  while (loadedCount() + 1 > optMaxLoaded || (optMaxGraphs > 0 && loadedGraphs() + (g->infoReady && g->ng > 0 ? g->ng : 1) > optMaxGraphs)) { Graph* v = lruIdle(); if (!v) { logf("graph %s: every loaded graph has a client, going over --max-loaded %d", g->sock.c_str(), optMaxLoaded); break; } startUnload(v, "least recently used"); }
  g->st = Graph::LOADING; post(J_LOAD, g); }
static void trimLoaded() { while (loadedCount() > optMaxLoaded || (optMaxGraphs > 0 && loadedGraphs() > optMaxGraphs)) { Graph* v = lruIdle(); if (!v) break; startUnload(v, "least recently used"); } }
static void graphWanted(Graph* g) {   // a client connected
  g->lastUse = now_ms();
  if (g->st == Graph::IDLE) startLoad(g);
  else if (g->st == Graph::UNLOADING) g->reload = true;
  else if (g->st == Graph::FAILED && now_ms() >= g->retryAt) startLoad(g); }
static void infoApply(Graph* g, const Parsed& p) { g->ng = p.ng; g->nt = p.nt; memcpy(g->bytes, p.bytes, sizeof g->bytes); g->name = p.name; g->stubDelayUs = p.stubDelayUs; g->infoReady = true; }
static void loadedApply(Graph* g, Result* r) { g->ng = r->p.ng; if (g->ng > 1) { for (int k = 0; k < 64; k++) g->GP[k] = r->gp[k]; memcpy(g->TP, r->p.TP, sizeof g->TP); logf("graph %s: ANF family of %d phase graphs in one context", g->sock.c_str(), g->ng); } infoApply(g, r->p); g->CTX = r->ctx; g->G = r->gh; g->sc = r->p.sc; r->p.sc = nullptr; g->bin.swap(r->p.bin); memcpy(g->T, r->p.T, sizeof g->T); g->st = Graph::LOADED; g->evictRetried = false; g->lastUse = now_ms(); }
static void graphFailed(Graph* g) {
  g->st = Graph::FAILED; g->retryAt = now_ms() + optGraphRetryMs; g->evictRetried = false;
  vector<Client*> mine; for (Client* c : clients) if (c->g == g && c->ready) mine.push_back(c);   // these were told SETUP_OK; a client before ALLOC gets the error as its answer
  for (Client* c : mine) { sendErr(c, E_SYS, 0, "graph could not be loaded"); dropClient(c, "graph load failed"); clientsDone--; } }
static void graphLoaded(Graph* g) {
  if (!gAllocReady) allocVkTest(g);
  vector<Client*> mine; for (Client* c : clients) if (c->g == g && c->ready && !c->registered) mine.push_back(c);
  for (Client* c : mine) { int st = registerTensors(c, c->setupFlags); if (st) { unreg(c); sendErr(c, st, 0, "register failed"); dropClient(c, "register failed"); clientsDone--; } else c->registered = true; } }
static void resultsApply() {
  for (;;) { Result* r; { std::lock_guard<std::mutex> l(gJobMu); if (gResults.empty()) return; r = gResults.front(); gResults.pop_front(); } Graph* g = r->g;
    if (r->type == J_INFO) { if (r->ok) { if (!g->infoReady) infoApply(g, r->p); }
      else if (g->st == Graph::IDLE) { logf("graph %s: file %s missing or bad, the other graphs are not affected", g->sock.c_str(), g->path.c_str()); g->st = Graph::FAILED; g->retryAt = now_ms() + optGraphRetryMs; } }
    else if (r->type == J_LOAD) {
      if (r->ok) { loadedApply(g, r); logf("graph %s (%s): context loaded %.1f ms; loaded now: %s", g->sock.c_str(), g->name.c_str(), r->ms, loadedList().c_str()); graphLoaded(g); statusLine(); }
      else { bool freed = false; if (r->ctxFail && !g->evictRetried) while (Graph* v = lruIdle()) { startUnload(v, "a load failed, freeing memory"); freed = true; }   // idle graphs may hold the memory: free them, then try once more
        if (freed) { g->evictRetried = true; post(J_LOAD, g); } else { logf("graph %s: load failed, next try after %d ms; the other graphs are not affected", g->sock.c_str(), optGraphRetryMs); graphFailed(g); } } }
    else { g->st = Graph::IDLE; g->CTX = nullptr; g->G = nullptr; logf("graph %s: unloaded %.1f ms; loaded now: %s", g->sock.c_str(), r->ms, loadedList().c_str()); statusLine();
      const bool again = g->reload || g->pins > 0; g->reload = false; if (again && !gExiting) startLoad(g); }
    delete r; } }

// ---------------- package 16: graph set (released, or the 4 MB-plan "low" set for units with a faulty upper VTCM window)
// Some SM8550 units give wrong NPU results with the released graphs: fixed bits in the upper part of the 8 MB VTCM window read
// wrong (GitHub issue 2; test 7 on an Odin 2 Portal: released and 5-7 MB plans wrong, 4 MB plans with the file's window size set
// to 8 MB correct in 20 of 20 runs). <graph-dir>/low4/ holds every graph planned for a 4 MB window (stored size patched to 8 MB,
// so the runtime puts the window at the bottom) and the file SELFCHECK: "<graph> <16 hex digits>" = the expected output hash of
// that graph for the fixed input of fsr4npu-check.
// --graph-set auto (default; env FSR4NPUD_GRAPH_SET): at start, before any socket is bound, the released <graph> runs 3 times
//   on the fixed input. Hash right: the released set (good units keep the faster graphs). Hash wrong: the same check on
//   low4/<graph>; right -> every graph is loaded from low4/; wrong or missing -> exit 4 (the unit does not restart; Vela then
//   runs FSR 4 on the GPU instead of showing a broken picture). No low4/SELFCHECK (the v79 folder, old images) or the check
//   cannot run (file, load or execute error): the released set, as before.
// --graph-set released: no check (package 15 behaviour).  --graph-set low: low4/ without a check (tests).
// Only with --graphs (the service); not in the one-graph --name mode.
static const char* kLowDir = "low4";
// 1 = right, 0 = wrong, -1 = could not run (why). Raw client buffers: no shared buffer, nothing registered, context freed again.
static int selfCheckRun(const string& path, uint64_t want, uint64_t* got, double* exMs, string* why) {
  *got = 0; *exMs = 0;
  if (!isQnn) {   // stub backend: FSR4NPUD_SELFCHECK_STUB="<released>,<low>" each ok | wrong | missing (tests of the choice only)
    static int nth = 0; const char* e = getenv("FSR4NPUD_SELFCHECK_STUB"); string s = e ? e : "ok,ok"; size_t c = s.find(',');
    const string v = nth++ == 0 ? s.substr(0, c) : (c == string::npos ? string("ok") : s.substr(c + 1));
    if (v == "missing") { *why = "file missing (stub)"; return -1; } *got = v == "ok" ? want : want ^ 0x5a5a; *exMs = 0.01; return v == "ok" ? 1 : 0; }
  if (access(path.c_str(), R_OK) != 0) { *why = "file missing"; return -1; }
  Graph g; g.path = path; g.sock = "selfcheck"; Parsed o; if (!parseGraph(&g, o)) { *why = "the graph file cannot be read"; return -1; }
  Qnn_ContextHandle_t ctx = nullptr; Qnn_GraphHandle_t G = nullptr; int res = -1; char b[96];
  { std::lock_guard<std::mutex> ql(gQnnMu); gLoadStart = now_ms();   // the hung-load watchdog covers the load
    Qnn_ErrorHandle_t e = Q.contextCreateFromBinary(BE, DV, nullptr, o.bin.data(), o.bin.size(), &ctx, nullptr); gLoadStart = 0;
    if (e != QNN_SUCCESS) { snprintf(b, sizeof b, "contextCreateFromBinary rc 0x%lx", (unsigned long)e); *why = b; ctx = nullptr; }
    else if ((e = Q.graphRetrieve(ctx, o.name.c_str(), &G)) != QNN_SUCCESS) { snprintf(b, sizeof b, "graphRetrieve rc 0x%lx", (unsigned long)e); *why = b; }
    else {
      vector<vector<uint8_t>> buf(o.nt); Qnn_Tensor_t T[FSR4NPU_MAXT];
      for (uint32_t k = 0; k < o.nt; k++) { buf[k].assign(o.bytes[k], 0); T[k] = o.T[k]; T[k].v1.memType = QNN_TENSORMEMTYPE_RAW;
        T[k].v1.clientBuf.data = buf[k].data(); T[k].v1.clientBuf.dataSize = (uint32_t)o.bytes[k]; }
      for (uint64_t i = 0; i < o.bytes[0]; i++) buf[0][i] = (uint8_t)((i * 7u + (i >> 12)) % 59u);   // the input of fsr4npu-check
      vector<double> ex; res = 1;
      for (int r = 0; r < 3 && res == 1; r++) {
        for (uint32_t k = 1; k < o.nt; k++) memset(buf[k].data(), 0xA5 ^ r, buf[k].size());   // a run that writes nothing cannot pass
        const double t0 = now_ms(); e = Q.graphExecute(G, &T[0], 1, &T[1], o.nt - 1, nullptr, nullptr); ex.push_back(now_ms() - t0);
        if (e != QNN_SUCCESS) { snprintf(b, sizeof b, "graphExecute rc 0x%lx", (unsigned long)e); *why = b; res = -1; break; }
        uint64_t hh = 1469598103934665603ULL; for (uint32_t k = 1; k < o.nt; k++) hh ^= fnv64(buf[k].data(), buf[k].size()) + k;   // as fsr4npu-check
        *got = hh; if (hh != want) res = 0; }
      sort(ex.begin(), ex.end()); *exMs = ex[ex.size() / 2]; }
    if (ctx) Q.contextFree(ctx, nullptr); }
  parsedFree(o); return res; }
// Returns 0 to go on, or the exit code. Sets the graphs' paths and gSetTxt.
static int graphSetChoose() {
  const string low = optGraphDir + "/" + kLowDir; auto useLow = [&]() { for (Graph* g : graphs) g->path = low + "/" + g->name + ".bin"; };
  if (optGraphSet == "released") { gSetTxt = "released (no self-check)"; logf("graph set: released, no self-check (--graph-set released)"); return 0; }
  if (optGraphSet == "low") { useLow(); gSetTxt = "low4 (forced)"; logf("graph set: LOW, the 4 MB-plan graphs of %s for every graph, no self-check (--graph-set low)", low.c_str()); return 0; }
  string gname; uint64_t want = 0;
  { FILE* f = fopen((low + "/SELFCHECK").c_str(), "r"); char n[64], h[32];
    if (f) { if (fscanf(f, "%63s %31s", n, h) == 2 && strlen(h) == 16 && strspn(h, "0123456789abcdef") == 16) { gname = n; want = strtoull(h, nullptr, 16); } fclose(f); } }
  if (gname.empty()) { logf("graph set: released, no self-check (no %s/SELFCHECK)", low.c_str()); return 0; }
  uint64_t got = 0, gotL = 0; double ex = 0, exL = 0; string why, whyL; const double t0 = now_ms();
  const int r = selfCheckRun(optGraphDir + "/" + gname + ".bin", want, &got, &ex, &why);
  if (r == 1) { gSetTxt = "released (self-check ok)";
    logf("graph set: released. Self-check ok: %s hash %016llx in 3 of 3 runs, execute %.2f ms without a power vote, %.0f ms in all", gname.c_str(), (unsigned long long)got, ex, now_ms() - t0); return 0; }
  if (r < 0) { logf("graph set: released. The self-check could not run (%s %s): nothing is known about this unit's NPU memory", gname.c_str(), why.c_str()); return 0; }
  const int rl = selfCheckRun(low + "/" + gname + ".bin", want, &gotL, &exL, &whyL);
  if (rl == 1) { useLow(); gSetTxt = "low4 (self-check: released wrong, low4 ok)";
    logf("graph set: LOW. SELF-CHECK FAILED for the released %s: hash %016llx, expected %016llx (this unit's NPU memory gives wrong results in the upper VTCM window). "
         "Every graph is loaded from %s (planned for a 4 MB window): its %s is right, execute %.2f ms (released %.2f ms), %.0f ms in all",
         gname.c_str(), (unsigned long long)got, (unsigned long long)want, low.c_str(), gname.c_str(), exL, ex, now_ms() - t0); return 0; }
  char lw[160]; if (rl < 0) snprintf(lw, sizeof lw, "the 4 MB-plan graph could not run (%s)", whyL.c_str()); else snprintf(lw, sizeof lw, "the 4 MB-plan graph is wrong too (hash %016llx)", (unsigned long long)gotL);
  logf("FAIL graph set: NONE. SELF-CHECK FAILED for the released %s (hash %016llx, expected %016llx) and %s. The NPU of this unit cannot give right results: "
       "exit 4, no restart; Vela runs FSR 4 on the GPU", gname.c_str(), (unsigned long long)got, (unsigned long long)want, lw);
  sdNotify("STATUS=FAIL NPU self-check: released and low4 graphs wrong, NPU off"); return 4; }

int main(int argc, char** argv) {
  T0 = now_ms(); setvbuf(stderr, nullptr, _IOLBF, 0);
  if (const char* pl = getenv("FSR4NPUD_POWER_LIVE")) optPowerLive = pl;   // NPUPWR: the control file ("" or off: none)
  if (const char* ps = getenv("FSR4NPUD_POWER_STATS_JOBS")) optPowerStatsJobs = atoi(ps);
  if (const char* gs = getenv("FSR4NPUD_GRAPH_SET")) optGraphSet = gs;
  { const string ge = govEnv(); if (!ge.empty()) { fprintf(stderr, "%s\n", ge.c_str()); return 2; } }   // R4 governor
  if (const char* pc = getenv("FSR4NPUD_POWER_CORNER")) if (*pc && !powerCornerParse(pc)) { fprintf(stderr, "FSR4NPUD_POWER_CORNER must be max, turbo_plus, turbo, nom_plus, nom, svs_plus, svs, svs2 or dcvs\n"); return 2; }
  for (int i = 1; i < argc; i++) { string a = argv[i]; auto nx = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "--name") optName = nx(); else if (a == "--backend") optBackend = nx(); else if (a == "--ctx") optCtx = nx();
    else if (a == "--stub-bytes") optStubBytes = strtoull(nx(), 0, 10); else if (a == "--stub-nout") optStubNout = atoi(nx()); else if (a == "--stub-delay-us") optStubDelayUs = atoi(nx()); else if (a == "--stub-load-ms") optStubLoadMs = atoi(nx());
    else if (a == "--graphs") optGraphs = nx(); else if (a == "--graph-dir") optGraphDir = nx(); else if (a == "--socket-prefix") optSocketPrefix = nx(); else if (a == "--max-loaded") optMaxLoaded = atoi(nx()); else if (a == "--max-loaded-graphs") optMaxGraphs = atoi(nx()); else if (a == "--load-concurrent") optLoadConc = true;
    else if (a == "--graph-idle-ms") optGraphIdleMs = atoi(nx()); else if (a == "--graph-retry-ms") optGraphRetryMs = atoi(nx()); else if (a == "--load-hard-timeout-ms") optLoadHardMs = atoi(nx());
    else if (a == "--max-clients") optMaxClients = atoi(nx()); else if (a == "--allow-uid") allowUid.push_back((uid_t)atoi(nx())); else if (a == "--allow-gid") allowGid.push_back((gid_t)atoi(nx())); else if (a == "--exec-timeout-ms") optExecTimeoutMs = atoi(nx());
    else if (a == "--exit-after-clients") optExitAfter = atoi(nx()); else if (a == "--max-seconds") optMaxSec = atoi(nx()); else if (a == "--qnn-log") optQnnLog = atoi(nx()); else if (a == "--alloc") optAlloc = nx();
    else if (a == "--exec-hard-timeout-ms") optExecHardMs = atoi(nx()); else if (a == "--power-grace-ms") optPowerGraceMs = atoi(nx());
    else if (a == "--graph-set") optGraphSet = nx(); else if (a == "--power-live") optPowerLive = nx(); else if (a == "--power-stats-jobs") optPowerStatsJobs = atoi(nx());
    else if (a == "--power-corner") { if (!powerCornerParse(nx())) { fprintf(stderr, "--power-corner must be max, turbo_plus, turbo, nom_plus, nom, svs_plus, svs, svs2 or dcvs\n"); return 2; } }
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; } }
  if (optAlloc != "auto" && optAlloc != "rpcmem" && optAlloc != "vk" && optAlloc != "memfd") { fprintf(stderr, "--alloc must be auto, rpcmem, vk or memfd\n"); return 2; }
  if (!optGraphs.empty() && (optGraphDir.empty() || !optCtx.empty())) { fprintf(stderr, "--graphs needs --graph-dir and excludes --ctx\n"); return 2; }
  if (optMaxLoaded < 1) { fprintf(stderr, "--max-loaded must be at least 1\n"); return 2; }
  if (optGraphSet != "auto" && optGraphSet != "released" && optGraphSet != "low") { fprintf(stderr, "--graph-set / FSR4NPUD_GRAPH_SET must be auto, released or low\n"); return 2; }
  if (optPowerLive == "off" || optPowerLive == "0") optPowerLive.clear();
  if (!optPowerLive.empty() && optPowerLive[0] != '/') { fprintf(stderr, "--power-live / FSR4NPUD_POWER_LIVE must be an absolute path or off\n"); return 2; }
  if (optPowerStatsJobs < 0) optPowerStatsJobs = optPowerLive.empty() ? 0 : 600;
  gPS0 = gPS = powerDefaults();   // NPUPWR: the start values = the package-15 vote of --power-corner
  govReset();
  gMemfd = optAlloc == "memfd";
  if (optExecHardMs < optExecTimeoutMs) optExecHardMs = optExecTimeoutMs;
  allowUid.push_back(getuid()); if (getenv("SUDO_UID")) allowUid.push_back((uid_t)atoi(getenv("SUDO_UID")));
  struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = onSig; sigaction(SIGTERM, &sa, nullptr); sigaction(SIGINT, &sa, nullptr); sigaction(SIGALRM, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);
  { struct sigaction cs; memset(&cs, 0, sizeof cs); cs.sa_flags = SA_SIGINFO | SA_RESETHAND;   // crash backtrace: print it on a fatal signal, then exit 139
    cs.sa_sigaction = [](int sg, siginfo_t* si, void*) { char b[128]; int n = snprintf(b, sizeof b, "CRASH fsr4npud signal %d fault addr %p backtrace:\n", sg, si->si_addr); if (write(2, b, n) < 0) {}
      void* bt[48]; int m = backtrace(bt, 48); backtrace_symbols_fd(bt, m, 2); _exit(139); };
    sigaction(SIGSEGV, &cs, nullptr); sigaction(SIGBUS, &cs, nullptr); sigaction(SIGABRT, &cs, nullptr); }
  if (optMaxSec > 0) alarm((unsigned)optMaxSec);
  if (optExecTimeoutMs > 0) std::thread([] {
    uint64_t fired = 0;
    for (;;) { usleep(50000);
      { double l0 = gLoadStart; if (l0 > 0 && now_ms() - l0 >= optLoadHardMs) { logf("FAIL a context load or free hung for > %d ms (HTP/DSP session wedged): exiting 75 so systemd restarts a fresh session", optLoadHardMs); sdNotify("STATUS=FAIL context load hung, restarting"); _exit(75); } }
      uint64_t id = gExecId; double s0 = gExecStart; if (s0 <= 0 || id != gExecId) continue; double el = now_ms() - s0;
      if (el >= optExecTimeoutMs && fired != id) { fired = id; gExecTimedOutId = id;
        logf("WARN execute running > %d ms: client told E_TIMEOUT (falls back); restart only if still stuck at %d ms", optExecTimeoutMs, optExecHardMs);
        fsr4npu_msg m; memset(&m, 0, sizeof m); m.magic = FSR4NPU_MAGIC; m.version = FSR4NPU_VERSION; m.type = M_ERROR; m.status = E_TIMEOUT; m.seq = gExecSeq;
        snprintf(m.text, sizeof m.text, "NPU execute slower than %d ms", optExecTimeoutMs);
        int so = gExecSock, dn = gExecDone; if (so >= 0 && send(so, &m, sizeof m, MSG_NOSIGNAL | MSG_DONTWAIT) < 0) {}
        if (dn >= 0) { uint64_t v = 2 + (uint64_t)(-E_TIMEOUT); if (write(dn, &v, 8) < 0) {} }
        sdNotify("STATUS=WARN execute slow"); }
      if (el >= optExecHardMs) { logf("FAIL execute hung for > %d ms (HTP/DSP session wedged): exiting 75 so systemd restarts a fresh session", optExecHardMs);
        sdNotify("STATUS=FAIL execute hung, restarting"); _exit(75); } } }).detach();
  isQnn = optBackend == "qnn";
  if (!isQnn && optBackend != "stub") { fprintf(stderr, "backend must be stub or qnn\n"); return 2; }
  if (!optGraphs.empty()) { size_t a = 0; while (a <= optGraphs.size()) { size_t b = optGraphs.find(',', a); if (b == string::npos) b = optGraphs.size(); string n = optGraphs.substr(a, b - a); a = b + 1; if (n.empty()) continue;
      Graph* g = new Graph(); g->sock = optSocketPrefix + n; g->path = optGraphDir + "/" + n + ".bin"; g->name = n; graphs.push_back(g); }
    if (graphs.empty()) { fprintf(stderr, "--graphs is empty\n"); return 2; } }
  else { Graph* g = new Graph(); g->sock = optName; g->path = optCtx; g->keep = true; graphs.push_back(g); }
  const bool oneGraph = graphs[0]->keep;
  if (getenv("NOTIFY_SOCKET")) sdNotify(isQnn ? "STATUS=opening the QNN backend" : "STATUS=stub backend");
  if (isQnn) { if (qnnOpen()) allocDecide(); else { logf("FAIL qnn init (is vela-npu-listener.service running? journalctl -u vela-npu-listener)"); sdNotify("STATUS=FAIL qnn init"); return 3; } }
  if (!oneGraph) { const int gr = graphSetChoose(); if (gr) { logf("fsr4npud exit %d", gr); return gr; } }   // package 16: before any socket is bound
  gWakeFd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK); if (gWakeFd < 0) { logf("FAIL eventfd: %s", strerror(errno)); return 1; }
  if (oneGraph) {   // one graph: loaded before READY, a failure ends the process, as always
    Graph* g = graphs[0]; Result r; r.g = g; double t0 = now_ms(); if (!contextLoad(&r)) { logf("FAIL qnn init (is vela-npu-listener.service running? journalctl -u vela-npu-listener)"); sdNotify("STATUS=FAIL qnn init"); return 3; }
    loadedApply(g, &r); logf("context loaded %.1f ms", now_ms() - t0); if (!gAllocReady) allocVkTest(g); }
  int bound = 0;
  for (Graph* g : graphs) { int ls = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0); sockaddr_un a{}; a.sun_family = AF_UNIX;
    if (g->sock.size() >= sizeof a.sun_path - 1) { logf("FAIL socket name too long: %s", g->sock.c_str()); return 2; } memcpy(a.sun_path + 1, g->sock.data(), g->sock.size());
    if (bind(ls, (sockaddr*)&a, (socklen_t)(offsetof(sockaddr_un, sun_path) + 1 + g->sock.size())) < 0) { logf("%s bind @%s: %s (another fsr4npud running?)", oneGraph ? "FAIL" : "SKIP", g->sock.c_str(), strerror(errno)); close(ls); continue; }
    listen(ls, 8); g->ls = ls; bound++; }
  if (!bound) { logf("FAIL no socket could be bound"); return 4; }
  { string bs; if (oneGraph) for (uint32_t k = 0; k < graphs[0]->nt; k++) bs += (k ? "," : "") + to_string(graphs[0]->bytes[k]);
    logf("fsr4npud READY name=@%s sockets=%d backend=%s graph=%s tensors=%u bytes=%s max-loaded=%d alloc=%s pid=%d uid=%u", graphs[0]->sock.c_str(), bound, optBackend.c_str(), oneGraph ? graphs[0]->name.c_str() : "on demand", graphs[0]->nt, bs.c_str(), optMaxLoaded, isQnn ? gAllocName.c_str() : gMemfd ? "memfd" : "vk", (int)getpid(), (unsigned)getuid()); }
  sdNotify("READY=1"); statusLine();
  logf("power_live: control file %s, checked every 200 ms; window stats every %d executes; start values %s", optPowerLive.empty() ? "off" : optPowerLive.c_str(), optPowerStatsJobs, psDesc(gPS0).c_str());
  if (optGov && !gPS0.gov) logf("power_gov OFF: --power-corner dcvs has no corner to steer");
  if (!optGov) logf("power_gov OFF: FSR4NPUD_POWER_GOV=0, the core corner stays at %s 0x%X (the vote of package 15)", cName(gPS0.core), gPS0.core);
  govLogCfg("start values");
  powerLiveCheck(true);
  std::thread loader(loaderMain);
  if (!oneGraph) for (Graph* g : graphs) if (g->ls >= 0) post(J_INFO, g);   // the sizes for HELLO and ALLOC, before the first client asks; a missing file shows in the log now
  while (!gStop) {
    vector<pollfd> p; vector<Graph*> lg; for (Graph* g : graphs) if (g->ls >= 0) { p.push_back({g->ls, POLLIN, 0}); lg.push_back(g); }
    const size_t nl = lg.size(); p.push_back({gWakeFd, POLLIN, 0}); vector<pair<Client*, int>> who;
    for (Client* c : clients) { Graph* g = c->g;
      // A client waits by not being read: until the sizes and the allocator are known, and after SETUP_OK until its handles are registered. Its kick stays in the eventfd. A hang-up is still seen.
      const bool wait = g->st != Graph::FAILED && (!g->infoReady || (isQnn && !gAllocReady) || (c->ready && !c->registered));
      p.push_back({c->sock, short(wait ? 0 : POLLIN), 0}); who.push_back({c, 0}); if (c->ready && (c->registered || c->g->ng > 1) && c->kick >= 0) { p.push_back({c->kick, POLLIN, 0}); who.push_back({c, 1}); } }
    int to = 1000; if (gVoted && clients.empty()) to = max(0, min(1000, (int)(gIdleSince + optPowerGraceMs - now_ms()) + 1));
    if (!optPowerLive.empty()) to = min(to, 200);   // NPUPWR: a control-file change lands within ~200 ms even between jobs
    int pr = poll(p.data(), p.size(), to); if (pr < 0) { if (errno == EINTR) continue; logf("poll: %s", strerror(errno)); gExitCode = 1; break; }
    powerLiveCheck(false);   // NPUPWR: before this round's executes (rate-limited to one lstat per 200 ms)
    if (gVoted && clients.empty() && now_ms() - gIdleSince >= optPowerGraceMs) powerVote(false, "no client");
    for (size_t li = 0; li < nl; li++) if (p[li].revents & POLLIN) { Graph* g = lg[li];
      int s = accept4(g->ls, nullptr, nullptr, SOCK_CLOEXEC);
      if (s >= 0) { ucred cr{}; socklen_t cl = sizeof cr; getsockopt(s, SOL_SOCKET, SO_PEERCRED, &cr, &cl);
        Client* c = new Client(); c->g = g; g->pins++; c->quiet = g->st == Graph::FAILED && now_ms() < g->retryAt; c->sock = s; c->uid = cr.uid; c->pid = cr.pid; c->id = ++clientSeq; c->tConn = now_ms();
        for (int k = 0; k < FSR4NPU_MAXT; k++) { c->fds[k] = -1; c->map[k] = nullptr; c->mh[k] = nullptr; c->regDma[k] = false; }
        clients.push_back(c);
        bool okUid = find(allowUid.begin(), allowUid.end(), cr.uid) != allowUid.end() || cr.uid == 0 || find(allowGid.begin(), allowGid.end(), cr.gid) != allowGid.end();
        if (!c->quiet) logf("client %d connected to @%s pid %d uid %u%s", c->id, g->sock.c_str(), (int)cr.pid, (unsigned)cr.uid, okUid ? "" : " (uid not allowed)");
        if (!okUid) { sendErr(c, E_PERM, 0, "uid not allowed"); dropClient(c, "uid not allowed"); clientsDone--; }
        else if (g->pins > optMaxClients) { sendErr(c, E_BUSY, 0, "too many clients"); dropClient(c, "busy"); clientsDone--; }
        else if (!c->quiet) { if (isQnn && !gVoted) { char w[48]; snprintf(w, sizeof w, "client %d connected", c->id); powerVote(true, w); } graphWanted(g); } }   // the vote first: the load runs at the top corner
    }
    for (size_t i = nl + 1; i < p.size(); i++) { Client* c = who[i - nl - 1].first; if (find(clients.begin(), clients.end(), c) == clients.end()) continue;
      if (who[i - nl - 1].second == 1) { if (p[i].revents & POLLIN) onKick(c); continue; }
      if (p[i].revents & POLLIN) handleMsg(c); else if (p[i].revents & (POLLHUP | POLLERR | POLLNVAL)) dropClient(c, "socket closed (client died?)"); }
    if (p[nl].revents & POLLIN) { uint64_t v; if (read(gWakeFd, &v, 8) != 8) {} resultsApply(); }
    { double tn = now_ms(); for (Graph* g : graphs) if (g->st == Graph::LOADED && g->pins == 0 && !g->keep && tn - g->lastUse >= optGraphIdleMs) startUnload(g, "idle"); }
    if (optExitAfter > 0 && clientsDone >= optExitAfter) { logf("served %d clients, exiting", clientsDone); break; }
  }
  if (gStop) logf("stop signal"); sdNotify("STOPPING=1"); winFlush("daemon exit");
  gExiting = true;
  { std::lock_guard<std::mutex> l(gJobMu); gJobQuit = true; gJobCv.notify_all(); } loader.join(); resultsApply();   // the job in hand ends first; queued jobs are dropped
  while (!clients.empty()) dropClient(clients.front(), "daemon exit");
  for (Graph* g : graphs) if (g->ls >= 0) close(g->ls);
  // release order: every client's memDeRegister -> munmap -> close fds -> free buffer (dropClient above), then the QNN
  // context and backend (closes the fastrpc session), then rpcmem, then our Vulkan device last (vk buffers are already freed).
  if (isQnn) { powerVote(false, "daemon exit"); if (gPerf) gPerf->destroyPowerConfigId(gPcid); for (Graph* g : graphs) if (g->CTX) { Q.contextFree(g->CTX, nullptr); g->CTX = nullptr; if (g->sc) S.systemContextFree(g->sc); g->sc = nullptr; } if (BE) Q.backendFree(BE); if (rDeinit) rDeinit(); }
  if (VA.ok) { ((PFN_vkDestroyDevice)VA.gipa(VA.inst, "vkDestroyDevice"))(VA.dev, nullptr); ((PFN_vkDestroyInstance)VA.gipa(VA.inst, "vkDestroyInstance"))(VA.inst, nullptr); }
  logf("fsr4npud exit %d", gExitCode);
  return gExitCode;
}
