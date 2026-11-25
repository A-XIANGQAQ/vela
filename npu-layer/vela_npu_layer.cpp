// VK_LAYER_VELA_npu: implicit Vulkan layer (aarch64 Linux, enabled by VELA_NPU=1) that runs the FSR 4.1 Lite and the full
// FSR 4.1.1 networks on the Hexagon NPU through the fsr4npud daemon.
//
// What it does (contract: vela_npu_abi.h v1 + vela_npu_ext.h):
//  * recognises the provider's vkCmdUpdateBuffer marker (256 B, magic VELAVPC1) and writes a VelaNpuAck in its place;
//  * PROBE: connects to fsr4npud, which allocates one dma-buf (input at offset 0, the outputs after it); the layer imports it
//    into the game's VkDevice (VK_EXT_external_memory_dma_buf, added at vkCreateDevice). The provider never sees that buffer;
//  * UPSCALE + F_TENSORS, exact: copy input image -> dma-buf, barrier (all writes -> HOST), SetEvent(E1), wait for E2 (set by
//    the worker), copy dma-buf -> out0/out1 images, FINAL ack.status + HTP us copied on the GPU timeline;
//    worker: E1 seen -> eventfd kick -> done -> status slot -> vkSetEvent(E2);
//  * F_ASYNC: the kick only (ack OK = kicked); OP_WAIT(seq) in a later command buffer does the wait + copy-out + final status;
//  * E2 is ALWAYS set (E1 timeout, HTP timeout, daemon gone, teardown, dropped producer): the game never hangs;
//    a non-OK final status makes the provider's select shader run the GPU network for that frame;
//  * E2 is reset on the host only after a later job's E1 proves the GPU passed the wait (Turnip: a vkCmdResetEvent recorded
//    before vkCmdWaitEvents is not yet visible to that wait, so the GPU ran ahead);
//  * a slow HTP execute (longer than VELA_NPU_SLOW_MS) releases its frame to the GPU network at once and keeps the connection;
//    until it ends, every queued job is released too and new markers get BUSY, so no frame waits for the HTP;
//  * daemon gone or HTP hung (no answer in VELA_NPU_NPU_TIMEOUT_MS) -> "no NPU" (NODAEMON / status 5), the worker reconnects
//    every VELA_NPU_RECONNECT_MS;
//  * zero overhead without markers: VELA_NPU=0 -> no hooks; else a size compare in vkCmdUpdateBuffer and an atomic
//    load in the submit/begin hooks.
// Graph choice: VELA_NPU_F_GRAPH picks the daemon per PROBE (fsr4npud_<graph>, VELA_NPUD_GRAPH_NAME overrides); 2-tensor
// sets (the full FSR 4.1.1 graph) get zero-copy and their VkDeviceAddress (CAP_BDA). Lite (no flag, 3 tensors) is unchanged.
// Bounded GPU wait: the GPU does not wait on E2 in the game's command buffer, because E2 is set only by this process: a game
// killed with a job in flight would leave the command processor in vkCmdWaitEvents until the kernel's hangcheck resets the GPU
// and gamescope dies. Instead the layer records KSTEP doubling steps of vkCmdDispatchIndirect whose x the command processor
// reads from the slot's CPU line (1 = armed at submit, 0 = the worker's "done"), each followed by a compute -> indirect
// barrier; all steps run = ~50 ms (680 MHz) .. ~160 ms (220 MHz), far below hangcheck.
// A shader cannot poll host memory on the A740 (its cache keeps the first line it read); the command processor does read
// indirect arguments from memory. Needs bufferDeviceAddress (vkd3d-proton, fsr4vk); otherwise the vkCmdWaitEvents wait is used.
// VELA_NPU_BOUNDED=0 forces that wait.
// Env: VELA_NPU (manifest enable; DISABLE_VELA_NPU=1 disables), VELA_NPUD_NAME (fsr4npud),
//      VELA_NPU_LOG_FILE (append; none -> stderr; the plugin parses the four "Vela NPU layer: " lines), VELA_NPU_CONNECT_MS (100),
//      VELA_NPU_RECONNECT_MS (150), VELA_NPU_E1_TIMEOUT_MS (200), VELA_NPU_SLOW_MS (20: a longer execute -> that frame on the GPU),
//      VELA_NPU_NPU_TIMEOUT_MS (400: no answer -> the HTP counts as hung, fresh connection).
// GPU identity override (build 14, opt-in): VELA_VK_SPOOF_GPU=nvidia makes vkGetPhysicalDeviceProperties / ...2 / ...2KHR report
// the GPU OptiScaler's own Vulkan spoof reports (values and source lines: spoof_props() below), for every caller in the process
// (native-Vulkan games under FEX whose calls OptiScaler's in-process hooks never see: Unity, Haste Demo). Anything else or unset:
// the three functions are not intercepted and vkCreateInstance asks the next layer for nothing more (build 12 behaviour).
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "fsr4npu.h"
#include "vela_npu_abi.h"
#include "vela_npu_ext.h"
#include "vela_npu_wait_spv.h"   // kWaitSpv (vela_npu_wait.comp)
#include "vela_npu_poll.h"       // CPUW: the E1 poll settings (async share / step from the environment)
#include "vela_npu_retire.h"     // A1: when a retired tensor set is freed
static NpuPollCfg g_poll;            // CPUW: set once per device before its worker starts (same environment for every device)
static NpuPollCfg g_poll_env;        // CPUW2: the environment's settings (the live file's keys go on top of these)
static void vlog(const char* f, ...);
#include <sys/stat.h>
// CPUW2: the worker pins itself (pthread_self) to the list, or to every CPU for "all"
static void worker_pin(const char* cpus, const char* why) {
  int list[64]; const int n = npu_cpu_list(cpus, list, 64, CPU_SETSIZE);
  cpu_set_t cs; CPU_ZERO(&cs);
  if (n) for (int i = 0; i < n; i++) CPU_SET(list[i], &cs);
  else { const long all = sysconf(_SC_NPROCESSORS_CONF); for (long c = 0; c < all && c < CPU_SETSIZE; c++) CPU_SET(c, &cs); }
  vlog("worker pinned to %s (%s): %s", n ? cpus : "all CPUs", why, pthread_setaffinity_np(pthread_self(), sizeof cs, &cs) ? "failed" : "ok");
}
// CPUW2: VELA_NPU_LIVE_FILE, looked at by the worker every NPU_LIVE_EVERY jobs (one stat); read again when its time changed
static void worker_live(long jobs) {
  static const char* path = getenv("VELA_NPU_LIVE_FILE");
  static struct timespec seen{0, 0}; static std::string cpusNow;
  if (!path || !*path || jobs % NPU_LIVE_EVERY) return;
  struct stat st;
  if (stat(path, &st) || (st.st_mtim.tv_sec == seen.tv_sec && st.st_mtim.tv_nsec == seen.tv_nsec)) return;
  seen = st.st_mtim;
  char buf[512]; size_t n = 0;
  if (FILE* f = fopen(path, "r")) { n = fread(buf, 1, sizeof buf - 1, f); fclose(f); }
  buf[n] = 0;
  std::string cpus; g_poll = npu_poll_live(g_poll_env, buf, &cpus);
  vlog("live file %s at job %ld: async sleep %.0f %%, step %ld us%s%s", path, jobs, g_poll.asyncShare * 100.0, g_poll.asyncStepNs / 1000L,
       cpus.empty() ? "" : ", worker_cpus ", cpus.c_str());
  if (!cpus.empty() && cpus != cpusNow) { worker_pin(cpus.c_str(), "live file"); cpusNow = cpus; }
}
#define VELA_NPU_ACK_ERROR_TIMEOUT 5u   /* layer-private final status: GPU/HTP timeout or device lost (provider reason "error") */

#define VL_EXPORT extern "C" __attribute__((visibility("default")))
#define VL_BUILD 26u   /* layer build number: reported in every ack and in the log; the manifest's implementation_version is the same number */
#define VL_BUILD_PREV 12u   /* previous shipped layer build number (13 = PUB/lay13, not shipped) */
#define NSLOT 8
#define KSTEP 11   /* steps of the bounded wait (111 << k iterations, k = 0..10: 227k in all) */

static double now_ms() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int env_int(const char* n, int def) { const char* v = getenv(n); return v && *v ? atoi(v) : def; }
static void* key_of(const void* h) { return *(void* const*)h; }

// ---------------------------------------------------------------- log (file under the game's /tmp)
static std::mutex g_logm; static FILE* g_log = nullptr; static volatile int g_logfd = -1;   // fd for the async-signal-safe line
static FILE* log_file() {  // VELA_NPU_LOG_FILE (plugin contract, append); no env -> stderr
  if (!g_log) { const char* p = getenv("VELA_NPU_LOG_FILE"); if (!p || !*p) p = getenv("VELA_NPU_LOG"); g_log = p && *p ? fopen(p, "a") : nullptr;
    if (!g_log) { const char* f4 = getenv("VELA_VK_FSR4"); if (f4 && !strcmp(f4, "1")) { char b[96]; snprintf(b, sizeof b, "/tmp/vela_npu_layer_%d.log", (int)getpid()); g_log = fopen(b, "a"); } }   // build 23: no env path -> a file under /tmp, not only stderr
    if (!g_log) g_log = stderr; g_logfd = fileno(g_log); }
  return g_log;
}
// Lines the plugin parses (loaded / daemon connected / daemon not reachable / job statistics): printed exactly, nothing before them on the line.
static void vline(const char* f, ...) {
  std::lock_guard<std::mutex> l(g_logm); FILE* o = log_file();
  fputs("Vela NPU layer: ", o); va_list a; va_start(a, f); vfprintf(o, f, a); va_end(a); fputc(10, o); fflush(o);
}
// Diagnostic lines: a different prefix, so the plugin's parser ignores them.
static void vlog(const char* f, ...) {
  std::lock_guard<std::mutex> l(g_logm); FILE* o = log_file();
  fprintf(o, "[vela-npu %d %.3f] ", (int)getpid(), now_ms());
  va_list a; va_start(a, f); vfprintf(o, f, a); va_end(a); fputc(10, o); fflush(o);
}

// Build 26: per-job events (kick recorded, executed secondary buffers merged, submit queued, worker got the job, E1 seen, kick sent, daemon done, WAIT recorded).
// On with VELA_NPU_JOBLOG=1, or with VELA_VK_FSR4=1 (the Vulkan path the plugin sets); first 40 seqs and every 600th, plus every abnormal event (<= 200).
static bool jl_env() { static const int on = [] { const char* v = getenv("VELA_NPU_JOBLOG"); if (v && *v) return v[0] != '0' ? 1 : 0; const char* f = getenv("VELA_VK_FSR4"); return f && !strcmp(f, "1") ? 1 : 0; }(); return on != 0; }
static bool jl_seq(uint64_t seq) { return jl_env() && (seq <= 40 || seq % 600 == 0); }
static std::atomic<int> g_jlBad{0};
static bool jl_bad() { return jl_env() && g_jlBad.fetch_add(1) < 200; }
// ---------------------------------------------------------------- state
// Contract: vela_npu_abi.h v1 + vela_npu_ext.h:
//   UPSCALE + F_TENSORS: color = HTP input, output = out0, motion = out1 (R32_UINT, GENERAL; word j at pixel (j%w, j/w)).
//   exact (no F_ASYNC): copy-in, E1, wait E2, copy-out; ack.status is the FINAL status, written on the GPU after the wait.
//   F_ASYNC: copy-in + E1 only, ack OK = "kicked".  OP_WAIT(seq): wait E2 of that kick, copy-out, final status after the wait.
//   ack bytes 32..35 = caps (CAP_ASYNC | CAP_TENSORS), bytes 36..39 = HTP time in us (GPU-written with the status).
static const uint32_t kCaps = VELA_NPU_CAP_ASYNC | VELA_NPU_CAP_TENSORS;
static const uint32_t kPending = 0x7FFFFFFFu;  // record-time status of a marker whose final status is copied after the wait
struct Inst { PFN_vkGetInstanceProcAddr gipa; VkInstance inst;
  // GPU identity override only (VELA_VK_SPOOF_GPU=nvidia): the next layer's physical-device functions; null otherwise
  PFN_vkGetPhysicalDeviceProperties gpdp = nullptr; PFN_vkGetPhysicalDeviceProperties2 gpdp2 = nullptr; PFN_vkGetPhysicalDeviceProperties2KHR gpdp2khr = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties edep = nullptr; };   // build 15: NVIDIA-only device extensions listed (spoof only)
struct TSet { fsr4npu* npu = nullptr; VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceAddress bda = 0; uint64_t offAlt = 0, offOut1 = 0;   // ping-pong: input slot 1, output slot 1 (0 = none)
  uint32_t gen = 0; double tRet = 0, tUse = 0; };   // A1: its generation; retired at; last zero-copy marker of the provider that named it (vela_npu_retire.h)
enum { P_FREE = 0, P_RECORDED, P_QUEUED, P_DONE, P_DROPPED };
struct Slot {
  VkEvent e1 = VK_NULL_HANDLE, e2 = VK_NULL_HANDLE;
  int refs = 0;            // recording hold + queued jobs + async wait hold + E2 hold + quarantine
  int prod = P_FREE; uint64_t seq = 0; uint32_t gen = 0;
  VkImage out0 = VK_NULL_HANDLE, out1 = VK_NULL_HANDLE; uint32_t w = 0, h = 0;   // copy-out targets (async: used by OP_WAIT)
  bool quarantine = false; double qSince = 0;
  bool zc = false;          // zero-copy job (the provider's pack/unpack use the dma-buf directly)
  bool e2Set = false; uint64_t e2After = 0;   // E2 left set: host-reset once a later job's E1 proves the GPU passed its wait
};
struct Job { int slot; uint64_t seq; uint32_t gen; double tSubmit; uint64_t qn; bool async = false; int in = 0, out = 0; bool released = false; int phase = -1; };   // in / out = ping-pong input / output slot of the kick   // async: its wait is a frame later   // released: frame already on the GPU network (slow execute), no kick
struct Consumer { int slot; uint64_t seq; };
struct CbRec { std::vector<Job> jobs; std::vector<Consumer> cons; bool submitted = false; };

struct Dev {
  VkDevice dev; PFN_vkGetDeviceProcAddr gdpa; VkPhysicalDeviceMemoryProperties mp{};
  bool fsr4 = false; std::vector<std::string> exts;   // build 23: FSR 4 device set made; names of the extensions enabled on the device
  bool on = false, extOk = false, bda = false;   // bda = the app enabled bufferDeviceAddress
  char want[64] = "";                           // daemon name of the last PROBE
#define F(n) PFN_vk##n n = nullptr;
  F(DestroyDevice) F(CreateBuffer) F(DestroyBuffer) F(GetBufferMemoryRequirements) F(AllocateMemory) F(FreeMemory)
  F(BindBufferMemory) F(MapMemory) F(GetMemoryFdPropertiesKHR) F(CreateEvent) F(DestroyEvent) F(SetEvent) F(ResetEvent)
  F(GetEventStatus) F(CmdSetEvent) F(CmdResetEvent) F(CmdWaitEvents) F(CmdPipelineBarrier) F(CmdUpdateBuffer) F(CmdCopyBuffer)
  F(CmdCopyImageToBuffer) F(CmdCopyBufferToImage)
  F(QueueSubmit) F(QueueSubmit2) F(QueueSubmit2KHR) F(BeginCommandBuffer) F(AllocateCommandBuffers) F(FreeCommandBuffers)
  F(ResetCommandPool) F(DeviceWaitIdle) F(GetBufferDeviceAddress)
  F(CreateShaderModule) F(DestroyShaderModule) F(CreatePipelineLayout) F(DestroyPipelineLayout) F(CreateComputePipelines) F(DestroyPipeline)
  F(CmdBindPipeline) F(CmdDispatchIndirect) F(CmdExecuteCommands)
  F(CmdDispatch) F(EndCommandBuffer) F(CmdBeginRenderPass) F(CmdBeginRenderPass2) F(CmdBeginRenderPass2KHR) F(CmdEndRenderPass) F(CmdEndRenderPass2) F(CmdEndRenderPass2KHR)
  F(CmdBeginRendering) F(CmdBeginRenderingKHR) F(CmdEndRendering) F(CmdEndRenderingKHR) F(CmdPipelineBarrier2) F(CmdPipelineBarrier2KHR) F(UpdateDescriptorSets) F(CmdPushDescriptorSetKHR)
#undef F
  // bounded wait (bw): GPU-only scratch for the step atomics, one pipeline per step length (no sets, no push constants)
  bool bw = false; VkBuffer wsBuf = VK_NULL_HANDLE; VkDeviceMemory wsMem = VK_NULL_HANDLE; VkShaderModule wsm = VK_NULL_HANDLE;
  VkPipelineLayout wpl = VK_NULL_HANDLE; VkPipeline wp[KSTEP] = {}; long bwWaits = 0, bwRestores = 0, bwNoRestore = 0;
  std::mutex mtx;
  int state = 0;                      // 0 never probed, 1 up, -1 down (reconnect pending), -2 lost/disabled
  uint32_t gen = 0; TSet set; std::vector<TSet> retired;
  uint32_t nt = 0; uint64_t off[3] = {}, bytes[3] = {};   // t[0] input, t[1] out0, t[2] out1 (graph may have 1 or 2 outputs)
  VkBuffer stBuf = VK_NULL_HANDLE; VkDeviceMemory stMem = VK_NULL_HANDLE; volatile uint32_t* st = nullptr;  // per slot: status, htp_us (CPU writes only)
  Slot slots[NSLOT]; uint32_t nextSlot = 0; int lastKick = -1;   // async kick waiting for its OP_WAIT
  std::unordered_map<VkCommandBuffer, CbRec> recs; std::atomic<int> nRec{0};
  std::unordered_map<VkCommandBuffer, VkCommandPool> cbPool;
  uint64_t qCounter = 0; std::thread worker; std::deque<Job> q; std::condition_variable cv; bool quit = false; double nextRetry = 0;
  bool held = false; long retFreed = 0;   // A1: connections held off (too many retired sets); retired sets freed before vkDestroyDevice
  bool slow = false; long slowJobs = 0, relJobs = 0;   // slow: the worker waits for a slow HTP execute (frames go to the GPU meanwhile)
  long l4At = 0, connFails = 0; long jobs = 0, okJobs = 0, e1Timeouts = 0, npuErrors = 0, noDaemon = 0, reconnects = 0, markers = 0, forced = 0;
  double sumE1 = 0, maxE1 = 0, sumNpu = 0, maxNpu = 0, sumRec = 0, maxRec = 0;
  // worker only: low envelope of submit->E1 (the worker sleeps through most of it instead of polling) + a diag window
  double e1Low = 0; long wJobs = 0, wPolls = 0, wAsync = 0; double wE1 = 0, wE1Max = 0, wNpu = 0, wNpuMax = 0, wSlept = 0;
};

static std::mutex g_mtx;
static std::unordered_map<void*, Inst*> g_inst;
struct DevEnt { std::atomic<void*> key{nullptr}; Dev* d = nullptr; };
static DevEnt g_dev[16];                                // lock-free lookup on the hot path
// A process that ends with a job in flight leaves behind the worker that sets E2 (and the done word of the bounded wait). With the
// vkCmdWaitEvents wait the GPU would then stay in the wait until the kernel's hangcheck ("hangcheck detected gpu lockup",
// "offending task: unknown") and the recovery kills gamescope (seen when Resident Evil 4 and Deep Rock Galactic were closed with
// FSR 4 on the NPU). vkDestroyDevice already sets every E2, but games end without it. Now every E2 is set when the process ends:
//  * exit() (the game's own quit, ExitProcess): the layer's destructor;
//  * SIGTERM / SIGHUP (anything but SIG_IGN) and SIGINT with the default action: a handler that chains to the previous one;
//  * after a release the worker never resets an E2 again (the GPU must not wait any more), the frames after it may show
//    a stale NPU output: only while the process ends.
// SIGKILL cannot be caught: with the vkCmdWaitEvents wait a SIGKILL with a job in flight still hangs; the bounded wait does not
// depend on this process.
static std::atomic<int> g_dying{0};
static void safe_line(const char* a, long v, const char* b) {   // async-signal-safe: write(2) of "[vela-npu exit] <a><v><b>\n"
  int fd = g_logfd; if (fd < 0) fd = 2;
  char buf[192]; size_t n = 0; auto put = [&](const char* s) { while (s && *s && n < sizeof buf - 1) buf[n++] = *s++; };
  put("[vela-npu exit] "); put(a);
  if (v >= 0) { char t[24]; int k = 0; long x = v; do { t[k++] = char('0' + x % 10); x /= 10; } while (x && k < 23); while (k && n < sizeof buf - 2) buf[n++] = t[--k]; }
  put(b); buf[n++] = 10; ssize_t r = write(fd, buf, n); (void)r;
}
static void release_all(const char* why, int sig) {   // async-signal-safe (atomics + vkSetEvent = a store in the event's BO)
  g_dying.store(1, std::memory_order_seq_cst);
  long n = 0, devs = 0;
  for (auto& e : g_dev) {
    if (!e.key.load(std::memory_order_acquire)) continue;
    Dev* d = e.d; if (!d || !d->stBuf || !d->SetEvent) continue;   // never probed: no events, nothing waits
    devs++;
    for (int i = 0; i < NSLOT; i++) if (d->slots[i].e2) { if (d->bw && d->st) d->st[i * 16 + 4] = 0; d->SetEvent(d->dev, d->slots[i].e2); n++; }   // and the done word
  }
  if (devs) { safe_line(why, sig, ""); safe_line("released E2 events: ", n, ""); }
}
static Dev* dev_of(const void* h) {
  void* k = key_of(h);
  for (auto& e : g_dev) if (e.key.load(std::memory_order_acquire) == k) return e.d;
  return nullptr;
}
// ---------------------------------------------------------------- build 21: crash trace of the FSR 4 provider (VELA_VK_FSR4=1), log only
// Haste dies inside the provider's first ffxDispatch (AUDIT/r11/IMPL.md). Trace on after the first provider pipeline (required subgroup size)
// is created: per frame <= 400 lines (reset at every vkQueueSubmit), 2500 in all, every line flushed. cb = the command buffer the call records into.
struct CbSt { VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; bool inRP = false; bool inDyn = false; };
static std::mutex g_dgm; static std::unordered_map<VkCommandBuffer, CbSt> g_cbst; static std::set<std::string> g_gpseen;
static std::atomic<int> g_prov{0}, g_dgn{0}, g_frn{0}, g_gpn{0};
static int g_fsr4d = -1;
static bool dg_fsr4() { if (g_fsr4d < 0) { const char* v = getenv("VELA_VK_FSR4"); g_fsr4d = v && !strcmp(v, "1"); } return g_fsr4d; }
#define DG(...) do { if (g_prov.load(std::memory_order_relaxed) && g_dgn.fetch_add(1) < 2500 && g_frn.fetch_add(1) < 400) vlog("dg " __VA_ARGS__); } while (0)
static CbSt cbs_get(VkCommandBuffer cb) { std::lock_guard<std::mutex> l(g_dgm); auto it = g_cbst.find(cb); return it == g_cbst.end() ? CbSt{} : it->second; }
static void cbs_set(VkCommandBuffer cb, int rp, int dyn) { std::lock_guard<std::mutex> l(g_dgm); CbSt& c = g_cbst[cb]; if (rp >= 0) c.inRP = rp; if (dyn >= 0) c.inDyn = dyn; }
static const char* dg_lay(int l) {
  switch (l) { case 0: return "UNDEFINED"; case 1: return "GENERAL"; case 2: return "COLOR_ATT"; case 3: return "DS_ATT"; case 4: return "DS_READONLY"; case 5: return "SHADER_READONLY";
    case 6: return "XFER_SRC"; case 7: return "XFER_DST"; case 8: return "PREINIT"; case 1000001002: return "PRESENT"; case 1000314000: return "DS_RO_STENCIL_ATT"; case 1000241000: return "DEPTH_ATT"; case 1000241001: return "DEPTH_RO"; case 1000024001: return "READONLY_2"; case 1000314001: return "ATT_OPT"; default: return "other"; } }
static const char* dg_st(const CbSt& c) { return c.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY ? "SECONDARY" : "primary"; }
#define DGCB(cb) CbSt cs = cbs_get(cb)
static VKAPI_ATTR void VKAPI_CALL vl_CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
  Dev* d = dev_of(cb); DGCB(cb); DG("dispatch cb %p %s inRP %d inDyn %d (%u,%u,%u)", (void*)cb, dg_st(cs), cs.inRP, cs.inDyn, x, y, z); d->CmdDispatch(cb, x, y, z);
}
static VKAPI_ATTR VkResult VKAPI_CALL vl_EndCommandBuffer(VkCommandBuffer cb) {
  Dev* d = dev_of(cb); DGCB(cb); DG("end cb %p %s inRP %d", (void*)cb, dg_st(cs), cs.inRP); cbs_set(cb, 0, 0); return d->EndCommandBuffer(cb);
}
static VKAPI_ATTR void VKAPI_CALL vl_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, VkSubpassContents c) {
  Dev* d = dev_of(cb); DGCB(cb); DG("beginRenderPass cb %p %s contents %d", (void*)cb, dg_st(cs), (int)c); cbs_set(cb, 1, -1); d->CmdBeginRenderPass(cb, bi, c);
}
static VKAPI_ATTR void VKAPI_CALL vl_CmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, const VkSubpassBeginInfo* sb) {
  Dev* d = dev_of(cb); DGCB(cb); DG("beginRenderPass2 cb %p %s contents %d", (void*)cb, dg_st(cs), sb ? (int)sb->contents : -1); cbs_set(cb, 1, -1); d->CmdBeginRenderPass2(cb, bi, sb);
}
static VKAPI_ATTR void VKAPI_CALL vl_CmdBeginRenderPass2KHR(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, const VkSubpassBeginInfo* sb) {
  Dev* d = dev_of(cb); DGCB(cb); DG("beginRenderPass2KHR cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, 1, -1); d->CmdBeginRenderPass2KHR(cb, bi, sb);
}
static VKAPI_ATTR void VKAPI_CALL vl_CmdEndRenderPass(VkCommandBuffer cb) { Dev* d = dev_of(cb); DGCB(cb); DG("endRenderPass cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, 0, -1); d->CmdEndRenderPass(cb); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo* e) { Dev* d = dev_of(cb); DGCB(cb); DG("endRenderPass2 cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, 0, -1); d->CmdEndRenderPass2(cb, e); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdEndRenderPass2KHR(VkCommandBuffer cb, const VkSubpassEndInfo* e) { Dev* d = dev_of(cb); DGCB(cb); DG("endRenderPass2KHR cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, 0, -1); d->CmdEndRenderPass2KHR(cb, e); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) { Dev* d = dev_of(cb); DGCB(cb); DG("beginRendering cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, -1, 1); d->CmdBeginRendering(cb, ri); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* ri) { Dev* d = dev_of(cb); DGCB(cb); DG("beginRenderingKHR cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, -1, 1); d->CmdBeginRenderingKHR(cb, ri); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdEndRendering(VkCommandBuffer cb) { Dev* d = dev_of(cb); DGCB(cb); DG("endRendering cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, -1, 0); d->CmdEndRendering(cb); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdEndRenderingKHR(VkCommandBuffer cb) { Dev* d = dev_of(cb); DGCB(cb); DG("endRenderingKHR cb %p %s", (void*)cb, dg_st(cs)); cbs_set(cb, -1, 0); d->CmdEndRenderingKHR(cb); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df, uint32_t nm, const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
  Dev* d = dev_of(cb); DGCB(cb);
  if (g_prov.load(std::memory_order_relaxed)) {
    DG("barrier cb %p %s inRP %d src 0x%x dst 0x%x mem %u buf %u img %u", (void*)cb, dg_st(cs), cs.inRP, (unsigned)ss, (unsigned)ds, nm, nb, ni);
    for (uint32_t i = 0; i < ni && i < 6; i++) DG("  imgbarrier image %p %s -> %s aspect 0x%x mips %u layers %u access 0x%x->0x%x", (void*)(uintptr_t)im[i].image, dg_lay(im[i].oldLayout), dg_lay(im[i].newLayout), (unsigned)im[i].subresourceRange.aspectMask, im[i].subresourceRange.levelCount, im[i].subresourceRange.layerCount, (unsigned)im[i].srcAccessMask, (unsigned)im[i].dstAccessMask);
  }
  d->CmdPipelineBarrier(cb, ss, ds, df, nm, m, nb, b, ni, im);
}
static void dg_barrier2(const char* nm, VkCommandBuffer cb, const VkDependencyInfo* di) {
  DGCB(cb);
  if (g_prov.load(std::memory_order_relaxed) && di) {
    DG("%s cb %p %s inRP %d mem %u buf %u img %u", nm, (void*)cb, dg_st(cs), cs.inRP, di->memoryBarrierCount, di->bufferMemoryBarrierCount, di->imageMemoryBarrierCount);
    for (uint32_t i = 0; i < di->imageMemoryBarrierCount && i < 6; i++) { const VkImageMemoryBarrier2& b = di->pImageMemoryBarriers[i];
      DG("  imgbarrier2 image %p %s -> %s aspect 0x%x mips %u layers %u stage 0x%llx->0x%llx", (void*)(uintptr_t)b.image, dg_lay(b.oldLayout), dg_lay(b.newLayout), (unsigned)b.subresourceRange.aspectMask, b.subresourceRange.levelCount, b.subresourceRange.layerCount, (unsigned long long)b.srcStageMask, (unsigned long long)b.dstStageMask); }
  }
}
static VKAPI_ATTR void VKAPI_CALL vl_CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di) { Dev* d = dev_of(cb); dg_barrier2("barrier2", cb, di); d->CmdPipelineBarrier2(cb, di); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* di) { Dev* d = dev_of(cb); dg_barrier2("barrier2KHR", cb, di); d->CmdPipelineBarrier2KHR(cb, di); }
static void dg_writes(const char* nm, void* who, uint32_t n, const VkWriteDescriptorSet* w) {
  if (!g_prov.load(std::memory_order_relaxed)) return;
  DG("%s %p: %u writes", nm, who, n);
  for (uint32_t i = 0; i < n && i < 40; i++) {
    const VkWriteDescriptorSet& x = w[i];
    bool img = x.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || x.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || x.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || x.descriptorType == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    if (!img || !x.pImageInfo) { DG("  write set %p binding %u type %d count %u", (void*)(uintptr_t)x.dstSet, x.dstBinding, (int)x.descriptorType, x.descriptorCount); continue; }
    for (uint32_t j = 0; j < x.descriptorCount && j < 4; j++) DG("  write set %p binding %u[%u] type %d view %p layout %s(%d)", (void*)(uintptr_t)x.dstSet, x.dstBinding, x.dstArrayElement + j, (int)x.descriptorType, (void*)(uintptr_t)x.pImageInfo[j].imageView, dg_lay(x.pImageInfo[j].imageLayout), (int)x.pImageInfo[j].imageLayout);
  }
}
static VKAPI_ATTR void VKAPI_CALL vl_UpdateDescriptorSets(VkDevice dev, uint32_t nw, const VkWriteDescriptorSet* w, uint32_t nc, const VkCopyDescriptorSet* c) { Dev* d = dev_of(dev); dg_writes("updateDescriptorSets", (void*)dev, nw, w); d->UpdateDescriptorSets(dev, nw, w, nc, c); }
static VKAPI_ATTR void VKAPI_CALL vl_CmdPushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t set, uint32_t nw, const VkWriteDescriptorSet* w) {
  Dev* d = dev_of(cb); { DGCB(cb); DG("pushDescriptorSet cb %p %s inRP %d set %u", (void*)cb, dg_st(cs), cs.inRP, set); } dg_writes("pushDescriptorSet", (void*)cb, nw, w); d->CmdPushDescriptorSetKHR(cb, bp, l, set, nw, w);
}
static void dg_frame(const char* what) { if (g_prov.load(std::memory_order_relaxed)) { g_frn.store(0); DG("-- %s", what); } }
static void dg_gp(const char* kind, const char* name, bool null) {   // vkGet{Device,Instance}ProcAddr answers, deduped, <= 400 lines
  if (!dg_fsr4() || !name) return;
  std::string k = std::string(kind) + name + (null ? "!" : "");
  { std::lock_guard<std::mutex> l(g_dgm); if (!g_gpseen.insert(k).second) return; }
  if (g_gpn.fetch_add(1) < 400) vlog("gp %s %s -> %s", kind, name, null ? "NULL" : "ok");
}
static int g_enabled = -1;
static bool enabled() { if (g_enabled < 0) { const char* v = getenv("VELA_NPU"); g_enabled = !(v && !strcmp(v, "0")); } return g_enabled; }

// ---------------------------------------------------------------- tensor set: daemon ALLOC -> import into the game device
static uint32_t mem_type(Dev* d, uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < d->mp.memoryTypeCount; i++) if ((bits >> i & 1) && (d->mp.memoryTypes[i].propertyFlags & want) == want) return i;
  return UINT32_MAX;
}
static void set_free(Dev* d, TSet& s) {
  if (s.npu) fsr4npu_close(s.npu);
  if (s.buf) d->DestroyBuffer(d->dev, s.buf, nullptr);
  if (s.mem) d->FreeMemory(d->dev, s.mem, nullptr);
  s = TSet{};
}
// Connect + ALLOC + import. No d->mtx held (blocks up to connect_ms). 0 or a status.
static bool pp_want(uint32_t nt) { static const bool off = [] { const char* v = getenv("VELA_NPU_PINGPONG"); return v && v[0] == '0'; }(); return !off && nt == 2; }
static int set_open(Dev* d, const char* name, int connect_ms, TSet& s, uint32_t* nt, uint64_t off[3], uint64_t bytes[3]) {
  bool bda = false;
  fsr4npu_info info{}; int fd = -1; uint64_t total = 0, o[FSR4NPU_MAXT] = {};
  VkMemoryRequirements mr{}; uint32_t mt = UINT32_MAX; VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  int st = fsr4npu_connect(name, connect_ms, &s.npu, &info);
  if (st) { s.npu = nullptr; return st; }
  if (info.ntensors < 2) { vlog("daemon graph '%s' has %u tensors", info.graph, info.ntensors); st = E_SIZE; goto fail; }
  s.offAlt = 0; s.offOut1 = 0;
  if ((st = pp_want(info.ntensors) ? fsr4npu_alloc_pp2(s.npu, 1, &fd, &total, o, &s.offAlt, &s.offOut1) : fsr4npu_alloc(s.npu, 1, &fd, &total, o))) { vlog("ALLOC: %s", fsr4npu_strerror(st)); goto fail; }
  *nt = std::min<uint32_t>(info.ntensors, 3);
  bda = d->bda && d->GetBufferDeviceAddress && info.ntensors == 2;   // only 1-output graphs (Lite unchanged)
  for (uint32_t k = 0; k < 3; k++) { off[k] = k < *nt ? o[k] : 0; bytes[k] = k < *nt ? info.bytes[k] : 0; }
  {
    VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; eb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bc.pNext = &eb; bc.size = total;
    bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (bda) bc.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (d->CreateBuffer(d->dev, &bc, nullptr, &s.buf) != VK_SUCCESS) { close(fd); vlog("vkCreateBuffer(external) failed"); st = E_SYS; goto fail; }
  }
  d->GetBufferMemoryRequirements(d->dev, s.buf, &mr);
  if (d->GetMemoryFdPropertiesKHR(d->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp) != VK_SUCCESS) { close(fd); vlog("vkGetMemoryFdPropertiesKHR failed"); st = E_FD; goto fail; }
  mt = mem_type(d, mr.memoryTypeBits & fp.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (mt == UINT32_MAX) mt = mem_type(d, mr.memoryTypeBits & fp.memoryTypeBits, 0);
  {
    VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR}; imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT; imp.fd = fd;
    VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fl.pNext = &imp; fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.pNext = bda ? (const void*)&fl : (const void*)&imp; ma.allocationSize = std::max<uint64_t>(total, mr.size); ma.memoryTypeIndex = mt;
    if (mt == UINT32_MAX || d->AllocateMemory(d->dev, &ma, nullptr, &s.mem) != VK_SUCCESS) { close(fd); vlog("dma-buf import failed (type %u)", mt); st = E_FD; goto fail; }
    if (d->BindBufferMemory(d->dev, s.buf, s.mem, 0) != VK_SUCCESS) { vlog("bind failed"); st = E_SYS; goto fail; }   // fd owned by the driver now
    if (bda) { VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bi.buffer = s.buf; s.bda = d->GetBufferDeviceAddress(d->dev, &bi); }
  }
  vlog("daemon '%s' graph '%s' backend %u: dma-buf %llu B imported; in @%llu %llu B, out0 @%llu %llu B, out1 @%llu %llu B", name, info.graph, info.backend,
       (unsigned long long)total, (unsigned long long)off[0], (unsigned long long)bytes[0], (unsigned long long)off[1], (unsigned long long)bytes[1],
       (unsigned long long)off[2], (unsigned long long)bytes[2]);
  return 0;
fail:
  set_free(d, s);
  return st ? st : E_SYS;
}
static void worker_main(Dev* d);
// the signals whose arrival ends the process; previous handlers are chained (Wine has its own for SIGINT / SIGQUIT)
static const int kExitSigs[3] = {SIGTERM, SIGHUP, SIGINT};
static struct sigaction g_oldsa[3];
static void exit_sig(int sig, siginfo_t* info, void* uc) {
  int k = sig == SIGTERM ? 0 : sig == SIGHUP ? 1 : 2; const struct sigaction& o = g_oldsa[k];
  const bool dfl = !(o.sa_flags & SA_SIGINFO) && o.sa_handler == SIG_DFL, ign = !(o.sa_flags & SA_SIGINFO) && o.sa_handler == SIG_IGN;
  if (!ign && (sig != SIGINT || dfl)) release_all("signal ", sig);
  if (dfl) { struct sigaction d{}; d.sa_handler = SIG_DFL; sigemptyset(&d.sa_mask); sigaction(sig, &d, nullptr); raise(sig); return; }   // delivered on return
  if (ign) return;
  if (o.sa_flags & SA_SIGINFO) o.sa_sigaction(sig, info, uc); else o.sa_handler(sig);
}
static void install_exit_hooks() {
  static std::once_flag once;
  std::call_once(once, [] {
    int ok = 0;
    for (int k = 0; k < 3; k++) {
      struct sigaction sa{}; sa.sa_sigaction = exit_sig; sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK; sigemptyset(&sa.sa_mask);
      if (sigaction(kExitSigs[k], nullptr, &g_oldsa[k]) == 0 && sigaction(kExitSigs[k], &sa, nullptr) == 0) ok++;
    }
    vlog("exit hooks: E2 release at exit() and on SIGTERM/SIGHUP/SIGINT (%d/3 handlers; previous SIGTERM %s)", ok,
         (g_oldsa[0].sa_flags & SA_SIGINFO) ? "handler" : g_oldsa[0].sa_handler == SIG_DFL ? "default" : g_oldsa[0].sa_handler == SIG_IGN ? "ignored" : "handler");
  });
}
__attribute__((destructor)) static void vl_fini() { release_all("exit() / unload, sig ", 0); }
static void slot_unref(Dev* d, int i) { Slot& s = d->slots[i]; if (s.refs > 0 && --s.refs == 0 && s.prod == P_RECORDED) s.prod = P_DROPPED; }
// bounded wait setup (d->mtx held, first PROBE). Any failure -> d->bw stays false = the vkCmdWaitEvents wait.
static void bw_free(Dev* d) {
  for (VkPipeline& p : d->wp) { if (p) d->DestroyPipeline(d->dev, p, nullptr); p = VK_NULL_HANDLE; }
  if (d->wpl) d->DestroyPipelineLayout(d->dev, d->wpl, nullptr); if (d->wsm) d->DestroyShaderModule(d->dev, d->wsm, nullptr);
  if (d->wsBuf) d->DestroyBuffer(d->dev, d->wsBuf, nullptr); if (d->wsMem) d->FreeMemory(d->dev, d->wsMem, nullptr);
  d->wpl = VK_NULL_HANDLE; d->wsm = VK_NULL_HANDLE; d->wsBuf = VK_NULL_HANDLE; d->wsMem = VK_NULL_HANDLE; d->bw = false;
}
static void bw_init(Dev* d) {
  const char* v = getenv("VELA_NPU_BOUNDED");
  if (v && v[0] == '0') { vlog("bounded GPU wait off (VELA_NPU_BOUNDED=0): host-event wait"); return; }
  if (!d->bda || !d->GetBufferDeviceAddress || !d->CreateComputePipelines || !d->CmdDispatchIndirect || !d->CmdBindPipeline) {
    vlog("bounded GPU wait unavailable (bufferDeviceAddress %s): host-event wait", d->bda ? "on" : "off"); return; }
  const char* why = nullptr; void* p = nullptr; VkDeviceAddress a = 0; uint32_t total = 0;
  {
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bc.size = 256; bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryRequirements mr{}; VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.pNext = &fl;
    if (d->CreateBuffer(d->dev, &bc, nullptr, &d->wsBuf) != VK_SUCCESS) { why = "scratch buffer"; goto fail; }
    d->GetBufferMemoryRequirements(d->dev, d->wsBuf, &mr); ma.allocationSize = mr.size;
    ma.memoryTypeIndex = mem_type(d, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);   // host-visible memory: the step timings in the header comment were measured with it
    if (ma.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->dev, &ma, nullptr, &d->wsMem) != VK_SUCCESS || d->BindBufferMemory(d->dev, d->wsBuf, d->wsMem, 0) != VK_SUCCESS) { why = "scratch memory"; goto fail; }
    if (d->MapMemory(d->dev, d->wsMem, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS && p) memset(p, 0, 256);
    VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bi.buffer = d->wsBuf; a = d->GetBufferDeviceAddress(d->dev, &bi);
    if (!a) { why = "scratch address"; goto fail; }
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize = sizeof kWaitSpv; sm.pCode = kWaitSpv;
    if (d->CreateShaderModule(d->dev, &sm, nullptr, &d->wsm) != VK_SUCCESS) { why = "shader module"; goto fail; }
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};   // empty: binding it disturbs no set and no push constant
    if (d->CreatePipelineLayout(d->dev, &pl, nullptr, &d->wpl) != VK_SUCCESS) { why = "pipeline layout"; goto fail; }
    for (int k = 0; k < KSTEP; k++) {
      const uint32_t sc[3] = {uint32_t(a), uint32_t(a >> 32), 111u << k}; total += sc[2];
      VkSpecializationMapEntry me[3]; for (int i = 0; i < 3; i++) { me[i].constantID = uint32_t(i); me[i].offset = uint32_t(i * 4); me[i].size = 4; }
      VkSpecializationInfo si{3, me, sizeof sc, sc};
      VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = d->wsm; ci.stage.pName = "main"; ci.stage.pSpecializationInfo = &si; ci.layout = d->wpl;
      if (d->CreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &ci, nullptr, &d->wp[k]) != VK_SUCCESS) { why = "compute pipeline"; goto fail; }
    }
  }
  d->bw = true;
  vlog("bounded GPU wait on: %d steps, %u iterations in all (~50 ms at 680 MHz, ~160 ms at 220 MHz; then the GPU network for that frame)", KSTEP, total);
  return;
fail:
  vlog("bounded GPU wait setup failed (%s): host-event wait", why); bw_free(d);
}
// the app's compute pipeline, restored after the steps. Thread-local (a command buffer records on one thread at a time);
// valid only for the recording this thread began (begin epoch in a small table keyed by the command buffer: a collision or a
// command buffer begun on another thread -> no restore, counted).
static std::atomic<uint64_t> g_epoch{0};
struct BeginEnt { std::atomic<VkCommandBuffer> cb{VK_NULL_HANDLE}; std::atomic<uint64_t> ep{0}; };
static BeginEnt g_begin[256];
static thread_local VkCommandBuffer tl_bcb = VK_NULL_HANDLE; static thread_local uint64_t tl_bep = 0;      // last Begin on this thread
static thread_local VkCommandBuffer tl_pcb = VK_NULL_HANDLE; static thread_local uint64_t tl_pep = 0; static thread_local VkPipeline tl_pipe = VK_NULL_HANDLE;
static unsigned cb_hash(VkCommandBuffer cb) { uintptr_t x = (uintptr_t)cb; return unsigned((x >> 4) ^ (x >> 12)) & 255u; }
static void note_begin(VkCommandBuffer cb) {
  const uint64_t e = g_epoch.fetch_add(1, std::memory_order_relaxed) + 1; BeginEnt& b = g_begin[cb_hash(cb)];
  b.cb.store(cb, std::memory_order_relaxed); b.ep.store(e, std::memory_order_release); tl_bcb = cb; tl_bep = e; if (tl_pcb == cb) tl_pcb = VK_NULL_HANDLE;
}
static VkPipeline app_pipe(VkCommandBuffer cb) {
  if (tl_pcb != cb || !tl_pipe) return VK_NULL_HANDLE;
  const BeginEnt& b = g_begin[cb_hash(cb)];
  return b.cb.load(std::memory_order_relaxed) == cb && b.ep.load(std::memory_order_acquire) == tl_pep ? tl_pipe : VK_NULL_HANDLE;
}
static bool ctx_init(Dev* d) {  // d->mtx held; first PROBE: status buffer, events, worker
  if (d->stBuf) return true;
  VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bc.size = NSLOT * 64; bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  if (d->CreateBuffer(d->dev, &bc, nullptr, &d->stBuf) != VK_SUCCESS) return false;
  VkMemoryRequirements mr; d->GetBufferMemoryRequirements(d->dev, d->stBuf, &mr);
  VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.allocationSize = mr.size;
  ma.memoryTypeIndex = mem_type(d, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  void* p = nullptr;
  if (ma.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->dev, &ma, nullptr, &d->stMem) != VK_SUCCESS ||
      d->BindBufferMemory(d->dev, d->stBuf, d->stMem, 0) != VK_SUCCESS || d->MapMemory(d->dev, d->stMem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return false;
  d->st = (volatile uint32_t*)p;
  for (Slot& s : d->slots) {
    VkEventCreateInfo ec{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    if (d->CreateEvent(d->dev, &ec, nullptr, &s.e1) != VK_SUCCESS || d->CreateEvent(d->dev, &ec, nullptr, &s.e2) != VK_SUCCESS) return false;
  }
  for (int i = 0; i < NSLOT; i++) { d->st[i * 16 + 0] = VELA_NPU_ACK_ERROR_TIMEOUT; d->st[i * 16 + 1] = 0; d->st[i * 16 + 4] = 0; d->st[i * 16 + 5] = 1; d->st[i * 16 + 6] = 1; }   // build 26: status NOT-OK until a worker job completes (a slot nobody wrote used to read as OK); indirect dispatch args (x, 1, 1)
  bw_init(d);             // bounded wait setup
  install_exit_hooks();   // E2 release at process end
  g_poll = g_poll_env = npu_poll_cfg(getenv("VELA_NPU_E1_SLEEP_PCT"), getenv("VELA_NPU_E1_STEP_US"));   // CPUW (PWR1 row 5)
  vlog("worker E1 poll: async sleep %.0f %% of the low submit->E1 (%s), then %d tight polls and %ld us steps (%s); exact: %.0f %%, %d tight, %ld us",
       g_poll.asyncShare * 100.0, g_poll.shareSet ? "VELA_NPU_E1_SLEEP_PCT" : "default", g_poll.asyncTight, g_poll.asyncStepNs / 1000L,
       g_poll.stepSet ? "VELA_NPU_E1_STEP_US" : "default", NpuPollCfg::kExactShare * 100.0, NpuPollCfg::kExactTight, NpuPollCfg::kExactStepNs / 1000L);
  d->worker = std::thread(worker_main, d);
  pthread_setname_np(d->worker.native_handle(), "vela-npu-w");   // visible in /proc/<pid>/task/*/comm for per-thread CPU
  if (const char* cpus = getenv("VELA_NPU_WORKER_CPUS")) {
    cpu_set_t cs; CPU_ZERO(&cs); int a = -1, n = 0; char* e = nullptr;
    for (const char* q = cpus; *q;) { long v = strtol(q, &e, 10); if (e == q) break; q = e;
      if (a >= 0) { for (long c = a; c <= v && c < CPU_SETSIZE; c++) { CPU_SET(c, &cs); n++; } a = -1; } else if (*q == '-') { a = int(v); } else { CPU_SET(v, &cs); n++; }
      if (*q == '-' || *q == ',') q++; }
    if (n) vlog("worker pinned to %s: %s", cpus, pthread_setaffinity_np(d->worker.native_handle(), sizeof cs, &cs) ? "failed" : "ok");
  }
  return true;
}
// A1: d->mtx held. Free the retired sets nothing can use any more (rules: vela_npu_retire.h). provenGen = generation of a job whose E1 was just seen, 0 = none.
static void retired_sweep(Dev* d, uint32_t provenGen) {
  if (d->retired.empty()) return; const double n = now_ms();
  for (size_t i = 0; i < d->retired.size();) { TSet& r = d->retired[i];
    if (!npu_retired_free(r.gen, r.tRet, r.tUse, provenGen, n)) { i++; continue; }
    const uint32_t g = r.gen; const double age = n - r.tRet; set_free(d, r); d->retired.erase(d->retired.begin() + i); d->retFreed++;
    vlog("retired gen %u freed after %.1f s (%s), %zu still retired", g, age / 1e3, provenGen && g < provenGen ? "a newer job passed the GPU" : "unused", d->retired.size()); }
}
static void go_down(Dev* d, const char* why) {  // d->mtx held: retire the set (GPU work may still use it: freed by retired_sweep or at vkDestroyDevice)
  if (d->state != 1) return;
  vlog("NPU DOWN (%s): gen %u retired, GPU path until fsr4npud is back", why, d->gen);
  if (d->set.npu) { fsr4npu_close(d->set.npu); d->set.npu = nullptr; }
  d->set.gen = d->gen; d->set.tRet = d->set.tUse = now_ms(); d->retired.push_back(d->set); d->set = TSet{};
  if (d->lastKick >= 0) { slot_unref(d, d->lastKick); d->lastKick = -1; }
  d->state = -1; d->nextRetry = now_ms() + env_int("VELA_NPU_RECONNECT_MS", 150);
}
static void try_up(Dev* d, int connect_ms) {  // d->mtx NOT held
  TSet ns; uint32_t nt = 0; uint64_t off[3], bytes[3]; char nm[64];
  { std::lock_guard<std::mutex> l(d->mtx); memcpy(nm, d->want, sizeof nm); }   // the daemon of the last PROBE
  double t0 = now_ms(); int st = set_open(d, nm, connect_ms, ns, &nt, off, bytes);
  std::lock_guard<std::mutex> l(d->mtx);
  if (st) {
    if (d->connFails++ % 200 == 0) vline("daemon not reachable (%s)", fsr4npu_strerror(st));   // about every 30 s
    if (d->state == 0) d->state = -1;
    d->nextRetry = now_ms() + env_int("VELA_NPU_RECONNECT_MS", 150); return;
  }
  if (d->state == -2 || npu_retire_hold(d->retired.size())) { set_free(d, ns); d->nextRetry = now_ms() + NpuRetire::kHoldMs; return; }   // A1: a brake with a retry (was: > 64 and never again)
  d->set = ns; d->nt = nt; for (int k = 0; k < 3; k++) { d->off[k] = off[k]; d->bytes[k] = bytes[k]; }
  if (d->state == -1) d->reconnects++;
  d->gen++; d->set.gen = d->gen; d->state = 1; d->connFails = 0; d->held = false;
  vline("daemon connected");
  vlog("NPU UP: gen %u, %u tensors, connect+alloc+import %.3f ms", d->gen, nt, now_ms() - t0);
}
// absolute sleep on CLOCK_MONOTONIC (now_ms() base)
static void sleep_until_ms(double t) {
  timespec ts; ts.tv_sec = time_t(t / 1e3); ts.tv_nsec = long((t - ts.tv_sec * 1e3) * 1e6); if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
}
// the worker's wake-up time for a job: the GPU reaches E1 no earlier than ~ the shortest recent submit->E1, so sleep
// through 85 % (async: its OP_WAIT is a frame later) / 70 % (exact: the GPU waits right after E1) of it, minus 0.25 ms.
// CPUW: the async share is g_poll.asyncShare (VELA_NPU_E1_SLEEP_PCT, default 85 %); exact stays 70 % (vela_npu_poll.h)
static double e1_wake(double tSubmit, double e1Low, bool async) { return npu_e1_wake(g_poll, tSubmit, e1Low, async); }
static double e1_low_next(double low, double e1) { return low <= 0 ? e1 : e1 < low ? 0.5 * low + 0.5 * e1 : 0.97 * low + 0.03 * e1; }  // fast down, slow up
static void line_diag(Dev* d) {  // d->mtx held; diag prefix: the plugin parser ignores it
  if (!d->wJobs) return;
  vlog("window %ld ok jobs: submit->E1 mean %.2f max %.2f ms (low %.2f, slept %.2f ms/job) | kick->done mean %.2f max %.2f ms | polls/job %.1f | async %ld",
       d->wJobs, d->wE1 / d->wJobs, d->wE1Max, d->e1Low, d->wSlept / d->wJobs, d->wNpu / d->wJobs, d->wNpuMax, double(d->wPolls) / d->wJobs, d->wAsync);
  d->wJobs = d->wPolls = d->wAsync = 0; d->wE1 = d->wE1Max = d->wNpu = d->wNpuMax = d->wSlept = 0;
}
static void line_bw(Dev* d) {  // diagnostic prefix
  if (d->bw) vlog("bounded waits %ld (app pipeline rebound %ld, not known %ld)", d->bwWaits, d->bwRestores, d->bwNoRestore);
}
static void line_slow(Dev* d) {  // d->mtx held; diagnostic prefix
  if (d->slowJobs || d->relJobs) vlog("slow NPU executes %ld (frames released to the GPU network while they ran: %ld)", d->slowJobs, d->relJobs);
}
static void line_l4(Dev* d) {  // d->mtx held
  vline("%ld jobs, HTP %.2f ms, timeouts %ld, errors %ld", d->jobs, d->okJobs ? d->sumNpu / d->okJobs : 0.0, d->e1Timeouts, d->npuErrors);
}

// ---------------------------------------------------------------- worker: E1 seen -> NPU -> final status -> E2 (always)
static void put_status(Dev* d, int slot, uint32_t ack, uint32_t htp_us) { d->st[slot * 16 + 0] = ack; d->st[slot * 16 + 1] = htp_us; }
// words 4..6 of the slot line = the steps' indirect args (x, 1, 1). Status first, then x (release + dsb: the CP reads memory).
static void put_x(Dev* d, int slot, uint32_t x) { __atomic_store_n((uint32_t*)&d->st[slot * 16 + 4], x, __ATOMIC_RELEASE); __asm__ volatile("dsb st" ::: "memory"); }
static void put_done(Dev* d, int slot) { if (d->bw) put_x(d, slot, 0); }
static void put_arm(Dev* d, int slot) { if (d->bw) { put_status(d, slot, VELA_NPU_ACK_ERROR_TIMEOUT, 0); put_x(d, slot, 1); } }   // before the driver submit
// final status TIMEOUT (the GPU network runs for that frame), then the GPU's wait ends: x = 0 and E2
static void put_release(Dev* d, int slot) { put_status(d, slot, VELA_NPU_ACK_ERROR_TIMEOUT, 0); put_done(d, slot); d->SetEvent(d->dev, d->slots[slot].e2); }
static void worker_main(Dev* d) {
  const double e1Tmo = env_int("VELA_NPU_E1_TIMEOUT_MS", 200); const int npuTmo = std::max(1, env_int("VELA_NPU_NPU_TIMEOUT_MS", 400));
  const int slowMs = std::min(npuTmo, std::max(1, env_int("VELA_NPU_SLOW_MS", 20)));
  std::unique_lock<std::mutex> lk(d->mtx);
  for (;;) {
    d->cv.wait_for(lk, std::chrono::milliseconds(50), [&] { return d->quit || !d->q.empty(); });
    for (int i = 0; i < NSLOT; i++) { Slot& s = d->slots[i];   // E1 timed out earlier: release once the GPU set it, or after 2 s
      if (s.quarantine && (d->GetEventStatus(d->dev, s.e1) == VK_EVENT_SET || now_ms() - s.qSince > 2000)) { d->ResetEvent(d->dev, s.e1); s.quarantine = false; slot_unref(d, i); } }
    if (d->q.empty()) {
      if (d->quit) return;
      retired_sweep(d, 0);   // A1: the age rule
      if (d->state == -1 && now_ms() >= d->nextRetry) {
        if (npu_retire_hold(d->retired.size())) { d->nextRetry = now_ms() + NpuRetire::kHoldMs;   // A1: no new buffer while this many old ones are still held
          if (!d->held) { d->held = true; vlog("NPU held off: %zu retired buffers are still in use by the GPU or the provider; next try every %.0f s", d->retired.size(), NpuRetire::kHoldMs / 1e3); } }
        else { d->nextRetry = now_ms() + 1e9; lk.unlock(); try_up(d, 20); lk.lock(); } }
      continue;
    }
    Job j = d->q.front(); d->q.pop_front();
    Slot& s = d->slots[j.slot];
    fsr4npu* npu = (d->state == 1 && j.gen == d->gen) ? d->set.npu : nullptr;
    const long jobsNow = d->jobs;
    lk.unlock();
    worker_live(jobsNow);   // CPUW2: test settings from VELA_NPU_LIVE_FILE (nothing without the variable)
    double t0 = now_ms(), deadline = t0 + e1Tmo; bool seen = false; VkResult er = VK_SUCCESS;
    // in a game the GPU reaches E1 10-20 ms after the submit (the frame's own work queued ahead); polling all of it (20 us
    // nanosleep steps + timer slack) keeps a game-process thread waking for that whole time. Sleep through the certain part, then poll (async: 8 tight + 100 us steps; exact: as before).
    const double wake = e1_wake(j.tSubmit, d->e1Low, j.async); long polls = 0;
    if (wake > t0) { sleep_until_ms(std::min(wake, deadline)); }
    const double slept = std::max(0.0, now_ms() - t0);
    for (int spin = 0; !d->quit && now_ms() < deadline; ++spin) {
      er = d->GetEventStatus(d->dev, s.e1); polls++;
      if (er == VK_EVENT_SET) { seen = true; break; }
      if (er < 0) break;                                                          // device lost
      if (spin > npu_tight_polls(g_poll, j.async)) { timespec ts{0, npu_step_ns(g_poll, j.async)}; nanosleep(&ts, nullptr); }   // tight, then steps (CPUW: async step VELA_NPU_E1_STEP_US)
    }
    double tE1 = now_ms(); int st = S_OK; uint32_t ack; bool slow = false;
    if (jl_seq(j.seq) || (!seen && jl_bad())) vlog("job seq %llu slot %d: worker got it, E1 %s after %.2f ms (submit->E1), %ld polls", (unsigned long long)j.seq, j.slot, seen ? "seen" : "NOT seen", tE1 - j.tSubmit, polls);
    if (seen) d->ResetEvent(d->dev, s.e1);
    if (!seen || j.released) ack = VELA_NPU_ACK_ERROR_TIMEOUT;
    else if (!npu) ack = VELA_NPU_ACK_NODAEMON;
    else { st = j.phase >= 0 ? fsr4npu_kick_phase(npu, j.in, j.out, j.phase) : (j.in || j.out) ? fsr4npu_kick_slots(npu, j.in, j.out) : fsr4npu_kick(npu); if (st == S_OK) st = fsr4npu_wait(npu, slowMs);
           // the poll ran out (the daemon's own E_TIMEOUT comes earlier): slow execute. Its frame and every queued one go to the GPU
           // network now; the wait for this execute goes on up to npuTmo on the same connection, so its "done" pairs with this kick.
           if (st == E_TIMEOUT && slowMs < npuTmo && now_ms() - tE1 >= slowMs - 0.5) {
             slow = true; put_release(d, j.slot);
             lk.lock(); d->slow = true; for (Job& o : d->q) if (!o.released) { put_release(d, o.slot); o.released = true; } lk.unlock();
             st = fsr4npu_wait(npu, std::max(1, int(tE1 + npuTmo - now_ms())));
           }
           ack = st == S_OK && !slow ? VELA_NPU_ACK_OK : st == E_DISCONNECTED || st == E_PROTO ? VELA_NPU_ACK_NODAEMON : VELA_NPU_ACK_ERROR_TIMEOUT; }
    double tDone = now_ms(), np = tDone - tE1;
    if (jl_seq(j.seq) || (ack != VELA_NPU_ACK_OK && jl_bad())) vlog("job seq %llu slot %d: kick sent, daemon %s after %.3f ms, ack %u (%s)", (unsigned long long)j.seq, j.slot, st == S_OK ? "exec done" : fsr4npu_strerror(st), np, ack, ack == VELA_NPU_ACK_OK ? "OK, status written" : "NOT OK");
    put_status(d, j.slot, ack, ack == VELA_NPU_ACK_OK ? uint32_t(np * 1000.0) : 0);   // coherent host memory, before the event
    put_done(d, j.slot);                                                          // the bounded wait's x = 0 (after the status)
    d->SetEvent(d->dev, s.e2);                                                    // ALWAYS: the GPU never waits on a dead or hung daemon
    lk.lock();
    d->slow = false; d->jobs++; s.prod = P_DONE;
    if (seen && !g_dying.load(std::memory_order_acquire)) for (int i = 0; i < NSLOT; i++) { Slot& o = d->slots[i];   // never after a release; a later E1 seen: the CP passed every earlier wait
      if (i != j.slot && o.e2Set && o.e2After < j.qn) { d->ResetEvent(d->dev, o.e2); o.e2Set = false; slot_unref(d, i); } }
    if (!s.e2Set) { s.e2Set = true; s.e2After = j.qn; s.refs++; }
    if (seen && !g_dying.load(std::memory_order_acquire)) retired_sweep(d, j.gen);   // A1: this job passed the GPU: sets of older generations are done
    if (ack == VELA_NPU_ACK_OK) { d->okJobs++; double e1 = tE1 - j.tSubmit; d->sumE1 += e1; d->sumNpu += np; d->maxE1 = std::max(d->maxE1, e1); d->maxNpu = std::max(d->maxNpu, np);
      d->e1Low = e1_low_next(d->e1Low, e1); d->wJobs++; d->wAsync += j.async ? 1 : 0; d->wPolls += polls; d->wE1 += e1; d->wE1Max = std::max(d->wE1Max, e1); d->wNpu += np; d->wNpuMax = std::max(d->wNpuMax, np); d->wSlept += slept; }
    else if (!seen) { d->e1Timeouts++; s.quarantine = true; s.qSince = now_ms(); s.refs++;
      vlog("job seq %llu slot %d: pack-done event not seen in %.0f ms%s", (unsigned long long)j.seq, j.slot, e1Tmo, er < 0 ? " (DEVICE LOST)" : "");
      if (er == VK_ERROR_DEVICE_LOST && d->state != -2) { go_down(d, "device lost"); d->state = -2; } }
    else if (j.released) d->relJobs++;
    else if (!npu) d->noDaemon++;
    else if (slow && st == S_OK) { d->slowJobs++;
      vlog("job seq %llu: slow NPU execute %.1f ms: frame on the GPU network from %d ms, connection kept", (unsigned long long)j.seq, np, slowMs); }
    else { d->npuErrors++; vlog("job seq %llu: NPU %s after %.1f ms -> fallback", (unsigned long long)j.seq, fsr4npu_strerror(st), np);
      if (j.gen == d->gen) go_down(d, fsr4npu_strerror(st)); }   // fresh connection: a late "done" must never pair with the next kick
    slot_unref(d, j.slot);
    if (d->jobs - d->l4At >= 600) { d->l4At = d->jobs; line_l4(d); line_diag(d); line_bw(d); line_slow(d); }
  }
}

// ---------------------------------------------------------------- recording at the marker
static const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, TR = VK_PIPELINE_STAGE_TRANSFER_BIT, HOST = VK_PIPELINE_STAGE_HOST_BIT;
static void mbar(Dev* d, VkCommandBuffer cb, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkAccessFlags sa, VkAccessFlags da) {
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = sa; b.dstAccessMask = da;
  d->CmdPipelineBarrier(cb, ss, ds, 0, 1, &b, 0, nullptr, 0, nullptr);
}
static int slot_take(Dev* d) {
  for (int k = 0; k < NSLOT; k++) { int i = int((d->nextSlot + k) % NSLOT); if (d->slots[i].refs == 0 && !d->slots[i].quarantine) { d->nextSlot = i + 1; return i; } }
  return -1;
}
// buffer <-> R32_UINT image (word j at pixel (j % w, j / w)): full rows + one partial row
static void tensor_copy(Dev* d, VkCommandBuffer cb, bool toImage, VkImage im, uint32_t w, uint32_t h, VkBuffer buf, uint64_t off, uint64_t bytes) {
  uint64_t words = std::min<uint64_t>(bytes / 4, uint64_t(w) * h); uint32_t rows = uint32_t(words / w), rest = uint32_t(words % w);
  VkBufferImageCopy r[2]{}; uint32_t n = 0;
  if (rows) { r[n].bufferOffset = off; r[n].bufferRowLength = w; r[n].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r[n].imageExtent = {w, rows, 1}; n++; }
  if (rest) { r[n].bufferOffset = off + uint64_t(rows) * w * 4; r[n].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r[n].imageOffset = {0, int32_t(rows), 0}; r[n].imageExtent = {rest, 1, 1}; n++; }
  if (!n) return;
  if (toImage) d->CmdCopyBufferToImage(cb, buf, im, VK_IMAGE_LAYOUT_GENERAL, n, r);
  else d->CmdCopyImageToBuffer(cb, im, VK_IMAGE_LAYOUT_GENERAL, buf, n, r);
}
// The bounded wait. Steps: indirect dispatch (x from the slot line, read by the CP) + compute -> indirect barrier (the next
// x is read only after the step ran). Then the same HOST_WRITE -> reads barrier as the old wait (Turnip invalidates the caches:
// the HTP's output and the worker's status line are read fresh). If x never turned 0 (process gone, worker late), the status
// stays the armed TIMEOUT -> the fork's select runs the GPU network for that frame. The app's compute pipeline is rebound.
static void bounded_wait(Dev* d, VkCommandBuffer cb, int si, VkPipelineStageFlags ds, VkAccessFlags da) {
  VkMemoryBarrier ib{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; ib.srcAccessMask = 0; ib.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  for (int k = 0; k < KSTEP; k++) {
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->wp[k]);
    d->CmdDispatchIndirect(cb, d->stBuf, uint64_t(si) * 64 + 16);
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &ib, 0, nullptr, 0, nullptr);
  }
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; b.dstAccessMask = da;
  d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | HOST, ds, 0, 1, &b, 0, nullptr, 0, nullptr);
  const VkPipeline ap = app_pipe(cb);
  if (ap) { d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ap); d->bwRestores++; } else d->bwNoRestore++;
  d->bwWaits++;
}
static void wait_and_copy_out(Dev* d, VkCommandBuffer cb, int si) {  // host-signalled E2: legacy vkCmdWaitEvents, srcStage HOST (outside a render pass)
  Slot& s = d->slots[si];
  if (s.zc) {   // zero-copy: no copy-out; make the HTP's writes visible to the provider's unpack (shader reads of the dma-buf)
    VkMemoryBarrier z{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; z.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; z.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    if (d->bw) bounded_wait(d, cb, si, ALL, z.dstAccessMask); else
    d->CmdWaitEvents(cb, 1, &s.e2, HOST, ALL, 1, &z, 0, nullptr, 0, nullptr);
    return;
  }
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
  if (d->bw) bounded_wait(d, cb, si, TR | ALL, b.dstAccessMask); else
  d->CmdWaitEvents(cb, 1, &s.e2, HOST, TR | ALL, 1, &b, 0, nullptr, 0, nullptr);
  const TSet& t = d->set;   // same gen as the kick (checked by the caller)
  tensor_copy(d, cb, true, s.out0, s.w, s.h, t.buf, d->off[1], d->bytes[1]);
  if (d->nt >= 3 && s.out1) tensor_copy(d, cb, true, s.out1, s.w, s.h, t.buf, d->off[2], d->bytes[2]);
  mbar(d, cb, TR, ALL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}
// zero-copy is offered when the set is up with in/out0/out1 at 16-byte offsets (uvec4 SSBO indexing); VELA_NPU_ZEROCOPY=0 disables.
static bool zc_ok(Dev* d) {  // d->mtx held
  static const bool off = [] { const char* v = getenv("VELA_NPU_ZEROCOPY"); return v && v[0] == '0'; }();
  return !off && d->state == 1 && d->set.buf && d->nt >= 2 && !(d->off[0] % 16) && !(d->off[1] % 16) && !(d->off[2] % 16);
}
static void zc_info(Dev* d, VelaNpuAck& a) {  // d->mtx held; every ack carries the CURRENT set (vela_npu_ext.h)
  if (!zc_ok(d)) return;
  uint32_t caps; memcpy(&caps, a.reserved, 4); caps |= VELA_NPU_CAP_ZEROCOPY; memcpy(a.reserved, &caps, 4);
  const uint64_t b = (uint64_t)(uintptr_t)d->set.buf; const uint32_t q[3] = {uint32_t(d->off[0] / 16), uint32_t(d->off[1] / 16), uint32_t(d->off[2] / 16)};
  memcpy(a.reserved + 8, &b, 8); memcpy(a.reserved + 16, &d->gen, 4); memcpy(a.reserved + 20, q, 12);
  if (d->set.bda) { caps |= VELA_NPU_CAP_BDA; memcpy(a.reserved, &caps, 4); memcpy(a.reserved + 32, &d->set.bda, 8); }   // device address
  if (d->set.offAlt && !(d->set.offAlt % 16)) { caps |= VELA_NPU_CAP_PINGPONG; memcpy(a.reserved, &caps, 4); const uint32_t qa = uint32_t(d->set.offAlt / 16); memcpy(a.reserved + 40, &qa, 4); }   // input ping-pong
  if (d->set.offOut1 && !(d->set.offOut1 % 16)) { caps |= VELA_NPU_CAP_OUTPINGPONG; memcpy(a.reserved, &caps, 4); const uint32_t qo = uint32_t(d->set.offOut1 / 16); memcpy(a.reserved + 44, &qo, 4); }   // output ping-pong
}
static bool img_ok(const VelaNpuImage& i) { return i.image && i.format == VK_FORMAT_R32_UINT && i.layout == VK_IMAGE_LAYOUT_GENERAL && i.width && i.height; }

// The daemon a PROBE asks for. VELA_NPU_F_GRAPH + cmd.reserved = graph ([a-z0-9_], <= 31) -> fsr4npud_<graph>
// (env VELA_NPUD_GRAPH_NAME overrides, for testing); no flag or a bad name -> VELA_NPUD_NAME or fsr4npud (Lite, unchanged).
static void probe_name(const VelaNpuCmd* c, char nm[64]) {
  const char* base = getenv("VELA_NPUD_NAME"); if (!base || !*base) base = FSR4NPU_DEFAULT_NAME;
  snprintf(nm, 64, "%s", base);
  if (!(c->flags & VELA_NPU_F_GRAPH)) return;
  char g[32]; size_t n = 0;
  for (; n < 31 && c->reserved[n]; n++) { const char ch = (char)c->reserved[n];
    if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_')) return; g[n] = ch; }
  if (!n) return; g[n] = 0;
  const char* ov = getenv("VELA_NPUD_GRAPH_NAME");
  if (ov && *ov) snprintf(nm, 64, "%s", ov); else snprintf(nm, 64, "fsr4npud_%s", g);
}
static void record_marker(Dev* d, VkCommandBuffer cb, const VelaNpuCmd* c, VkBuffer mk, VkDeviceSize off) {
  VelaNpuAck a{}; a.magic = VELA_NPU_MAGIC_ACK; a.version = VELA_NPU_VERSION; a.seq = c->seq; a.layerBuild = VL_BUILD; a.slot = ~0u;
  memcpy(a.reserved, &kCaps, 4);
  int copySlot = -1;
  if (c->version != VELA_NPU_VERSION) a.status = VELA_NPU_ACK_BADARGS;
  else if (c->op == VELA_NPU_OP_PROBE) {
    char nm[64]; probe_name(c, nm); bool first, sw = false, hold = false;
    { std::lock_guard<std::mutex> l(d->mtx); first = d->state == 0 && d->extOk && ctx_init(d); if (!d->extOk) d->state = -2;
      if (strcmp(nm, d->want)) { const bool had = d->want[0] != 0; memcpy(d->want, nm, sizeof nm);   // another daemon (graph)
        if (had && d->state == 1) go_down(d, "graph switch");
        sw = had && d->state == -1; }
      hold = npu_retire_hold(d->retired.size()); }   // A1
    // A2: the connect retries (20 ms steps up to the budget) only at the device's FIRST probe, where a daemon may still be starting. A probe for
    // another graph tries once: with no daemon it used to stall the recording thread 100 ms per graph name, twice every 300 frames; the worker
    // keeps retrying in the background.
    if ((first || sw) && !hold) try_up(d, first ? env_int("VELA_NPU_CONNECT_MS", 100) : 0);
    std::lock_guard<std::mutex> l(d->mtx);
    a.status = d->state == 1 ? VELA_NPU_ACK_OK : d->state == -2 ? VELA_NPU_ACK_ERROR_TIMEOUT : VELA_NPU_ACK_NODAEMON;
    if (a.status == VELA_NPU_ACK_OK) zc_info(d, a);
  } else if (c->op == VELA_NPU_OP_WAIT) {
    std::lock_guard<std::mutex> l(d->mtx);
    int si = d->lastKick;
    if (si < 0 || d->slots[si].seq != c->seq || d->slots[si].gen != d->gen || d->state != 1) { a.status = VELA_NPU_ACK_NODAEMON;   // nothing kicked for this seq
      if (jl_bad()) vlog("WAIT seq %llu in cb %p: nothing kicked for it (lastKick %d, state %d) -> NODAEMON", (unsigned long long)c->seq, (void*)cb, si, d->state); }
    else {
      CbRec& rec = d->recs[cb]; if (rec.jobs.empty() && rec.cons.empty()) d->nRec++;
      wait_and_copy_out(d, cb, si); rec.cons.push_back(Consumer{si, c->seq});
      if (jl_seq(c->seq)) vlog("job seq %llu slot %d: WAIT recorded in cb %p", (unsigned long long)c->seq, si, (void*)cb);
      d->lastKick = -1; slot_unref(d, si);                                    // the GPU order now holds the slot (E2 hold)
      a.slot = si; a.status = kPending; copySlot = si;
    }
    zc_info(d, a);
  } else if (c->op != VELA_NPU_OP_UPSCALE) a.status = VELA_NPU_ACK_BADARGS;
  else {
    std::lock_guard<std::mutex> l(d->mtx);
    const bool async = c->flags & VELA_NPU_F_ASYNC, zc = c->flags & VELA_NPU_F_ZEROCOPY; int si = -1;
    if (zc && !d->retired.empty()) { const double n = now_ms(); for (TSet& r : d->retired) if (r.gen == c->pad0) r.tUse = n; }   // A1: the provider still packs into that buffer
    if (!(c->flags & VELA_NPU_F_TENSORS) || !img_ok(c->color) || !img_ok(c->output) || (c->motion.image && !img_ok(c->motion)) ||
        c->output.width != c->color.width || c->output.height != c->color.height) a.status = VELA_NPU_ACK_BADARGS;
    else if (d->state == -2 || !d->stBuf) a.status = d->state == -2 ? VELA_NPU_ACK_ERROR_TIMEOUT : VELA_NPU_ACK_NODAEMON;
    else if (d->state != 1) { a.status = VELA_NPU_ACK_NODAEMON; d->noDaemon++; }
    else if (zc && (c->pad0 != d->gen || !zc_ok(d))) a.status = VELA_NPU_ACK_BADARGS;   // stale buffer (reconnect): GPU network this frame
    else if (d->slow || (si = slot_take(d)) < 0) a.status = VELA_NPU_ACK_BUSY;   // slow execute running: GPU network, no kick
    else {
      Slot& s = d->slots[si]; CbRec& rec = d->recs[cb]; if (rec.jobs.empty() && rec.cons.empty()) d->nRec++;
      put_status(d, si, VELA_NPU_ACK_ERROR_TIMEOUT, 0);   // build 26: NOT-OK before every kick; only a completed worker job writes OK
      if (jl_seq(c->seq)) vlog("job seq %llu slot %d: kick recorded in cb %p (flags 0x%x%s)", (unsigned long long)c->seq, si, (void*)cb, c->flags, async ? " async" : " exact");
      if (d->lastKick >= 0) { slot_unref(d, d->lastKick); d->lastKick = -1; }   // an older kick nobody waited for
      // copy-in: the fork's pack wrote the input image; release everything to the HOST domain; E1
      if (zc) mbar(d, cb, ALL, HOST, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);   // zero-copy: the pack wrote the dma-buf
      else {
      mbar(d, cb, ALL, TR, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
      tensor_copy(d, cb, false, (VkImage)c->color.image, c->color.width, c->color.height, d->set.buf, d->off[0], d->bytes[0]);
      mbar(d, cb, TR | ALL, HOST, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
      }
      d->CmdSetEvent(cb, s.e1, ALL);
      s.refs = 1; s.prod = P_RECORDED; s.seq = c->seq; s.gen = d->gen;
      s.out0 = (VkImage)c->output.image; s.out1 = (VkImage)c->motion.image; s.w = c->output.width; s.h = c->output.height; s.zc = zc;
      rec.jobs.push_back(Job{si, c->seq, d->gen, 0, 0, async});
      if (zc && (c->flags & VELA_NPU_F_INSLOT) && d->set.offAlt) rec.jobs.back().in = c->reserved[32] & 1;   // input ping-pong slot
      if (zc && (c->flags & VELA_NPU_F_OUTSLOT) && d->set.offOut1) rec.jobs.back().out = c->reserved[33] & 1;   // output ping-pong slot
      if (zc && (c->flags & VELA_NPU_F_PHASE)) rec.jobs.back().phase = c->reserved[34];   // ANF family phase
      a.slot = si;
      if (!async) { wait_and_copy_out(d, cb, si); a.status = kPending; copySlot = si; rec.cons.push_back(Consumer{si, c->seq}); }
      else { a.status = VELA_NPU_ACK_OK; d->lastKick = si; s.refs++; }        // "kicked"; hold until its OP_WAIT
    }
    zc_info(d, a);
  }
  if (mk) {
    d->CmdUpdateBuffer(cb, mk, off, sizeof a, &a);
    if (copySlot >= 0) {   // FINAL status + HTP us, from the worker's status slot, after the wait (GPU timeline)
      VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; bb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; bb.buffer = mk; bb.offset = off; bb.size = sizeof a;
      d->CmdPipelineBarrier(cb, TR, TR, 0, 0, nullptr, 1, &bb, 0, nullptr);
      VkBufferCopy r[2] = {{uint64_t(copySlot) * 64, off + offsetof(VelaNpuAck, status), 4}, {uint64_t(copySlot) * 64 + 4, off + offsetof(VelaNpuAck, reserved) + 4, 4}};
      d->CmdCopyBuffer(cb, d->stBuf, mk, 2, r);
      mbar(d, cb, TR, ALL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }
  }
}

// ---------------------------------------------------------------- hooks
static VKAPI_ATTR void VKAPI_CALL vl_CmdVelaNpuUpscale(VkCommandBuffer cb, const VelaNpuCmd* c) {  // native route, no ack
  Dev* d = dev_of(cb); if (d && c && c->magic == VELA_NPU_MAGIC_CMD) record_marker(d, cb, c, VK_NULL_HANDLE, 0);
}
static VKAPI_ATTR void VKAPI_CALL vl_CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize off, VkDeviceSize size, const void* data) {
  Dev* d = dev_of(cb);
  if (size != sizeof(VelaNpuCmd) || ((const VelaNpuCmd*)data)->magic != VELA_NPU_MAGIC_CMD) { d->CmdUpdateBuffer(cb, dst, off, size, data); return; }
  double t0 = now_ms();
  record_marker(d, cb, (const VelaNpuCmd*)data, dst, off);
  double dt = now_ms() - t0; if (((const VelaNpuCmd*)data)->op == VELA_NPU_OP_PROBE) return;
  std::lock_guard<std::mutex> l(d->mtx); d->markers++; d->sumRec += dt; d->maxRec = std::max(d->maxRec, dt);
}
static void drop_rec(Dev* d, VkCommandBuffer cb) {  // d->mtx held
  auto it = d->recs.find(cb); if (it == d->recs.end()) return;
  if (!it->second.submitted) for (Job& j : it->second.jobs) slot_unref(d, j.slot);
  d->recs.erase(it); d->nRec--;
}
static VKAPI_ATTR VkResult VKAPI_CALL vl_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi) {
  Dev* d = dev_of(cb);
  if (dg_fsr4()) { CbSt cs = cbs_get(cb); cbs_set(cb, 0, 0);
    if (g_prov.load(std::memory_order_relaxed)) { if (cs.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY && bi && bi->pInheritanceInfo) DG("begin cb %p SECONDARY flags 0x%x inherit renderPass %p subpass %u fb %p", (void*)cb, (unsigned)bi->flags, (void*)(uintptr_t)bi->pInheritanceInfo->renderPass, bi->pInheritanceInfo->subpass, (void*)(uintptr_t)bi->pInheritanceInfo->framebuffer); else DG("begin cb %p %s flags 0x%x", (void*)cb, dg_st(cs), bi ? (unsigned)bi->flags : 0u); } }
  note_begin(cb);   // remember the begin epoch
  if (d->nRec.load(std::memory_order_relaxed)) { std::lock_guard<std::mutex> l(d->mtx); drop_rec(d, cb); }
  return d->BeginCommandBuffer(cb, bi);
}
static VKAPI_ATTR void VKAPI_CALL vl_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {   // remembers the app's compute pipeline
  Dev* d = dev_of(cb);
  if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) { if (tl_bcb == cb) { tl_pcb = cb; tl_pep = tl_bep; tl_pipe = p; } else if (tl_pcb == cb) tl_pcb = VK_NULL_HANDLE; }
  d->CmdBindPipeline(cb, bp, p);
}
static VKAPI_ATTR VkResult VKAPI_CALL vl_AllocateCommandBuffers(VkDevice dev, const VkCommandBufferAllocateInfo* ai, VkCommandBuffer* cbs) {
  Dev* d = dev_of(dev); VkResult r = d->AllocateCommandBuffers(dev, ai, cbs);
  if (r == VK_SUCCESS && dg_fsr4()) { std::lock_guard<std::mutex> l(g_dgm); for (uint32_t i = 0; i < ai->commandBufferCount; i++) { CbSt& c = g_cbst[cbs[i]]; c = CbSt{}; c.level = ai->level; } }
  if (r == VK_SUCCESS) { std::lock_guard<std::mutex> l(d->mtx); for (uint32_t i = 0; i < ai->commandBufferCount; i++) d->cbPool[cbs[i]] = ai->commandPool; }
  return r;
}
static VKAPI_ATTR void VKAPI_CALL vl_FreeCommandBuffers(VkDevice dev, VkCommandPool pool, uint32_t n, const VkCommandBuffer* cbs) {
  Dev* d = dev_of(dev);
  { std::lock_guard<std::mutex> l(d->mtx); for (uint32_t i = 0; i < n; i++) if (cbs[i]) { d->cbPool.erase(cbs[i]); if (d->nRec) drop_rec(d, cbs[i]); } }
  d->FreeCommandBuffers(dev, pool, n, cbs);
}
static VKAPI_ATTR VkResult VKAPI_CALL vl_ResetCommandPool(VkDevice dev, VkCommandPool pool, VkCommandPoolResetFlags f) {
  Dev* d = dev_of(dev);
  if (d->nRec.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> l(d->mtx); std::vector<VkCommandBuffer> v;
    for (auto& r : d->recs) { auto p = d->cbPool.find(r.first); if (p != d->cbPool.end() && p->second == pool) v.push_back(r.first); }
    for (auto cb : v) drop_rec(d, cb);
  }
  return d->ResetCommandPool(dev, pool, f);
}
// Build 26: the Unity Vulkan path records the provider's markers into a SECONDARY command buffer that a primary one executes; the jobs and consumers
// were keyed by the secondary and never seen at vkQueueSubmit (kit e2ehc: daemon frames 0, status slot never written = OK). vkCmdExecuteCommands moves
// the secondary's record into the primary's. A secondary re-executed without a re-record has nothing left to move (its status then stays NOT-OK).
static VKAPI_ATTR void VKAPI_CALL vl_CmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* secs) {
  Dev* d = dev_of(cb);
  if (d->nRec.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> l(d->mtx);
    for (uint32_t i = 0; i < n; i++) {
      auto it = d->recs.find(secs[i]); if (it == d->recs.end() || secs[i] == cb) continue;
      CbRec src = std::move(it->second); d->recs.erase(it); d->nRec--;
      if (src.jobs.empty() && src.cons.empty()) continue;
      CbRec& dst = d->recs[cb]; if (dst.jobs.empty() && dst.cons.empty()) d->nRec++;
      if (jl_env() && (jl_seq(src.jobs.empty() ? src.cons[0].seq : src.jobs[0].seq) || jl_bad()))
        vlog("vkCmdExecuteCommands: primary cb %p took secondary cb %p: %zu jobs, %zu waits", (void*)cb, (void*)secs[i], src.jobs.size(), src.cons.size());
      for (Job& j : src.jobs) dst.jobs.push_back(j);
      for (Consumer& c : src.cons) dst.cons.push_back(c);
    }
  }
  d->CmdExecuteCommands(cb, n, secs);
}
// Before the driver submit: a wait whose producer was never submitted (dropped / reused) is released now (status NODAEMON).
// After: queue the producer jobs for the worker.
template <class GetCb> static VkResult submit_common(Dev* d, uint32_t n, GetCb cbs, VkResult (*call)(void*), void* ctx) {
  if (!d->nRec.load(std::memory_order_relaxed)) return call(ctx);
  std::vector<std::pair<VkCommandBuffer, CbRec*>> mine; std::vector<int> forcedSlots;
  {
    std::lock_guard<std::mutex> l(d->mtx); std::vector<int> batchProd;
    for (uint32_t i = 0; i < n; i++) cbs(i, [&](VkCommandBuffer cb) {
      auto it = d->recs.find(cb); if (it == d->recs.end()) return;
      for (Job& j : it->second.jobs) { batchProd.push_back(j.slot); put_arm(d, j.slot); }   // x = 1 before the GPU can reach the wait
      for (Consumer& c : it->second.cons) {
        Slot& s = d->slots[c.slot]; bool inBatch = std::find(batchProd.begin(), batchProd.end(), c.slot) != batchProd.end();
        if (!inBatch && (s.seq != c.seq || s.prod == P_RECORDED || s.prod == P_DROPPED)) {
          put_status(d, c.slot, VELA_NPU_ACK_NODAEMON, 0); put_done(d, c.slot); d->SetEvent(d->dev, s.e2); d->forced++;
          if (!s.e2Set) { s.e2Set = true; s.e2After = ~0ull; s.refs++; forcedSlots.push_back(c.slot); }
        }
      }
      mine.push_back({cb, &it->second});
    });
  }
  VkResult r = call(ctx);
  std::lock_guard<std::mutex> l(d->mtx); double t = now_ms(); bool any = false;
  if (r == VK_ERROR_DEVICE_LOST && d->state != -2) { go_down(d, "device lost at submit"); d->state = -2; }
  for (auto& m : mine) {
    auto it = d->recs.find(m.first); if (it == d->recs.end()) continue;
    for (Job j : it->second.jobs)
      if (r == VK_SUCCESS) { j.tSubmit = t; j.qn = ++d->qCounter; d->slots[j.slot].refs++; d->slots[j.slot].prod = P_QUEUED;
        if (d->slow) { put_release(d, j.slot); j.released = true; }   // recorded before a slow execute began: GPU network
        { if (jl_seq(j.seq)) vlog("job seq %llu slot %d: queued for the worker at submit (cb %p)", (unsigned long long)j.seq, j.slot, (void*)m.first); }
        d->q.push_back(j); any = true; }
    if (r == VK_SUCCESS && !it->second.submitted) { it->second.submitted = true; for (Job& j : it->second.jobs) slot_unref(d, j.slot); }
  }
  for (int i : forcedSlots) d->slots[i].e2After = d->qCounter;   // proof: E1 of a job from a later submit
  if (any) d->cv.notify_one();
  return r;
}
struct S1 { Dev* d; VkQueue q; uint32_t n; const VkSubmitInfo* si; VkFence f; };
struct S2 { Dev* d; VkQueue q; uint32_t n; const VkSubmitInfo2* si; VkFence f; PFN_vkQueueSubmit2 fn; };
static VKAPI_ATTR VkResult VKAPI_CALL vl_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* si, VkFence f) {
  Dev* d = dev_of(q); S1 c{d, q, n, si, f}; dg_frame("vkQueueSubmit");
  return submit_common(d, n, [&](uint32_t i, auto fn) { for (uint32_t k = 0; k < si[i].commandBufferCount; k++) fn(si[i].pCommandBuffers[k]); },
                       [](void* p) { S1* s = (S1*)p; return s->d->QueueSubmit(s->q, s->n, s->si, s->f); }, &c);
}
static VkResult submit2(VkQueue q, uint32_t n, const VkSubmitInfo2* si, VkFence f, bool khr) {
  Dev* d = dev_of(q); S2 c{d, q, n, si, f, khr ? d->QueueSubmit2KHR : d->QueueSubmit2}; dg_frame("vkQueueSubmit2");
  return submit_common(d, n, [&](uint32_t i, auto fn) { for (uint32_t k = 0; k < si[i].commandBufferInfoCount; k++) fn(si[i].pCommandBufferInfos[k].commandBuffer); },
                       [](void* p) { S2* s = (S2*)p; return s->fn(s->q, s->n, s->si, s->f); }, &c);
}
static VKAPI_ATTR VkResult VKAPI_CALL vl_QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* si, VkFence f) { return submit2(q, n, si, f, false); }
static VKAPI_ATTR VkResult VKAPI_CALL vl_QueueSubmit2KHR(VkQueue q, uint32_t n, const VkSubmitInfo2* si, VkFence f) { return submit2(q, n, si, f, true); }

static VKAPI_ATTR void VKAPI_CALL vl_DestroyDevice(VkDevice dev, const VkAllocationCallbacks* alloc) {
  Dev* d = dev_of(dev); if (!d) return;
  if (d->worker.joinable()) {
    { std::lock_guard<std::mutex> l(d->mtx); d->quit = true; } d->cv.notify_all(); d->worker.join();
  }
  if (d->on && d->stBuf) {
    for (int i = 0; i < NSLOT; i++) { put_done(d, i); if (d->slots[i].e2) d->SetEvent(dev, d->slots[i].e2); }  // release any GPU wait still pending
    line_l4(d); line_bw(d); line_slow(d);
    vlog("stats dev %p: markers %ld (record mean %.4f max %.3f ms) | jobs %ld ok %ld | E1 timeouts %ld, NPU errors %ld, no-daemon %ld, forced releases %ld, reconnects %ld | "
         "submit->E1 seen mean %.3f max %.3f ms | kick->done mean %.3f max %.3f ms | retired sets freed early %ld, left %zu", (void*)dev, d->markers, d->markers ? d->sumRec / d->markers : 0, d->maxRec,
         d->jobs, d->okJobs, d->e1Timeouts, d->npuErrors, d->noDaemon, d->forced, d->reconnects,
         d->okJobs ? d->sumE1 / d->okJobs : 0, d->maxE1, d->okJobs ? d->sumNpu / d->okJobs : 0, d->maxNpu, d->retFreed, d->retired.size());
  }
  if (d->DeviceWaitIdle) d->DeviceWaitIdle(dev);
  for (Slot& s : d->slots) { if (s.e1) d->DestroyEvent(dev, s.e1, nullptr); if (s.e2) d->DestroyEvent(dev, s.e2, nullptr); }
  bw_free(d);   // bounded wait resources
  set_free(d, d->set);
  for (auto& s : d->retired) set_free(d, s);
  if (d->stBuf) d->DestroyBuffer(dev, d->stBuf, nullptr);
  if (d->stMem) d->FreeMemory(dev, d->stMem, nullptr);
  PFN_vkDestroyDevice next = d->DestroyDevice;
  for (auto& e : g_dev) if (e.key.load() == key_of(dev)) { e.key.store(nullptr); e.d = nullptr; }
  delete d; next(dev, alloc);
}

// ---------------------------------------------------------------- GPU identity override (VELA_VK_SPOOF_GPU=nvidia)
// The values are OptiScaler's own Vulkan spoof (FT9/fx24/src/OptiScaler):
//   deviceName  "NVIDIA GeForce RTX 4090"   Config.h:498 SpoofedGPUName default; written at spoofing/Vulkan_Spoofing.cpp:111-112, 140-141
//   vendorID    0x10DE                       Config.h:494 SpoofedVendorId default VendorId::Nvidia = 0x10DE (OptiTypes.h:233); Vulkan_Spoofing.cpp:114, 142
//   deviceID    0x2684                       Config.h:495 SpoofedDeviceId default; Vulkan_Spoofing.cpp:115, 143
//   driverVersion VK_MAKE_API_VERSION(999, 99, 0, 0)                    Vulkan_Spoofing.cpp:116, 144
//   VkPhysicalDeviceDriverProperties: driverID VK_DRIVER_ID_NVIDIA_PROPRIETARY, driverName "NVIDIA", driverInfo "999.99"
//                                                                       Vulkan_Spoofing.cpp:153-158 (2KHR: 201-206)
// OptiScaler patches only VkPhysicalDeviceDriverProperties in the chain; the same three fields of VkPhysicalDeviceVulkan12Properties
// get the same values here (a Vulkan 1.2 app reads them there). Nothing else changes (deviceType, limits, features, extensions, UUIDs).
static int g_spoof = -1;
static bool spoof_on() { if (g_spoof < 0) { const char* v = getenv("VELA_VK_SPOOF_GPU"); g_spoof = v && !strcmp(v, "nvidia"); } return g_spoof; }
static const char kSpoofName[] = "NVIDIA GeForce RTX 4090";
static const uint32_t kSpoofVendor = 0x10DE, kSpoofDevice = 0x2684, kSpoofDriverVersion = VK_MAKE_API_VERSION(999, 99, 0, 0);
// Build 17 handshake for the OptiScaler fork (native Vulkan FSR 4): a device made with the FSR 4 set (VELA_VK_FSR4=1) is marked in the
// driver description. Without the mark the fork must not call the provider: winevulkan hands out a thunk for every function name.
static const char kFsr4Mark[] = " | VELA layer 17 fsr4dev";
static std::mutex g_fsr4m; static std::vector<VkPhysicalDevice> g_fsr4pd;   // physical devices a device with the FSR 4 set was created on
static bool fsr4_pd(VkPhysicalDevice pd) { std::lock_guard<std::mutex> l(g_fsr4m); for (auto p : g_fsr4pd) if (p == pd) return true; return false; }
static void spoof_driver(VkDriverId* id, char* name, char* info, bool mark) {
  *id = VK_DRIVER_ID_NVIDIA_PROPRIETARY;
  snprintf(name, VK_MAX_DRIVER_NAME_SIZE, "%s", "NVIDIA"); snprintf(info, VK_MAX_DRIVER_INFO_SIZE, "%s%s", "999.99", mark ? kFsr4Mark : "");
}
static void spoof_props(VkPhysicalDevice pd, VkPhysicalDeviceProperties* p, void* chain, const char* via) {
  const bool mark = fsr4_pd(pd);
  if (mark) { static std::atomic<bool> once{false}; if (!once.exchange(true)) vlog("FSR 4 handshake served (first call %s): driverInfo gets '%s'", via, kFsr4Mark); }
  static std::atomic<bool> once{false};
  if (!once.exchange(true))   // one line per process: the real identity and what every caller sees from now on
    vlog("GPU identity override (VELA_VK_SPOOF_GPU=nvidia, first call %s): vendor 0x%04x device 0x%04x '%s' driver 0x%08x -> "
         "vendor 0x%04x device 0x%04x '%s' driver 0x%08x, driverID NVIDIA_PROPRIETARY", via, p->vendorID, p->deviceID, p->deviceName,
         p->driverVersion, kSpoofVendor, kSpoofDevice, kSpoofName, kSpoofDriverVersion);
  snprintf(p->deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE, "%s", kSpoofName);
  p->vendorID = kSpoofVendor; p->deviceID = kSpoofDevice; p->driverVersion = kSpoofDriverVersion;
  for (auto* n = (VkBaseOutStructure*)chain; n; n = n->pNext) {
    if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) { auto* x = (VkPhysicalDeviceDriverProperties*)n; spoof_driver(&x->driverID, x->driverName, x->driverInfo, mark); }
    else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES) { auto* x = (VkPhysicalDeviceVulkan12Properties*)n; spoof_driver(&x->driverID, x->driverName, x->driverInfo, mark); }
  }
}
static Inst* inst_of_pd(VkPhysicalDevice pd) {   // a physical device shares its instance's dispatch key (vl_CreateDevice relies on it too)
  std::lock_guard<std::mutex> l(g_mtx); auto it = g_inst.find(key_of(pd)); return it == g_inst.end() ? nullptr : it->second;
}
static VKAPI_ATTR void VKAPI_CALL vl_GetPhysicalDeviceProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties* p) {
  Inst* in = inst_of_pd(pd); if (!in || !in->gpdp) return;
  in->gpdp(pd, p); spoof_props(pd, p, nullptr, "vkGetPhysicalDeviceProperties");
}
static VKAPI_ATTR void VKAPI_CALL vl_GetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p) {
  Inst* in = inst_of_pd(pd); if (!in || !in->gpdp2) return;
  in->gpdp2(pd, p); spoof_props(pd, &p->properties, p->pNext, "vkGetPhysicalDeviceProperties2");
}
static VKAPI_ATTR void VKAPI_CALL vl_GetPhysicalDeviceProperties2KHR(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p) {
  Inst* in = inst_of_pd(pd); if (!in || !in->gpdp2khr) return;
  in->gpdp2khr(pd, p); spoof_props(pd, &p->properties, p->pNext, "vkGetPhysicalDeviceProperties2KHR");
}

// Build 15: the device extensions an NVIDIA driver lists and a DLSS integration checks before it offers DLSS (Unity's NVIDIA
// module in Haste Demo stops after NGX init without them). Same three names and the same idea as OptiScaler's own Vulkan
// extension spoof (Vulkan_Spoofing.cpp:599-727 lists them, 443-446 removes them again in vkCreateDevice): listed here when
// the driver lacks them, never passed to the driver.
static const VkExtensionProperties kNvxExt[] = {
  {VK_NVX_BINARY_IMPORT_EXTENSION_NAME, VK_NVX_BINARY_IMPORT_SPEC_VERSION},
  {VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME, VK_NVX_IMAGE_VIEW_HANDLE_SPEC_VERSION},
  {VK_NVX_MULTIVIEW_PER_VIEW_ATTRIBUTES_EXTENSION_NAME, VK_NVX_MULTIVIEW_PER_VIEW_ATTRIBUTES_SPEC_VERSION}};
static bool is_nvx_fake(const char* n) { for (auto& e : kNvxExt) if (!strcmp(n, e.extensionName)) return true; return false; }
static VKAPI_ATTR VkResult VKAPI_CALL vl_EnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char* layer, uint32_t* cnt, VkExtensionProperties* props) {
  Inst* in = inst_of_pd(pd); if (!in || !in->edep) return VK_ERROR_INITIALIZATION_FAILED;
  if (layer) return in->edep(pd, layer, cnt, props);
  uint32_t n = 0; VkResult r = in->edep(pd, nullptr, &n, nullptr); if (r != VK_SUCCESS) return r;
  std::vector<VkExtensionProperties> all(n); r = in->edep(pd, nullptr, &n, all.data()); if (r < 0) return r; all.resize(n);
  for (auto& e : kNvxExt) { bool have = false; for (auto& a : all) have |= !strcmp(a.extensionName, e.extensionName); if (!have) all.push_back(e); }
  static std::atomic<bool> once{false};
  if (!once.exchange(true)) vline("GPU identity override: device extensions %u -> %u (NVX binary_import, image_view_handle, multiview_per_view_attributes listed)", n, uint32_t(all.size()));
  if (!props) { *cnt = uint32_t(all.size()); return VK_SUCCESS; }
  uint32_t w = *cnt < all.size() ? *cnt : uint32_t(all.size());
  for (uint32_t i = 0; i < w; i++) props[i] = all[i];
  *cnt = w; return w < all.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

// ---------------------------------------------------------------- instance / device plumbing
static VKAPI_ATTR VkResult VKAPI_CALL vl_CreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks* a, VkInstance* out) {
  auto* li = (VkLayerInstanceCreateInfo*)ci->pNext;
  while (li && !(li->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && li->function == VK_LAYER_LINK_INFO)) li = (VkLayerInstanceCreateInfo*)li->pNext;
  if (!li) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = li->u.pLayerInfo->pfnNextGetInstanceProcAddr; li->u.pLayerInfo = li->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
  VkInstanceCreateInfo ci2 = *ci; VkApplicationInfo ai2{};   // build 20: VELA_VK_FSR4=1 raises the app's Vulkan 1.1 / 1.2 to 1.3 when the loader has it (the provider's SPIR-V is 1.5)
  { const char* fv = getenv("VELA_VK_FSR4"); const char* nv = getenv("VELA_VK_APIUP");
    if (fv && !strcmp(fv, "1") && (nv && !strcmp(nv, "1")) && ci->pApplicationInfo && ci->pApplicationInfo->apiVersion >= VK_API_VERSION_1_1 && ci->pApplicationInfo->apiVersion < VK_API_VERSION_1_3) {
      uint32_t iv = VK_API_VERSION_1_0; auto eiv = (PFN_vkEnumerateInstanceVersion)gipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"); if (eiv) eiv(&iv);
      vline("instance: the app asks for %u.%u, loader offers %u.%u", VK_VERSION_MAJOR(ci->pApplicationInfo->apiVersion), VK_VERSION_MINOR(ci->pApplicationInfo->apiVersion), VK_VERSION_MAJOR(iv), VK_VERSION_MINOR(iv));
      if (iv >= VK_API_VERSION_1_3) { ai2 = *ci->pApplicationInfo; ai2.apiVersion = VK_API_VERSION_1_3; ci2.pApplicationInfo = &ai2; vline("instance: apiVersion raised to 1.3"); } } }
  VkResult r = create(&ci2, a, out); if (r != VK_SUCCESS) return r;
  Inst* ni = new Inst{gipa, *out};
  { const char* fv = getenv("VELA_VK_FSR4"); if (fv && !strcmp(fv, "1")) { const uint32_t av = ci->pApplicationInfo ? ci->pApplicationInfo->apiVersion : 0; vline("instance %p: the app asks for Vulkan %u.%u.%u (engine '%s')", (void*)*out, VK_VERSION_MAJOR(av), VK_VERSION_MINOR(av), VK_VERSION_PATCH(av), ci->pApplicationInfo && ci->pApplicationInfo->pEngineName ? ci->pApplicationInfo->pEngineName : ""); } }  // build 18: log only
  if (spoof_on()) {   // GPU identity override: the next layer's physical-device functions (off: nothing asked, as build 12)
    ni->gpdp = (PFN_vkGetPhysicalDeviceProperties)gipa(*out, "vkGetPhysicalDeviceProperties");
    ni->gpdp2 = (PFN_vkGetPhysicalDeviceProperties2)gipa(*out, "vkGetPhysicalDeviceProperties2");
    ni->gpdp2khr = (PFN_vkGetPhysicalDeviceProperties2KHR)gipa(*out, "vkGetPhysicalDeviceProperties2KHR");
    ni->edep = (PFN_vkEnumerateDeviceExtensionProperties)gipa(*out, "vkEnumerateDeviceExtensionProperties");
  }
  { std::lock_guard<std::mutex> l(g_mtx); g_inst[key_of(*out)] = ni; }
  static std::atomic<bool> once{false};
  if (enabled() && !once.exchange(true)) vline("loaded v%u", VL_BUILD);
  return r;
}
static VKAPI_ATTR void VKAPI_CALL vl_DestroyInstance(VkInstance inst, const VkAllocationCallbacks* a) {
  Inst* in; { std::lock_guard<std::mutex> l(g_mtx); in = g_inst[key_of(inst)]; g_inst.erase(key_of(inst)); }
  if (!in) return;
  auto fn = (PFN_vkDestroyInstance)in->gipa(inst, "vkDestroyInstance"); delete in; fn(inst, a);
}
static bool bda_enabled(const VkDeviceCreateInfo* ci) {  // the app enabled bufferDeviceAddress (vkd3d-proton and fsr4vk do)
  for (auto* p = (const VkBaseInStructure*)ci->pNext; p; p = p->pNext) {
    if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES && ((const VkPhysicalDeviceVulkan12Features*)p)->bufferDeviceAddress) return true;
    if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES && ((const VkPhysicalDeviceBufferDeviceAddressFeatures*)p)->bufferDeviceAddress) return true;
  }
  return false;
}
// ---------------------------------------------------------------- build 17: the FSR 4 device set (VELA_VK_FSR4=1)
// The FSR 4 provider (amd_fidelityfx_upscaler_vk.dll, round 11 design AUDIT/r11/NOTES.md 4.1) needs extensions and features that can only be
// switched on at vkCreateDevice. vkd3d-proton does that for a DX12 game; a native Vulkan game (Unity) does not, and no hook of the fork sees
// its vkCreateDevice. Here, under winevulkan, every caller is seen. Everything is find-or-add so the pNext chain stays valid; structs of the
// app's chain are const, their bits are set through a cast and put back right after the next layer's vkCreateDevice returned.
// The 1.3.275 headers of the build image predate VK_KHR_compute_shader_derivatives; its features struct has the NV one's sType (1000201000) and fields.
static const char kKhrCsDerivName[] = "VK_KHR_compute_shader_derivatives";
static int g_fsr4 = -1;
static bool fsr4_on() { if (g_fsr4 < 0) { const char* v = getenv("VELA_VK_FSR4"); g_fsr4 = v && !strcmp(v, "1"); } return g_fsr4; }
struct Fsr4Set {
  VkPhysicalDeviceFeatures core{};
  VkPhysicalDeviceShaderFloat16Int8Features f16{};
  VkPhysicalDevice8BitStorageFeatures s8{};
  VkPhysicalDeviceDescriptorIndexingFeatures di{};
  VkPhysicalDeviceBufferDeviceAddressFeatures bda{};
  VkPhysicalDeviceSynchronization2Features s2{};
  VkPhysicalDeviceShaderIntegerDotProductFeatures dot{};
  VkPhysicalDeviceDescriptorBufferFeaturesEXT db{};
  VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mut{};
  VkPhysicalDeviceComputeShaderDerivativesFeaturesNV cd{};
  VkPhysicalDeviceSubgroupSizeControlFeatures ssc{};   // build 19: the provider's pipelines carry a required subgroup size (Khronos validation, k2r11c/d)
  std::vector<std::pair<VkBool32*, VkBool32>> undo;   // the app's own struct bits changed, with their old values
  std::vector<const char*> exts;                       // final extension list (the caller's plus the added ones)
  std::string addedExt, setFeat, structs;
  void undoAll() { for (auto& u : undo) *u.first = u.second; undo.clear(); }
};
// Fills S and the create info `dci` (a copy of the caller's). false: nothing usable, `why` says what is missing.
static bool fsr4_build(Inst* in, VkPhysicalDevice pd, PFN_vkGetInstanceProcAddr gipa, const VkDeviceCreateInfo* ci, const std::vector<const char*>& have,
                       Fsr4Set& S, VkDeviceCreateInfo& dci, std::string& why) {
  auto enumExt = (PFN_vkEnumerateDeviceExtensionProperties)gipa(in->inst, "vkEnumerateDeviceExtensionProperties");
  auto gpdf2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(in->inst, "vkGetPhysicalDeviceFeatures2");
  if (!gpdf2) gpdf2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(in->inst, "vkGetPhysicalDeviceFeatures2KHR");
  if (!enumExt || !gpdf2) { why = "no vkEnumerateDeviceExtensionProperties / vkGetPhysicalDeviceFeatures2"; return false; }
  uint32_t ne = 0; enumExt(pd, nullptr, &ne, nullptr); std::vector<VkExtensionProperties> ep(ne); enumExt(pd, nullptr, &ne, ep.data()); ep.resize(ne);
  auto listed = [&](const char* n) { for (auto& e : ep) if (!strcmp(e.extensionName, n)) return true; return false; };
  auto enabled_ = [&](const char* n) { for (auto* e : have) if (!strcmp(e, n)) return true; return false; };
  // extensions: the three unpromoted ones must exist; the rest is added when listed (promoted ones are core on a 1.2 / 1.3 driver)
  const char* must[] = {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME, kKhrCsDerivName};
  std::string miss;
  for (const char* m : must) {
    bool ok = listed(m) || (!strcmp(m, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME) && (listed("VK_VALVE_mutable_descriptor_type") || enabled_("VK_VALVE_mutable_descriptor_type")));
    if (!ok) miss += std::string(miss.empty() ? "" : " ") + m;
  }
  if (!miss.empty()) { why = "driver lacks " + miss; return false; }
  if (enabled_("VK_EXT_buffer_device_address")) { why = "the app enabled VK_EXT_buffer_device_address (invalid together with the KHR / core one)"; return false; }
  const char* add[] = {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME, kKhrCsDerivName,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
    VK_KHR_8BIT_STORAGE_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME, VK_KHR_MAINTENANCE_3_EXTENSION_NAME, VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
    VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
  const bool valveMut = enabled_("VK_VALVE_mutable_descriptor_type");
  S.exts = have;
  for (const char* w : add) {
    if (valveMut && !strcmp(w, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME)) continue;
    if (!listed(w) || enabled_(w)) continue;
    S.exts.push_back(w); S.addedExt += (S.addedExt.empty() ? "" : " "); S.addedExt += w;
  }
  // support of every feature
  VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceShaderFloat16Int8Features qf16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  VkPhysicalDevice8BitStorageFeatures qs8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
  VkPhysicalDeviceDescriptorIndexingFeatures qdi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
  VkPhysicalDeviceBufferDeviceAddressFeatures qbda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceSynchronization2Features qs2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
  VkPhysicalDeviceShaderIntegerDotProductFeatures qdot{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
  VkPhysicalDeviceDescriptorBufferFeaturesEXT qdb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
  VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT qmut{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT};
  VkPhysicalDeviceComputeShaderDerivativesFeaturesNV qcd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV};
  VkPhysicalDeviceSubgroupSizeControlFeatures qssc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
  q.pNext = &qf16; qf16.pNext = &qs8; qs8.pNext = &qdi; qdi.pNext = &qbda; qbda.pNext = &qs2; qs2.pNext = &qdot; qdot.pNext = &qdb; qdb.pNext = &qmut; qmut.pNext = &qcd; qcd.pNext = &qssc;
  gpdf2(pd, &q);
  struct { const char* n; VkBool32 v; } need[] = {
    {"shaderInt16", q.features.shaderInt16}, {"shaderStorageImageReadWithoutFormat", q.features.shaderStorageImageReadWithoutFormat},
    {"shaderStorageImageWriteWithoutFormat", q.features.shaderStorageImageWriteWithoutFormat}, {"shaderFloat16", qf16.shaderFloat16},
    {"shaderInt8", qf16.shaderInt8}, {"storageBuffer8BitAccess", qs8.storageBuffer8BitAccess}, {"runtimeDescriptorArray", qdi.runtimeDescriptorArray},
    {"descriptorBindingVariableDescriptorCount", qdi.descriptorBindingVariableDescriptorCount}, {"bufferDeviceAddress", qbda.bufferDeviceAddress},
    {"synchronization2", qs2.synchronization2}, {"shaderIntegerDotProduct", qdot.shaderIntegerDotProduct}, {"descriptorBuffer", qdb.descriptorBuffer},
    {"mutableDescriptorType", qmut.mutableDescriptorType}, {"computeDerivativeGroupLinear", qcd.computeDerivativeGroupLinear}};
  for (auto& n : need) if (!n.v) miss += std::string(miss.empty() ? "" : " ") + n.n;
  if (!miss.empty()) { why = "driver lacks features " + miss; return false; }

  auto on = [&](VkBool32& b, const char* nm) { if (!b) { S.undo.push_back({&b, b}); b = VK_TRUE; S.setFeat += (S.setFeat.empty() ? "" : " "); S.setFeat += nm; } };
  auto fresh = [&](VkBool32& b, const char* nm) { b = VK_TRUE; S.setFeat += (S.setFeat.empty() ? "" : " "); S.setFeat += nm; };
  // walk the app's chain (const in the API; restored by the caller)
  bool hSSC = false, hF2 = false, h12 = false, h13 = false, hF16 = false, hS8 = false, hDI = false, hBDA = false, hS2 = false, hDOT = false, hDB = false, hMUT = false, hCD = false;
  for (auto* n = (VkBaseOutStructure*)ci->pNext; n; n = n->pNext) {
    switch (n->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: { auto* x = (VkPhysicalDeviceFeatures2*)n; hF2 = true;
        on(x->features.shaderInt16, "shaderInt16"); on(x->features.shaderStorageImageReadWithoutFormat, "shaderStorageImageReadWithoutFormat");
        on(x->features.shaderStorageImageWriteWithoutFormat, "shaderStorageImageWriteWithoutFormat"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: { auto* x = (VkPhysicalDeviceVulkan12Features*)n; h12 = true;
        on(x->shaderFloat16, "shaderFloat16"); on(x->shaderInt8, "shaderInt8"); on(x->storageBuffer8BitAccess, "storageBuffer8BitAccess");
        on(x->descriptorIndexing, "descriptorIndexing"); on(x->runtimeDescriptorArray, "runtimeDescriptorArray");
        on(x->descriptorBindingVariableDescriptorCount, "descriptorBindingVariableDescriptorCount"); on(x->bufferDeviceAddress, "bufferDeviceAddress"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: { auto* x = (VkPhysicalDeviceVulkan13Features*)n; h13 = true;
        on(x->synchronization2, "synchronization2"); on(x->shaderIntegerDotProduct, "shaderIntegerDotProduct");
        if (qssc.subgroupSizeControl) { on(x->subgroupSizeControl, "subgroupSizeControl"); if (qssc.computeFullSubgroups) on(x->computeFullSubgroups, "computeFullSubgroups"); } break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: { auto* x = (VkPhysicalDeviceSubgroupSizeControlFeatures*)n; hSSC = true;
        if (qssc.subgroupSizeControl) { on(x->subgroupSizeControl, "subgroupSizeControl"); if (qssc.computeFullSubgroups) on(x->computeFullSubgroups, "computeFullSubgroups"); } break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: { auto* x = (VkPhysicalDeviceShaderFloat16Int8Features*)n; hF16 = true;
        on(x->shaderFloat16, "shaderFloat16"); on(x->shaderInt8, "shaderInt8"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: { auto* x = (VkPhysicalDevice8BitStorageFeatures*)n; hS8 = true; on(x->storageBuffer8BitAccess, "storageBuffer8BitAccess"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES: { auto* x = (VkPhysicalDeviceDescriptorIndexingFeatures*)n; hDI = true;
        on(x->runtimeDescriptorArray, "runtimeDescriptorArray"); on(x->descriptorBindingVariableDescriptorCount, "descriptorBindingVariableDescriptorCount"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: { auto* x = (VkPhysicalDeviceBufferDeviceAddressFeatures*)n; hBDA = true; on(x->bufferDeviceAddress, "bufferDeviceAddress"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES: { auto* x = (VkPhysicalDeviceSynchronization2Features*)n; hS2 = true; on(x->synchronization2, "synchronization2"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: { auto* x = (VkPhysicalDeviceShaderIntegerDotProductFeatures*)n; hDOT = true; on(x->shaderIntegerDotProduct, "shaderIntegerDotProduct"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT: { auto* x = (VkPhysicalDeviceDescriptorBufferFeaturesEXT*)n; hDB = true; on(x->descriptorBuffer, "descriptorBuffer"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT: { auto* x = (VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT*)n; hMUT = true; on(x->mutableDescriptorType, "mutableDescriptorType"); break; }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV: { auto* x = (VkPhysicalDeviceComputeShaderDerivativesFeaturesNV*)n; hCD = true; on(x->computeDerivativeGroupLinear, "computeDerivativeGroupLinear"); break; }
      default: break;
    }
    S.structs += (S.structs.empty() ? "" : ",") + std::to_string((unsigned)n->sType);
  }
  // core features without a Features2 struct: pEnabledFeatures (a local copy)
  if (!hF2) {
    if (ci->pEnabledFeatures) S.core = *ci->pEnabledFeatures;
    fresh(S.core.shaderInt16, "shaderInt16(core)"); fresh(S.core.shaderStorageImageReadWithoutFormat, "shaderStorageImageReadWithoutFormat(core)");
    fresh(S.core.shaderStorageImageWriteWithoutFormat, "shaderStorageImageWriteWithoutFormat(core)");
    dci.pEnabledFeatures = &S.core;
  }
  // new structs, prepended; the loader's link info stays in the chain behind them
  const void* head = dci.pNext;
  auto pre = [&](auto& st) { st.pNext = const_cast<void*>(head); head = &st; };
  if (!h12) {   // with a 1.2 struct in the chain it covers these; a separate struct of the same feature next to it would be invalid
    if (!hF16) { S.f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES; fresh(S.f16.shaderFloat16, "shaderFloat16"); fresh(S.f16.shaderInt8, "shaderInt8"); pre(S.f16); }
    if (!hS8) { S.s8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES; fresh(S.s8.storageBuffer8BitAccess, "storageBuffer8BitAccess"); pre(S.s8); }
    if (!hDI) { S.di.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES; fresh(S.di.runtimeDescriptorArray, "runtimeDescriptorArray");
      fresh(S.di.descriptorBindingVariableDescriptorCount, "descriptorBindingVariableDescriptorCount"); pre(S.di); }
    if (!hBDA) { S.bda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES; fresh(S.bda.bufferDeviceAddress, "bufferDeviceAddress"); pre(S.bda); }
  }
  if (!h13) {
    if (!hS2) { S.s2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES; fresh(S.s2.synchronization2, "synchronization2"); pre(S.s2); }
    if (!hDOT) { S.dot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES; fresh(S.dot.shaderIntegerDotProduct, "shaderIntegerDotProduct"); pre(S.dot); }
  }
  if (!hDB) { S.db.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT; fresh(S.db.descriptorBuffer, "descriptorBuffer"); pre(S.db); }
  if (!hMUT) { S.mut.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT; fresh(S.mut.mutableDescriptorType, "mutableDescriptorType"); pre(S.mut); }
  if (!h13 && !hSSC && qssc.subgroupSizeControl) { S.ssc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES; fresh(S.ssc.subgroupSizeControl, "subgroupSizeControl"); if (qssc.computeFullSubgroups) fresh(S.ssc.computeFullSubgroups, "computeFullSubgroups"); pre(S.ssc); }
  if (!hCD) { S.cd.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV; fresh(S.cd.computeDerivativeGroupLinear, "computeDerivativeGroupLinear"); pre(S.cd); }
  dci.pNext = head;
  dci.enabledExtensionCount = uint32_t(S.exts.size()); dci.ppEnabledExtensionNames = S.exts.data();
  vlog("FSR 4 device set: app chain sTypes [%s]", S.structs.c_str());
  return true;
}

static VKAPI_ATTR VkResult VKAPI_CALL vl_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks* a, VkDevice* out) {
  auto* li = (VkLayerDeviceCreateInfo*)ci->pNext;
  while (li && !(li->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && li->function == VK_LAYER_LINK_INFO)) li = (VkLayerDeviceCreateInfo*)li->pNext;
  if (!li) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = li->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr gdpa = li->u.pLayerInfo->pfnNextGetDeviceProcAddr; li->u.pLayerInfo = li->u.pLayerInfo->pNext;
  Inst* in = nullptr; { std::lock_guard<std::mutex> l(g_mtx); auto it = g_inst.find(key_of(pd)); if (it != g_inst.end()) in = it->second; }
  if (!in) return VK_ERROR_INITIALIZATION_FAILED;
  const bool on = enabled();
  std::vector<const char*> exts(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
  int have = 0;
  if (spoof_on()) {  // build 15: the listed-only NVIDIA extensions never reach the driver (it has none of them)
    size_t before = exts.size();
    exts.erase(std::remove_if(exts.begin(), exts.end(), [](const char* e) { return is_nvx_fake(e); }), exts.end());
    if (exts.size() != before) vline("GPU identity override: %u listed-only NVX extension(s) removed from vkCreateDevice", unsigned(before - exts.size()));
    // Build 16: the device extensions OptiScaler's own vkCreateDevice hook adds for its upscalers (FT9/fx24/src/OptiScaler/
    // spoofing/Vulkan_Spoofing.cpp hkvkCreateDevice: push_descriptor, get_memory_requirements2, external memory / semaphore,
    // timeline semaphore, dedicated allocation, float16_int8, integer dot product, mutable descriptor type, synchronization2).
    // A game whose Vulkan calls bypass OptiScaler's hooks (Unity x64 under FEX) never gets them, and SGSR 2 / FSR 3 / FSR 2
    // then fail to create or draw nothing (Haste k2lsp4). Added here only when the driver has them and the game did not.
    // Not added: OptiScaler's EXT_buffer_device_address (the game may use the core/KHR one; both together are invalid) and
    // the Win32 / NV / AMD ones the host driver does not have.
    auto enumExt = (PFN_vkEnumerateDeviceExtensionProperties)gipa(in->inst, "vkEnumerateDeviceExtensionProperties");
    uint32_t ne = 0; enumExt(pd, nullptr, &ne, nullptr); std::vector<VkExtensionProperties> ep(ne); enumExt(pd, nullptr, &ne, ep.data()); ep.resize(ne);
    const char* add[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
      VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
      VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME,
      VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME};
    bool valveMut = false; for (auto* e : exts) valveMut |= !strcmp(e, "VK_VALVE_mutable_descriptor_type");
    std::string added;
    for (const char* w : add) {
      if (valveMut && !strcmp(w, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME)) continue;
      bool sup = false, en = false;
      for (auto& e : ep) sup |= !strcmp(e.extensionName, w);
      for (auto* e : exts) en |= !strcmp(e, w);
      if (sup && !en) { exts.push_back(w); added += added.empty() ? "" : " "; added += w; }
    }
    vline("GPU identity override: vkCreateDevice gets OptiScaler's upscaler extensions: %s", added.empty() ? "none missing" : added.c_str());
  }
  if (on) {  // add the dma-buf import extensions when the driver has them and the app did not enable them (vkd3d does not)
    auto enumExt = (PFN_vkEnumerateDeviceExtensionProperties)gipa(in->inst, "vkEnumerateDeviceExtensionProperties");
    uint32_t ne = 0; enumExt(pd, nullptr, &ne, nullptr); std::vector<VkExtensionProperties> ep(ne); enumExt(pd, nullptr, &ne, ep.data());
    const char* want[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
    for (const char* w : want) {
      bool sup = false, en = false;
      for (auto& e : ep) sup |= !strcmp(e.extensionName, w);
      for (auto* e : exts) en |= !strcmp(e, w);
      if (sup && !en) exts.push_back(w);
      have += sup;
    }
  }
  VkDeviceCreateInfo dci = *ci; dci.enabledExtensionCount = uint32_t(exts.size()); dci.ppEnabledExtensionNames = exts.data();
  auto create = (PFN_vkCreateDevice)gipa(in->inst, "vkCreateDevice");
  // Build 17: the FSR 4 device set. Off (VELA_VK_FSR4 unset): nothing below runs and this is build 16.
  Fsr4Set fs; VkDeviceCreateInfo dciF = dci; bool fsr4 = false; VkResult r = VK_ERROR_INITIALIZATION_FAILED;
  if (fsr4_on()) {
    std::string why;
    if (fsr4_build(in, pd, gipa, ci, exts, fs, dciF, why)) {
      r = create(pd, &dciF, a, out); fs.undoAll();
      if (r == VK_SUCCESS) {
        fsr4 = true; { std::lock_guard<std::mutex> l(g_fsr4m); g_fsr4pd.push_back(pd); }
        { std::string all; for (uint32_t k = 0; k < dciF.enabledExtensionCount; k++) { all += dciF.ppEnabledExtensionNames[k]; all += ' '; } vline("device %p: all %u enabled extensions: %s", (void*)*out, dciF.enabledExtensionCount, all.c_str()); }  // build 18: log only
        vline("device %p: FSR 4 device set: added extensions: %s; features set: %s", (void*)*out, fs.addedExt.empty() ? "none" : fs.addedExt.c_str(), fs.setFeat.empty() ? "none" : fs.setFeat.c_str());
      } else vline("FSR 4 device set refused by the driver (vkCreateDevice %d); retrying with the app's own create info: no FSR 4 on this device (extensions %s; features %s)", (int)r, fs.addedExt.c_str(), fs.setFeat.c_str());
    } else { fs.undoAll(); vline("FSR 4 device set not available: %s", why.c_str()); }
  }
  if (!fsr4) { r = create(pd, &dci, a, out); if (r != VK_SUCCESS) return r; }
  const VkDeviceCreateInfo& made = fsr4 ? dciF : dci;
  Dev* d = new Dev; d->fsr4 = fsr4; for (uint32_t k = 0; k < made.enabledExtensionCount; k++) d->exts.push_back(made.ppEnabledExtensionNames[k]);
  d->dev = *out; d->gdpa = gdpa; d->on = on; d->extOk = have == 2; d->bda = fsr4 || bda_enabled(&made);   // build 24: the FSR 4 device set always enables bufferDeviceAddress (undoAll had put the app's bits back)
  ((PFN_vkGetPhysicalDeviceMemoryProperties)gipa(in->inst, "vkGetPhysicalDeviceMemoryProperties"))(pd, &d->mp);
#define F(n) d->n = (PFN_vk##n)gdpa(*out, "vk" #n);
  F(DestroyDevice) F(CreateBuffer) F(DestroyBuffer) F(GetBufferMemoryRequirements) F(AllocateMemory) F(FreeMemory)
  F(BindBufferMemory) F(MapMemory) F(GetMemoryFdPropertiesKHR) F(CreateEvent) F(DestroyEvent) F(SetEvent) F(ResetEvent)
  F(GetEventStatus) F(CmdSetEvent) F(CmdResetEvent) F(CmdWaitEvents) F(CmdPipelineBarrier) F(CmdUpdateBuffer) F(CmdCopyBuffer)
  F(CmdCopyImageToBuffer) F(CmdCopyBufferToImage)
  F(QueueSubmit) F(QueueSubmit2) F(QueueSubmit2KHR) F(BeginCommandBuffer) F(AllocateCommandBuffers) F(FreeCommandBuffers)
  F(ResetCommandPool) F(DeviceWaitIdle) F(GetBufferDeviceAddress)
  F(CreateShaderModule) F(DestroyShaderModule) F(CreatePipelineLayout) F(DestroyPipelineLayout) F(CreateComputePipelines) F(DestroyPipeline)
  F(CmdBindPipeline) F(CmdDispatchIndirect) F(CmdExecuteCommands)
  F(CmdDispatch) F(EndCommandBuffer) F(CmdBeginRenderPass) F(CmdBeginRenderPass2) F(CmdBeginRenderPass2KHR) F(CmdEndRenderPass) F(CmdEndRenderPass2) F(CmdEndRenderPass2KHR)
  F(CmdBeginRendering) F(CmdBeginRenderingKHR) F(CmdEndRendering) F(CmdEndRenderingKHR) F(CmdPipelineBarrier2) F(CmdPipelineBarrier2KHR) F(UpdateDescriptorSets) F(CmdPushDescriptorSetKHR)
#undef F
  if (!d->GetBufferDeviceAddress) { d->GetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)gdpa(*out, "vkGetBufferDeviceAddressKHR"); vline("device %p: vkGetBufferDeviceAddress core name NULL, KHR %s", (void*)*out, d->GetBufferDeviceAddress ? "used" : "NULL too"); }   // build 24: an API 1.1 device (Unity) answers NULL for the core name -> no CAP_BDA -> the provider's 'badargs'
  if (!d->GetMemoryFdPropertiesKHR) d->extOk = false;
  bool placed = false;
  { std::lock_guard<std::mutex> l(g_mtx); for (auto& e : g_dev) if (!e.key.load()) { e.d = d; e.key.store(key_of(*out), std::memory_order_release); placed = true; break; } }
  if (!placed) { d->on = false; vlog("more than 16 devices: device %p not tracked", (void*)*out); }  // lookup fails -> GDPA passes through
  if (on) vlog("device %p created (layer build %u): dma-buf import %s", (void*)*out, VL_BUILD, d->extOk ? "on" : "MISSING -> no NPU");
  return r;
}

// Build 22: the provider's SPIR-V 1.4+ modules (Unity's are older) are logged BEFORE the driver sees them (size, version, capabilities, extensions, execution modes), and every compute
// pipeline from #50 on is logged BEFORE vkCreateComputePipelines (module, entry, flags, required subgroup size), flushed: the last such line before the death names the call that kills the process (build 21: the trace
// after provider pipeline #63 stayed empty, so the death is inside the pipeline / module creation loop, not in command recording).
static std::atomic<int> g_modn{0}, g_prel{0}; static std::unordered_map<uint64_t, int> g_modseq;
static VKAPI_ATTR VkResult VKAPI_CALL vl_CreateShaderModule(VkDevice dev, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a, VkShaderModule* out) {
  Dev* d = dev_of(dev); const uint32_t* w = ci->pCode; const size_t nw = ci->codeSize / 4;
  const bool big = nw > 5 && w[0] == 0x07230203u && w[1] >= 0x10400u; int seq = -1;
  if (big) {
    seq = g_modn.fetch_add(1);
    if (seq < 300) {
      std::string caps, exts, modes; size_t i = 5;
      while (i < nw) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) break;
        if (op == 17) { caps += std::to_string(w[i + 1]) + ' '; }
        else if (op == 10) { exts += (const char*)&w[i + 1]; exts += ' '; }
        else if (op == 16) { modes += "mode" + std::to_string(w[i + 2]); for (uint32_t j = 3; j < len; j++) modes += "." + std::to_string(w[i + j]); modes += ' '; }
        else if (op != 11 && op != 14 && op != 15) break;
        i += len; }
      vlog("sm #%d PRE %zu words spirv 0x%x bound %u caps: %s| exts: %s| exec modes: %s", seq, nw, (unsigned)w[1], (unsigned)w[3], caps.c_str(), exts.c_str(), modes.c_str());
    }
  }
  VkResult r = d->CreateShaderModule(dev, ci, a, out);
  if (big && seq < 300) { vlog("sm #%d -> result %d module %p", seq, (int)r, r == VK_SUCCESS ? (void*)(uintptr_t)*out : nullptr); if (r == VK_SUCCESS) { std::lock_guard<std::mutex> l(g_dgm); g_modseq[(uint64_t)*out] = seq; } }
  return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL vl_CreateComputePipelines(VkDevice dev, VkPipelineCache pc, uint32_t n, const VkComputePipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* out) {
  Dev* d = dev_of(dev);
  static std::atomic<int> cnt{0}; int k = cnt.fetch_add(1);
  uint32_t ssz = 0;
  for (uint32_t i = 0; i < n; i++) for (auto* p = (const VkBaseInStructure*)ci[i].stage.pNext; p; p = p->pNext)
    if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO) ssz = ((const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*)p)->requiredSubgroupSize;
  if (k >= 50 && g_prel.fetch_add(1) < 500) {
    int ms = -1; { std::lock_guard<std::mutex> l(g_dgm); auto it = g_modseq.find((uint64_t)ci[0].stage.module); if (it != g_modseq.end()) ms = it->second; }
    vlog("cp #%d PRE count %u flags0 0x%x module %p (sm #%d) entry %s stageflags 0x%x requiredSubgroupSize %u layout %p", k, n, (unsigned)ci[0].flags, (void*)(uintptr_t)ci[0].stage.module, ms, ci[0].stage.pName ? ci[0].stage.pName : "?", (unsigned)ci[0].stage.flags, ssz, (void*)(uintptr_t)ci[0].layout);
  }
  VkResult r = d->CreateComputePipelines(dev, pc, n, ci, a, out);
  if (ssz && !g_prov.load()) { g_prov.store(1); vlog("dg trace on: first provider pipeline (#%d, required subgroup size %u)", k, ssz); }
  if (k >= 50 && g_prel.load() <= 500) vlog("cp #%d -> result %d", k, (int)r);
  if (k < 60 || r != VK_SUCCESS) { int nn = 0; for (uint32_t i = 0; r == VK_SUCCESS && i < n; i++) if (!out[i]) nn++; vline("vkCreateComputePipelines #%d: count %u result %d null pipelines %d flags0 0x%x", k, n, (int)r, nn, n ? (unsigned)ci[0].flags : 0u); }
  return r;
}

// ---------------------------------------------------------------- build 23: core 1.2 / 1.3 names answered with the KHR / EXT pointer
// The Unity device is API 1.1; winevulkan answers NULL for the core names of later versions but not for their KHR names. A provider that asks
// vkGetDeviceProcAddr("vkGetBufferDeviceAddress") then calls a null pointer. Under VELA_VK_FSR4=1 the layer answers the core name with the
// KHR / EXT pointer when the extension is enabled on the device (same semantics: these are the promoted functions).
struct Alias { const char* core; const char* ext; const char* extname; };
static const Alias kAliases[] = {
  {"vkGetBufferDeviceAddress", "vkGetBufferDeviceAddressKHR", "VK_KHR_buffer_device_address"},
  {"vkGetBufferOpaqueCaptureAddress", "vkGetBufferOpaqueCaptureAddressKHR", "VK_KHR_buffer_device_address"},
  {"vkGetDeviceMemoryOpaqueCaptureAddress", "vkGetDeviceMemoryOpaqueCaptureAddressKHR", "VK_KHR_buffer_device_address"},
  {"vkCmdPipelineBarrier2", "vkCmdPipelineBarrier2KHR", "VK_KHR_synchronization2"},
  {"vkCmdWriteTimestamp2", "vkCmdWriteTimestamp2KHR", "VK_KHR_synchronization2"},
  {"vkQueueSubmit2", "vkQueueSubmit2KHR", "VK_KHR_synchronization2"},
  {"vkCmdSetEvent2", "vkCmdSetEvent2KHR", "VK_KHR_synchronization2"},
  {"vkCmdResetEvent2", "vkCmdResetEvent2KHR", "VK_KHR_synchronization2"},
  {"vkCmdWaitEvents2", "vkCmdWaitEvents2KHR", "VK_KHR_synchronization2"},
  {"vkCmdBeginRendering", "vkCmdBeginRenderingKHR", "VK_KHR_dynamic_rendering"},
  {"vkCmdEndRendering", "vkCmdEndRenderingKHR", "VK_KHR_dynamic_rendering"},
  {"vkCmdCopyBuffer2", "vkCmdCopyBuffer2KHR", "VK_KHR_copy_commands2"},
  {"vkCmdCopyImage2", "vkCmdCopyImage2KHR", "VK_KHR_copy_commands2"},
  {"vkCmdCopyBufferToImage2", "vkCmdCopyBufferToImage2KHR", "VK_KHR_copy_commands2"},
  {"vkCmdCopyImageToBuffer2", "vkCmdCopyImageToBuffer2KHR", "VK_KHR_copy_commands2"},
  {"vkCmdBlitImage2", "vkCmdBlitImage2KHR", "VK_KHR_copy_commands2"},
  {"vkCmdResolveImage2", "vkCmdResolveImage2KHR", "VK_KHR_copy_commands2"},
  {"vkGetDeviceBufferMemoryRequirements", "vkGetDeviceBufferMemoryRequirementsKHR", "VK_KHR_maintenance4"},
  {"vkGetDeviceImageMemoryRequirements", "vkGetDeviceImageMemoryRequirementsKHR", "VK_KHR_maintenance4"},
  {"vkGetDeviceImageSparseMemoryRequirements", "vkGetDeviceImageSparseMemoryRequirementsKHR", "VK_KHR_maintenance4"},
  {"vkCmdPushDescriptorSet", "vkCmdPushDescriptorSetKHR", "VK_KHR_push_descriptor"},
  {"vkCmdPushDescriptorSetWithTemplate", "vkCmdPushDescriptorSetWithTemplateKHR", "VK_KHR_push_descriptor"},
  {"vkResetQueryPool", "vkResetQueryPoolEXT", "VK_EXT_host_query_reset"},
  {"vkGetSemaphoreCounterValue", "vkGetSemaphoreCounterValueKHR", "VK_KHR_timeline_semaphore"},
  {"vkWaitSemaphores", "vkWaitSemaphoresKHR", "VK_KHR_timeline_semaphore"},
  {"vkSignalSemaphore", "vkSignalSemaphoreKHR", "VK_KHR_timeline_semaphore"},
  {"vkCreateRenderPass2", "vkCreateRenderPass2KHR", "VK_KHR_create_renderpass2"},
  {"vkCmdBeginRenderPass2", "vkCmdBeginRenderPass2KHR", "VK_KHR_create_renderpass2"},
  {"vkCmdNextSubpass2", "vkCmdNextSubpass2KHR", "VK_KHR_create_renderpass2"},
  {"vkCmdEndRenderPass2", "vkCmdEndRenderPass2KHR", "VK_KHR_create_renderpass2"},
  {"vkCmdDrawIndirectCount", "vkCmdDrawIndirectCountKHR", "VK_KHR_draw_indirect_count"},
  {"vkCmdDrawIndexedIndirectCount", "vkCmdDrawIndexedIndirectCountKHR", "VK_KHR_draw_indirect_count"},
};
static const Alias* alias_of(const char* name) { for (auto& a : kAliases) if (!strcmp(a.core, name)) return &a; return nullptr; }
static bool dev_has_ext(const Dev* d, const char* e) { for (auto& x : d->exts) if (x == e) return true; return false; }
static void alias_log(const char* via, const Alias* a) {
  static std::mutex m; static std::set<std::string> seen; std::string k = std::string(via) + a->core;
  { std::lock_guard<std::mutex> l(m); if (!seen.insert(k).second) return; }
  vlog("alias %s: %s -> %s (%s enabled)", via, a->core, a->ext, a->extname);
}
VL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vl_GetDeviceProcAddr(VkDevice dev, const char* name);
static PFN_vkVoidFunction gdpa_impl(VkDevice dev, const char* name) {
  if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vl_GetDeviceProcAddr;
  Dev* d = dev_of(dev); if (!d) return nullptr;
  if (!strcmp(name, "vkDestroyDevice")) return (PFN_vkVoidFunction)vl_DestroyDevice;
  if (d->on) {  // VELA_NPU=0 -> every other call goes straight to the driver (zero overhead)
#define H(n, f) if (!strcmp(name, n)) return (PFN_vkVoidFunction)(f);
    H(VELA_NPU_ENTRY_NAME, vl_CmdVelaNpuUpscale)  // private: answered before asking the next layer
    H("vkCmdUpdateBuffer", vl_CmdUpdateBuffer) H("vkQueueSubmit", vl_QueueSubmit) H("vkBeginCommandBuffer", vl_BeginCommandBuffer)
    if (fsr4_on() && d->CreateComputePipelines && !strcmp(name, "vkCreateComputePipelines")) return (PFN_vkVoidFunction)vl_CreateComputePipelines;
    if (fsr4_on() && d->CreateShaderModule && !strcmp(name, "vkCreateShaderModule")) return (PFN_vkVoidFunction)vl_CreateShaderModule;
    if (fsr4_on()) {   // build 21: provider trace hooks (only where the next layer has the function, else null as it would answer)
#define HD(n, f, fld) if (d->fld && !strcmp(name, n)) return (PFN_vkVoidFunction)(f);
      HD("vkCmdDispatch", vl_CmdDispatch, CmdDispatch) HD("vkEndCommandBuffer", vl_EndCommandBuffer, EndCommandBuffer) HD("vkCmdPipelineBarrier", vl_CmdPipelineBarrier, CmdPipelineBarrier)
      HD("vkCmdBeginRenderPass", vl_CmdBeginRenderPass, CmdBeginRenderPass) HD("vkCmdBeginRenderPass2", vl_CmdBeginRenderPass2, CmdBeginRenderPass2) HD("vkCmdBeginRenderPass2KHR", vl_CmdBeginRenderPass2KHR, CmdBeginRenderPass2KHR)
      HD("vkCmdEndRenderPass", vl_CmdEndRenderPass, CmdEndRenderPass) HD("vkCmdEndRenderPass2", vl_CmdEndRenderPass2, CmdEndRenderPass2) HD("vkCmdEndRenderPass2KHR", vl_CmdEndRenderPass2KHR, CmdEndRenderPass2KHR)
      HD("vkCmdBeginRendering", vl_CmdBeginRendering, CmdBeginRendering) HD("vkCmdBeginRenderingKHR", vl_CmdBeginRenderingKHR, CmdBeginRenderingKHR) HD("vkCmdEndRendering", vl_CmdEndRendering, CmdEndRendering) HD("vkCmdEndRenderingKHR", vl_CmdEndRenderingKHR, CmdEndRenderingKHR)
      HD("vkCmdPipelineBarrier2", vl_CmdPipelineBarrier2, CmdPipelineBarrier2) HD("vkCmdPipelineBarrier2KHR", vl_CmdPipelineBarrier2KHR, CmdPipelineBarrier2KHR)
      HD("vkUpdateDescriptorSets", vl_UpdateDescriptorSets, UpdateDescriptorSets) HD("vkCmdPushDescriptorSetKHR", vl_CmdPushDescriptorSetKHR, CmdPushDescriptorSetKHR)
#undef HD
    }
    H("vkCmdBindPipeline", vl_CmdBindPipeline)   // remembers the app's compute pipeline
    if (d->CmdExecuteCommands && !strcmp(name, "vkCmdExecuteCommands")) return (PFN_vkVoidFunction)vl_CmdExecuteCommands;   // build 26
    H("vkAllocateCommandBuffers", vl_AllocateCommandBuffers) H("vkFreeCommandBuffers", vl_FreeCommandBuffers) H("vkResetCommandPool", vl_ResetCommandPool)
    if (!strcmp(name, "vkQueueSubmit2")) return d->QueueSubmit2 ? (PFN_vkVoidFunction)vl_QueueSubmit2 : nullptr;
    if (!strcmp(name, "vkQueueSubmit2KHR")) return d->QueueSubmit2KHR ? (PFN_vkVoidFunction)vl_QueueSubmit2KHR : nullptr;
#undef H
  }
  return d->gdpa(dev, name);
}
VL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vl_GetDeviceProcAddr(VkDevice dev, const char* name) {
  PFN_vkVoidFunction r = gdpa_impl(dev, name);
  if (!r && name && fsr4_on()) { Dev* d = dev_of(dev); const Alias* a = d && d->fsr4 ? alias_of(name) : nullptr;
    if (a && dev_has_ext(d, a->extname)) { r = gdpa_impl(dev, a->ext); if (r) alias_log("gdpa", a); } }
  dg_gp("dev", name, r == nullptr); return r;
}
VL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vl_GetInstanceProcAddr(VkInstance inst, const char* name);
static PFN_vkVoidFunction gipa_impl(VkInstance inst, const char* name) {
  if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vl_GetInstanceProcAddr;
  if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)vl_CreateInstance;
  if (!strcmp(name, "vkDestroyInstance")) return (PFN_vkVoidFunction)vl_DestroyInstance;
  if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)vl_CreateDevice;
  if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vl_GetDeviceProcAddr;
  if (!inst) return nullptr;
  Inst* in; { std::lock_guard<std::mutex> l(g_mtx); auto it = g_inst.find(key_of(inst)); in = it == g_inst.end() ? nullptr : it->second; }
  if (in && spoof_on()) {   // GPU identity override: only where the next layer has the function (else null, as it would answer)
    if (!strcmp(name, "vkGetPhysicalDeviceProperties")) return in->gpdp ? (PFN_vkVoidFunction)vl_GetPhysicalDeviceProperties : nullptr;
    if (!strcmp(name, "vkGetPhysicalDeviceProperties2")) return in->gpdp2 ? (PFN_vkVoidFunction)vl_GetPhysicalDeviceProperties2 : nullptr;
    if (!strcmp(name, "vkGetPhysicalDeviceProperties2KHR")) return in->gpdp2khr ? (PFN_vkVoidFunction)vl_GetPhysicalDeviceProperties2KHR : nullptr;
    if (!strcmp(name, "vkEnumerateDeviceExtensionProperties")) return in->edep ? (PFN_vkVoidFunction)vl_EnumerateDeviceExtensionProperties : nullptr;
  }
  return in ? in->gipa(inst, name) : nullptr;
}
VL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vl_GetInstanceProcAddr(VkInstance inst, const char* name) {
  PFN_vkVoidFunction r = gipa_impl(inst, name);
  if (!r && inst && name && fsr4_on()) { const Alias* a = alias_of(name);
    if (a) { bool ok = false; for (auto& e : g_dev) { Dev* d = e.key.load(std::memory_order_acquire) ? e.d : nullptr; if (d && d->fsr4 && dev_has_ext(d, a->extname)) { ok = true; break; } }
      if (ok) { r = gipa_impl(inst, a->ext); if (r) alias_log("gipa", a); } } }
  dg_gp("inst", name, r == nullptr); return r;
}
VL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v) {
  if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
  v->pfnGetInstanceProcAddr = vl_GetInstanceProcAddr; v->pfnGetDeviceProcAddr = vl_GetDeviceProcAddr; v->pfnGetPhysicalDeviceProcAddr = nullptr;
  return VK_SUCCESS;
}
