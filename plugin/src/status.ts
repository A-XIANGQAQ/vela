// PANELWHY (0.8.23): the status card of the panel, out of index.tsx so it runs under Node in tests
// (tests/test_panelwhy.py compiles this file and strings.ts and checks the card for each state).
// The card tells the last run's truth: what really ran, and in plain words why the picked model did not.
import { menuHint, MODEL_NAME, npuReason, reasonText, t, tOr } from "./strings";

// ---- what the backend sends (main.py game_state, py_modules/sgsr3.py log_summary) ----------------------------------------
export type NpuLog = { state: "on" | "fallback" | "off" | null; mode: string | null; reason: string | null };
export type S3Log = {
  api?: string | null; out_format?: boolean;
  sgsr3: boolean; render: string | null; output: string | null; mode: string | null; calibrated?: boolean;
  fsr4_fallback?: string | null; fsr3_fallback?: string | null; sgsr2_fallback?: string | null;
  dittosr_fallback?: string | null;  // DITTO: DittoSR did not run (FSR 3 instead)
  // PANELWHY: why FSR 4 did not start (sgsr3.FSR4_WHY) and which FSR 4 it was (fsr4 / fsr4fast)
  fsr4_why?: string | null; sgsr2_instead?: string | null;
  not_offered?: string | null; npu?: NpuLog | null;
};
export type Live = { model?: string; npu?: number; npuexact?: number };
export type S3State = { installed: boolean; intact: boolean | null; log: S3Log | null; menu_hint: string; static_fsr2?: object | null; live?: Live };
export type Check = { kind: string; reason: string | null; dx11_only?: boolean };
// the part of the game state the card reads
export type StatusIn = {
  running: boolean; sgsr3: S3State; check: Check; scan?: { inputs?: string[] | null }; model: string;
  layer_on?: boolean; not_in_launch?: "wrapped" | "left_out" | null;
  run_secs?: number | null;  // PANELWHY: how long this run has been going (main.py)
};

// text = the first line; sub and why = the lines under it, in this order
export type Status = { dot: string; text: string; sub?: string; why?: string };
export const GREEN = "#50c060", YELLOW = "#e0b040", RED = "#e05050", GREY = "#888", WAIT = "#8fb8e8";
// PANELWHY: a run this long that never sent a frame through the game's upscaler is not "in the menu" any more
export const WAIT_LONG_S = 300;

const px = (s?: string | null) => (s ?? "?").replace(/x/, "×");
// what the log says runs: fsr2 = the game's own FSR 2 (RE4 interceptor), anything else unknown = an older DLL's fallback
const runName = (m?: string | null) => !m ? t("other_upscaler") : m === "fsr2" ? t("game_fsr2") : MODEL_NAME[m] ?? t("other_upscaler");
const FSR4 = ["fsr4", "fsr4fast"];
// PANELWHY: the reason FSR 4 did not start, one plain line (an unknown or missing id: "send a report")
export const fellBackWhy = (why: string | null | undefined, want: string) =>
  tOr(`fb.why.${why ?? "unknown"}`, t("fb.why.unknown"), { want });

export function statusOf(gs: StatusIn): Status {
  const s3 = gs.sgsr3, c = gs.check;
  if (gs.layer_on) return { dot: YELLOW, text: t("st.layer"), sub: gs.running ? t("st.layerRunning") : t("st.layerOff") };
  if (c.kind === "none") return { dot: GREY, text: t("not_supported"), sub: reasonText(c.reason) };
  if (!s3.installed) return { dot: GREY, text: t("st.off") };  // NOFG (2026-11-15): no frame-generation / any-game status any more
  if (s3.intact === false) return { dot: RED, text: t("st.filesChanged"), sub: t("st.filesChangedSub") };
  // V03c: set up, but Vela is not in this game's launch (options that wrap %command% in a shell command, or the launcher left the
  // kit out of the running game). Never "On" / "Works". The wording is the owner's (2026-11-10, NIGHT/QUESTIONS.md answer 3):
  // the two proposals as written, in English here (the panel is English only)
  if (gs.not_in_launch) return { dot: YELLOW, text: t(gs.not_in_launch === "wrapped" ? "st.launchWrapped" : "st.notConnected") };
  const want = s3.live?.model ?? gs.model;
  const wantName = MODEL_NAME[want] ?? want;
  const L = s3.log;
  const isStatic = !!s3.static_fsr2;
  const runs = L ? (isStatic && !L.sgsr3 ? t("game_fsr2") : runName(L.mode)) : "";
  // the FSR 4 that fell back (FSR 4 or FSR 4 Fast): the log's, else the one picked
  const failedName = MODEL_NAME[L?.sgsr2_instead ?? ""] ?? wantName;
  if (!gs.running) {
    // PANELWHY: after the run, the card keeps the last run's truth while an FSR 4 is still the pick (the next run tries again)
    if (L?.fsr4_fallback && FSR4.includes(want))
      return { dot: YELLOW, text: t("st.onIdle", { m: wantName }), sub: t("st.lastFellBack", { want: failedName, runs }),
        why: fellBackWhy(L.fsr4_why, failedName) };
    return { dot: GREEN, text: t("st.onIdle", { m: wantName }), sub: t("st.onIdleSub") };
  }
  if (!L) return { dot: YELLOW, text: t("st.nextStart") };
  const sizes = L.render && L.output ? `${px(L.render)} → ${px(L.output)}` : undefined;
  // the chosen model could not run: the one short reason the log gives
  if (L.not_offered) return { dot: YELLOW, text: t("st.notOffered", { key: L.not_offered, runs }), sub: sizes };
  // PANELWHY: FSR 4 fell back: what runs instead, then why in plain words
  if (L.fsr4_fallback) return { dot: YELLOW, text: t("st.fellBack", { want: failedName, runs }), sub: fellBackWhy(L.fsr4_why, failedName), why: sizes };
  if (L.sgsr2_fallback) return { dot: YELLOW, text: t("st.failed", { want: wantName, runs }), sub: sizes };
  if (L.fsr3_fallback) return { dot: YELLOW, text: t("st.failed", { want: MODEL_NAME.fsr3, runs }), sub: sizes, why: t("st.fsr3FailedSub") };
  if (L.dittosr_fallback) return { dot: YELLOW, text: t("st.failed", { want: MODEL_NAME.dittosr, runs }), sub: sizes, why: t("st.dittosrFailedSub") };
  // the last run was DirectX 11 in a game that has DirectX 12: no upscaler of Vela's started (the hint says what to switch).
  // PANELWHY / DX11FSR: a DirectX 11 game (dx11_only) runs Vela's models in DX11: there it is the waiting state below
  if (!L.sgsr3 && L.api === "dx11" && !c.dx11_only) return { dot: YELLOW, text: t("st.dx11") };
  // no frame through the game's upscaler yet (its menu): waiting, not an error
  if (!L.sgsr3) {
    // PANELWHY: still nothing after a long run: the game uses an upscaler input Vela does not take
    if (!isStatic && (gs.run_secs ?? 0) >= WAIT_LONG_S) return { dot: YELLOW, text: t("st.neverAsked"), sub: t("st.neverAskedSub") };
    const what = isStatic ? t("st.waitingFsr2") : t("st.waitingMenu", { hint: menuHint(gs.scan?.inputs, s3.menu_hint) });
    return { dot: WAIT, text: t("st.waiting", { m: wantName }), sub: t("st.waitingSub", { what }) };
  }
  const model = runName(L.mode);
  if (L.mode && MODEL_NAME[L.mode] && want !== L.mode) return { dot: GREEN, text: t("st.switching", { from: model, to: wantName }), sub: sizes };
  if (L.mode === "fsr4" || L.mode === "fsr4fast") {
    const n = L.npu;
    if (n?.state === "on") return { dot: GREEN, text: `${t("st.works")} · ${model} ${t("st.onNpu")}`, sub: sizes };
    if (n?.state === "fallback") return { dot: YELLOW, text: `${t("st.works")} · ${model} ${t("st.onGpu")}`, sub: sizes, why: npuReason(n.reason) };
    // the provider reports the NPU at the first frame it upscales: before it, FSR 4 is set up and waits
    if (s3.live?.npu && n?.state !== "off") return { dot: WAIT, text: t("st.waiting", { m: model }), sub: sizes };
  }
  return { dot: GREEN, text: `${t("st.works")} · ${model} ${t("st.onGpu")}`, sub: sizes };
}
