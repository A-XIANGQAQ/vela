# Changelog

## 0.8.23 — 2026-11-22

- **DirectX 11 games get FSR 4.** FSR 4, FSR 4 Fast, FSR 3 and SGSR 2 now work in DirectX 11 games, the same as in DirectX 12, and switch while you play. Checked in Ghostrunner and Bright Memory: Infinite.
- **Unreal games that only have their own FSR 2 in DirectX 11** (Bright Memory: Infinite with -dx11) now run Vela's upscaler instead of it.
- **New model: DittoSR.** A sharp upscaler without a neural network, as fast as SGSR 2, offered wherever FSR 3 is (DirectX 12 and DirectX 11 games). By Vancete, github.com/Vancete/DittoSR.
- Games with a 24-bit or 16-bit depth buffer no longer fall back to SGSR 2 when FSR 4 is picked. This was the cause in a Kingdom Come: Deliverance II report; the game itself wasn't retested here.
- Bright Memory: Infinite no longer crashes at start in DirectX 11 with Vela on.
- DirectX 11 games no longer show a black or white picture with FSR 4.
- Games that send motion vectors in a signed 16-bit format are accepted.
- Non-Steam shortcuts made by Heroic and Lutris are supported.
- When a model can't run in a game, the panel says why and what runs instead.
- Reports say which part of the game crashed, so crashes can be traced.
- Known issues: Vulkan games get nothing from Vela yet; work on it has started. Some games crash early on Snapdragon 8 Elite devices (Witcher 3, Monster Hunter Rise, Tokyo Xtreme Racer, PRAGMATA); the new crash reports should show why. On some Retroid Pocket 6 units the NPU part's check fails (github.com/zrprod9-cmyk/vela/issues/2), being worked on. Clair Obscur: Expedition 33 scenes without a depth buffer still use FSR 1; picking FSR 3.1 or DLSS in the game's menu instead of XeSS may avoid it. Cyberpunk 2077 with FSR 4 Fast: black rectangles on 8 Elite, not seen on 8 Gen 2, not solved.

## 0.8.22 — 2026-11-16

Mostly fixes from player reports.

- Death Stranding Director's Cut starts with Vela on. Vela no longer overrides OptiScaler's own list of games that break when it pretends to be a different GPU, which should also help other games on that list.
- Returnal no longer crashes about 25 seconds after launch (same cause).
- Deep Rock Galactic: Vela launches it in DirectX 12 (Steam's default DirectX 11 launch crashed). Turning Vela off puts back the game's anti-aliasing setting.
- Turning Vela off keeps launch options you added while it was on, doesn't leave a game half set up if something fails, and leaves a read-only settings file alone.
- The panel no longer says "On" when Vela isn't actually running in the game.
- Reports carry the logs, OptiScaler.ini, launch options and GPU kernel messages of the run that had the problem.
- Uninstalling the plugin turns Vela off in your games first.
- A failed update check is retried after an hour instead of the next day.
- Frame generation (test builds only) is removed. Updating from such a build turns it off and restores that game's launch options.
- Known issues: Clair Obscur: Expedition 33 has scenes where DLSS gets no depth buffer; Vela uses FSR 1 there for now (small bright specks on sharp edges are possible), a proper fix is in progress. PRAGMATA crashes with Vela on (seen on 8 Elite). Cyberpunk 2077 with FSR 4 Fast: reports of black rectangles, being checked.

## 0.8.21 — 2026-11-05

- **FSR 4 on the NPU at five more game resolutions:** 1600x904, 1440x810, 1280x800, 1152x648 and 960x540. 960x540 has run in a game so far, the others passed the installer's check. All five are experimental on 8 Elite: not tested there yet.
- **One NPU service instead of seven.** The NPU part (package 14) runs all its FSR 4 networks from a single service.
- **The NPU part needs one reinstall with sudo.** In Desktop mode open Konsole and run the command the panel shows:
  `curl -fsSL https://github.com/zrprod9-cmyk/vela/releases/latest/download/install-vela-npu.sh | sudo sh`
  Until then package 13 keeps working at the resolutions it already had.
- Cyberpunk 2077: the panel no longer sits on “Loading…”.
- The panel sees the NPU part as soon as it is installed, without a restart.
- The panel no longer asks for 1920x1080 while the NPU runs the game's resolution. If the resolution only needs a newer NPU
  part, it says to update that instead.
- Non-Steam shortcuts (Heroic, Lutris…) now say they are not supported yet.
- The panel no longer asks for a report when the NPU only has no network for the game's resolution: the game runs on the GPU and the panel's hint says so.
- Known issue: PRAGMATA crashes with Vela on, seen on 8 Elite. Play it with Vela off. The cause is not found yet.

## 0.8.20 — 2026-11-04

- **Vela no longer writes into game folders.** Its files are shown to the game only while it runs (an overlay), so
  Steam's file check has nothing to repair and **Off** leaves nothing behind. Games set up by an older Vela move over by
  themselves the next time you open the panel with the game closed.
- Games that hand over their image in a compact HDR format (for example PRAGMATA) no longer crash with FSR 4.
- The panel says when a game ran in DirectX 11 (Vela needs DirectX 12) and how to switch it.

## 0.8.19 — 2026-11-03

- **Fix for games hanging at start on Snapdragon 8 Elite devices** (AYN Odin 3, KONKR Pocket FIT Elite). If a game still
  hangs on your device, please send a report.
- **Updates from the panel.** Vela checks for a new version once a day, or when you press **Check now**, and updates with
  one press. The daily check can be turned off.
- Vela stays out of games with anti-cheat and says why in the panel.

## 0.8.18 — 2026-11-03

- **Your launch options are safe.** Vela now adds and removes only its own launch arguments; whatever you had before, or add
  later, stays where it was. If an older version removed something from a game you have not edited since, it comes back.
- The NPU part installer accepts the Snapdragon 8 Elite (Odin 3) results, which are correct but not bit-identical.
- The panel's NPU install command always points to the latest release.

## 0.8.17 — 2026-11-03

NPU part only; the plugin stays `vela-0.8.16.zip`.

- **AYN Odin 3 (Snapdragon 8 Elite): FSR 4 on the NPU, experimental.** The NPU part (package 13) now carries the FSR 4
  network for both chips and picks the right one by itself.
- Snapdragon 8 Gen 2 devices: no change.

## 0.8.16 — 2026-11-02

First public release.

- **FSR 4 Fast**: the same FSR 4 network in a lighter mode, faster than FSR 4.
- **DLSS input works with FSR 4** in more games.
- **Game profiles** switch the in-game upscaler on for Control, Resident Evil 4 and Deep Rock Galactic (your own setting is kept).
- **Chip detection**: each device gets the models its chip can run.
- **Report a problem** button in the panel.
- The panel is in English.
- Godot games are shown as unsupported.
- The NPU part waits for the GPU only for a bounded time, so closing a game can no longer hang the GPU.
