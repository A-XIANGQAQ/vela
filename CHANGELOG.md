# Changelog

## 0.8.26 — 2026-11-24

Games started from launchers: Heroic, Xodus Nexus and other non-Steam shortcuts. Contributed by indiesaudi (#3, #4, #5), tested on an Odin 2 mini by iloveswiming909.

New
- **Launcher games (#3).** Vela now finds the game behind a non-Steam shortcut made by a launcher: through the launcher's own config (Heroic, Lutris) or, when there is none (Xodus Nexus, store clients inside a Proton prefix), from the running game itself. The game gets its own Vela setup like a Steam game.
- **Live settings in launcher games (#4).** Model changes in the Vela panel now reach games whose launcher hides the Steam id, including Heroic games started through umu.
- **DirectX 12 for Unreal launcher games (#5).** For an Unreal game started from a launcher, Vela turns on DirectX 12 through the game's Engine.ini while Vela is on, and puts the file back byte for byte when it is off.

Checked on our Retroid Pocket 6: a Haste Demo shortcut made by Heroic 2.22.4 (found through Heroic's config, live model changes reached the game), and Steam games unchanged:

| game | model | 0.8.25 | 0.8.26 |
|---|---|---|---|
| Deep Rock Galactic 1920x1080 | FSR 4 | 66.9 | 77.7 (cool start) |
| Deep Rock Galactic 1920x1080 | SGSR 2 | 99.5 | 103.1 (cool start) |
| Manor Lords 1280x720 | FSR 4 | 59.6 | 59.4 |
| Manor Lords 1280x720 | SGSR 2 | 59.8 | 59.7 |

The 0.8.26 Deep Rock Galactic run started on a cool device, so it is higher than a warm run; the launcher change does not touch the upscaler.

Known issues
- DirectX 12 through Engine.ini (#5) is covered by tests only; we have no Unreal game from a launcher on our device. Please report how it goes.
- Native Vulkan games on Unity: Vela can draw through their DLSS path, but Unity only offers DLSS on NVIDIA GPUs, so players can't reach it yet.
- FSR 3 and DittoSR in native Vulkan games don't start on our device; Vela uses SGSR 2 there. FSR 4 is not offered in native Vulkan games yet.
- On some Retroid Pocket 6 units the NPU gives wrong results even with the smaller graphs (#2); there the NPU stays off and FSR 4 runs on the GPU.
- Scenes without a depth buffer still use FSR 1.
- FSR 4 Fast: black rectangles in some games on 8 Elite. We don't see them on 8 Gen 2.

## 0.8.25 — 2026-11-24

Faster NPU graphs, a fix for units whose NPU gives wrong FSR 4 output, and a fix for SGSR 2 colours.

New
- **NPU package 17.** Faster FSR 4 on the NPU with the same picture bit for bit. NPU time per frame on our Retroid Pocket 6: 1280x720 output -5 %, 1280x800 -5 %, 1920x1080 Ultra Performance -10 %, 1152x648 -31 %, 960x540 -16 %.
- **Units whose NPU gives wrong FSR 4 output (#2).** When the NPU service starts it checks one graph with a known input. If the result is wrong, it switches to graphs planned for a smaller memory window, which give correct results on the units tested in #2. If those are wrong too, the NPU stays off and Vela runs FSR 4 on the GPU instead of showing a broken picture.
- The NPU power governor is on by default: it lowers the NPU clock when the game has time to spare. FSR4NPUD_POWER_GOV=0 turns it off.
- FSR 4: the GPU part does a little less work per frame (about 0.1 ms), same picture bit for bit.

Our Retroid Pocket 6 (Snapdragon 8 Gen 2), average fps, each version run twice in the order old, new, new, old:

| game | model | 0.8.24 + NPU package 15 | 0.8.25 + NPU package 17 |
|---|---|---|---|
| Deep Rock Galactic 1920x1080 | FSR 4 | 65.8 | 66.9 |
| Deep Rock Galactic 1920x1080 | SGSR 2 | 98.9 | 99.5 |
| Manor Lords 1280x720 | FSR 4 | 59.5 | 59.6 |
| Manor Lords 1280x720 | SGSR 2 | 59.7 | 59.8 |

Manor Lords runs at its frame cap here, so it shows that nothing got slower rather than how much faster.

Fixed
- SGSR 2: wrong brightness and tint in games that report their own exposure (Haste, Ghostrunner, Into the Dead, Manor Lords, Deep Rock Galactic). SGSR 2 got the game's exposure value where it expects the change of exposure between frames.
- Vulkan: when an upscaler can't start, Vela no longer retries it every frame; it falls back to SGSR 2 once.
- Vulkan: SGSR 2 reads depth buffers that also carry stencil correctly.

Known issues
- Native Vulkan games on Unity: Vela can now draw through their DLSS path (the 3D scene used to stay empty), but Unity only offers DLSS on NVIDIA GPUs, so players can't reach it yet.
- FSR 3 and DittoSR in native Vulkan games don't start on our device: AMD's Vulkan module finds no memory type it accepts on the console's GPU driver. Vela uses SGSR 2 there.
- FSR 4 is not offered in native Vulkan games yet.
- Unity Vulkan games: menu backgrounds with many small bright points look a little darker with SGSR 2 than without upscaling.
- On some Retroid Pocket 6 units the NPU gives wrong results even with the smaller graphs (#2); there the NPU stays off and FSR 4 runs on the GPU. We are testing those units.
- Scenes without a depth buffer still use FSR 1.
- FSR 4 Fast: black rectangles in some games on 8 Elite. We don't see them on 8 Gen 2.

## 0.8.24 — 2026-11-23

New
- **Snapdragon 8 Gen 3 (SM8650) devices can run FSR 4 on the NPU.** NPU package 15 adds the Qualcomm runtime part for this chip. Contributed by enij90 (github.com/zrprod9-cmyk/vela/pull/6), who checked it on a Lenovo Legion Tab Gen 3.

Changed
- NPU package 15: while a game uses the NPU, it runs one power level lower. When the device is hot this leaves more power to the GPU: about 6 % more fps with FSR 4 on the NPU at 1920x1080 in our test, and no loss at 1280x720. The daemon option `--power-corner max` (or `FSR4NPUD_POWER_CORNER=max`) gives the old level.
- With 0.8.24 and NPU package 15 our Retroid Pocket 6 ran 1 to 11 % more fps than with 0.8.23 in the games we measured (table in the release notes).
- Reports say which upscaler input a game uses.

Fixed
- Games that Steam starts through the Rockstar Games Launcher: the launcher no longer fails with Vela on.

Known issues
- Native Vulkan games: FSR 3, DittoSR and SGSR 2 are built in, but not yet checked in a game on our device; FSR 4 is not offered in Vulkan games yet.
- Snapdragon 8 Gen 3: we have no such device, so package 15 isn't confirmed on a device by us yet. Please send the installer's last lines if you try it.
- Unreal games with XeSS and no depth buffer: Vela now takes the depth from the engine, but this path hasn't run in a game yet.
- Some games crash early on Snapdragon 8 Elite devices; the crash reports say where.
- On some Retroid Pocket 6 and Odin 2 Portal units the NPU part's check fails (github.com/zrprod9-cmyk/vela/issues/2). We are collecting tests from those units.
- Scenes without a depth buffer still use FSR 1.
- FSR 4 Fast: black rectangles in some games on 8 Elite. We don't see them on 8 Gen 2.

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
