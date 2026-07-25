import { app, shell, BrowserWindow, ipcMain, Tray, Menu, nativeImage } from 'electron'
import type { MenuItemConstructorOptions } from 'electron'
import { spawn, ChildProcessByStdio } from 'child_process'
import type { Readable } from 'stream'
import { createServer } from 'net'
import { createWriteStream, mkdirSync, existsSync, WriteStream } from 'fs'
import { homedir } from 'os'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { autoUpdater } from 'electron-updater'
import icon from '../../resources/icon.png?asset'
import trayIconAsset from '../../resources/tray-icon.png?asset'
// TODO(windows): Windows' system tray convention is a much smaller icon
// (16x16, traditionally .ico) than Linux's SNI/AppIndicator convention
// (which reads fine at 32-48px, verified on this machine). tray-icon-16.png
// exists in resources/ for this but isn't wired up/tested anywhere -- no
// Windows machine available in this environment. If the tray icon looks
// oversized/blurry on Windows, switch to it there, e.g.:
//   trayIconAsset (Linux/macOS) vs. a 16px asset on win32.
import trayIconWin from '../../resources/tray-icon-16.png?asset'

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

// Backend can run two ways:
//   1. Bundled (production / packaged): a self-contained PyInstaller onedir
//      build of pleiades.webui, built from desktop/backend-build/backend.spec.
//      Needs nothing pre-installed on the target machine -- no system Python,
//      no venv, no pip install.
//   2. Dev-mode fallback: shells out to the developer's own ~/Pleiades/.venv
//      checkout with `python3.12 -m pleiades.webui` (POSIX) or
//      `python.exe -m pleiades.webui` (Windows), same as before. Used
//      automatically whenever the bundled binary isn't present, so local
//      development against a live venv (fast iteration on the Python side)
//      keeps working unmodified.
const PLEIADES_ROOT = process.env.PLEIADES_ROOT ?? join(homedir(), 'Pleiades')
// venv layout differs by platform: POSIX puts executables in bin/ (and this
// repo's dev venv is pinned to python3.12 specifically -- see repo CLAUDE.md
// on the python3.12/python3.13 split); Windows' `python -m venv` always
// lays executables out under Scripts/python.exe regardless of version.
const PYTHON_BIN =
  process.platform === 'win32'
    ? join(PLEIADES_ROOT, '.venv', 'Scripts', 'python.exe')
    : join(PLEIADES_ROOT, '.venv', 'bin', 'python3.12')

const BACKEND_BIN_NAME = 'pleiades-backend'

/**
 * Where the bundled backend lives, in dev vs. packaged builds.
 *  - Packaged: electron-builder's `extraResources` (see electron-builder.yml)
 *    copies desktop/backend-build/dist/pleiades-backend/ to
 *    <resources>/backend/, so process.resourcesPath/backend/pleiades-backend
 *    is the launcher.
 *  - Dev: point straight at the freshly-built onedir output so `npm run dev`
 *    can exercise the real bundle without a packaging step, if it's been
 *    built (desktop/backend-build/build.sh). Falls back to the venv (below)
 *    if it hasn't.
 */
function bundledBackendPath(): string {
  if (is.dev) {
    return join(__dirname, '../../backend-build/dist/pleiades-backend', BACKEND_BIN_NAME)
  }
  return join(process.resourcesPath, 'backend', BACKEND_BIN_NAME)
}

const BACKEND_READY_TIMEOUT_MS = 20_000
const BACKEND_POLL_INTERVAL_MS = 400
const SHUTDOWN_GRACE_MS = 5_000

interface BackendState {
  phase: 'starting' | 'ready' | 'error'
  port: number | null
  url: string | null
  error: string | null
}

const backendState: BackendState = { phase: 'starting', port: null, url: null, error: null }

// See the "Auto-update" section further down for how this gets populated.
interface UpdateStatus {
  phase: 'idle' | 'checking' | 'available' | 'downloading' | 'ready' | 'not-available' | 'error'
  version: string | null // the version being downloaded/ready, once known
  progressPercent: number | null
  error: string | null
}

let updateStatus: UpdateStatus = {
  phase: 'idle',
  version: null,
  progressPercent: null,
  error: null
}

type BackendProcess = ChildProcessByStdio<null, Readable, Readable>

let backendProcess: BackendProcess | null = null
let mainWindow: BrowserWindow | null = null
let tray: Tray | null = null
// Set true once we've committed to a real, full app quit (via the tray's
// Quit item, Ctrl/Cmd+Q, or the app menu). While false, closing the window
// just hides it (minimize-to-tray) instead of tearing anything down.
let quitting = false
let logStream: WriteStream | null = null

function logLine(line: string): void {
  const msg = `[${new Date().toISOString()}] ${line}`
  console.log(msg)
  logStream?.write(msg + '\n')
}

function initLogging(): void {
  const logDir = join(app.getPath('userData'), 'logs')
  if (!existsSync(logDir)) mkdirSync(logDir, { recursive: true })
  logStream = createWriteStream(join(logDir, 'backend.log'), { flags: 'a' })
}

function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.once('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const address = srv.address()
      const port = typeof address === 'object' && address ? address.port : null
      srv.close(() => {
        if (port) resolve(port)
        else reject(new Error('Could not determine a free port'))
      })
    })
  })
}

function spawnBackend(port: number): void {
  const bundledBin = bundledBackendPath()
  const useBundled = existsSync(bundledBin)

  if (!useBundled && !existsSync(PYTHON_BIN)) {
    backendState.phase = 'error'
    backendState.error =
      `Neither the bundled backend (${bundledBin}) nor a dev venv Python ` +
      `(${PYTHON_BIN}) was found. Build the bundle with ` +
      `desktop/backend-build/build.sh, or check out ~/Pleiades with its venv built.`
    logLine(`[backend] ${backendState.error}`)
    return
  }

  const args = ['--host', '127.0.0.1', '--port', String(port), '--no-browser']
  const command = useBundled ? bundledBin : PYTHON_BIN
  const spawnArgs = useBundled ? args : ['-m', 'pleiades.webui', ...args]
  // The bundled binary is fully self-contained (no venv/site-packages to
  // resolve relative to a cwd); the dev fallback still needs to run from
  // PLEIADES_ROOT the way `pleiades.webui` expects.
  const cwd = useBundled ? join(bundledBin, '..') : PLEIADES_ROOT

  logLine(
    `[backend] spawning ${useBundled ? 'bundled' : 'dev venv'} backend: ` +
      `${command} ${spawnArgs.join(' ')} (cwd=${cwd})`
  )

  const child = spawn(command, spawnArgs, {
    cwd,
    stdio: ['ignore', 'pipe', 'pipe']
  }) as BackendProcess

  backendProcess = child

  child.stdout.on('data', (chunk: Buffer) =>
    logLine(`[backend stdout] ${chunk.toString().trimEnd()}`)
  )
  child.stderr.on('data', (chunk: Buffer) =>
    logLine(`[backend stderr] ${chunk.toString().trimEnd()}`)
  )

  child.on('error', (err) => {
    logLine(`[backend] spawn error: ${err.message}`)
    backendState.phase = 'error'
    backendState.error = `Failed to launch backend: ${err.message}`
  })

  child.on('exit', (code, signal) => {
    logLine(`[backend] exited (code=${code}, signal=${signal})`)
    backendProcess = null
    if (!quitting && backendState.phase !== 'ready') {
      backendState.phase = 'error'
      backendState.error = `Backend process exited early (code=${code}, signal=${signal}). Check ${join(
        app.getPath('userData'),
        'logs',
        'backend.log'
      )}.`
    } else if (!quitting) {
      // Backend died after having been ready - surface that too.
      backendState.phase = 'error'
      backendState.error = `Backend process exited unexpectedly (code=${code}, signal=${signal}).`
    }
  })
}

async function pollBackendReady(port: number): Promise<void> {
  const deadline = Date.now() + BACKEND_READY_TIMEOUT_MS
  const url = `http://127.0.0.1:${port}`

  while (Date.now() < deadline) {
    if (backendState.phase === 'error') return
    try {
      const res = await fetch(`${url}/api/status`, { signal: AbortSignal.timeout(2000) })
      if (res.ok) {
        backendState.phase = 'ready'
        logLine(`[backend] ready on ${url}`)
        return
      }
    } catch {
      // not up yet, keep polling
    }
    await new Promise((r) => setTimeout(r, BACKEND_POLL_INTERVAL_MS))
  }

  if (backendState.phase !== 'ready') {
    backendState.phase = 'error'
    backendState.error = `Backend did not respond on ${url}/api/status within ${BACKEND_READY_TIMEOUT_MS / 1000}s.`
    logLine(`[backend] ${backendState.error}`)
  }
}

async function startBackend(): Promise<void> {
  try {
    const port = await getFreePort()
    backendState.port = port
    backendState.url = `http://127.0.0.1:${port}`
    spawnBackend(port)
    if (backendState.phase !== 'error') {
      await pollBackendReady(port)
    }
  } catch (err) {
    backendState.phase = 'error'
    backendState.error = `Could not allocate a port for the backend: ${(err as Error).message}`
    logLine(`[backend] ${backendState.error}`)
  }
}

/** SIGTERM the backend, escalate to SIGKILL after a grace period. */
function stopBackend(): Promise<void> {
  return new Promise((resolve) => {
    if (!backendProcess) {
      resolve()
      return
    }
    const proc = backendProcess
    let settled = false
    const finish = (): void => {
      if (settled) return
      settled = true
      resolve()
    }

    proc.once('exit', finish)

    logLine('[backend] sending SIGTERM')
    proc.kill('SIGTERM')

    setTimeout(() => {
      if (!settled && backendProcess === proc) {
        logLine('[backend] still alive after grace period, sending SIGKILL')
        proc.kill('SIGKILL')
      }
    }, SHUTDOWN_GRACE_MS)

    // Safety net in case 'exit' never fires.
    setTimeout(finish, SHUTDOWN_GRACE_MS + 2000)
  })
}

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

/**
 * The renderer runs on its own origin (the Vite dev-server URL in
 * development, a file:// / "null" origin once packaged) while the backend
 * is a separate localhost port, so plain `fetch()` calls from the renderer
 * are cross-origin. pleiades/webui/server.py now runs CORSMiddleware
 * scoped to http(s)://localhost|127.0.0.1(:*) / file:// / null origins
 * (see server.py's create_app()), so the browser's normal CORS/preflight
 * checks succeed against our own spawned backend without needing to
 * relax webSecurity. CSP (`connect-src`/`img-src` scoped to 'self' +
 * http://127.0.0.1:*) provides defense in depth on top of that.
 */

// ---------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------

function registerIpcHandlers(): void {
  ipcMain.handle('pleiades:get-backend-url', () => backendState.url)
  ipcMain.handle('pleiades:get-backend-status', () => ({ ...backendState }))
  ipcMain.handle('pleiades:get-app-version', () => app.getVersion())

  // Custom title bar window controls (see createWindow's frame: false).
  ipcMain.handle('pleiades:window-minimize', () => mainWindow?.minimize())
  ipcMain.handle('pleiades:window-maximize-toggle', () => {
    if (!mainWindow) return
    if (mainWindow.isMaximized()) mainWindow.unmaximize()
    else mainWindow.maximize()
  })
  ipcMain.handle('pleiades:window-close', () => mainWindow?.close())
  ipcMain.handle('pleiades:window-is-maximized', () => mainWindow?.isMaximized() ?? false)

  // Auto-update (see the "Auto-update" section below for the full flow).
  ipcMain.handle('pleiades:check-for-updates', () => {
    autoUpdater.checkForUpdates().catch((e) => logLine(`[update] manual check failed: ${e}`))
  })
  ipcMain.handle('pleiades:get-update-status', () => updateStatus)
  ipcMain.handle('pleiades:install-update-and-restart', () => {
    // quitAndInstall's own 'before-quit'-equivalent path bypasses our normal
    // before-quit handler (which stops the backend first) -- stop it
    // ourselves before handing off, so the update doesn't land on top of a
    // still-running backend process on the next launch.
    stopBackend().finally(() => autoUpdater.quitAndInstall())
  })
}

// ---------------------------------------------------------------------------
// Auto-update
//
// electron-updater's GitHub provider (see electron-builder.yml's `publish`
// block) compares this build's own version against the newest GitHub
// Release on Fleabag515/Pleiades that has electron-builder's latest*.yml
// metadata attached -- `electron-builder --publish always` produces both
// the installers AND that metadata in one pass; a release built without
// `--publish` needs the latest*.yml files from dist/ uploaded to the
// release by hand alongside the installers, or updates will never be seen.
//
// Flow: check on startup (packaged builds only -- checking against a real
// GitHub release from an unpackaged dev run would either 404 confusingly or
// offer to "update" a dev checkout, neither useful) -- autoDownload is on,
// so once a newer release is found it fetches it in the background with no
// user action needed. The renderer gets every state change broadcast via
// 'pleiades:update-status-changed' (Settings' Updates section renders it;
// Shell puts a small dot on the gear icon the instant it's anything other
// than idle/checking/not-available, so an update doesn't sit undiscovered
// behind a panel nobody happened to open). The ONE click the owner asked
// for is installUpdateAndRestart's IPC handler above, which is
// autoUpdater.quitAndInstall() after stopping the backend cleanly first.
// ---------------------------------------------------------------------------

function setUpdateStatus(patch: Partial<UpdateStatus>): void {
  updateStatus = { ...updateStatus, ...patch }
  mainWindow?.webContents.send('pleiades:update-status-changed', updateStatus)
}

function setupAutoUpdater(): void {
  autoUpdater.autoDownload = true
  autoUpdater.autoInstallOnAppQuit = false // only via the explicit one-click action, never a surprise quit

  autoUpdater.on('checking-for-update', () => setUpdateStatus({ phase: 'checking', error: null }))
  autoUpdater.on('update-available', (info) =>
    setUpdateStatus({ phase: 'available', version: info.version, error: null })
  )
  autoUpdater.on('update-not-available', () =>
    setUpdateStatus({ phase: 'not-available', version: null, progressPercent: null })
  )
  autoUpdater.on('download-progress', (p) =>
    setUpdateStatus({ phase: 'downloading', progressPercent: Math.round(p.percent) })
  )
  autoUpdater.on('update-downloaded', (info) =>
    setUpdateStatus({ phase: 'ready', version: info.version, progressPercent: 100 })
  )
  autoUpdater.on('error', (err) => {
    logLine(`[update] ${err}`)
    setUpdateStatus({ phase: 'error', error: String(err?.message ?? err) })
  })

  if (app.isPackaged) {
    // A few seconds after launch, not immediately -- startBackend() is
    // already doing real work in this same window; no need to compete
    // with it for the very first moments of startup.
    setTimeout(() => {
      autoUpdater.checkForUpdates().catch((e) => logLine(`[update] startup check failed: ${e}`))
    }, 10_000)
  }
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    show: false,
    autoHideMenuBar: true,
    frame: false,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  // Custom title bar needs to reflect maximize/restore state for its icon.
  mainWindow.on('maximize', () =>
    mainWindow?.webContents.send('pleiades:window-maximized-changed', true)
  )
  mainWindow.on('unmaximize', () =>
    mainWindow?.webContents.send('pleiades:window-maximized-changed', false)
  )

  // Follow-up to the "can't type after reopening" fix in showMainWindow():
  // that function's explicit webContents.focus() call only runs on paths
  // that go through THIS app's own JS -- tray click, the app-menu/global
  // shortcut, or `app.on('activate')`. But per the 'close' handler just
  // below, this is a tray-resident app that minimizes (never destroys) its
  // window, and minimize() deliberately keeps it in _NET_CLIENT_LIST so
  // it's *also* restorable straight from the window manager itself: Alt+Tab,
  // the taskbar/window-list applet, or switching back to it after a
  // workspace/monitor change. All of those hand the window its OS-level
  // focus back without ever calling showMainWindow(), so the previous fix
  // never ran on that path -- almost certainly why the bug was still
  // reproducible after that patch landed.
  //
  // Hooking the BrowserWindow's own native 'focus' event closes that gap:
  // it fires no matter *how* the window regains OS focus, JS-driven or not.
  // It also self-heals the narrower race in showMainWindow() itself --
  // show()/restore() are synchronous Electron calls, but the underlying
  // X11 focus grant from the window manager is not, so a focus()/
  // webContents.focus() pair called immediately afterward can (intermittently
  // -- "sometimes", matching the report) run before the window manager has
  // actually finished granting focus. Whenever real focus does land, this
  // listener re-syncs Chromium's focused element regardless of what
  // triggered the show or how the timing lined up.
  mainWindow.on('focus', () => {
    mainWindow?.webContents.focus()
  })

  // Tray-resident app pattern: clicking the OS window-close button hides the
  // window instead of destroying it, so the backend keeps running and the
  // app stays reachable from the tray icon. A real quit (tray "Quit", the
  // app menu, or Ctrl/Cmd+Q) sets `quitting` via the existing before-quit
  // handler *before* it re-invokes app.quit(), so by the time Electron
  // actually closes this window as part of that quit sequence, `quitting`
  // is already true and this handler steps out of the way.
  mainWindow.on('close', (event) => {
    if (quitting) return
    event.preventDefault()
    // Verified on this machine (Linux Mint Cinnamon/Muffin, xapp-status
    // panel applet): the panel hides this app's tray icon the instant the
    // window receives ANY close request (WM_DELETE_WINDOW / EWMH
    // _NET_CLOSE_WINDOW) -- confirmed identical across hide(), minimize(),
    // and even letting the window actually destroy; the panel reacts to
    // the close request itself, not to what we do afterward, and it's
    // unrelated to the SNI's own registration (still "Active" per
    // busctl/dbus-send introspection the whole time). That's a Cinnamon/
    // xapp-status quirk outside this app's control.
    //
    // minimize() (rather than hide()) is the mitigation: it keeps the
    // window in _NET_CLIENT_LIST, so it stays reachable via Alt+Tab / the
    // window-list applet even if the tray icon itself is invisible --
    // guaranteeing the app is never truly unreachable, which a plain
    // hide() could not guarantee on this desktop (verified: wmctrl -l
    // shows nothing and the tray slot disappears from the panel layout
    // entirely, blind clicks at its old coordinates do nothing).
    mainWindow?.minimize()
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  // Never let the renderer navigate to or open remote content.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    logLine(`[window] blocked window.open to ${url}`)
    void shell.openExternal(url).catch(() => {})
    return { action: 'deny' }
  })
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('http://127.0.0.1:') && !url.startsWith('file://') && !is.dev) {
      event.preventDefault()
    }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function showMainWindow(): void {
  if (!mainWindow) {
    createWindow()
    return
  }
  if (mainWindow.isMinimized()) mainWindow.restore()
  if (!mainWindow.isVisible()) mainWindow.show()
  mainWindow.focus()
  // `focus()` above only grants the BrowserWindow OS-level window focus --
  // on Linux (Cinnamon/Muffin, confirmed environment) that doesn't
  // reliably re-route keyboard input to the page's own focused element
  // after a hide()/show() cycle. Explicitly focusing the webContents
  // (distinct from window-level focus) covers the common case immediately.
  // It is NOT the whole fix, though: this call only runs on the showMainWindow()
  // path (tray click / activate), and even here can race the window
  // manager's own (async) focus grant. createWindow()'s `mainWindow.on('focus', ...)`
  // listener is the belt-and-suspenders piece that covers every other way
  // this window can regain OS focus (Alt+Tab, taskbar/window-list applet,
  // workspace switch) plus self-heals this call if it fired too early.
  mainWindow.webContents.focus()
}

function toggleMainWindow(): void {
  if (mainWindow && mainWindow.isVisible() && !mainWindow.isMinimized()) {
    mainWindow.hide()
  } else {
    showMainWindow()
  }
}

// ---------------------------------------------------------------------------
// Tray
// ---------------------------------------------------------------------------

function createTray(): void {
  // TODO(windows): verify this looks right when a Windows build is
  // actually testable -- see the TODO on the trayIconWin import above.
  const iconPath = process.platform === 'win32' ? trayIconWin : trayIconAsset
  const image = nativeImage.createFromPath(iconPath)
  tray = new Tray(image)
  tray.setToolTip('Pleiades')

  const contextMenu = Menu.buildFromTemplate([
    { label: 'Open Pleiades', click: () => showMainWindow() },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        // Reuses the exact same before-quit -> stopBackend() -> app.quit()
        // path as the app menu / Ctrl+Q -- no separate shutdown logic here.
        app.quit()
      }
    }
  ])
  tray.setContextMenu(contextMenu)

  // On Linux (StatusNotifierItem/AppIndicator), left-click support varies by
  // desktop environment; where it's not delivered as a distinct 'click'
  // event, users still have "Open Pleiades" one right-click away via the
  // context menu above. Where 'click' *is* delivered (verified on this
  // machine's Cinnamon/Muffin session), it toggles show/hide for parity with
  // the macOS/Windows tray convention.
  tray.on('click', () => toggleMainWindow())
  // Explicit popUpContextMenu() on 'right-click' is a defensive no-op on
  // most desktops (setContextMenu already wires this up), but a few Linux
  // tray hosts need it triggered explicitly to show anything at all.
  //
  // NOTE (verified on this machine, Linux Mint Cinnamon/Muffin, xapp-status
  // panel applet hosting the StatusNotifierItem): right-clicking the icon
  // here shows Cinnamon's own generic applet-management menu (Move to
  // other monitor / Applet preferences / Minimize / Maximize / Close)
  // instead of the menu below, and this handler never fires -- Cinnamon
  // intercepts the right-click at the panel level before Electron sees it.
  // That's a desktop-environment limitation, not something fixable from
  // here. Left-click show/hide (above) is unaffected and is the primary,
  // reliable way to reach the window from the tray on this machine; a
  // real quit is always additionally reachable once the window is open
  // via Ctrl/Cmd+Q or File > Quit.
  tray.on('right-click', () => tray?.popUpContextMenu(contextMenu))
}

// ---------------------------------------------------------------------------
// Application menu (mainly: a reliable CmdOrCtrl+Q that always does a real,
// full quit -- including backend shutdown -- regardless of the minimize-to-
// tray behavior on the window's own close button).
// ---------------------------------------------------------------------------

function createAppMenu(): void {
  const isMac = process.platform === 'darwin'

  const template: MenuItemConstructorOptions[] = [
    ...(isMac
      ? ([
          {
            label: app.name,
            submenu: [{ role: 'about' }, { type: 'separator' }, { role: 'quit' }]
          }
        ] as MenuItemConstructorOptions[])
      : []),
    {
      label: 'File',
      submenu: [
        {
          label: 'Quit',
          accelerator: 'CmdOrCtrl+Q',
          click: () => app.quit()
        }
      ]
    },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' }
  ]

  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(() => {
  initLogging()
  electronApp.setAppUserModelId('com.pleiades.desktop')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  registerIpcHandlers()
  createAppMenu()
  createWindow()
  createTray()
  setupAutoUpdater()
  void startBackend()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
    else showMainWindow()
  })
})

// With a tray icon present, closing the window (handled above via 'close')
// keeps the app alive, so this normally won't fire from that path anymore.
// Left in place as a safety net for the (non-mac) case where every window
// has genuinely been destroyed outright.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', (event) => {
  if (quitting) return
  quitting = true
  if (!backendProcess) {
    tray?.destroy()
    tray = null
    return
  }
  event.preventDefault()
  logLine('[app] before-quit: stopping backend before exit')
  stopBackend().finally(() => {
    tray?.destroy()
    tray = null
    app.quit()
  })
})

// Make sure Ctrl+C in a terminal / `kill -TERM` on the Electron process also
// tears down the Python child instead of leaving it orphaned.
process.on('SIGINT', () => app.quit())
process.on('SIGTERM', () => app.quit())
