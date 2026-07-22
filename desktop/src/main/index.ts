import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { spawn, ChildProcessByStdio } from 'child_process'
import type { Readable } from 'stream'
import { createServer } from 'net'
import { createWriteStream, mkdirSync, existsSync, WriteStream } from 'fs'
import { homedir } from 'os'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

// Backend can run two ways:
//   1. Bundled (production / packaged): a self-contained PyInstaller onedir
//      build of pleiades.webui, built from desktop/backend-build/backend.spec.
//      Needs nothing pre-installed on the target machine -- no system Python,
//      no venv, no pip install.
//   2. Dev-mode fallback: shells out to the developer's own ~/Pleiades/.venv
//      checkout with `python3.12 -m pleiades.webui`, same as before. Used
//      automatically whenever the bundled binary isn't present, so local
//      development against a live venv (fast iteration on the Python side)
//      keeps working unmodified.
const PLEIADES_ROOT = process.env.PLEIADES_ROOT ?? join(homedir(), 'Pleiades')
const PYTHON_BIN = join(PLEIADES_ROOT, '.venv', 'bin', 'python3.12')

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

type BackendProcess = ChildProcessByStdio<null, Readable, Readable>

let backendProcess: BackendProcess | null = null
let mainWindow: BrowserWindow | null = null
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

  child.stdout.on('data', (chunk: Buffer) => logLine(`[backend stdout] ${chunk.toString().trimEnd()}`))
  child.stderr.on('data', (chunk: Buffer) => logLine(`[backend stderr] ${chunk.toString().trimEnd()}`))

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
  createWindow()
  void startBackend()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', (event) => {
  if (quitting) return
  if (!backendProcess) return
  event.preventDefault()
  quitting = true
  logLine('[app] before-quit: stopping backend before exit')
  stopBackend().finally(() => app.quit())
})

// Make sure Ctrl+C in a terminal / `kill -TERM` on the Electron process also
// tears down the Python child instead of leaving it orphaned.
process.on('SIGINT', () => app.quit())
process.on('SIGTERM', () => app.quit())
