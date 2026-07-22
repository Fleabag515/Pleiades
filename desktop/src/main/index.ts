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

// TODO(phase3): this shells out to the developer's existing ~/Pleiades/.venv
// checkout. A packaged installer needs a self-contained backend (PyInstaller
// or Nuitka build of pleiades.webui) bundled as an extraResource so the app
// works on machines that don't have this exact venv/repo checkout.
const PLEIADES_ROOT = process.env.PLEIADES_ROOT ?? join(homedir(), 'Pleiades')
const PYTHON_BIN = join(PLEIADES_ROOT, '.venv', 'bin', 'python3.12')

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
  if (!existsSync(PYTHON_BIN)) {
    backendState.phase = 'error'
    backendState.error = `Python interpreter not found at ${PYTHON_BIN}. Is ~/Pleiades checked out with its venv built?`
    logLine(`[backend] ${backendState.error}`)
    return
  }

  logLine(`[backend] spawning ${PYTHON_BIN} -m pleiades.webui --host 127.0.0.1 --port ${port} --no-browser (cwd=${PLEIADES_ROOT})`)

  const child = spawn(
    PYTHON_BIN,
    ['-m', 'pleiades.webui', '--host', '127.0.0.1', '--port', String(port), '--no-browser'],
    { cwd: PLEIADES_ROOT, stdio: ['ignore', 'pipe', 'pipe'] }
  ) as BackendProcess

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
 * are cross-origin.
 *
 * Phase B tried the "inject permissive CORS response headers" approach
 * first (see git history), scoped to responses from our own spawned
 * 127.0.0.1 backend port. It is not sufficient: pleiades/webui/server.py
 * has no CORSMiddleware/OPTIONS route, so any *preflighted* request (any
 * POST/DELETE carrying `Content-Type: application/json`, which is exactly
 * what chat creation, sending a message, tool approval, and chat deletion
 * all need) gets a bare `405 Method Not Allowed` on the OPTIONS preflight
 * itself. Chromium requires a 2xx preflight response before it will even
 * look at injected Access-Control-* headers, so header injection alone
 * cannot fix this without also touching the Python backend (out of scope
 * per the brief).
 *
 * Fix: disable Chromium's same-origin policy for this window instead
 * (`webSecurity: false` below), which removes CORS/preflight enforcement
 * entirely — no more OPTIONS round-trip, no more header dance. This does
 * NOT widen what the renderer can reach: the page's CSP
 * (`connect-src 'self' http://127.0.0.1:*`, `img-src` likewise) still
 * hard-limits every fetch/img load to same-origin plus our own spawned
 * localhost backend, contextIsolation/sandbox/nodeIntegration stay on, and
 * navigation is still locked down via will-navigate below. In short: this
 * app only ever talks to a backend it spawned itself on 127.0.0.1, so
 * disabling a same-origin check that exists to protect against *other*
 * origins carries no real risk here.
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
      // See the CORS comment above `registerIpcHandlers` for why this is
      // off: it's required for POST/DELETE JSON requests against our own
      // spawned backend to work at all, and CSP still scopes network
      // access to 'self' + http://127.0.0.1:*.
      webSecurity: false
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
