'use strict';

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const DAEMON_JS = path.join(__dirname, 'daemon.js');
const NODE_BIN = process.execPath;
const IS_WINDOWS = process.platform === 'win32';

async function install() {
  if (IS_WINDOWS) {
    await installWindows();
  } else if (process.platform === 'darwin') {
    installMacOS();
  } else {
    installLinux();
  }
}

function installLinux() {
  if (process.getuid && process.getuid() !== 0) {
    console.error('error: anamnesis install requires root — run with sudo');
    process.exit(1);
  }
  const unit = `[Unit]
Description=Anamnesis — multi-character memory proxy daemon
After=network.target

[Service]
Type=simple
User=${os.userInfo().username}
ExecStart=${NODE_BIN} ${DAEMON_JS}
Restart=on-failure
RestartSec=5
Environment=ANAMNESIS_LOG=info
StandardOutput=journal
StandardError=journal
SyslogIdentifier=anamnesis

[Install]
WantedBy=multi-user.target
`;
  fs.writeFileSync('/etc/systemd/system/anamnesis.service', unit, 'utf8');
  execSync('systemctl daemon-reload');
  execSync('systemctl enable anamnesis');
  execSync('systemctl restart anamnesis');
  console.log('✓ anamnesis service installed and started');
  console.log('  check status: systemctl status anamnesis');
}

async function installWindows() {
  // Use Task Scheduler (schtasks) — no extra dependencies, works without elevation
  // for per-user ONLOGON tasks. Falls back to a warning if schtasks isn't available.
  const { execSync } = require('child_process');
  const taskName = 'Anamnesis Daemon';

  // Delete existing task silently before recreating
  try { execSync(`schtasks /Delete /TN "${taskName}" /F`, { stdio: 'pipe' }); } catch {}

  const xmlPath = path.join(os.tmpdir(), 'anamnesis-task.xml');
  const xml = `<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>${NODE_BIN.replace(/\\/g, '\\\\')}</Command>
      <Arguments>${DAEMON_JS.replace(/\\/g, '\\\\')}</Arguments>
      <WorkingDirectory>${path.dirname(DAEMON_JS).replace(/\\/g, '\\\\')}</WorkingDirectory>
    </Exec>
  </Actions>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
</Task>`;

  fs.writeFileSync(xmlPath, xml, 'utf16le');
  try {
    execSync(`schtasks /Create /TN "${taskName}" /XML "${xmlPath}" /F`, { stdio: 'pipe' });
    execSync(`schtasks /Run /TN "${taskName}"`, { stdio: 'pipe' });
    fs.unlinkSync(xmlPath);
    console.log('✓ Anamnesis daemon registered as Task Scheduler logon task and started');
    console.log('  It will auto-start on every login.');
    console.log(`  To check: schtasks /Query /TN "${taskName}"`);
    console.log(`  Logs: ${path.join(os.homedir(), '.anamnesis', 'daemon.log')}`);
  } catch (err) {
    console.error('error: failed to create scheduled task:', err.message);
    console.error('Try running as Administrator, or start manually: anamnesis start <name>');
    process.exit(1);
  }
}

const MACOS_LABEL = 'com.anamnesis.daemon';
const MACOS_PLIST = `/Library/LaunchDaemons/${MACOS_LABEL}.plist`;

function installMacOS() {
  if (process.getuid && process.getuid() !== 0) {
    console.error('error: anamnesis install requires root — run with sudo');
    process.exit(1);
  }
  const logPath = path.join(os.homedir(), '.anamnesis', 'daemon.log');
  const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${MACOS_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${NODE_BIN}</string>
    <string>${DAEMON_JS}</string>
  </array>
  <key>UserName</key><string>${os.userInfo().username}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>${logPath}</string>
  <key>StandardErrorPath</key><string>${logPath}</string>
  <key>EnvironmentVariables</key>
  <dict><key>ANAMNESIS_LOG</key><string>info</string></dict>
</dict>
</plist>
`;
  fs.writeFileSync(MACOS_PLIST, plist, 'utf8');
  try { execSync(`launchctl bootout system/${MACOS_LABEL}`, { stdio: 'pipe' }); } catch { /* not loaded */ }
  try {
    execSync(`launchctl bootstrap system ${MACOS_PLIST}`);
  } catch {
    // Pre-Big-Sur macOS: bootstrap may be unavailable; legacy load -w still works.
    execSync(`launchctl load -w ${MACOS_PLIST}`);
  }
  console.log('✓ anamnesis service installed and started');
  console.log(`  check status: launchctl print system/${MACOS_LABEL}`);
}

function uninstallMacOS() {
  try {
    execSync(`launchctl bootout system/${MACOS_LABEL}`, { stdio: 'pipe' });
  } catch {
    try { execSync(`launchctl unload ${MACOS_PLIST}`, { stdio: 'pipe' }); } catch { /* not loaded */ }
  }
  try {
    fs.unlinkSync(MACOS_PLIST);
  } catch {
    /* already gone */
  }
  console.log('✓ anamnesis service uninstalled (data preserved in ~/.anamnesis/)');
}

async function uninstall() {
  if (IS_WINDOWS) {
    await uninstallWindows();
  } else if (process.platform === 'darwin') {
    uninstallMacOS();
  } else {
    uninstallLinux();
  }
}

function uninstallLinux() {
  try {
    execSync('systemctl stop anamnesis');
  } catch {
    /* not running */
  }
  try {
    execSync('systemctl disable anamnesis');
  } catch {
    /* not enabled */
  }
  try {
    fs.unlinkSync('/etc/systemd/system/anamnesis.service');
  } catch {
    /* already gone */
  }
  try {
    execSync('systemctl daemon-reload');
  } catch {
    /* ignore */
  }
  console.log('✓ anamnesis service uninstalled (data preserved in ~/.anamnesis/)');
}

async function uninstallWindows() {
  const { execSync } = require('child_process');
  const taskName = 'Anamnesis Daemon';
  try {
    execSync(`schtasks /End /TN "${taskName}"`, { stdio: 'pipe' });
  } catch {}
  try {
    execSync(`schtasks /Delete /TN "${taskName}" /F`, { stdio: 'pipe' });
    console.log('✓ Anamnesis scheduled task removed (data preserved)');
  } catch {
    console.log('No scheduled task found — nothing to remove');
  }
}

module.exports = { install, uninstall };
