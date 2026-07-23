# desktop

An Electron application with React and TypeScript

## Recommended IDE Setup

- [VSCode](https://code.visualstudio.com/) + [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) + [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)

## Project Setup

### Install

```bash
$ npm install
```

### Development

```bash
$ npm run dev
```

### Build

The bundled Python backend (PyInstaller onedir, see `backend-build/`) must be
built first, on the target OS -- PyInstaller does not cross-compile, so a
Windows backend bundle has to be produced by running `backend-build/build.sh`
(or a Windows equivalent -- none exists yet, see repo audit notes) on an
actual Windows machine with the project's venv.

```bash
# Linux (.deb, AppImage)
$ npm run dist:linux

# Windows (NSIS installer) -- untested end-to-end; see repo audit notes
$ npm run dist:win
```

There is currently no macOS build script wired up in `package.json`
(`electron-builder.yml` has a `mac:`/`dmg:` section, but no `dist:mac` npm
script exists yet, and it is unsigned/unnotarized).

