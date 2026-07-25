'use strict'
const fs = require('fs')
const path = require('path')

/**
 * Copies the pruned Anamnesis payload (anamnesis/build-native/dist/anamnesis
 * -- see that script for what "pruned" means) into the packaged app as a
 * plain filesystem operation, bypassing electron-builder's own file-matching
 * engine entirely.
 *
 * Why not just use `extraResources` like the sibling anamnesis-node-runtime
 * entry does? Verified directly (two separate full builds, the second with
 * an explicit `filter: ['**\/*']` override): electron-builder detects the
 * package.json at the root of that directory and applies its own "only
 * production dependencies are included" node_modules resolution instead of
 * a raw copy -- and that resolution can't walk a foreign, already-pruned
 * node_modules/ it didn't create, silently producing ZERO packages rather
 * than partial or full inclusion. This is a known class of electron-builder
 * gotcha (electron-userland/electron-builder#6392, #3185, #867), not
 * something specific to a typo here. The `filter` override did not fix it,
 * so extraResources is not used for this entry at all -- afterPack's plain
 * fs.cpSync sidesteps the matching engine completely.
 *
 * anamnesis-node-runtime (a single `node` binary, no package.json in that
 * directory) is unaffected by this and keeps using ordinary extraResources.
 */
exports.default = async function afterPack(context) {
  // desktop/build-resources/afterPack.cjs -> desktop -> repo root
  const repoRoot = path.resolve(__dirname, '..', '..')
  const src = path.join(repoRoot, 'anamnesis', 'build-native', 'dist', 'anamnesis')
  const dest = path.join(context.appOutDir, 'resources', 'anamnesis')

  if (context.electronPlatformName !== 'linux' && context.electronPlatformName !== 'win32') {
    // build.sh/build.ps1 only produce Linux x64 / Windows x64 payloads today
    // -- see their own platform guards. Nothing to do (and nothing to
    // break) on other platforms until a matching build script exists.
    console.log(`[afterPack] skipping Anamnesis bundling for ${context.electronPlatformName} (Linux/Windows x64 only for now)`)
    return
  }

  if (!fs.existsSync(src)) {
    const script = context.electronPlatformName === 'win32'
      ? 'anamnesis\\build-native\\build.ps1'
      : 'anamnesis/build-native/build.sh'
    throw new Error(
      `afterPack: expected a pruned Anamnesis build at ${src} but it doesn't exist. ` +
        `Run ${script} first.`
    )
  }

  fs.rmSync(dest, { recursive: true, force: true })
  fs.cpSync(src, dest, { recursive: true })

  const nodeModulesDir = path.join(dest, 'node_modules')
  const packageCount = fs.existsSync(nodeModulesDir) ? fs.readdirSync(nodeModulesDir).length : 0
  console.log(`[afterPack] copied Anamnesis to ${dest} (node_modules: ${packageCount} packages)`)

  if (packageCount === 0) {
    // Fail loudly rather than silently shipping a build that can't
    // actually start Anamnesis -- this exact failure mode is what
    // extraResources produced silently before this hook existed.
    throw new Error(
      `afterPack: copied Anamnesis to ${dest} but its node_modules/ is empty or missing. ` +
        'Refusing to produce a build that cannot start Anamnesis.'
    )
  }
}
