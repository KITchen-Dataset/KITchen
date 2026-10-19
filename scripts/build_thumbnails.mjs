// Render a transparent PNG thumbnail for every GLB in static/models/ using
// <model-viewer> in headless Chromium.
//
//   npm i playwright            (or use a global install)
//   python -m http.server 8000  (from the repo root, in another shell)
//   node scripts/build_thumbnails.mjs http://localhost:8000
//
// Writes static/models/thumbs/obj_XXXXXX.png (480x480); convert to 320 px WebP for the site.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const base = process.argv[2] || 'http://localhost:8000';
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modelsDir = path.join(repo, 'static', 'models');
const outDir = path.join(modelsDir, 'thumbs');
fs.mkdirSync(outDir, { recursive: true });
const only = (process.env.ONLY || '').split(',').filter(Boolean).map(n => `obj_${n.padStart(6, '0')}.glb`);
const glbs = fs.readdirSync(modelsDir).filter(f => /^obj_\d{6}\.glb$/.test(f) && (!only.length || only.includes(f))).sort();

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 480, height: 480 }, deviceScaleFactor: 1 });
if (process.env.MV_LOCAL) {
  await page.route(/model-viewer\.min\.js$/, r => r.fulfill({ path: process.env.MV_LOCAL, contentType: 'text/javascript' }));
}
await page.goto(`${base}/scripts/thumb.html`);
await page.waitForFunction(() => customElements.get('model-viewer'));
for (const f of glbs) {
  const dataUrl = await page.evaluate(async (src) => {
    const mv = document.querySelector('model-viewer');
    await new Promise((res, rej) => {
      const t = setTimeout(() => rej(new Error('timeout ' + src)), 60000);
      mv.addEventListener('load', () => { clearTimeout(t); res(); }, { once: true });
      mv.src = src;
    });
    await new Promise(r => setTimeout(r, 400));
    const blob = await mv.toBlob({ mimeType: 'image/png', idealAspect: true });
    return await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  }, `../static/models/${f}`);
  fs.writeFileSync(path.join(outDir, f.replace('.glb', '.png')), Buffer.from(dataUrl.split(',')[1], 'base64'));
  process.stdout.write(f + ' ');
}
await browser.close();
console.log('\ndone', glbs.length);
