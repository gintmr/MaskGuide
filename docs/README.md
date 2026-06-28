# MaskGuide — Project Page

Static project page for **MaskGuide: Efficient Distillation for Deployable Lightweight
Segmentation in Marine Environments** (accepted at IEEE Robotics and Automation Letters, RA-L 2026).

Pure HTML/CSS/JS — no build step. Marine editorial theme with a deep-ocean dark mode.

## Files
- `index.html` — page content
- `styles.css` — theme + layout (light seafoam / dark abyss; teal accent, coral highlights)
- `script.js` — theme toggle, scroll reveal, figure album, lightbox, copy-BibTeX
- `assets/` — figures from the paper, the logo, and the camera-ready PDF
  (`MaskGuide_RAL2026.pdf`, offered for download in the hero, citation, and footer)
- `favicon.svg`, `.nojekyll`

## Preview locally
```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy to GitHub Pages
1. Push this repo to GitHub.
2. Settings → Pages → Source: `Deploy from a branch`.
3. Branch: `main`, folder: `/docs`. Save.
4. Site goes live at `https://<user>.github.io/<repo>/`.

All figures are sourced **only** from the compiled paper
(`MaskGuide-latex/figure/`): `overview`, `method-pipeline` (Combine-3figs),
`problems` (3Problems-mini), `qualitative` (Response_fig2), and the two
training-curve JPGs. Transparent PNGs are given a light canvas so their black
labels stay readable in dark mode.

## TODO before publishing
- All six author emails are filled (from the paper's author block, hover tooltip
  + bottom list in the hero `.authors` block).
- Confirm the **GitHub** link (`https://github.com/gintmr/MaskGuide`).
- The paper link (`https://ieeexplore.ieee.org/document/11479799`) is wired into
  the nav button, the hero "Accepted" badge, the "Read the Paper" CTA, and the
  footer. The BibTeX is complete (vol. 11, no. 6, pp. 6775–6782, June 2026,
  DOI 10.1109/LRA.2026.3683334).
