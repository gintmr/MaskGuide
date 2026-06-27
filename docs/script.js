/* MaskGuide project page — interactions */
(function () {
  "use strict";

  /* ---------- theme toggle (persisted) ---------- */
  var root = document.documentElement;
  var toggle = document.getElementById("theme-toggle");
  var icon = document.getElementById("theme-icon");

  var sunPath = '<circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8"/>';
  var moonPath = '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>';

  function apply(theme) {
    root.setAttribute("data-theme", theme);
    if (icon) icon.innerHTML = theme === "dark" ? sunPath : moonPath;
  }

  var saved = null;
  var qp = new URLSearchParams(window.location.search).get("theme");
  if (qp === "light" || qp === "dark") { saved = qp; }
  try { if (!saved) saved = localStorage.getItem("mg-theme"); } catch (e) {}
  if (!saved) {
    saved = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  apply(saved);

  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      apply(next);
      try { localStorage.setItem("mg-theme", next); } catch (e) {}
    });
  }

  /* ---------- reveal on scroll ---------- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in"); obs.unobserve(en.target); }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    reveals.forEach(function (el) { obs.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- copy bibtex ---------- */
  var copyBtn = document.getElementById("copy-cite");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = document.getElementById("cite-text").innerText;
      var done = function () {
        copyBtn.textContent = "Copied ✓";
        setTimeout(function () { copyBtn.textContent = "Copy"; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  }

  /* ---------- lightbox ---------- */
  var lb = document.getElementById("lightbox");
  var lbImg = document.getElementById("lb-img");
  var lbCap = document.getElementById("lb-cap");
  var lbClose = document.getElementById("lb-close");

  function openLightbox(src, cap) {
    lbImg.setAttribute("src", src);
    lbImg.setAttribute("alt", cap || "");
    lbCap.textContent = cap || "";
    lb.classList.add("open");
    document.body.style.overflow = "hidden";
  }
  function closeLightbox() {
    lb.classList.remove("open");
    lbImg.setAttribute("src", "");
    document.body.style.overflow = "";
  }

  document.querySelectorAll(".gcard").forEach(function (card) {
    card.addEventListener("click", function () {
      var img = card.querySelector("img");
      if (img) openLightbox(img.getAttribute("src"), card.getAttribute("data-cap") || img.getAttribute("alt"));
    });
  });

  /* ---------- figure album ---------- */
  var album = document.getElementById("fig-album");
  if (album) {
    var tabs = Array.prototype.slice.call(album.querySelectorAll("#album-tabs button"));
    var aImg = document.getElementById("album-img");
    var aTitle = document.getElementById("album-title");
    var aIdx = document.getElementById("album-idx");
    var aTotal = document.getElementById("album-total");
    var aCap = document.getElementById("album-cap");
    var cur = 0;
    if (aTotal) aTotal.textContent = tabs.length;
    function showFig(i) {
      cur = (i + tabs.length) % tabs.length;
      var b = tabs[cur];
      aImg.setAttribute("src", b.getAttribute("data-src"));
      aImg.setAttribute("alt", b.getAttribute("data-title"));
      aTitle.textContent = b.getAttribute("data-title");
      aCap.textContent = b.getAttribute("data-cap");
      if (aIdx) aIdx.textContent = cur + 1;
      tabs.forEach(function (t) { t.classList.toggle("active", t === b); });
    }
    var autoTimer = null, hovering = false;
    var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function stopAuto() { if (autoTimer) { clearInterval(autoTimer); autoTimer = null; } }
    function startAuto() {
      stopAuto();
      if (reduceMotion || hovering) return;
      autoTimer = setInterval(function () { showFig(cur + 1); }, 5200);
    }

    tabs.forEach(function (t, i) { t.addEventListener("click", function () { showFig(i); startAuto(); }); });
    var aPrev = document.getElementById("album-prev");
    var aNext = document.getElementById("album-next");
    if (aPrev) aPrev.addEventListener("click", function () { showFig(cur - 1); startAuto(); });
    if (aNext) aNext.addEventListener("click", function () { showFig(cur + 1); startAuto(); });
    if (aImg) aImg.addEventListener("click", function () { stopAuto(); openLightbox(aImg.getAttribute("src"), aTitle.textContent); });

    album.addEventListener("mouseenter", function () { hovering = true; stopAuto(); });
    album.addEventListener("mouseleave", function () { hovering = false; startAuto(); });
    album.addEventListener("focusin", stopAuto);
    album.addEventListener("focusout", function () { if (!hovering) startAuto(); });

    document.addEventListener("keydown", function (e) {
      if (lb && lb.classList.contains("open")) return;
      if (!(hovering || album.contains(document.activeElement))) return;
      if (e.key === "ArrowLeft") { showFig(cur - 1); startAuto(); e.preventDefault(); }
      else if (e.key === "ArrowRight") { showFig(cur + 1); startAuto(); e.preventDefault(); }
    });

    showFig(0);
    startAuto();
  }

  if (lbClose) lbClose.addEventListener("click", closeLightbox);
  if (lb) lb.addEventListener("click", function (e) { if (e.target === lb) closeLightbox(); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeLightbox(); });
})();
