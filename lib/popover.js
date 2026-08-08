/* Citation popovers for confidence badges.
 *
 * Any element with a data-cite attribute (HTML string) gets a hover/focus
 * popover carrying that citation. The element keeps its normal click
 * behaviour — on a linked badge the popover is supplementary and the click
 * still navigates to the full proof.
 *
 * One shared popover node is reused for every trigger. It is pointer-events:
 * none, so it never steals the hover and never flickers.
 */
(function () {
  "use strict";

  function init() {
    var triggers = document.querySelectorAll("[data-cite]");
    if (!triggers.length) return;

    var pop = document.createElement("div");
    pop.className = "cite-popover";
    pop.setAttribute("role", "tooltip");
    document.body.appendChild(pop);

    var current = null;

    function place(el) {
      var r = el.getBoundingClientRect();
      var pr = pop.getBoundingClientRect();
      var vw = document.documentElement.clientWidth;
      var margin = 8;
      var gap = 10; // room for the arrow

      var left = window.scrollX + r.left + r.width / 2 - pr.width / 2;
      var minLeft = window.scrollX + margin;
      var maxLeft = window.scrollX + vw - pr.width - margin;
      if (maxLeft < minLeft) maxLeft = minLeft;
      left = Math.max(minLeft, Math.min(left, maxLeft));

      var below = r.top - pr.height - gap < 0;
      var top = below
        ? window.scrollY + r.bottom + gap
        : window.scrollY + r.top - pr.height - gap;

      pop.style.left = left + "px";
      pop.style.top = top + "px";
      pop.classList.toggle("below", below);

      var arrowX = window.scrollX + r.left + r.width / 2 - left;
      pop.style.setProperty("--arrow-x", arrowX + "px");
    }

    function show(el) {
      current = el;
      pop.innerHTML = el.getAttribute("data-cite");
      place(el);
      pop.classList.add("is-visible");
    }

    function hide() {
      current = null;
      pop.classList.remove("is-visible");
    }

    triggers.forEach(function (el) {
      el.addEventListener("mouseenter", function () { show(el); });
      el.addEventListener("mouseleave", hide);
      el.addEventListener("focus", function () { show(el); });
      el.addEventListener("blur", hide);
    });

    window.addEventListener(
      "scroll",
      function () { if (current) place(current); },
      { passive: true }
    );
    window.addEventListener("resize", function () { if (current) place(current); });
    // Dismiss on Escape for keyboard users.
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && current) { current.blur(); hide(); }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
