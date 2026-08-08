# "Proven" tag audit

Every live `<span class="conf c-proven">Proven</span>` badge in the corpus, the claim it
sits on, and a recommendation for its tooltip citation + hyperlink target.

Scope: `intro/` book. The legend instance at `intro/index.html:49` is excluded (it defines
the badge, it doesn't make a claim). Two chapters carry real Proven badges: the new
`where-the-choices-show` chapter (3) and `addition-dial` (2). Total: **5 claims**.

> Note: `papers/4-the-chart-you-cant-write-down` still uses the **retired** `c-solid`
> "Solid" badge (lines 117, 187). Out of scope for this task, but flagging it — per
> commit `c3360db` that label was retired. Worth a follow-up sweep.

---

## 1. Rigidity — distributivity forces ordinary ×
`intro/where-the-choices-show/index.html:51`

> "Chase that back and the multiplication is forced to be ordinary ×." (Every field you
> reach by re-charting is ℝ in disguise.)

- **Whose:** yours in framing, but the mathematical core is classical: distributivity ⇒
  each "multiply-by-a" map is additive ⇒ (barring pathological monsters) linear ⇒ ordinary
  ×. This is the Cauchy functional-equation / additive-implies-linear result.
- **Home today:** none. Paper 4 (*The Chart You Can't Write Down*) is the closest thematic
  fit but does **not** actually state or prove this theorem.
- **Recommendation:** **write a short new article** — "Why there's no new field" — since
  nothing in the corpus currently carries the proof, and it's load-bearing enough (the whole
  "everything collapses to ℝ" thesis rests on it) to deserve its own home. Tooltip meanwhile
  should cite the classical backing: **Aczél, *Lectures on Functional Equations and Their
  Applications*** (additive ⇒ linear). Link the badge to the new article once written; until
  then, link to Paper 4 as the nearest context.

## 2. Tropical is the idempotent limit at the edge
`intro/where-the-choices-show/index.html:72`

> "No reversible re-charting of + can be idempotent, so tropical is … the limit at its
> edge, the point where the addition stops being invertible."

- **Whose:** yours, but again resting on an elementary fact: an idempotent element in a
  cancellative (reversible) operation is trivial, so no reversible re-charting of + is
  idempotent.
- **Home today:** none.
- **Recommendation:** **bundle with claim 1** — it's the corollary of the same rigidity
  theorem (the edge case where reversibility is given up). Same new article, "tropical"
  section. Tooltip: "idempotent + cancellative ⇒ trivial; tropical lives on the boundary."
  Standard external backing for the semiring side: **Maclagan & Sturmfels, *Introduction to
  Tropical Geometry***.

## 3. 0^ω is the multiplication's own −1
`intro/where-the-choices-show/index.html:93`

> "0^ω is the multiplication's own '−1' — its unique nontrivial square root of the identity,
> its order-two element."

- **Whose:** **yours** — this is core COTT (the balance principle behind `0^w = -1`, commit
  `420bcd5`; canon in memory *traction-cancellation-canon*).
- **Home today:** **exists.** `intro/powers-of-zero/` develops `0^{2ω}=1` / the order-two
  element. Paper 1 (*What Falls Out of an Invertible Zero*) also covers it.
- **Recommendation:** **link to existing** — `intro/powers-of-zero/`. No new article needed.
  Tooltip: "The unique nontrivial square root of the multiplicative identity — 0^{2ω}=1."

## 4. Multiplication distributes over the entire dial
`intro/addition-dial/index.html:89`

> "a · S_p(x, y) = S_p(a·x, a·y) — for every setting of the dial at once."

- **Whose:** **yours** — this is Paper 3's territory exactly.
- **Home today:** **exists.** Paper 3 (*The Sum That Chooses Its Own Zero*), §08 "The dial:
  every addition at once."
- **Recommendation:** **link to existing** —
  `papers/3-the-sum-that-chooses-its-zero/#s08`. Tooltip: "One multiplication is homogeneous
  over the whole family of additions, not just ordinary +."

## 5. The sign of the knob decides where "nothing" sits
`intro/addition-dial/index.html:103`

> "A positive setting puts the identity at 0, a negative setting puts it at ω."

- **Whose:** **yours** — Paper 3.
- **Home today:** **exists.** Paper 3, §11 "The striking part — the dial already knew 0
  and ω."
- **Recommendation:** **link to existing** —
  `papers/3-the-sum-that-chooses-its-zero/#s11`. Tooltip: "sign(p) selects which pole (0 or
  ω) holds the additive identity."

---

## Summary — implemented

Resolution: the relevant papers/ proofs were rewritten as a single intro-book appendix,
`intro/proofs/` (*The Proven Results*), added to the TOC as Appendix A. Each Proven badge is
now an `<a class="conf c-proven">` that (a) hyperlinks to its proof and (b) carries a rich
citation in a `data-cite` attribute, surfaced by a **custom hover/focus popover** — not the
native `title` tooltip. Popover engine: `lib/popover.js` (one shared, pointer-events:none
card, keyboard-focusable, Escape-dismissable). Popover + badge-link styling in
`lib/papers.css` (`.cite-popover`). Script is loaded on the two chapter pages that carry
linked badges (`addition-dial`, `where-the-choices-show`).

| # | Location | Whose | Tooltip citation | Links to |
|---|----------|-------|------------------|----------|
| 1 | where-the-choices-show:51 | classical, your framing | distributivity ⇒ additive ⇒ linear (Cauchy) | `intro/proofs/#rigidity` |
| 2 | where-the-choices-show:72 | classical, your framing | cancellative ⇒ not idempotent; semiring edge | `intro/proofs/#tropical-edge` |
| 3 | where-the-choices-show:93 | yours | 0^{2ω}=1, order-two element | `intro/powers-of-zero/` (existing) |
| 4 | addition-dial:89 | yours | homogeneity of × over every S_p | `intro/proofs/#distributes` |
| 5 | addition-dial:103 | yours | sign(p) solves eᵖ=0 → 0 or ω | `intro/proofs/#zero-flip` |

External citations live as footnotes in the appendix: Aczél (Cauchy functional equation)
for #rigidity; Maclagan–Sturmfels (tropical geometry) for #tropical-edge.

Follow-ups still open:
- `papers/4` still uses the retired `c-solid` "Solid" badge (lines 117, 187).
- `where-the-choices-show` remains uncommitted (3 of these 5 badges live there).
- The tooltip is a native `title` attribute. If you want a styled hover-card instead, that's
  a later CSS pass.
