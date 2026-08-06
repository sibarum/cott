# COTT / Traction — Consolidated Book Plan

*A restructuring plan for a single, coherent, linked series of chapters — a knowledge base for COTT, Traction Theory, Mirror Calculus, and the surrounding ideas — pitched at curious non-mathematicians. Built from an audit of the full legacy corpus and the current papers.*

Status: **plan approved (outline v3), not yet written.** The legacy pages stay as a historical record and are phased out later; this book consolidates everything worth keeping and fixes what's broken.

---

## 1. Purpose & audience

A reader who likes math but isn't a mathematician should be able to start at Chapter 1 and be *carried* — each chapter forced by the one before — from "what is a number for?" to the frontier, and be able to branch sideways through cross-links like a wiki. Chapters have no length quota; each has one well-defined topic. Plain topic titles (index-friendly) with an evocative subtitle (the house voice).

## 2. The reconciled spine (the one thread)

> Treat **operations** as primary and a **type** as "the total, reversible operations it stays closed under." The single axiom that violates this is **absorption** (`0·a = 0`). Delete only that. Cancellation then becomes an *operation* **⊘**, and `0` and `1` fall out as its two **residues**. Zero becomes **invertible** (`1/0 = ω`, `0·ω = 1`) and **graded**. From the grade rail come an exact algebraic circle (the Chebyshev/dyadic tower), multiplication solved by grades, and addition regrown as a **dial** `S_p = (xᵖ+yᵖ)^(1/p)`. Read every "addition" as a **chart** and the derivative, log-derivative, and iteration become one operator `D_α f = (α∘f)′`. What the charts can't cover globally becomes an **obstruction** (residue / associator), and the thread ends at an honest frontier (`ω ≠ ∞`; Γ, ζ ceded to limits) resting on two admitted conjectures.

## 3. Confidence & candor mechanism

- Keep the three-tier tag: **Solid** (proven / machine-checked), **Plausible** (load-bearing bet), **Wild** (past the proof).
- **Speak only as confidently as the evidence supports.** Reframes and caveats go in **footnotes/asides**, not hedged into every sentence — but never omit a known problem. Specifically: `0^ω = −1` reads confidently as "the natural normalization," with a footnote that it is a branch/character choice, not a forced theorem; the Chebyshev chapter states the correct math plainly, with a footnote that it holds for any reciprocal pair; borrowed vocabulary (monodromy, gauge, eigenvalue) is defined as local jargon with a caveat or replaced.
- **"Machine-checked" must be scoped honestly**: it covers the exact rational/algebraic fragment, the tower, and the associator cocycle — *not* the consistency/projection conjectures.

## 4. Global notation canon (one fix that touches everything)

Legacy docs drift; fix once, use everywhere:

| Symbol | Canonical meaning | Kill / avoid |
|---|---|---|
| `0` | the invertible, graded, non-absorbing zero (grade +1) | never "the additive identity"; never conflate with erasure |
| `ω` | `1/0`, reciprocal of zero (grade −1) | stop writing `w`; **never reuse the same letter for `0^{1/4}`** (the Chebyshev-tower symbol clash) |
| `⊘` | cancellation as an *operation* (annihilation); leaves a residue | not an element |
| residue | the `0` or `1` deposited by `⊘` (subtractive-in-product → `0`; divisive-in-sum → `1`) | keep distinct from the complex-analysis "residue" (name-overlap flagged where used) |
| grade | integer valuation; `grade(0ⁿ) = +n` | one convention across all docs |
| `S_p` | the addition dial / **grade sum** `(xᵖ+yᵖ)^(1/p)` | — |
| `⊕` | harmonic addition `ab/(a+b)` (`S_{-1}`) | — |
| `= / ≈ / :=` | reversible rewrite / lossy projection / structure-introduction | keep this discipline explicit |
| tower symbols | pick distinct letters for `0^{1/2}`, `0^{1/4}`, `s`, `t` | resolve `v`/`w`/`u` overloads from `theory/chebyshev` |

Also: settle the metatheory name — **"Principled Emergence Metatheory"** (drop the "Emergent" variant).

## 5. Cross-cutting issues ledger

| # | Problem (source) | Fix |
|---|---|---|
| 1 | "What `0` *is*" told 3 ways: additive identity vs `null`/erasure vs displacement (`intro`, `theory/traction`, `void-calculus`) | Adopt the ⊘/residue canon uniformly (Ch 4). No primitive 0. |
| 2 | `0^ω = −1` "forced by closure" — circular (log₀ *is* base-0 exp's inverse); really a branch choice (CS Constraint 05) | Reframe as natural normalization; footnote the circularity (Ch 6). |
| 3 | `0²` contradiction: `−1` vs `0²` vs `−1−i`; the `0^a+0^b = 0^{ab}` step is invalid (`intro §7`, `chebyshev §2`) | Drop the broken `0²=−1−i`/`0³=−2i`; standardize `0ⁿ = u^{2n}` (Ch 6, 8). |
| 4 | Chebyshev "coincidence = evidence" — a tautology for any `uv=1` pair | Keep the correct math; footnote it's natural structure, not proof (Ch 8). |
| 5 | Borrowed vocabulary: "type theory"→ZFC (Cantor); monodromy/holonomy/gauge/eigenvalue for finite objects; "non-archimedean number field"→function field | Define as local jargon w/ caveat, or replace (Ch 8, 13, 17, 18). |
| 6 | "machine-checked" scope vague | State exactly what's checked vs conjectured (Ch 22 + throughout). |
| 7 | Notation drift: PEM name; `w`=ω vs `0^{1/4}` | §4 canon. |
| 8 | Honest Logic requires "proven true AND disproven false" (contradiction); PEM elegance metric circular/undefined | Repair truth conditions; scope PEM as heuristic (Ch 3, 21). |
| 9 | Stubs: `zero`, `vector`, `browserai`, `analogsnn`, half of `metatheory` | Not carried forward; content written fresh where needed. |
| 10 | Two load-bearing conjectures buried at the end | Foreground as *the* open problems (Ch 22), cross-linked from every Solid claim. |
| 11 | `0/0 = 1`, "ω is not infinity", "surjective to ℂ" asserted as derived | Mark stipulations/intuitions as such; scope the tower's reach (Ch 5, 8). |

---

## 6. Chapter plan

Fields per chapter — **Thesis** (one breath) · **Tag** · **Consolidates** (sources) · **Keep** (gems) · **Fix** (reframes) · **Links**.

### Front matter

**0 · How to Read This** — *The rock, the rope, and the receipts* — **this is `intro/index.html`, the landing page**
- **Thesis:** what this is, and how to read the confidence tags; the "counterexamples are information, not defeat" ethos. Doubles as the **table of contents**: the chapter list (grouped by Part) with a one-line evocative summary each, plus the tag legend.
- **Tag:** —
- **Consolidates:** `papers/0` disclaimer (confidence-as-altitude), `theory/honest` (ethos).
- **Keep:** the altitude framing; the debt-honesty stance.
- **Fix:** state the tag definitions precisely up front.
- **Links:** → every chapter (it's the TOC); esp. → 1, 21, 22.

### The One Idea (Solid)

**1 · Types as Operations** — *What a number is for, not what it is*
- **Thesis:** a type is defined by the total, reversible operations it stays closed under, not by its elements.
- **Tag:** Solid.
- **Consolidates:** `intro §1`, `theory/reference §4.3`, `papers/1 §01` (the Type definition).
- **Keep:** the ℕ→ℤ→ℚ→ℂ progression ("types emerge from which operations must stay total") — the strongest intuition pump in the corpus.
- **Fix:** —
- **Links:** → 2, 4.

**2 · Totality and Reversibility** — *The one thing we refuse to do*
- **Thesis:** demand every operation be total and reversible (no information destroyed); that single refusal drives everything.
- **Tag:** Solid.
- **Consolidates:** `intro §1`, `papers/0 §I`, `papers/1 §01`.
- **Keep:** "limits as a reconciliation ritual" (manipulate beyond the domain, then forgive by a limit).
- **Fix:** reversibility ≠ bijectivity; state precisely what "no irreversible collapse" means (currently informal).
- **Links:** → 3, 4, 15 (where limits legitimately return).

**3 · The Method (PEM)** — *Delete one axiom, keep everything else*
- **Thesis:** a design discipline — seeds (a one-sentence intent), minimal axioms justified from the seed, reuse-by-reframing; hence "delete only absorption."
- **Tag:** Plausible (as a stated method).
- **Consolidates:** `theory/reference §1`, `theory/metatheory §1`.
- **Keep:** the "seed / reconstruct axioms from a metarule" idea; minimality narrative.
- **Fix:** unify the name; be honest that "elegance as a measurable criterion" has **no metric yet** — present as heuristic, not theorem; note the metatheory doc is incomplete.
- **Links:** → 2, 21.

**4 · Cancellation as an Operation** — *Zero and one are what's left behind*
- **Thesis:** ⊘ (annihilation) is an operation, not an element; `0` and `1` are its two residues (subtractive-in-a-product → `0`; divisive-in-a-sum → `1`).
- **Tag:** Solid.
- **Consolidates:** `papers/1 §02`, cheatsheet "Foundation".
- **Keep:** the observation that classical `0` fuses three jobs (identity + annihilator + no-quantity); the reversibility argument that a residue *must* be left; the two-residues additive/multiplicative duality; the `=`/`≈`/`:=` discipline.
- **Fix:** **the linchpin** — this canon resolves the three-way "what is 0" conflict across the legacy docs; state it once, definitively.
- **Links:** → 5, 9, 11, 16.

**5 · The Invertible Zero** — *A reciprocal for nothing — and why it isn't infinity*
- **Thesis:** with absorption gone, `1/0 := ω`, `0·ω = 1`; zero carries a grade; ω is symbolic (a direction), not analytic (`∞`).
- **Tag:** Solid (core); the ω≠∞ line is the sharpest conceptual move.
- **Consolidates:** `intro §4`, `theory/reference §3.2`, `papers/1 §03`, `papers/0 §II`, `papers/4 §V`, cheatsheet grade rail.
- **Keep:** the ω≠∞ distinction (wraps the circle, `0^{2ω}=1`); grades as valuation.
- **Fix:** specify the multiplication table for `0·(finite)` (never fully given); mark `0/0 = 1` as a *stipulation* (vs Wheel theory's ⊥, Meadows), not a derivation.
- **Links:** → 4, 9, 15, 19.

### The Circle for Free (Solid, one caveat)

**6 · Powers of the Zero** — *Where −1 comes from*
- **Thesis:** the integer power ladder of 0 and ω is forced; the one free value is fixed by a natural normalization: `0^ω = −1`.
- **Tag:** Solid (ladder) / **caveat** on `0^ω`.
- **Consolidates:** `theory/reference §3.3, §4.4`, `intro §5`, `papers/1 §04`, `papers/0 §II`, cheatsheet.
- **Keep:** the tidy minimality enumeration (only `0^ω`, `ω^ω` free).
- **Fix:** **reframe honestly** — `0^ω=−1` is a branch/character choice (cheatsheet Constraint 05), and the log₀ "confirmation" is circular (log₀ is *defined* as base-0 exp's inverse). **Drop** the broken `0²=−1−i`, `0³=−2i` and the invalid `0^a+0^b=0^{ab}` step (`intro §7`).
- **Links:** → 7, 8, 22.

**7 · The Imaginary Unit** — *A zero seen edge-on*
- **Thesis:** `i = 0^{ω/2}`; the imaginary unit is a unit orthogonal to the reals whose shadow is zero; the discarded coordinate is a "receipt."
- **Tag:** Plausible ("a change of eye," not a theorem).
- **Consolidates:** `intro §7`, `theory/reference §5`, `papers/0 §III`, `papers/1 §06`.
- **Keep:** the shadow/receipt reframe (memorable, and it motivates the whole conservation stance).
- **Fix:** present as intuition; note the ambient space in which "0 is a unit at an angle" is precise remains to be built (open).
- **Links:** → 6, 8.

**8 · The Exact Circle** — *Phase without limits; roots of unity by hand*
- **Thesis:** powers of the half-step generator build an exact algebraic model of phase — the Chebyshev/dyadic tower — with roots of unity as exact rationals, no analytic continuation.
- **Tag:** Solid (the math is verified).
- **Consolidates:** `theory/chebyshev`, `theory/chebyshev/database`, `papers/1 §05`, `papers/0 §II`.
- **Keep:** the correct ring `ℚ[s][u]/(u²−su+1)`, recurrence `aₙ = s·aₙ₋₁ − aₙ₋₂`, norm/conjugation, the dyadic doubling, and the verified roots-of-unity database (`0^{ω·x} ↔ e^{iπx}`).
- **Fix:** footnote that this holds for *any* pair with `uv=1` (a Lucas/Dickson identity), so it's natural structure, **not** independent evidence; fix the garbled Discovery table (`chebyshev §2`); "non-archimedean number field" → function-field/Galois tower; "modified Chebyshev of the first kind" → `2Tₙ(s/2)` / Dickson; scope "surjective to ℂ" (only dyadic exponents reached); resolve the `w` symbol clash.
- **Links:** → 6, 16, 17.

### The Two Operations (Solid)

**9 · Multiplication, Solved** — *Bookkeeping with grades*
- **Thesis:** multiplication multiplies coefficients and adds grades; its only indeterminate `0·ω` is cured by one integer.
- **Tag:** Solid.
- **Consolidates:** `papers/0 §IV`, `theory/reference §3.2`.
- **Keep:** the "one integer closes the multiplicative wound" framing.
- **Fix:** reconcile grade-bookkeeping with the exponentiation results (why are `0·ω=1` and `0^ω=−1` both "grade 0" yet distinct — what invariant separates them?); give the coefficient arithmetic explicitly.
- **Links:** → 5, 10.

**10 · The Addition Problem** — *The wound the grades can't close*
- **Thesis:** ordinary addition fails in a graded world because leading-term cancellation hides the grade of what remains; addition must be regrown.
- **Tag:** Solid.
- **Consolidates:** `papers/0 §IV`, `papers/3 §1`.
- **Keep:** the additive-vs-multiplicative wound asymmetry — "one integer vs the whole tail," a valuation vs a germ — the single cleanest motivating insight in the corpus.
- **Fix:** —
- **Links:** → 9, 11.

**11 · The Addition Dial** — *Every average is a different plus*
- **Thesis:** `S_p(x,y) = (xᵖ+yᵖ)^(1/p)` is ordinary `+` conjugated by `x↦xᵖ`; multiplication distributes over *every* `S_p`, and the sign of `p` selects the additive zero.
- **Tag:** Solid (the two theorems are genuinely proven).
- **Consolidates:** `papers/0 §V`, `papers/3 §3, §8`, cheatsheet.
- **Keep:** Theorems 1 (`a·S_p(x,y)=S_p(ax,ay)`) & 2 (sign of p → additive zero); harmonic `⊕=ab/(a+b)`; the power-means / Kolmogorov–Nagumo unification; the named rungs (harmonic / quadrature / tropical max-min / log→×). Author's preferred name: **grade sum**.
- **Fix:** reconcile "which is *the* addition" (reciprocal field vs phase addition) — the compactness argument (`papers/3 §7`) favors the elliptic one; keep the choice honest, not asserted.
- **Links:** → 10, 13, 16.

### Calculus Without Limits (Solid → Plausible)

**12 · The Structural Differential** — *Differentiation as multiplication by zero*
- **Thesis:** `0·f(x) := f(x+0x) − f(x)` yields a derivative-like object with no limit; `0·x² = 2x·0x + (0x)²`, and the `2x` is read off the linear component.
- **Tag:** Solid.
- **Consolidates:** `intro §2`, `papers/1 §02`, `void-calculus §2`.
- **Keep:** the worked `0·x²` example (best pedagogy in the corpus); the dual-numbers-but-retains-higher-order positioning.
- **Fix:** state plainly it is jet/dual-number-like; keep `≈` = "select the linear component" explicit.
- **Links:** → 4, 13, 14.

**13 · The Chart** — *One derivative wearing three hats*
- **Thesis:** `D_α f = (α∘f)′` unifies the ordinary derivative (`α=id`), the log-derivative (`α=log`), and iteration (via Abel/Schröder) as one primitive read through a coordinate α.
- **Tag:** Solid (reduction) / Plausible (the "spectral content" reading).
- **Consolidates:** `papers/0 §VI`, `papers/2 §01`, `papers/4 §I`.
- **Keep:** the three-calculi-as-one-chart unification; the two differentials and `[D,θ]=D`.
- **Fix:** "eigenvalue equation / spectrum" for Schröder is loose — footnote it as suggestive, not established.
- **Links:** → 11, 15, 16.

**14 · Void Calculus in Practice** — *Slopes and roots without a limit*
- **Thesis:** treating `0` as a graded infinitesimal and extracting with `ω`, then projecting with `Q`, recovers derivatives, log-derivatives, critical points, and roots.
- **Tag:** Plausible (method) / Solid (the specific results).
- **Consolidates:** `void-calculus §2–3`.
- **Keep:** the correct worked derivative and root-finding examples (`x²+3x−4 → {1,−4}`); the author's honesty that factoring is easier and the value is "isolating complexity to a single square root."
- **Fix:** `Q` is defined only informally — either formalize the projection or flag that the manipulations aren't yet rule-licensed.
- **Links:** → 12, 13.

**15 · The Four Registers** — *Where the method stops and analysis begins*
- **Thesis:** a chart is writable / implicit (functional-equation solution) / obstructed (local-only) / analytic (limit-only); the method owns the first three, and cedes the fourth (`Γ`, `ζ`, the continuum) to limits — the `ω`-vs-`∞` line.
- **Tag:** Solid (taxonomy) / Plausible (the "one axis" claim).
- **Consolidates:** `papers/4` (all), `papers/2 §08`.
- **Keep:** the register taxonomy, honestly sourced (Kolmogorov–Nagumo, Koenigs–Schröder–Abel, Écalle–Voronin all pre-existing); the `ω≠∞` boundary; `x^0 = 1+0·ln x` ("burns the derivative, keeps the ash").
- **Fix:** "hypertranscendental ζ" is overstated relative to Γ/Hölder — footnote; keep "holonomic/D-finite" usage (that one's fine).
- **Links:** → 5, 13, 22.

### Obstructions (Plausible, marked)

**16 · Phases That Won't Close** — *The residue as a winding number*
- **Thesis:** the complex-analysis residue and the `2πi` are, in this reading, the winding `2ω` — the monodromy of log₀ — recovered by finite orthogonality at the tower's roots of unity.
- **Tag:** Solid (P0 framing) / Plausible (P2 "traction contour" is unbuilt).
- **Consolidates:** `papers/2 §03–04`, `papers/0 §VI–VII`.
- **Keep:** residue and `+C` as the two receipts antidifferentiation emits (at ω and 0).
- **Fix:** the "traction contour" is not yet constructed (Plausible); footnote the name-overlap between the two "residues" (persuasion, not proof).
- **Links:** → 8, 11, 17, 18.

**17 · The Octonion Associator** — *A sign that survives every disguise*
- **Thesis:** the octonion associator is a closed, re-signing-invariant ℤ/2 sign, first nonzero at 𝕆, valued in `0^ω=−1`; harmonic addition stays comm+assoc up the Cayley–Dickson tower, while distributivity fails exactly at the sedenions.
- **Tag:** Solid (machine-checked).
- **Consolidates:** `papers/3 §12–13`.
- **Keep:** the exact cocycle counts (0/64 quaternion, 168/512 octonion anti-associating; closed on 4096 loops; invariant under 256 re-signings); the tower table; the author's **self-correction** about *why* sedenions fail (composition failure, not zero-divisor norm) — a model of the ethos.
- **Fix:** "monodromy/holonomy/gauge" are borrowed for a finite group-cohomology object — define as local jargon with a caveat (there is no loop/connection/transport).
- **Links:** → 8, 18.

**18 · One Object, Three Faces?** — *A conjecture about residues, associators, and moduli*
- **Thesis:** the residue's `2πi` (ℤ), the octonion associator (ℤ/2), and the Écalle–Voronin moduli might be one object at a "unipotent corner."
- **Tag:** Plausible — presented explicitly as a **conjecture/analogy**.
- **Consolidates:** `papers/0 §VII`, `papers/3 §12`, `papers/4 §III`.
- **Keep:** the structural intuition and its honest labeling.
- **Fix:** this is the single most-repeated red flag (metaphor-as-identity) — foreground that these live in different categories and the identification is unproven; do not state as fact.
- **Links:** → 16, 17, 22.

### The Frontier (Wild, fenced)

**19 · Reversible vs One-Way** — *Keep the receipt, or burn it*
- **Thesis:** reversibility and one-wayness are two settings of one switch; the discrete logarithm is the implicit chart of `x↦ax` mod n — a burned receipt, the trapdoor cryptography sells.
- **Tag:** Plausible.
- **Consolidates:** `papers/0 §VIII`.
- **Keep:** the framing (genuinely illuminating) and the honest corollary that traction, keeping every receipt, is therefore *not* a code-breaker.
- **Fix:** —
- **Links:** → 5, 15.

**20 · Order, Time, and Physics** — *Past the proof*
- **Thesis:** commutativity is the lossy quotient that forgets order; time is the retained order; the arrow is a global monodromy — plus gravity/Bell speculations.
- **Tag:** Wild (fenced, explicitly).
- **Consolidates:** `papers/0 §IX`, `papers/2 §06–07`.
- **Keep:** the "block universe = commutative shadow" image; clearly-labeled bets.
- **Fix:** the Bell claim as written misstates Bell (local hidden-variable models are bounded regardless of ordering) — correct or heavily caveat in a footnote.
- **Links:** → 17, 18.

### Judging the Work (meta, at the end)

**21 · Honest Logic** — *Receipts, and what it takes to call something true*
- **Thesis:** proofs as non-erasing "receipts"; a rubric for axiom quality (Parsimony, Empirical Fit, Provenance, Net-Complexity Convenience).
- **Tag:** Plausible (method).
- **Consolidates:** `theory/honest`.
- **Keep:** the four axiom-quality axes; the AC-vs-PA illustration.
- **Fix:** repair the incoherent objective-truth condition ("proven true **and** disproven false" are mutually exclusive — almost certainly a drafting slip); drop or defend the "Clifford > quaternions empirical fit" aside.
- **Links:** → 3, 22.

**22 · The Debt Ledger** — *The two things everything rests on*
- **Thesis:** the whole edifice is one implication with two unproven premises: **consistency / joint-refusability** (absorption, commutativity, associativity, norm-multiplicativity can all be declined together) and the **projection theorem** (classical arithmetic is this structure modulo imposed absorption).
- **Tag:** Plausible — "this is the ballgame."
- **Consolidates:** `papers/0 §X`, `papers/1 §08`.
- **Keep:** the candor verbatim in spirit ("a conjecture wearing a symbol rather than a theorem forcing one"); the "checkable-first" discipline.
- **Fix:** here is where "machine-checked" gets its precise scope; cross-link *back* from every Solid claim that ultimately leans on these.
- **Links:** ← every Solid chapter.

---

## 7. Appendices

- **A · Notation glossary** — the §4 canon, one page, authoritative.
- **B · Old → new map** — table below; which legacy page each chapter absorbs, and which pages become pure historical record.
- **C · [pending] The Solver** — the `project-to-ℂ-last` pipeline, `[coefficient, phase]` encoding, and known limitations (`solver/`, `implementation.html`) are rigorous and worth a home — **held until the decision on whether the solver is updated to the new theory.**
- **D · [optional] Opinions** — the salvageable kernels of the hot-takes, reframed and corrected: "unbounded ≠ actually-infinite" (Cantor) and "qubit count is a bad metric" (Quantum). Clearly labeled as opinion, separate from the theory. *Include only if wanted; otherwise these stay solely in the historical-record pages.*

## 8. Stays as historical record (not consolidated, phased out later)

`theory/traction`, `theory/reference`, `theory/chebyshev(+database)`, `theory/zero`, `theory/metatheory`, `theory/honest`, `theory/void-calculus`, `hot/cantor`, `hot/quantum`, `software/*`, `inventions/analogsnn`, `solver/*`, and the current `papers/0–4`. The new book supersedes them chapter by chapter; nothing is edited or deleted.

**Exception — `intro/`:** the book is *built into* `intro/` (see §9.1). The current `intro/index.html` and `intro/traction.html` are historical record only until the new landing page and chapters are ready, at which point they are replaced/removed in place. New chapters are added as `intro/<slug>/` without touching the old files until the final swap.

### Old → new source map (abbreviated)

| Legacy / current source | Feeds chapters |
|---|---|
| `intro/index.html` | 1, 2, 5, 6, 7, 12 |
| `intro/traction.html` (stub) | (generators note → 6/8) |
| `theory/traction/index.html` | 4, 5, 9 |
| `theory/reference/index.html` | 1, 3, 5, 6, 7, 8 |
| `theory/chebyshev(+database)` | 8 |
| `theory/metatheory` | 3 |
| `theory/honest` | 0, 21 |
| `theory/void-calculus` | 12, 14 |
| `hot/cantor`, `hot/quantum` | Appendix D (optional) |
| `solver/*`, `implementation.html` | Appendix C (pending) |
| `papers/0 The Conserved World` | 1–2, 4–5, 9–10, 16–20, 22 (the spine) |
| `papers/1 …Invertible Zero` | 2, 4, 6, 8, 22 |
| `papers/2 …Calculus of the Pole` | 13, 16, 20 |
| `papers/3 …Sum That Chooses Its Zero` | 10, 11, 17, 18 |
| `papers/4 …Chart You Can't Write Down` | 13, 15, 18 |
| `ln-zero-cheatsheet` | 4, 5, 6, 11 |

---

## 9. Open decisions (before drafting chapters)

1. **Book home — DECIDED: `/intro/`.** The book lives in the `intro/` folder and *becomes* the new intro. Each chapter is a directory with `index.html` for clean URLs (e.g. `/cott/intro/invertible-zero/`, matching the site convention). **`intro/index.html` becomes the landing page: a table of contents with a one-line evocative summary per chapter** (grouped by the Parts above), plus the tag legend. The current `intro/index.html` and `intro/traction.html` remain as historical record until the new pages are ready, then are replaced/removed. New chapters can be built in `intro/<slug>/` alongside the old files without disturbing them; the `index.html` swap is the last step.
2. **Solver** (Appendix C) — updated to the new theory, or omitted for now?
3. **Opinions** (Appendix D) — include the reframed kernels, or leave hot-takes solely as historical record?
4. **Drafting order** — recommend Chapters 4–5 and 11 first (the strongest, most self-contained cores) to set voice and notation before the derivation spine.
