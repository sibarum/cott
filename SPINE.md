# Proposed new spine for `intro/` — DEAD

> **Superseded by [SPINE-2.md](SPINE-2.md), which is the target.** The prescription below
> — move machinery *earlier* — is retired. Do not implement it.
>
> This file is retained, not deleted, because `SPINE-2.md` references it in about a dozen
> places and does not restate what it borrows: the **D1–D4 defect analysis** below is the
> shared diagnosis both spines were built on, and several `[RW]` rewrite obligations
> (Ch 3's diagnosis, Ch 8's `0·ω` derivation, Ch 11's ladder, the `≈` fixes) are carried
> into SPINE-2 by pointer. Read those sections as live; read the ordering as dead.
>
> Also stale on its own terms: written before the working-note thread, so its Part V
> lists two notes where there are now five, and its argument for the new Ch 5 is weaker
> than what `anchor-shadow` and `weaves` later established — evaluation *is* projection,
> which makes join-don't-resolve and the `=`/`≈` discipline one rule rather than two
> conveniences.

## Original proposal (retired)

A re-ordering of the book, as a diff against the current table of contents
(`intro/index.html:54–111`), with the dependency violation motivating each move.

Scope: ordering and the rewrite obligations the ordering exposes. The `=`/`≈` audit is
a separate document; where a move *requires* a rewrite, it is flagged here as **[RW]**.

The re-order is **fork-independent** — nothing below depends on resolving the
infinitesimal-vs-elliptic question in `slot-closure`'s "What is owed" item 1.

---

## The four defects the current order has

**D1 — The indictment precedes the ontology.**
Ch 2 (`totality-reversibility`) makes the case against absorption using `0·a = 0`, with
`0` as the annihilator (`:62`, `:70`). Ch 5 (`cancellation`) then says `0` was *never* the
annihilator — `⊘` is, and `0` is one of its two residues (`:56–88`). The book's founding
crime is described for three chapters in an ontology it later dismantles.

**D2 — The circle is built on machinery introduced after it.**
`powers-of-zero:50` asserts "`0ⁿ` sits at grade `+n`, exactly as the bookkeeping demands" —
but the bookkeeping is Ch 10 (`multiplication`), three chapters downstream. The entire
graded ladder of Part II runs on Part III's machinery.

**D3 — The qualification arrives five chapters after the claim.**
Ch 7 sells `0^ω = −1` as "the choice that closes" (`:74–78`). `where-the-choices-show` then
establishes that `0^ω` is the fingerprint of a *chosen* multiplication, not a constant
(`:116`). A reader who stops at Ch 9 — the end of the book's headline result — has been
misled by sequencing alone.

**D4 — There is no chapter saying what a legal identity is.**
`x⁰ = 1` and `x¹ = x` are used as forced steps in Ch 7 and Ch 10. `slot-closure` later
bars both as *evaluations* — slot-discharges that break the chain. Nothing in Part I tells
the reader that writing an identity can be illegal, so the ladder gets built out of
discharges before the rule against them exists.

---

## The new spine

### Part I — The Refusal

| # | Chapter | Was | Note |
|---|---|---|---|
| 1 | Types as Operations | 1 | unchanged |
| 2 | **Cancellation as an Operation** | 5 | **moved up** — fixes D1 |
| 3 | Totality and Reversibility | 2 | **[RW]** |
| 4 | What a Value Is | 3 | — |
| 5 | **What May Be Written** | *new* | fixes D4 |
| 6 | The Method | 4 | — |

**Ch 2 before Ch 3.** `⊘` and its two residues have to exist before the book indicts
absorption, so the indictment can name the actual defendant: not "zero swallows things"
but "the identity and the annihilator were fused onto one glyph." Ch 5's opening already
reads as the true Chapter 2 — "Before we can hand zero a reciprocal, we have to notice
something strange about it."

**[RW] Ch 3** — rewrite `:62` and `:70`. The `a = b ⟶ 0·a = 0·b ⟶ 0 = 0` demonstration
survives intact as a demonstration; what changes is the diagnosis, which can now say
*fusion* rather than *absorption-as-primitive*, and forward-reference nothing.

**Ch 5, new — *What May Be Written*.** Two things merge here:

- The three-signs discipline, extracted from `cancellation:104–108`. It is currently
  declared inside another chapter and then used **once in the entire corpus** — the `≈`
  glyph appears only in the sentence defining it. Given its own chapter immediately before
  the first projection, it becomes enforceable.
- **Join, don't resolve** — from `slot-closure:67–74`. That an identity may be refused
  because writing it discharges a slot out of the language. Stated as a principle only;
  the eight-slot machinery stays in Part V.

This is the chapter that makes `x⁰ = 1` illegal *before* Ch 11 wants to write it, and it
needs none of the fork to say so.

### Part II — The Two Operations *(moved ahead of the circle)*

| # | Chapter | Was | Note |
|---|---|---|---|
| 7 | The Invertible Zero | 6 | — |
| 8 | Multiplication, Solved | 10 | **[RW]** |
| 9 | The Addition Problem | 11 | — |
| 10 | The Addition Dial | 12 | — |

The Part swap is the substance of the proposal, and D2 is the argument: grades are the
machinery, the circle is the result, and the book currently presents the result first.
`invertible-zero` already introduces grades informally (`:100`) and already fixes `0·ω = 1`
by definition, correctly signed with `:=` (`:61`). Ch 8 then owns the bookkeeping properly,
and Ch 11 can spend it.

**[RW] Ch 8** — two obligations.

1. `multiplication:66` derives `0·ω = (1·0¹)·(1·0⁻¹) = 1·0⁰ = 1`, which routes the book's
   founding identity through *both* barred evaluations. Under Ch 5 this derivation is
   illegal. It is also unnecessary: Ch 7 already fixed `0·ω = 1` by `:=`. The section
   becomes a *consistency check* — the grades add to zero — rather than a derivation.
2. `multiplication:83` ("A fair worry: two things at grade zero") compares `0·ω = 1`
   against `0^ω = −1` as a backward reference. After the swap that is a forward reference.
   Either move the section to Ch 11, or leave a pointer and pay it off there.

This is the one place the swap costs something, and it is worth naming plainly.

### Part III — The Circle

| # | Chapter | Was | Note |
|---|---|---|---|
| 11 | Powers of the Zero | 7 | **[RW]** |
| 12 | Where the Choices Show | † | **promoted** — fixes D3 |
| 13 | The Imaginary Unit | 8 | — |
| 14 | The Invariant Coordinate | *new* | — |
| 15 | The Exact Circle | 9 | — |

**[RW] Ch 11** — the ladder at `:44–45` is eight evaluations. Under Ch 5 it is rewritten as
links, and the honest scope of the forced claim (integers only) moves from footnote 2 into
the body, where it changes how the reader reads the table. The bijection argument at
`:74–78` needs the same treatment; note that `slot-closure`'s "What is owed" item 4 already
observes that argument dies once the carrier grows past four, so this rewrite is owed
regardless of the reorder.

**Ch 12, promoted from the dagger slot.** It stops being an appendix to Part III and
becomes the chapter that qualifies `0^ω = −1` on arrival. Its dependency on the dial is
satisfied — the dial is now Ch 10.

**Ch 14, new — *The Invariant Coordinate*.** Chebyshev is currently derived twice, both
times in working notes downstream of the chapters that need it: as the deck-invariant chart
in `reversible-one:166–187`, and as the trace coordinate of the norm-one subgroup in
`slot-closure:188`. It belongs upstream of the exact circle, in one place. The
`reversible-one` treatment is the harvestable one; the trace-coordinate reading is
fork-entangled and stays in Part V.

> **Title collision.** Current Ch 14 is also called *The Chart*. Rename one — either this
> new chapter to *The Invariant Coordinate* (as above), or the calculus chapter to
> *Three Hats*, after its own section heading.

### Part IV — Calculus Without Limits

| # | Chapter | Was |
|---|---|---|
| 16 | The Structural Differential | 13 |
| 17 | The Chart | 14 |
| 18 | Void Calculus in Practice | 15 |
| 19 | The Four Registers *(soon)* | 16 |

Internally unchanged; all dependencies (grades, `ω`, the dial) now resolve backwards.
**[RW]** on 16 and 18 is notational only — `structural-differential:106` and
`void-calculus-in-practice:65` both perform projections with `=` and want `≈` once Ch 5
makes the sign enforceable.

### Part V — What May Be Written

| # | Chapter | Was |
|---|---|---|
| 20 | The Reversible One | *unfiled note* |
| 21 | The Slot-Closure Formulation | *unfiled note* |

Both keep their **Wild** badges. The arc this creates is the point: Ch 5 states the
principle, the book runs on it for fifteen chapters, and Part V asks the systematic
question — which identities close the system without collapsing it. The fork stays
quarantined here, where a reader has the whole apparatus in hand and the "What is owed"
list reads as live work rather than as a hole in the foundations.

### Unchanged

Obstructions (22–24), The Frontier (25–26), Judging the Work (27–28), Appendix A.
Renumber only.

---

## Summary of the diff

- **Move:** 5 → 2 (cancellation ahead of the indictment)
- **Move:** Part "Two Operations" ahead of Part "The Circle" — 10,11,12 → 8,9,10
- **Promote:** † `where-the-choices-show` → Ch 12, adjacent to the claim it qualifies
- **Promote:** both working notes → Part V, Ch 20–21
- **New:** Ch 5 *What May Be Written* (three signs + join-don't-resolve)
- **New:** Ch 14 *The Invariant Coordinate* (Chebyshev, harvested from `reversible-one`)
- **Rewrite obligations:** Ch 3 (diagnosis), Ch 8 (§`0·ω` derivation, §grade-zero worry),
  Ch 11 (ladder as links, scope of the forced claim), Ch 16 & 18 (`≈`)

Chapter count goes 15 live → 17 live, plus 2 promoted notes.
