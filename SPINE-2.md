# Cold-open spine for `intro/`

Supersedes [SPINE.md](SPINE.md) as a proposal. Same diagnosis, opposite prescription.

SPINE.md found four real defects (D1–D4) and fixed them by moving machinery *earlier* —
textbook order, dependencies resolve backwards. This spine fixes the same four defects by
moving machinery *later and smaller*: results first, explanation on demand, formalization
at the end. The governing rules:

1. **The reader sees the mind-bending thing before any apparatus.**
2. **Machinery is introduced just-in-time, at the smallest size that makes the current
   equation inevitable** — two lines of grade arithmetic where the circle needs them, not
   a chapter of bookkeeping ahead of it.
3. **Honesty is inline, not deferred** — every headline claim carries its one-sentence
   caveat on arrival, with a pointer to the back matter that discharges it.
4. **The formal spine (three signs, slot-closure, what may be written, the fork) is the
   *finale*, not the toll booth.**
5. **No monologuing.** Chapters do not narrate their own significance. If a result is
   shocking, show it, then explain it; delete the sentences that announce it will be.

How this discharges SPINE.md's defects:

- **D1** (indictment precedes ontology): the indictment stops being a standalone act.
  ⊘ and the fusion diagnosis are introduced *at the moment the reader first asks* "but
  doesn't 0·a = 0?" — inside the tour, two paragraphs, not a chapter earlier.
- **D2** (circle built on later machinery): the circle chapter introduces grade addition
  inline (the two lines it actually uses). The full bookkeeping chapter comes later and
  opens with "you have already been using this."
- **D3** (qualification five chapters late): `0^ω = −1` carries its caveat in the same
  breath — "this is the fingerprint of a *chosen* multiplication; Chapter N shows the
  choice." No reader can stop reading misled.
- **D4** (no rule for legal identities): the rule is stated as a one-line house rule the
  first time an identity is *refused* in front of the reader, and gets its full chapter in
  the back matter, where it reads as the system's deepest property rather than a hazing.

---

## The new spine

### Part I — The Shock

| # | Chapter | Source | Note |
|---|---|---|---|
| 1 | **The Results** | *new* | the cold open |
| 2 | The Invertible Zero | was 6 | **[RW]** first tour stop |

**Ch 1, new — *The Results*.** Almost no prose. The book's five headline identities,
each stated bare, each with a one-line "not a typo" gloss and a pointer:

- `0 · ω = 1` — zero has a reciprocal, and it is not infinity. *(→ Ch 2)*
- `0^ω = −1` — minus one is a power of zero. *(→ Ch 4; a chosen multiplication's
  fingerprint, → Ch 12)*
- `i = 0^(ω/2)` (stated in whatever form Ch 8's derivation actually licenses) — the
  imaginary unit is a zero seen edge-on. *(→ Ch 5)*
- The exact circle: roots of unity by hand, phase without limits. *(→ Ch 6)*
- `d f = ` differentiation as multiplication by zero — a derivative with no limit
  anywhere in it. *(→ Ch 9)*

Plus one paragraph, at the end, stating the single price of admission: *one axiom is
deleted (absorption); nothing else changes; the rest of the book is what mathematics
looks like after the deletion.* That paragraph is the entire surviving role of the old
Part I. Types-as-operations, values, the method — their content redistributes into the
tour (see Dissolutions below).

**[RW] Ch 2** — `invertible-zero` becomes the first full stop, so it inherits the
just-in-time introductions it used to receive from upstream: ⊘ and the two residues
(two paragraphs, harvested from `cancellation`'s opening move), and the fusion diagnosis
(one paragraph, the corrected form of `totality-reversibility:62,70`) at the exact moment
the reader objects "but 0·a = 0." The chapter already introduces grades informally
(`:100`) and fixes `0·ω = 1` by `:=` (`:61`) — both stay.

### Part II — The Circle

| # | Chapter | Source | Note |
|---|---|---|---|
| 3 | Cancellation as an Operation | was 5 | **[RW]** trimmed to its result |
| 4 | Powers of the Zero | was 7 | **[RW]** |
| 5 | The Imaginary Unit | was 8 | — |
| 6 | The Exact Circle | was 9 | **[RW]** inline grade arithmetic |
| 7 | The Invariant Coordinate | *new* | Chebyshev, harvested from `reversible-one:166–187` |

**[RW] Ch 3** — with its opening move already spent in Ch 2, `cancellation` refocuses on
what only it delivers: zero and one as the two residues, and the first *refused* identity
— which is where the one-line house rule ("an identity may be refused because writing it
discharges a slot out of the language — the full rule is Ch 15") enters the book. That
single line is Part I's answer to SPINE.md's new Ch 5; the chapter it promised moves to
the back matter.

**[RW] Ch 4** — same obligations as SPINE.md's Ch 11 rewrite (ladder as links, integer
scope of the forced claim moved into the body), *plus* the D3 caveat inline: `0^ω = −1`
is introduced together with its status as a fingerprint of choice, one sentence, pointer
to Ch 12. The bijection argument at `:74–78` gets the rewrite `slot-closure`'s "What is
owed" item 4 already demands.

**[RW] Ch 6** — where `exact-circle` leans on grade bookkeeping, it states the two lines
it needs (grades add under multiplication; the check that they sum to zero) as local
facts, flagged "the full bookkeeping is Ch 8." This is the D2 fix without the Part swap.

**Ch 7, new** — as in SPINE.md's Ch 14, unchanged in content: the deck-invariant chart
from `reversible-one` is the harvestable treatment; the trace-coordinate reading stays
fork-entangled in the back matter. Title-collision note carries over: rename this one
*The Invariant Coordinate* or rename the calculus chapter *Three Hats*.

### Part III — The Two Operations

| # | Chapter | Source | Note |
|---|---|---|---|
| 8 | Multiplication, Solved | was 10 | **[RW]** |
| 9 | The Addition Problem | was 11 | — |
| 10 | The Addition Dial | was 12 | — |

**[RW] Ch 8** — opens by *collecting* the grade facts Chs 2–6 used informally ("you have
been doing this since Chapter 2; here is the whole system"). Inherits both obligations
from SPINE.md's Ch 8 rewrite: the `0·ω` derivation at `multiplication:66` routes through
two barred evaluations and is demoted to a consistency check (Ch 2 fixed the identity by
`:=`); the grade-zero worry at `:83` now points *backwards* to Ch 4 — under this order
that reference is naturally satisfied, which the Part swap in SPINE.md could not say.

### Part IV — Calculus Without Limits

| # | Chapter | Source |
|---|---|---|
| 11 | The Structural Differential | was 13 |
| 12 | Where the Choices Show | was † |
| 13 | The Chart | was 14 |
| 14 | Void Calculus in Practice | was 15 |

**Ch 12, promoted** — placed after the differential rather than adjacent to Ch 4, because
under rule 3 the *caveat* already arrived with the claim; this chapter is the *payoff* of
the caveat, and it lands best once the dial (Ch 10) and the differential (Ch 11) have both
shown multiplications being chosen. It discharges the Ch 4 pointer explicitly in its
opening.

**[RW] on 11 and 14** — notational only, carried over from SPINE.md: the projections at
`structural-differential:106` and `void-calculus-in-practice:65` want `≈` once the three
signs become enforceable — which is now a *back-matter* event, so these chapters use `≈`
with a footnote ("the sign discipline is Ch 15") rather than after a prior chapter's
definition.

### Part V — The Rules of the Game *(the formalization, moved to the end)*

| # | Chapter | Source | Note |
|---|---|---|---|
| 15 | What May Be Written | *new* | three signs + join-don't-resolve, in full |
| 16 | The Reversible One | unfiled note | keeps **Wild** badge |
| 17 | The Slot-Closure Formulation | unfiled note | keeps **Wild** badge |
| 18 | The Four Registers *(soon)* | was 16 | — |

**Ch 15** — SPINE.md's new Ch 5, relocated to where it reads as revelation instead of
regulation. Its obligations grow accordingly: it must *reconcile*, explicitly, every
informal introduction the tour made — the one-line house rule from Ch 3, the local grade
facts from Ch 6, the `≈`-with-footnote uses in Chs 11 and 14 — in a short "what you were
actually doing" section. This is the price of just-in-time introduction, and it is paid
here, once, in one place.

**Chs 16–17** — as in SPINE.md's Part V: the fork stays quarantined here, "What is owed"
reads as live work. The arc inverts, though: instead of *state the principle, run on it,
then systematize*, it is *run on results, then discover the principle they were secretly
obeying* — which is the arc the reader of a cold-open book is actually on.

### Unchanged

Obstructions, The Frontier, Judging the Work, Appendix A — renumbered after Ch 18.

### Dissolutions *(chapters that stop being chapters)*

- **Types as Operations** (was 1) → its thesis compresses into Ch 1's closing paragraph
  and Ch 2's framing; any remainder becomes the opening of Ch 15.
- **Totality and Reversibility** (was 2) → the demonstration `a = b ⟶ 0·a = 0·b ⟶ 0 = 0`
  survives as the objection-and-answer beat inside Ch 2; the corrected *fusion* diagnosis
  (SPINE.md's Ch 3 [RW]) is written there. The refusal-as-principle prose goes to Ch 15.
- **What a Value Is** (was 3) → the digit-shadow definition of number is introduced the
  first time a deferred value must be *read* — inside Ch 2 or Ch 4, wherever the first
  projection happens — as a section, not a chapter.
- **The Method** (was 4) → one paragraph in Ch 1 (the "delete one axiom" price of
  admission) plus a short front-note on how claims are badged (Proven/Plausible/Wild),
  which the index already carries.

These four dissolutions are the anti-monologue work: each currently spends a chapter
preparing the reader for significance the results can deliver themselves.

---

## Summary of the diff (against SPINE.md's proposal)

- **Reject:** the Part swap (machinery before circle). Replaced by inline grade facts in
  the circle chapters + a "collecting" opening in Multiplication.
- **Reject:** What May Be Written as front-matter Ch 5. Relocated to back matter (Ch 15)
  with a one-line house rule planted in Ch 3 and a reconciliation obligation added.
- **Keep:** both new chapters (The Invariant Coordinate, What May Be Written), both
  note promotions, the † promotion, and every [RW] obligation — several relocate.
- **New:** Ch 1 *The Results* (the cold open).
- **New:** four dissolutions — old Chs 1–4 stop being chapters; their content
  redistributes just-in-time.
- Chapter count: 15 live → 14 live + 2 promoted notes (net *smaller* front, bigger back).

## Risks, named plainly

1. **Double introduction drift.** Informal grade facts (Ch 6) vs. the full system (Ch 8),
   house rule (Ch 3) vs. full rule (Ch 15) — the informal version must never contradict
   the formal one. Mitigation: Ch 15's reconciliation section is a *checklist*, and each
   informal introduction carries a forward pointer so drift is greppable.
2. **The cold open can read as crankery** to a mathematician who opens the book at Ch 1
   and sees `0^ω = −1` with no apparatus. Mitigation: the "not a typo" glosses are doing
   real work — each must state, in one line, what kind of claim it is (definition,
   theorem, choice) and where it's cashed. The badge system helps here; use it in Ch 1.
3. **Dissolving The Method** removes the one place the epistemics were stated together.
   The front-note on badges must survive somewhere the reader can't miss — the index
   already has it; Ch 1 should echo it in one line.
