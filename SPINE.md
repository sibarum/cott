# Spine for `intro/`

**Status: the single authoritative plan.** Supersedes the machinery-earlier proposal
(rejected), the cold-open proposal (adopted, then outgrown), and the `§7` patch that was
appended to it after the working notes landed. Self-contained.

What changed: the cold-open spine was a *reordering* of the existing book, and the notes
arrived afterwards as five entries each labelled "a change to the plan, not just the
content." A plan that names its organizing object in its second-to-last section wants
rewriting, not patching. Nothing was published in the meantime — `the-results` is an orphan
file, no `[RW]` obligation was executed, and the note thread links only to itself — so the
cost of redoing the plan is one file and this document.

The genre also changes. This is not a tour with a mystery at the front. It is a manual:
**how to make an algebra total and reversible, and how not to.** Step by step. No
monologuing.

---

## 1 · The frame

Doing algebra is three acts, and only the middle one can be exact.

**Invention.** You write an expression down. You choose what to include and what to leave
out — which operations must stay total, what the objects are, where the identity sits, what
you are willing to remember. This act is *contextual and imprecise*: nothing forces it, and
what you leave out is gone before any manipulation begins.

**Manipulation.** You rewrite the symbols. If the algebra is **total** (no operation is
undefined) and **reversible** (no rewrite loses information), then nothing is lost in
transit — every step is undoable, and the expression at the end carries exactly what the
expression at the start carried. *This is the only place the corpus claims exactness.*

**Measurement.** Eventually you must read a value out. That act is *necessarily*
information-destructive: a reading is a single number, the expression was more than a single
number, and the difference is discarded. Contextual and imprecise, like invention — and for
the same reason. It is a choice about what to keep.

Two consequences, both load-bearing.

1. **The claim is narrow and checkable.** COTT does not make mathematics exact. It makes the
   *middle* exact, and it makes the two ends *visible* — the price is paid where you can see
   it charged, instead of being smeared across every step. A skeptic can now be told
   precisely what is and is not being claimed, which is a better defence than a badge.
2. **Invention and measurement are one knob seen from two ends.** Both choose what to keep.
   The book has a chapter that demonstrates this rather than asserting it — the dial (Ch 15),
   where the choice of addition is simultaneously a choice of what the front of the
   expression commits to and a choice of what the back of it reads.

## 2 · What the frame buys

**It collapses D1–D4 into one defect.** The four sequencing failures diagnosed below are
four symptoms of the same thing: the book never tells the reader there are three stages, so
it cannot flag a move that belongs to one stage being performed in another.

**It turns the error ledger into content.** Every entry in `papers/PLAN.md §5` and every
retraction in the note thread is a stage violation, and they sort cleanly:

| Error | Stage violation |
|---|---|
| `0^(0ω)` gives both `0` and `1`, so `0 = 1` (`reversible-one:52–55`) | the fatal line is "split the exponent, **then** `x⁰ = 1`" — a *measurement* performed mid-manipulation |
| `x⁰ = 1`, `x¹ = x` barred as slot-discharges (`slot-closure`) | the same violation, stated as a general rule |
| `0^a + 0^b = 0^{ab}` (ledger #3) | an illegal *manipulation* — no stage confusion, just wrong |
| `0² = −1−i`, `0³ = −2i` (ledger #3) | consequences of the above; dropped |
| `0^ω = −1` "forced by closure" (ledger #2) | an *invention* choice sold as a manipulation result |
| `0/0 = 1`, "ω is not infinity" asserted as derived (ledger #11) | *invention* (stipulation) sold as manipulation |
| `0·a = 0` prosecuted as the founding crime (D1) | a *measurement* residue treated as the object |
| Chebyshev "coincidence = evidence" (ledger #4) | measurement-stage agreement read as manipulation-stage proof |
| the `Proven` badge at `where-the-choices-show:93` | rests on a normalization that is a projection — i.e. a measurement |

This is the "how not to" half, and it is not an appendix of shame: it is the argument for the
frame. Each failure is the frame's own prediction, made after the fact but reproducibly.

**It houses the two homeless findings.** Both were open debts in the previous plan:

- **The correspondence principle** — every upstairs distinction must vanish downstairs into
  exactly the classical value. Previously "epistemics, belongs somewhere in the back." It is
  the *measurement stage's law*, full stop, and it states the falsifiability guard in the
  only place it can be tested.
- **"The type is whatever must exist to keep the operations working"** — previously a rule
  stranded in a dissolved chapter, cited by name in `the-doubling`. It is the *invention
  stage's law*. Ch 3 states it; Ch 20 cites Ch 3.

Manipulation's own law is the one `weaves` proves: **information-conservative modelling is
precisely equivalence of categories.** So each stage gets exactly one law, and the book can
end by cashing all three.

## 3 · The diagnosis

Four defects in the current order, restated as stage confusions.

**D1 — the indictment precedes the ontology.** Ch 2 (`totality-reversibility:62,70`)
prosecutes absorption using `0·a = 0` with `0` as annihilator; Ch 5 (`cancellation:56–88`)
then says `⊘` is the annihilator and `0` is one of its two residues. *Stage reading:* the
founding crime is stated in measurement-stage vocabulary three chapters before the reader is
given the invention-stage object it is really about.

**D2 — the circle is built on machinery introduced after it.** `powers-of-zero:50` leans on
bookkeeping that arrives in Ch 10. *Stage reading:* grades are the manipulation stage's
coordinate; the chapter uses them without saying that is what they are.

**D3 — the qualification arrives five chapters after the claim.** Ch 7 sells `0^ω = −1` as
"the choice that closes" (`:74–78`); `where-the-choices-show:116` later makes it the
fingerprint of a *chosen* multiplication. *Stage reading:* one claim, two stages, stated five
chapters apart as if they competed. The position (`ω` carries `0` to the half-turn) is
derived at the manipulation stage; the value (`−1`) is a projection at the measurement stage.
Both true, in one breath, no hedging.

**D4 — no chapter says what a legal identity is.** `x⁰ = 1` and `x¹ = x` are used as forced
steps in Chs 7 and 10; `slot-closure` later bars both. *Stage reading:* this is D1–D3's
common cause, and it is the whole reason the frame has to be stated up front.

## 4 · The prescription

1. **Results before apparatus.** Ch 1 is the payoff, stated bare.
2. **The frame immediately after.** Ch 2 is the three stages and the price of admission. It
   is short, and it is the only abstraction the reader meets before work starts.
3. **Every chapter declares its stage** in the chapter-meta line. This is the mechanism that
   makes D1–D4 unrepeatable, and it costs one word per chapter.
4. **Machinery just-in-time, at the smallest size that makes the current equation
   inevitable** — two lines of grade arithmetic where the circle needs them, not a chapter of
   bookkeeping ahead of it.
5. **How-not-to is inline and specific.** Where a stage violation was actually committed in
   this corpus, the chapter shows the wrong line, not a warning in the abstract.
6. **Exactness is claimed only for the middle**, every time it is claimed.
7. **No monologuing.** Chapters do not narrate their own significance. Delete the sentences
   announcing that a result will be shocking; show it, then explain it.

## 5 · The spine

`[RW]` = rewrite required, itemized in §7. Stage labels are the chapter-meta declarations
from rule 3.

| # | Chapter | Source | Note |
|---|---|---|---|
| 1 | **The Results** | `the-results` | the cold open; exists, currently orphaned |
| 2 | **The Three Stages** | *new* | §1, plus the price of admission |

**Ch 1** keeps its five bare identities and one-line "not a typo" glosses. **Ch 2** replaces
the old Ch 1–4 block entirely: the three stages, the single deletion (absorption), and the
sentence that scopes every later claim — *the middle is exact; the ends are choices, and this
book tells you which is which every time.*

### Part I — Invention: choosing what to write down

| # | Chapter | Source | Note |
|---|---|---|---|
| 3 | What an Expression Commits You To | `types-as-operations` + `values` | **[RW]** merge; states the invention law |
| 4 | Cancellation as an Operation | `cancellation` | **[RW]** `⊘`, the two residues, the three kinds of equals |
| 5 | The Invertible Zero | `invertible-zero` | **[RW]** the `:=`, honestly labelled; grades introduced as bookkeeping we *choose* |

Part I closes with its how-not-to: stipulations sold as derivations (ledger #2, #11).

### Part II — Manipulation: rewriting without loss

| # | Chapter | Source | Note |
|---|---|---|---|
| 6 | Powers of the Zero | `powers-of-zero` | **[RW]** ladder as links, not evaluations |
| 7 | The Imaginary Unit | `imaginary-unit` | — |
| 8 | The Exact Circle | `exact-circle` | **[RW]** inline grade facts; "exact" scoped |
| 9 | Multiplication, Solved | `multiplication` | **[RW]** collecting opening |
| 10 | What May Be Written | `slot-closure` | promoted; slots not terms, pivots as the only rule |
| 11 | How This Goes Wrong | `reversible-one` | promoted; the `0^(0ω)` collapse and its escape |
| 12 | The Invariant Coordinate | *new* | Chebyshev, from `reversible-one:166–187` |
| 13 | The Addition Problem | `addition-problem` | the wound — manipulation's honest limit |

**Ch 10 and 11 are the change of shape.** In the previous plan `slot-closure` and
`reversible-one` were back-matter Wild notes at positions 17–18. Under the frame they are
Part II's formal centre and Part II's worked failure: the rule for legal rewriting, and the
one place in the corpus where breaking it produced `0 = 1`. They keep their **Wild** badges.
Ch 11 shows the reader the fatal line and names the stage it came from; it reaches
`one-curve` for the check table and the `x = 0` analysis.

**Ch 13 ends the part unresolved on purpose.** The grades cannot close addition. That is not
a defect to apologize for — it is the boundary of the exact middle, and Part III opens by
crossing it.

### Part III — Measurement: reading a value, and paying for it

| # | Chapter | Source | Note |
|---|---|---|---|
| 14 | Evaluation Is Projection | `anchor-shadow` | promoted; the two floors, and which claims live on each |
| 15 | The Addition Dial | `addition-dial` | heals Ch 13's wound by a *readout* choice |
| 16 | Where the Choices Show | `where-the-choices-show` | rigidity: the options are fewer than they look |
| 17 | The Chart | `the-chart` | — |

**Ch 14 first**, because it supplies the part's one idea: discharging a slot out of the
language *is* stepping down a floor, so evaluation, projection, and the `≈` sign are one
thing under three names. Every measurement-stage claim in the book routes through it.

**Ch 15 is where §1's second consequence is earned.** The dial heals the wound by choosing
how to read, and the same knob turns out to fix what the front of the expression committed
to. That is the invention/measurement identity, demonstrated rather than asserted, and it
discharges Ch 4's forward pointer.

**Ch 16 discharges D3's pointer explicitly in its opening.**

### Part IV — One problem, all three stages

| # | Chapter | Source | Note |
|---|---|---|---|
| 18 | The Structural Differential | `structural-differential` | **[RW]** notational |
| 19 | Void Calculus in Practice | `void-calculus-in-practice` | **[RW]** notational |
| — | The Four Registers *(soon)* | was 16 | keep as forward-marked |

You invent the chart, manipulate to the differential, measure to get the slope. The part is
the frame run once, end to end, on a problem the reader can check by hand. It is also the
answer to "so what" — a derivative with no limit in it, produced by the discipline the first
three parts built.

### Part V — What it turned out to be

| # | Chapter | Source | Note |
|---|---|---|---|
| 20 | The Scalar That Doubles | `the-doubling` | invention forced: cites Ch 3's law by name |
| 21 | Completion Is a Weave | `weaves` | the three laws, cashed |

**Ch 21 is the finale, and it is a theorem, not a retraction list.** `weaves` proves that
information-conservative modelling is equivalence of categories — manipulation's law, in the
formal form Ch 2 promised — and carries the correspondence principle, measurement's law.
Ch 3 stated invention's. The book ends by paying its opening promissory note.

The **Clifford torus** is the picture Part V draws, not its organizing object: `ℍ = ℂ ⊕ ℂj`,
so a unit quaternion lives on `S³`, and inside `S³` the locus `{|z| = |w| = 1/√2}` is
coordinatized by `(arg z, arg w)` — exactly the (base, exponent) phase torus the figures
already show. It places two loose findings: `i = log₀ ∘ r` (two reflections composing to a
quarter-turn — a second, operations-first route to `i`, cross-referenced from Ch 7) and
`ω^ω = 1/(−1)` (the one intrinsically-ordered-pair cell, hence a torus intersection point,
since `(0,ω)` and `(ω,0)` are distinct — Ch 20 or 21).

### Unchanged

Obstructions, The Frontier, Judging the Work, Appendix A — renumbered after Ch 21.

## 6 · Dissolutions

- **Types as Operations** (was 1) → the ladder and lens framing compress into Ch 1's close
  and Ch 2's framing; the rule *"the type is whatever must exist to keep the operations
  working"* survives as **Ch 3's invention law**, stated in a quotable sentence because Ch 20
  cites it by name.
- **Totality and Reversibility** (was 2) → the definitions move to Ch 2 (they are the
  manipulation stage's definition). The demonstration `a = b ⟶ 0·a = 0·b ⟶ 0 = 0` survives
  as the objection-and-answer beat inside Ch 5, with the corrected *fusion* diagnosis. The
  refusal-as-principle prose goes to Ch 10.
- **What a Value Is** (was 3) → the digit-shadow definition merges into Ch 3, which is where
  the question "what did you commit to by writing that down" is already being asked.
- **The Method** (was 4) → the inverse principle and the axiom-cost argument become Ch 2's
  price-of-admission section; the badging front-note moves to the index, strengthened by the
  correspondence principle.

Four chapters out, two in (Ch 2, Ch 12), five notes promoted. Front matter shrinks from four
chapters of preparation to one chapter of frame.

## 7 · The `[RW]` obligations

| Ch | File | Obligation |
|---|---|---|
| 1 | `the-results` | Link it — reachable from nothing today. Badging by kind is **already done** (`:42` definition, `:53` choice, `:63` consequence), so §9.2 is half-satisfied; add the stage half. **But `:53–54` state the superseded reading** — "A *choice*, not a discovery … the fingerprint of a chosen multiplication" — which is precisely what D3 corrects, sitting in the most prominent gloss in the book. Rewrite as *position derived, value projected*. Retire the part name "The Shock" (`:23`); that Part no longer exists. |
| 2 | *new* | Write it. §1 verbatim in substance, plus the totality/reversibility definitions inherited from the dissolved Ch 2, plus the axiom-cost argument from the dissolved Ch 4. Short — it is a frame, not a treatise. |
| 3 | *new (merge)* | Merge `types-as-operations` + `values`. Must end with the invention law in one quotable sentence. Plant the one-line house rule pointing to Ch 10. |
| 4 | `cancellation` | Keep `⊘`, the two residues, the three kinds of equals. Add the corrected fusion diagnosis (from `totality-reversibility:62,70`) at the reader's objection. Forward-point to Ch 15 for the front/back-knob identity. |
| 5 | `invertible-zero` | Keep `:100` grades and the `:=` at `:61`; label the `:=` explicitly as an invention-stage stipulation. Absorb the `a = b ⟶ 0·a = 0·b` beat. |
| 6 | `powers-of-zero` | Ladder at `:44–45` rewritten as links, not evaluations. Integer scope moves from fn2 into the body. Bijection argument at `:74–78` rewritten per `slot-closure` item 4 (it dies once the carrier exceeds four). `0^ω` caveat inline and two-part: **position derived, value projected** — no hedging, pointer to Ch 16 for the coordinate-dependence. |
| 8 | `exact-circle` | State inline the two grade facts it uses, flagged "full bookkeeping is Ch 9." "Exact" means roots of unity — torsion, cyclotomic — not `S¹`; and the circle closes *downstairs*, so `0^(2ω) ≈ 1`. |
| 9 | `multiplication` | Collecting opening ("you have been doing this since Chapter 5"). The `0·ω` derivation at `:66` routes through two barred evaluations — demote to a consistency check, since Ch 5 fixes the identity by `:=`. The grade-zero worry at `:83` points back to Ch 6 and is satisfied there. |
| 10 | `slot-closure` | Promote out of working-note voice into chapter voice; keep the Wild badge. Absorb the refusal-as-principle prose from the dissolved Ch 2. Its "fork underneath" stays an open question, stated as one. |
| 11 | `reversible-one` | Promote. Reframe the lead: the contradiction is not evidence against an invertible zero, it is a stage violation with a named line. Keep the escape (`x⁰ = 1^x`) and the reciprocal-of-one consequence flagged provisional. Reach `one-curve` for the check table. |
| 12 | *new* | Write from `reversible-one:166–187`. The old title collision (*The Invariant Coordinate* vs. *Three Hats*) is gone — Ch 17 keeps *The Chart*, so no rename is needed. |
| 14 | `anchor-shadow` | Promote; lead with the two-floor criterion, then the identity *evaluation = projection*. This chapter is why the `≈`/`=`/`:=` discipline exists, so it must state the discipline, not merely use it. |
| 16 | `where-the-choices-show` | The **Proven** badge at `:93` rests on a projection — downgrade it and record in `PROVEN-AUDIT.md`. Open by discharging D3's pointer. The "fingerprint of a chosen multiplication" thesis now reads as the measurement-stage half of a two-stage claim, not a correction of Ch 6. |
| 18, 19 | `structural-differential:106`, `void-calculus-in-practice:65` | Projections written with `=` want `≈`, footnoted to Ch 14. |
| all | — | Stage declaration in the chapter-meta line (rule 3). The slot exists — `chapter-meta` is already `Chapter N &middot; <span class="part">…</span>` with an optional third `<span class="conf c-wild">` — so this is a third span plus one rule in `lib/papers.css` beside `.chapter-meta .part` at `:292`. Part names all change (Invention / Manipulation / Measurement / …), so the stage word and the part name should not be redundant: keep the part name descriptive and let the stage be the bare noun. |

**Retired obligation.** The previous plan owed a Ch 15 *"what you were actually doing"*
reconciliation section — the price of just-in-time introduction. Under the frame there is no
such chapter and no such debt: the reconciliation is distributed, because every informal
introduction now carries a stage label, and the stage label *is* the pointer. Ch 14 collects
the sign discipline; Ch 9 collects the grade bookkeeping. Nothing is left over.

## 8 · The tension worth recording

Partitioning by stage fights dependency order in exactly one place. The dial (Ch 15) is
measurement-stage material by function, but it needs the wound (Ch 13), which needs the
grades (Ch 9). Strict stage-partitioning of the *whole* book would therefore require moving
the grades forward — which is the machinery-earlier proposal, already rejected.

**Decision: stages order the Parts, dependencies order the chapters inside them.** The seam
lands between Ch 13 and Ch 14, which is the best available place for it — Part II ends on the
limit of exact manipulation and Part III opens by crossing that limit. The seam is content,
not compromise.

The residual risk is that a reader expects Part I to contain every invention-stage chapter
and finds Chs 15 and 16 doing invention-stage work late. Ch 2 pre-empts this in one sentence:
*the stages are not three chunks of the book, they are three things you are always doing; the
parts are named for which one is under the microscope.*

## 9 · Risks

1. **Three stages is still an abstraction on page two.** Mitigation: Ch 1 comes first, and Ch
   2 is short, concrete, and immediately cashed — its examples are the identities the reader
   just saw. If Ch 2 runs longer than the shortest existing chapter, it is too long.
2. **The cold open can read as crankery.** Mitigation unchanged, and strengthened: each "not
   a typo" gloss states what *kind* of claim it is and which stage it lives in. "Minus one is
   a power of zero — the position is derived, the value is a projection" is a sentence a
   skeptic can engage with.
3. **Stage labels could become decoration.** If every chapter is labelled and no chapter's
   content changes, the frame has bought nothing. Mitigation: the label is only earned where
   §7 shows a rewrite. A chapter whose obligation is *only* the label is a chapter that did
   not need the frame — find those and drop the label there.
4. **Double-introduction drift.** Informal grade facts (Ch 8) vs. the full system (Ch 9).
   Mitigation: forward pointers so drift is greppable, and Ch 9's collecting opening is the
   checklist.
5. **Part V is Wild and ends the book.** Less acute than before — two chapters instead of
   five, and the last one is a theorem. Ch 21 still needs a short closing section stating
   plainly what survived: the results with computations behind them.

## 10 · Order of work

The mechanical layer goes **last**. Every live chapter carries a hand-wired
`<nav class="chapter-nav">` with prev/next titles (e.g. `exact-circle:159–163`) and a
`chapter-meta` line with a hard-coded number and part name. Renumbering 15 live chapters into
21 slots touches all of it, and doing that before the chapter set is final means doing it
twice.

1. **Ch 2, *The Three Stages*.** The one genuinely new load-bearing chapter, and every other
   chapter's rewrite is stated relative to its vocabulary. Nothing else can be labelled until
   the three words are fixed in prose.
2. **Ch 1's correction** (§7) — small, and it removes the most prominent superseded claim in
   the corpus.
3. **The stage-bearing rewrites**, in spine order: Chs 3–6, then 8–11, then 14 and 16. Each
   is self-contained once Ch 2 exists.
4. **Promotions** — Chs 10, 11, 14, 20, 21 out of working-note voice, badges kept.
5. **The new chapters** — Ch 3's merge, Ch 12.
6. **The mechanical layer** — TOC, nav, numbers, part names, the `.stage` rule.
7. **The title** (§11), once the shape has stopped moving.

## 11 · Debts

- **`the-results` is unlinked and `intro/index.html` is untouched.** Both are the first
  concrete edits, and neither is a rewrite.
- **`PROVEN-AUDIT.md`** — downgrade the `0^(2ω) = 1` badge (§7, Ch 16). Also still open from
  that file: `papers/4` uses the retired `c-solid` badge at `:117,187`.
- **The book's own title and subtitle.** "How to make an algebra total and reversible — and
  how not to" is the frame's natural title, and the current front matter does not say it.
  Left unresolved on purpose; it is the last thing to fix, not the first.
- **The two completions: bridged, not joined.** Whether the covering completion (Ch 21) and
  the doubling completion (Ch 20) are one structure. The Clifford torus supplies the venue,
  but it is a *slice* of `S³` — the locus of equal magnitudes — so it is the meeting place
  rather than the whole of either completion. The step that would close it: show that the
  `ℤ/2` relating the two torus decompositions **is** conjugation by `j`, the element the
  doubling adjoins. If so, the covering picture's twist and the doubling's generator are one
  element. Unchecked, and worth checking first.
- **`one-curve` is superseded** and is not a chapter. It stays in the tree for its check
  table and `x = 0` analysis, reachable from Ch 11.
