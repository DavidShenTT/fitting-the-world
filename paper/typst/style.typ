// style.typ: house style for "Fitting the World to the Model".
// Aim: a clean, editorial document with sans-serif structure and a serif reading text.
// It shouldn't imitate LaTeX. Colours match the figures (blue wild, orange wear, green shaping by eye).
// Fonts: Source Serif 4 (text) and Inter (structure), both SIL Open Font License; compile with
// --font-path pointing at a folder holding them. Fallbacks are listed after each.

#let ink    = rgb("#1c2230")
#let muted  = rgb("#5d6676")
#let faint  = rgb("#8a93a3")
#let hair   = rgb("#dfe3ea")
#let tint   = rgb("#f4f6f9")
#let accent = rgb("#4b5160")   // dark slate grey: labels, numbers, bullets
#let rule-c = rgb("#c3c8d0")   // pale grey: heading rules, box borders
#let spec-ink  = rgb("#8a6a12")
#let spec-tint = rgb("#fbf7ec")
#let spec-bar  = rgb("#d4a62a")

#let sans  = ("Inter", "Segoe UI", "Helvetica Neue", "Arial")
#let serif = ("Source Serif 4", "Georgia", "Cambria")
#let mono  = ("JetBrains Mono", "Cascadia Mono", "DejaVu Sans Mono")

// ---------- page furniture -------------------------------------------------
#let doc(
  title: [],
  subtitle: [],
  author: [],
  kicker: [Preprint · November 2026],
  short-title: [],
  meta-title: "Fitting the World to the Model",
  meta-author: (),
  compact: false,   // tighter page for the one-page summary
  body,
) = {
  set document(title: meta-title, author: meta-author)
  set page(
    paper: "a4",
    margin: if compact { (x: 22mm, top: 18mm, bottom: 16mm) } else { (left: 26mm, right: 26mm, top: 24mm, bottom: 24mm) },
    header: context {
      if counter(page).get().first() > 1 {
        set text(font: sans, size: 7.5pt, fill: faint, tracking: 0.02em)
        grid(columns: (1fr, auto), short-title, counter(page).display())
        v(-5pt)
        line(length: 100%, stroke: 0.4pt + hair)
      }
    },
    footer: none,
  )

  // reading text
  set text(font: serif, size: if compact { 9.6pt } else { 10.25pt }, fill: ink, lang: "en", hyphenate: true,
           number-type: "lining")
  set par(justify: true, leading: if compact { 0.6em } else { 0.68em }, spacing: if compact { 0.8em } else { 0.95em }, linebreaks: "optimized")

  // run-in labels ("Items.", "Upkeep:") and emphasis
  show strong: set text(font: sans, weight: 500, size: 0.92em)
  // (no show rule for emph: emph already toggles italic, and a `set text(style: "italic")` rule on it
  //  makes the surrounding text italic first, so emph toggles it back to upright)

  // headings: sans, a short accent rule above each section
  show heading.where(level: 1): it => {
    v(1.6em, weak: true)
    block(below: 0.9em, {
      line(length: 18pt, stroke: 1.6pt + rule-c)
      v(5pt)
      set text(font: sans, size: 13pt, weight: 600, fill: ink, tracking: -0.005em)
      it.body
    })
  }
  show heading.where(level: 2): it => {
    v(1.15em, weak: true)
    block(below: 0.6em,
      text(font: sans, size: 10.5pt, weight: 600, fill: ink, it.body))
  }

  // lists
  set list(indent: 0.2em, body-indent: 0.6em, spacing: 0.55em,
           marker: (text(font: sans, fill: faint, size: 0.85em, baseline: -0.05em)[●], text(fill: faint)[–]))   // ● is in Inter, not in Source Serif 4
  set enum(indent: 0.2em, body-indent: 0.6em, spacing: 0.55em,
           numbering: n => text(font: sans, size: 0.88em, weight: 600, fill: accent)[#n.])

  // code, links
  show raw: set text(font: mono, size: 0.84em, fill: muted)
  show link: set text(fill: ink)

  // figures: float to page top or bottom, captions small, sans, left-aligned
  set figure(placement: auto, gap: 0.7em)
  show figure: set block(breakable: false, above: 1.5em, below: 1.5em)
  show figure.caption: it => {
    set text(font: sans, size: 8pt, fill: muted)
    set par(justify: false, leading: 0.5em)
    show strong: set text(fill: ink, weight: 600, size: 1em)
    align(left, it.body)   // caption text already carries "Figure n."
  }
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): set figure(placement: none)
  show figure.where(kind: table): set block(breakable: false)

  // tables: no grid, a firm rule under the header, hairlines between rows
  set table(
    inset: (x: 6pt, y: 4.2pt),
    align: left + horizon,
    fill: (x, y) => if y == 0 { tint },
    stroke: (x, y) => (
      top: if y == 0 { 0.8pt + ink } else { none },
      bottom: if y == 0 { 0.6pt + ink } else { 0.3pt + hair },
    ),
  )
  show table: set text(font: sans, size: 8pt)
  show table: set par(justify: false, leading: 0.5em)
  show table.cell.where(y: 0): set text(weight: 600)

  // ---------- title block ----------
  block(width: 100%, below: 1.4em, align(center, {
    set par(justify: false)
    text(font: sans, size: 7.5pt, weight: 600, fill: accent, tracking: 0.12em, upper(kicker))
    v(10pt)
    text(font: ("Inter Display", "Inter") + sans, size: if compact { 17pt } else { 24pt }, weight: 700, fill: ink,
         tracking: -0.015em, title)
    if subtitle != [] {
      v(4pt)
      text(font: sans, size: 14pt, weight: 400, fill: muted, tracking: -0.005em, subtitle)
    }
    if author != [] {
      v(14pt)
      text(font: sans, size: 9.5pt, weight: 500, fill: ink, author)
    }
    v(12pt)
    line(length: 100%, stroke: 0.5pt + hair)
  }))

  body
}

// ---------- components used in the text ----------------------------------------

// The abstract: slightly larger, unjustified, with a small label.
#let abstract(body) = block(width: 100%, below: 1.2em, {
  text(font: sans, size: 7.5pt, weight: 600, fill: faint, tracking: 0.12em, "ABSTRACT")
  v(4pt)
  set text(size: 10.75pt)
  set par(justify: false, leading: 0.7em)
  body
})

// The abstract's list of costs, as boxes stacked one below the other.
// Each card: (label, body). The text is exactly the abstract's list item, so no number is duplicated.
#let costs(..cards) = block(above: 1.1em, below: 1.1em, stack(spacing: 6pt,
  ..cards.pos().map(c => block(
    width: 100%, fill: tint, inset: (x: 11pt, top: 8pt, bottom: 9pt), stroke: 0.6pt + rule-c, radius: 3pt,
    {
      set par(justify: false, leading: 0.55em)
      text(font: sans, size: 7pt, weight: 600, fill: accent, tracking: 0.1em, upper(c.at(0)))
      v(4pt)
      text(font: sans, size: 8.6pt, fill: ink, c.at(1))
    },
  )),
))

// A visibly marked speculative section (§6).
#let speculation(body) = block(
  width: 100%, inset: (x: 12pt, y: 11pt), radius: 3pt, fill: spec-tint,
  stroke: (left: 2.4pt + spec-bar),
  {
    text(font: sans, size: 7pt, weight: 600, fill: spec-ink, tracking: 0.12em, "SPECULATION")
    v(4pt)
    body
  },
)

// References and appendices: smaller text, hanging indent for references.
#let references(body) = {
  set text(size: 8.75pt)
  set par(justify: false, hanging-indent: 1.4em, leading: 0.55em, spacing: 0.6em)
  show list: it => for item in it.children {
    block(above: 0.6em, below: 0pt, par(hanging-indent: 1.4em, item.body))
  }
  body
}

// pandoc's Typst writer emits this for a Markdown horizontal rule.
#let horizontalrule = block(above: 1.2em, below: 1.2em, line(length: 100%, stroke: 0.5pt + hair))
