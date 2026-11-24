// Pandoc template: pandoc fills this in; style.typ does the design.
#import "typst/style.typ": *
#show: doc.with(
  title: [$title$],
$if(subtitle)$
  subtitle: [$subtitle$],
$endif$
$if(author)$
  author: [$for(author)$$author$$sep$, $endfor$],
$endif$
$if(date)$
  kicker: [$date$],
$endif$
  short-title: [$short-title$],
$if(doi)$
  doi: "$doi$",
$endif$
$if(compact)$
  compact: true,
$endif$
$if(pagetitle)$
  meta-title: "$pagetitle$",
$endif$
)

$body$
