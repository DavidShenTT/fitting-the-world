-- preprint.lua: pandoc Lua filter used only for the Typst build of preprint.md.
-- It changes layout, never wording or numbers:
--   * splits the title at <br> into title and subtitle;
--   * sets the abstract in a labelled block, and its bulleted list of costs as a 2 x 2 grid of cards;
--   * attaches each italic "Table n. ..." paragraph to the table below it as a caption;
--   * makes "Figure n." / "Table n." bold, and uses the SVG version of each figure;
--   * unwraps the HTML-only .sidefig divs (those figures are set at 62% width instead);
--   * boxes the speculative section, sets the references small, and starts the appendices on a new page.

local stringify = pandoc.utils.stringify

local function to_typst(inlines)
  local out = pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), 'typst')
  return (out:gsub('%s+$', ''))
end

local function raw(s) return pandoc.RawBlock('typst', s) end

local function bold_prefix(inlines, word)
  if #inlines >= 3 and inlines[1].t == 'Str' and inlines[1].text == word
      and inlines[2].t == 'Space' and inlines[3].t == 'Str' and inlines[3].text:match('^%u?%d+%.$') then
    local strong = pandoc.Strong({ inlines[1], inlines[2], inlines[3] })
    inlines:remove(1); inlines:remove(1); inlines:remove(1)
    inlines:insert(1, strong)
  end
  return inlines
end

local function capitalise_first(inlines)
  for _, el in ipairs(inlines) do
    if el.t == 'Str' then
      el.text = el.text:sub(1, 1):upper() .. el.text:sub(2)
      return inlines
    end
  end
  return inlines
end

local function blocks_to_typst(blocks)
  local out = pandoc.write(pandoc.Pandoc(blocks), 'typst', { wrap_text = 'none' })
  return (out:gsub('^%s+', ''):gsub('%s+$', ''))
end

-- Figures and tables are written as raw Typst here, so the result doesn't depend on
-- which pandoc version's Typst writer is installed.
local function fig_raw(fig, width)
  local src
  fig:walk({ Image = function(img) src = src or img.src end })
  src = src:gsub('%.png$', '.svg')
  local cap = ''
  if fig.caption and fig.caption.long and fig.caption.long[1] then
    local first = fig.caption.long[1]
    first.content = bold_prefix(first.content, 'Figure')
    cap = blocks_to_typst(fig.caption.long)
  end
  return raw('#figure(image("' .. src .. '", width: ' .. (width or '100%') .. '),\n  caption: [' .. cap .. '])')
end

local function table_raw(tbl, caption_inlines)
  local ncol = #tbl.colspecs
  local lens, counts = {}, {}
  for c = 1, ncol do lens[c] = 0; counts[c] = 0 end
  local rows = {}
  local function collect(row, is_head)
    local cells = {}
    for c, cell in ipairs(row.cells) do
      local txt = stringify(cell.contents)
      lens[c] = math.max(lens[c], #txt)
      table.insert(cells, '[' .. blocks_to_typst(cell.contents) .. ']')
    end
    table.insert(rows, { head = is_head, text = table.concat(cells, ', ') })
  end
  for _, row in ipairs(tbl.head.rows) do collect(row, true) end
  for _, body in ipairs(tbl.bodies) do for _, row in ipairs(body.body) do collect(row, false) end end
  local cols = {}
  for c = 1, ncol do
    if lens[c] <= 22 then cols[c] = 'auto' else cols[c] = tostring(math.min(lens[c], 90)) .. 'fr' end
  end
  local parts = { 'table(', '  columns: (' .. table.concat(cols, ', ') .. ',),' }
  for _, r in ipairs(rows) do
    if r.head then
      table.insert(parts, '  table.header(' .. r.text .. '),')
    else
      table.insert(parts, '  ' .. r.text .. ',')
    end
  end
  table.insert(parts, ')')
  local body = table.concat(parts, '\n')
  local cap = ''
  if caption_inlines then cap = ',\n  caption: [' .. to_typst(bold_prefix(caption_inlines, 'Table')) .. ']' end
  return raw('#figure(kind: table,\n' .. body .. cap .. ')')
end

function Meta(m)
  if m.title then
    local before, after, seen = pandoc.Inlines({}), pandoc.Inlines({}), false
    for _, el in ipairs(m.title) do
      if el.t == 'RawInline' and el.text:match('<br') then
        seen = true
      elseif seen then
        after:insert(el)
      else
        before:insert(el)
      end
    end
    if seen then
      local last = before[#before]
      if last and last.t == 'Str' then last.text = last.text:gsub(':$', '') end
      while after[1] and (after[1].t == 'Space' or after[1].t == 'SoftBreak') do after:remove(1) end
      m.title = before
      m.subtitle = after
    end
    if not m['short-title'] then m['short-title'] = m.title end
  end
  return m
end

local function is_cost_list(b)
  if b.t ~= 'BulletList' then return false end
  for _, item in ipairs(b.content) do
    local first = item[1]
    if not first or not first.content or not first.content[1] or first.content[1].t ~= 'Strong' then
      return false
    end
  end
  return true
end

local function cost_cards(b)
  local cards = {}
  for _, item in ipairs(b.content) do
    local inl = item[1].content:clone()
    local label = inl[1].content:clone()
    local lastl = label[#label]
    if lastl and lastl.t == 'Str' then lastl.text = lastl.text:gsub(':$', '') end
    inl:remove(1)
    while inl[1] and inl[1].t == 'Space' do inl:remove(1) end
    capitalise_first(inl)
    table.insert(cards, '([' .. to_typst(label) .. '], [' .. to_typst(inl) .. '])')
  end
  return raw('#costs(\n  ' .. table.concat(cards, ',\n  ') .. ',\n)')
end

function Pandoc(doc)
  local out = pandoc.List()
  local mode = nil
  local blocks = doc.blocks
  local skip_next = false

  local function close()
    if mode then out:insert(raw(']')); mode = nil end
  end

  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    local handled = false

    if b.t == 'Header' then
      close()
      local txt = stringify(b):lower()
      if txt == 'abstract' then
        out:insert(raw('#abstract['))
        mode = 'abstract'
        handled = true
      elseif txt:match('speculation') then
        out:insert(b); out:insert(raw('#speculation['))
        mode = 'spec'
        handled = true
      elseif txt == 'references' then
        out:insert(b); out:insert(raw('#references['))
        mode = 'refs'
        handled = true
      elseif txt:match('^appendix a') then
        out:insert(raw('#pagebreak(weak: true)'))
      end
    elseif mode == 'abstract' and is_cost_list(b) then
      out:insert(cost_cards(b))
      handled = true
    elseif b.t == 'Div' and b.classes:includes('sidefig') then
      for _, inner in ipairs(b.content) do
        if inner.t == 'Figure' then out:insert(fig_raw(inner, '62%')) else out:insert(inner) end
      end
      handled = true
    elseif b.t == 'Figure' then
      out:insert(fig_raw(b))
      handled = true
    elseif (b.t == 'Para' or b.t == 'Plain') and #b.content == 1 and b.content[1].t == 'Emph'
        and stringify(b.content[1]):match('^Table %u?%d+%.') and blocks[i + 1] and blocks[i + 1].t == 'Table' then
      out:insert(table_raw(blocks[i + 1], b.content[1].content:clone()))
      i = i + 1
      handled = true
    end

    if not handled and b.t == 'Table' then
      out:insert(table_raw(b, nil))
      handled = true
    end
    if not handled then out:insert(b) end
    i = i + 1
  end
  close()
  doc.blocks = out
  return doc
end

return {
  { Meta = Meta },
  { Pandoc = Pandoc },
}
