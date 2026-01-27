let s:BLACK = 0
let s:RED = 1
let s:GREEN = 2
let s:BLUE = 4
let s:BRIGHT = 8
let s:YELLOW = s:RED + s:GREEN
let s:MAGENTA = s:RED + s:BLUE
let s:CYAN = s:GREEN + s:BLUE
let s:WHITE = s:RED + s:GREEN + s:BLUE
let s:GREY = s:BRIGHT + s:BLACK

" r, g, b: integer 0-5 inclusive
function! s:Rgb(r, g, b)
    return 16 + a:r * 36 + a:g * 6 + a:b
endfunction

" shade: integer 0-23 inclusive
function! s:Grey(shade)
    return 232 + a:shade
endfunction

function! s:CreateTheme(theme)
    let g:colors_name = "custom"
    set background=dark
    set notermguicolors

    let l:highlights = {}
    let l:CTERM_ATTRS = ['bold', 'italic', 'strikethrough', 'underline', 'undercurl', 'reverse']

    function! s:AddHighlight(k, v, theme, highlights) closure
        if a:v == v:null
            throw 'highlight not defined in config: ' . a:k
        endif

        if type(a:v) == type('')
            let l:opts = has_key(a:highlights, a:v) ? a:highlights[a:v] : s:AddHighlight(a:v, a:theme[a:v], a:theme, a:highlights)
            let a:highlights[a:k] = l:opts
            execute 'hi! link ' . a:k . ' ' . a:v
            return l:opts
        else
            let l:opts = {}
            
            if empty(a:v)
                let l:opts = {
                    \ 'ctermfg': v:null,
                    \ 'ctermbg': v:null,
                    \ 'cterm_attrs': []
                    \ }
                execute 'hi ' . a:k . ' NONE'
                let a:highlights[a:k] = l:opts
                return l:opts
            endif
            
            let l:opts = {
                \ 'ctermfg': v:null,
                \ 'ctermbg': v:null,
                \ 'cterm_attrs': []
                \ }
            
            if has_key(a:v, 'extends')
                let l:extends_val = a:v.extends
                let l:extends_list = []
                
                if type(l:extends_val) == type('')
                    let l:extends_list = [l:extends_val]
                else
                    let l:extends_list = l:extends_val
                endif
                
                for l:base_name in l:extends_list
                    let l:base_opts = has_key(a:highlights, l:base_name) ? a:highlights[l:base_name] : s:AddHighlight(l:base_name, a:theme[l:base_name], a:theme, a:highlights)
                    let l:opts.ctermfg = l:opts.ctermfg != v:null ? l:opts.ctermfg : l:base_opts.ctermfg
                    let l:opts.ctermbg = l:opts.ctermbg != v:null ? l:opts.ctermbg : l:base_opts.ctermbg
                    let l:opts.cterm_attrs = l:opts.cterm_attrs + l:base_opts.cterm_attrs
                endfor
            endif
            
            if has_key(a:v, 'fg')
                let l:opts.ctermfg = a:v.fg
            endif
            if has_key(a:v, 'bg')
                let l:opts.ctermbg = a:v.bg
            endif
            
            for l:attr in l:CTERM_ATTRS
                if has_key(a:v, l:attr)
                    call add(l:opts.cterm_attrs, l:attr)
                endif
            endfor
            
            let l:cmd = 'hi ' . a:k
            if l:opts.ctermfg != v:null
                let l:cmd .= ' ctermfg=' . l:opts.ctermfg
            endif
            if l:opts.ctermbg != v:null
                let l:cmd .= ' ctermbg=' . l:opts.ctermbg
            endif
            if !empty(l:opts.cterm_attrs)
                let l:cmd .= ' cterm=' . join(l:opts.cterm_attrs, ',')
            endif
            execute l:cmd
            
            let a:highlights[a:k] = l:opts
            
            return l:opts
        endif
    endfunction

    for [l:k, l:v] in items(a:theme)
        if !has_key(l:highlights, l:k)
            call s:AddHighlight(l:k, l:v, a:theme, l:highlights)
        endif
    endfor
endfunction

call s:CreateTheme({
    \ 'Constant': {'fg': s:CYAN},
    \ 'Literal': {'fg': s:YELLOW},
    \ 'Number': 'Literal',
    \ 'Character': 'Literal',
    \ 'Boolean': 'Literal',
    \ 'String': {'fg': s:GREEN},
    \
    \ 'Identifier': {},
    \ 'Variable': 'Identifier',
    \ 'Function': {'fg': s:BLUE},
    \
    \ 'Statement': {'fg': s:MAGENTA},
    \ 'Conditional': 'Statement',
    \ 'Repeat': 'Statement',
    \ 'Label': 'Statement',
    \ 'Operator': 'Statement',
    \ 'Keyword': 'Statement',
    \ 'Exception': {'fg': s:CYAN},
    \
    \ 'PreProc': {},
    \ 'Define': {'fg': s:MAGENTA},
    \ 'Macro': {'fg': s:MAGENTA},
    \ 'PreCondit': {'fg': s:BRIGHT + s:YELLOW},
    \ 'Include': {'fg': s:BLUE},
    \
    \ 'Type': {'fg': s:BRIGHT + s:YELLOW},
    \ 'StorageClass': 'Type',
    \ 'Structure': 'Type',
    \ 'Typedef': {},
    \
    \ 'Error': {'fg': s:RED},
    \ 'Warning': {'fg': s:YELLOW},
    \ 'Info': {'fg': s:CYAN},
    \ 'Hint': {'fg': s:CYAN},
    \ 'Success': {'fg': s:GREEN},
    \
    \ 'ErrorMsg': 'Error',
    \ 'WarningMsg': 'Warning',
    \ 'InfoMsg': 'Info',
    \ 'HintMsg': 'Hint',
    \ 'SuccessMsg': 'Success',
    \
    \ 'SpellBad': {'extends': 'Error', 'undercurl': v:true},
    \ 'SpellCap': {'extends': 'Warning', 'undercurl': v:true},
    \
    \ 'Special': {'fg': s:MAGENTA},
    \ 'SpecialKey': 'Special',
    \
    \ 'Title': {'fg': s:GREEN},
    \ 'Todo': 'Special',
    \ 'Question': 'Special',
    \ 'Comment': {'fg': s:GREY, 'italic': v:true},
    \ 'SpecialComment': 'GreyFg2',
    \ 'SpecialChar': 'GreyFg3',
    \
    \ 'Directory': {'fg': s:BLUE},
    \
    \ 'GreyBg1': {'bg': s:Grey(1)},
    \ 'GreyBg2': {'bg': s:Grey(2)},
    \ 'GreyBg3': {'bg': s:Grey(3)},
    \ 'GreyBg4': {'bg': s:Grey(5)},
    \
    \ 'GreyFg1': {'fg': s:Grey(4)},
    \ 'GreyFg2': {'fg': s:Grey(6)},
    \ 'GreyFg3': {'fg': s:Grey(12)},
    \
    \ 'Cursor': {'reverse': v:true},
    \ 'Visual': 'GreyBg3',
    \
    \ 'CursorLine': 'GreyBg1',
    \ 'CursorLineNr': 'CursorLine',
    \ 'CursorLineSign': 'CursorLine',
    \ 'CursorLineFold': 'CursorLine',
    \ 'CursorColumn': 'CursorLine',
    \
    \ 'WildMenu': {'bg': s:BLUE},
    \ 'MatchParen': {'extends': 'GreyBg2', 'fg': s:MAGENTA},
    \ 'QuickFixLine': {'bold': v:true},
    \
    \ 'Search': 'GreyBg2',
    \ 'CurSearch': {'extends': 'Search', 'fg': s:BRIGHT + s:YELLOW},
    \ 'IncSearch': 'CurSearch',
    \
    \ 'Pmenu': 'GreyBg2',
    \ 'PmenuSbar': 'GreyBg2',
    \ 'PmenuThumb': 'GreyBg3',
    \ 'PmenuSel': 'GreyBg4',
    \
    \ 'TabLine': 'GreyBg2',
    \ 'TabLineSel': 'GreyBg2',
    \ 'TabLineFill': 'GreyBg2',
    \
    \ 'StatusLine': 'GreyBg2',
    \ 'StatusLineNC': 'GreyFg2',
    \
    \ 'NormalFloat': 'GreyBg2',
    \ 'FloatBorder': 'GreyFg1',
    \
    \ 'NonText': 'GreyFg1',
    \ 'VertSplit': 'GreyFg1',
    \ 'Border': 'GreyFg1',
    \ 'LineNr': 'GreyFg1',
    \ 'WinSeparator': 'GreyFg1',
    \
    \ 'Folded': 'GreyFg2',
    \
    \ 'DiffDelete': {'bg': s:Rgb(1, 0, 0)},
    \ 'DiffAdd': {'bg': s:Rgb(0, 1, 0)},
    \ 'DiffChange': {},
    \ })
