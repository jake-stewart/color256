vim9script

const BLACK = 0
const RED = 1
const GREEN = 2
const BLUE = 4
const BRIGHT = 8
const YELLOW = RED + GREEN
const MAGENTA = RED + BLUE
const CYAN = GREEN + BLUE
const WHITE = RED + GREEN + BLUE
const GREY = BRIGHT + BLACK

# r, g, b: integer 0-5 inclusive
def Rgb(r: number, g: number, b: number): number
    return 16 + r * 36 + g * 6 + b
enddef

# shade: integer 0-23 inclusive
def Grey(shade: number): number
    return 232 + shade
enddef

def CreateTheme(theme: dict<any>)
    g:colors_name = "custom"
    set background=dark
    set notermguicolors

    var highlights: dict<dict<any>> = {}
    const CTERM_ATTRS = ['bold', 'italic', 'strikethrough', 'underline', 'undercurl', 'reverse']

    def AddHighlight(k: string, v: any): dict<any>
        if v == null
            throw 'highlight not defined in config: ' .. k
        endif

        if type(v) == type('')
            var opts = has_key(highlights, v) ? highlights[v] : AddHighlight(v, theme[v])
            highlights[k] = opts
            execute 'hi! link ' .. k .. ' ' .. v
            return opts
        else
            var opts: dict<any>
            
            if empty(v)
                opts = {
                    ctermfg: null,
                    ctermbg: null,
                    cterm_attrs: []
                }
                execute 'hi ' .. k .. ' NONE'
                highlights[k] = opts
                return opts
            endif
            
            opts = {
                ctermfg: null,
                ctermbg: null,
                cterm_attrs: []
            }
            
            if has_key(v, 'extends')
                var extends_val = v.extends
                var extends_list: list<string>
                
                if type(extends_val) == type('')
                    extends_list = [extends_val]
                else
                    extends_list = extends_val
                endif
                
                for base_name in extends_list
                    var base_opts = has_key(highlights, base_name) ? highlights[base_name] : AddHighlight(base_name, theme[base_name])
                    opts.ctermfg = opts.ctermfg ?? base_opts.ctermfg
                    opts.ctermbg = opts.ctermbg ?? base_opts.ctermbg
                    opts.cterm_attrs = opts.cterm_attrs + base_opts.cterm_attrs
                endfor
            endif
            
            if has_key(v, 'fg')
                opts.ctermfg = v.fg
            endif
            if has_key(v, 'bg')
                opts.ctermbg = v.bg
            endif
            
            for attr in CTERM_ATTRS
                if has_key(v, attr)
                    add(opts.cterm_attrs, attr)
                endif
            endfor
            
            var cmd = 'hi ' .. k
            if opts.ctermfg != null
                cmd ..= ' ctermfg=' .. opts.ctermfg
            endif
            if opts.ctermbg != null
                cmd ..= ' ctermbg=' .. opts.ctermbg
            endif
            if !empty(opts.cterm_attrs)
                cmd ..= ' cterm=' .. join(opts.cterm_attrs, ',')
            endif
            execute cmd
            
            highlights[k] = opts
            
            return opts
        endif
    enddef

    for [k, v] in items(theme)
        if !has_key(highlights, k)
            AddHighlight(k, v)
        endif
    endfor
enddef

CreateTheme({
    Constant: {fg: CYAN},
    Literal: {fg: YELLOW},
    Number: 'Literal',
    Character: 'Literal',
    Boolean: 'Literal',
    String: {fg: GREEN},

    Identifier: {},
    Variable: 'Identifier',
    Function: {fg: BLUE},

    Statement: {fg: MAGENTA},
    Conditional: 'Statement',
    Repeat: 'Statement',
    Label: 'Statement',
    Operator: 'Statement',
    Keyword: 'Statement',
    Exception: {fg: CYAN},

    PreProc: {},
    Define: {fg: MAGENTA},
    Macro: {fg: MAGENTA},
    PreCondit: {fg: BRIGHT + YELLOW},
    Include: {fg: BLUE},

    Type: {fg: BRIGHT + YELLOW},
    StorageClass: 'Type',
    Structure: 'Type',
    Typedef: {},

    Error: {fg: RED},
    Warning: {fg: YELLOW},
    Info: {fg: CYAN},
    Hint: {fg: CYAN},
    Success: {fg: GREEN},

    ErrorMsg: 'Error',
    WarningMsg: 'Warning',
    InfoMsg: 'Info',
    HintMsg: 'Hint',
    SuccessMsg: 'Success',

    SpellBad: {extends: 'Error', undercurl: true},
    SpellCap: {extends: 'Warning', undercurl: true},

    Special: {fg: MAGENTA},
    SpecialKey: 'Special',

    Title: {fg: GREEN},
    Todo: 'Special',
    Question: 'Special',
    Comment: {fg: GREY, italic: true},
    SpecialComment: 'GreyFg2',
    SpecialChar: 'GreyFg3',

    Directory: {fg: BLUE},

    GreyBg1: {bg: Grey(1)},
    GreyBg2: {bg: Grey(2)},
    GreyBg3: {bg: Grey(3)},
    GreyBg4: {bg: Grey(5)},

    GreyFg1: {fg: Grey(4)},
    GreyFg2: {fg: Grey(6)},
    GreyFg3: {fg: Grey(12)},

    Cursor: {reverse: true},
    Visual: 'GreyBg3',

    CursorLine: 'GreyBg1',
    CursorLineNr: 'CursorLine',
    CursorLineSign: 'CursorLine',
    CursorLineFold: 'CursorLine',
    CursorColumn: 'CursorLine',

    WildMenu: {bg: BLUE},
    MatchParen: {extends: 'GreyBg2', fg: MAGENTA},
    QuickFixLine: {bold: true},

    Search: 'GreyBg2',
    CurSearch: {extends: 'Search', fg: BRIGHT + YELLOW},
    IncSearch: 'CurSearch',

    Pmenu: 'GreyBg2',
    PmenuSbar: 'GreyBg2',
    PmenuThumb: 'GreyBg3',
    PmenuSel: 'GreyBg4',

    TabLine: 'GreyBg2',
    TabLineSel: 'GreyBg2',
    TabLineFill: 'GreyBg2',

    StatusLine: 'GreyBg2',
    StatusLineNC: 'GreyFg2',

    NormalFloat: 'GreyBg2',
    FloatBorder: 'GreyFg1',

    NonText: 'GreyFg1',
    VertSplit: 'GreyFg1',
    Border: 'GreyFg1',
    LineNr: 'GreyFg1',
    WinSeparator: 'GreyFg1',

    Folded: 'GreyFg2',

    DiffDelete: {bg: Rgb(1, 0, 0)},
    DiffAdd: {bg: Rgb(0, 1, 0)},
    DiffChange: {},
})
