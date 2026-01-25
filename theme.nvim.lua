--- @diagnostic disable unused locals
local BLACK = 0
local RED = 1
local GREEN = 2
local BLUE = 4
local BRIGHT = 8
local YELLOW = RED + GREEN
local MAGENTA = RED + BLUE
local CYAN = GREEN + BLUE
local WHITE = RED + GREEN + BLUE
local GREY = BRIGHT + BLACK

--- @param r integer 0-5 inclusive
--- @param g integer 0-5 inclusive
--- @param b integer 0-5 inclusive
local function rgb(r, g, b)
    return 16 + r * 36 + g * 6 + b
end

--- @param shade integer 0-23 inclusive
local function grey(shade)
    return 232 + shade
end
--- @diagnostic enable

---@class (exact) HighlightOpts
---@field fg? integer
---@field bg? integer
---@field bold? true
---@field italic? true
---@field underline? true
---@field undercurl? true
---@field strikethrough? true
---@field reverse? true
---@field [integer] string Groups to extend from

---@param t table<string, string | HighlightOpts>
local function create_theme(t)
    vim.g.colors_name = "custom"
    vim.o.background = "dark"
    vim.o.termguicolors = false

    local hl = vim.api.nvim_set_hl
    local highlights = {}
    local function add_highlight(k, v)
        if v == nil then
            error("highlight not defined in config: " .. k)
        end
        if type(v) == "string" then
            local opts = highlights[v] or add_highlight(v, t[v])
            highlights[k] = opts
            hl(0, k, { link = v })
            return opts
        else
            local opts
            if #v ~= 1 or v.fg or v.bg
                or v.bold or v.italic or v.underline
                or v.undercurl or v.strikethrough or v.reverse
            then
                opts = {
                    ctermfg = v.fg,
                    ctermbg = v.bg,
                    cterm = {
                        bold = v.bold,
                        italic = v.italic,
                        strikethrough = v.strikethrough,
                        underline = v.underline,
                        undercurl = v.undercurl,
                        reverse = v.reverse,
                    }
                }
                for _, name in ipairs(v) do
                    local h = highlights[name] or add_highlight(name, t[name])
                    opts = {
                        ctermfg = opts.ctermfg or h.ctermfg,
                        ctermbg = opts.ctermbg or h.ctermbg,
                        cterm = {
                            bold = opts.cterm.bold or h.cterm.bold,
                            italic = opts.cterm.italic or h.cterm.italic,
                            strikethrough = opts.cterm.strikethrough or h.cterm.strikethrough,
                            underline = opts.cterm.underline or h.cterm.underline,
                            undercurl = opts.cterm.undercurl or h.cterm.undercurl,
                            reverse = opts.cterm.reverse or h.cterm.reverse,
                        }
                    }
                end
                hl(0, k, opts)
            else
                opts = highlights[v[1]] or add_highlight(v[1], t[v[1]])
                hl(0, k, { link = v[1] })
            end
            highlights[k] = opts
            return opts
        end
    end
    for k, v in pairs(t) do
        if not highlights[k] then
            add_highlight(k, v)
        end
    end
end

create_theme({
    Constant = { fg = CYAN },
    Literal = { fg = YELLOW },
    Number = "Literal",
    Character = "Literal",
    Boolean = "Literal",
    String = { fg = GREEN },

    Identifier = {},
    Variable = "Identifier",
    Function = { fg = BLUE },

    Statement = { fg = MAGENTA },
    Conditional = "Statement",
    Repeat = "Statement",
    Label = "Statement",
    Operator = "Statement",
    Keyword = "Statement",
    Exception = { fg = CYAN },

    PreProc = {},
    Define = { fg = MAGENTA },
    Macro = { fg = MAGENTA },
    PreCondit = { fg = BRIGHT + YELLOW },
    Include = { fg = BLUE },

    Type = { fg = BRIGHT + YELLOW  },
    StorageClass = "Type",
    Structure = "Type",
    Typedef = {},

    Error = { fg = RED },
    Warning = { fg = YELLOW },
    Info = { fg = CYAN },
    Hint = { fg = CYAN },
    Success = { fg = GREEN },

    ErrorMsg = "Error",
    WarningMsg = "Warning",
    InfoMsg = "Info",
    HintMsg = "Hint",
    SuccessMsg = "Success",

    DiagnosticError = "Error",
    DiagnosticWarn = "Warning",
    DiagnosticInfo = "Info",
    DiagnosticHint = "Hint",
    DiagnosticOk = "Success",

    DiagnosticUnderlineError = "DiagnosticError",
    DiagnosticUnderlineWarn = "DiagnosticWarn",
    DiagnosticUnderlineHint = "DiagnosticHint",
    DiagnosticUnderlineInfo = "DiagnosticInfo",
    DiagnosticUnderlineOk = "DiagnosticOk",

    DiagnosticSignError = "DiagnosticError",
    DiagnosticSignWarn = "DiagnosticWarn",
    DiagnosticSignHint = "DiagnosticHint",
    DiagnosticSignInfo = "DiagnosticInfo",
    DiagnosticSignOk = "DiagnosticOk",

    DiagnosticUnnecessary = { fg = GREY },

    SpellBad = { "Error", undercurl = true },
    SpellCap = { "Warning", undercurl = true },

    Special = { fg = MAGENTA },
    SpecialKey = "Special",

    Title = { fg = GREEN },
    Todo = "Special",
    Question = "Special",
    Comment = { fg = GREY, italic = true },
    SpecialComment = "GreyFg2",
    SpecialChar = "GreyFg3",

    Directory = { fg = BLUE },

    GreyBg1 = { bg = grey(1) },
    GreyBg2 = { bg = grey(2) },
    GreyBg3 = { bg = grey(3) },
    GreyBg4 = { bg = grey(5) },

    GreyFg1 = { fg = grey(4) },
    GreyFg2 = { fg = grey(6) },
    GreyFg3 = { fg = grey(12) },

    Cursor = { reverse = true },
    Visual = "GreyBg3",

    CursorLine = "GreyBg1",
    CursorLineNr = "CursorLine",
    CursorLineSign = "CursorLine",
    CursorLineFold = "CursorLine",
    CursorColumn = "CursorLine",

    WildMenu = { bg = BLUE },
    MatchParen = { "GreyBg2", fg = MAGENTA },
    QuickFixLine = { bold = true },

    Search = "GreyBg2",
    CurSearch = { "Search", fg = BRIGHT + YELLOW  },
    IncSearch = "CurSearch",

    Pmenu = "GreyBg2",
    PmenuSbar = "GreyBg2",
    PmenuThumb = "GreyBg3",
    PmenuSel = "GreyBg4",

    TabLine = "GreyBg2",
    TabLineSel = "GreyBg2",
    TabLineFill = "GreyBg2",

    StatusLine = "GreyBg2",
    StatusLineNC = "GreyFg2",

    NormalFloat = "GreyBg2",
    FloatBorder = "GreyFg1",

    NonText = "GreyFg1",
    VertSplit = "GreyFg1",
    Border = "GreyFg1",
    LineNr = "GreyFg1",
    WinSeparator = "GreyFg1",

    Folded = "GreyFg2",

    DiffDelete = { fg = rgb(1, 0, 0) },
    DiffAdd = { bg = rgb(0, 1, 0) },
    DiffChange = {},

    ["@constructor"] = "Function",
    ["@function.builtin"] = "Function",
    ["@variable.builtin"] = "Variable",
    ["@constant.builtin"] = "Constant",
    ["@type.builtin"] = "Type",

    ["@constructor.lua"] = {}
})
