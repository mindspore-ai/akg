const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const env = { ...process.env, FORCE_COLOR: "1" };

// Generate each piece using oh-my-logo (same tool as original AKG CLI logo)
function generate(text) {
    return execSync(`npx oh-my-logo "${text}" mint --filled`, { env }).toString("utf-8");
}

// Parse ANSI art into rows of colored-character arrays
// Each element: { color: "...", char: "X" }
function parseAnsi(raw) {
    const lines = raw.replace(/\r/g, "").split("\n");
    const artLines = lines.filter(l => /[█╗╔═╚╝║]/.test(l));

    const rows = [];
    for (const line of artLines) {
        const chars = [];
        let currentColor = "";
        let i = 0;
        while (i < line.length) {
            if (line[i] === "\x1b" && line[i + 1] === "[") {
                let j = i + 2;
                while (j < line.length && line[j] !== "m" && !"hHJK".includes(line[j])) j++;
                const code = line.substring(i, j + 1);
                if (code.includes("38;2;")) currentColor = code;
                else if (code === "\x1b[39m" || code === "\x1b[0m") currentColor = "";
                i = j + 1;
            } else {
                chars.push({ color: currentColor, char: line[i] });
                i++;
            }
        }
        rows.push(chars);
    }
    return rows;
}

// Render rows back to ANSI string
function renderRows(rows) {
    const lines = [];
    for (const row of rows) {
        let line = "";
        let lastColor = "";
        for (const { color, char } of row) {
            if (color !== lastColor) {
                line += color;
                lastColor = color;
            }
            line += char;
        }
        line += "\x1b[39m";
        lines.push(line);
    }
    return lines.join("\r\n");
}

// Interpolate color between two RGB values
function lerpColor(r1, g1, b1, r2, g2, b2, t) {
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return `\x1b[38;2;${r};${g};${b}m`;
}

// oh-my-logo mint gradient endpoints
const MINT_START = [0, 210, 255];   // leftmost color
const MINT_END = [58, 123, 213];    // rightmost color

// ANSI Shadow font characters for [ and ]
const BRACKET_L = [
    "██████╗",
    "██╔═══╝",
    "██║    ",
    "██║    ",
    "██╚═══╗",
    "██████╝",
];
const BRACKET_R = [
    "██████╗",
    "╚═══██║",
    "    ██║",
    "    ██║",
    "╔═══██║",
    "██████╝",
];

console.log("Generating M...");
const rawM = generate("M");
const mRows = parseAnsi(rawM);
console.log(`  M: ${mRows.length} rows, ${mRows[0]?.length} cols`);

console.log("Generating S...");
const rawS = generate("S");
const sRows = parseAnsi(rawS);
console.log(`  S: ${sRows.length} rows, ${sRows[0]?.length} cols`);

console.log("Generating AKG...");
const rawCLI = generate("AKG");
const cliRows = parseAnsi(rawCLI);
console.log(`  AKG: ${cliRows.length} rows, ${cliRows[0]?.length} cols`);

const NUM_ROWS = 6;

// Build bracket characters with gradient color
function makeBracket(template, colStart, totalCols) {
    return template.map(rowStr => {
        const chars = [...rowStr];
        return chars.map((ch, ci) => {
            const t = (colStart + ci) / totalCols;
            const color = lerpColor(...MINT_START, ...MINT_END, t);
            return { color, char: ch };
        });
    });
}

// Compact S using half-block characters (▄▀█) — 3 rows tall, positioned at top
// Each row packs 2 visual rows via half-blocks, so 3 terminal rows = 6 visual rows
const SMALL_S = [
    "▄▀▀▀▀",  // top bar extended one cell right
    " ▀▀▀▄",  // middle → right side (unchanged)
    "▄▄▄▄▀",  // bottom bar extended one cell left
];

function makeSmallS(colStart, totalCols) {
    const result = [];
    for (let r = 0; r < NUM_ROWS; r++) {
        if (r < SMALL_S.length) {
            const rowChars = [...SMALL_S[r]];
            const colored = rowChars.map((ch, ci) => {
                const t = (colStart + ci) / totalCols;
                const color = lerpColor(...MINT_START, ...MINT_END, t);
                return { color, char: ch };
            });
            result.push(colored);
        } else {
            // Empty rows below the compact S
            const width = [...SMALL_S[0]].length;
            result.push(Array(width).fill(null).map(() => ({ color: "", char: " " })));
        }
    }
    return result;
}

// Make a gap column
function makeGap(width, rows) {
    return Array(rows).fill(null).map(() =>
        Array(width).fill(null).map(() => ({ color: "", char: " " }))
    );
}

// Calculate total visible columns for gradient
const bracketW = BRACKET_L[0].length;  // 7
const mW = mRows[0]?.length || 12;
const smallSW = [...SMALL_S[0]].length;  // width of compact S
const cliW = cliRows[0]?.length || 20;
const gap1 = 1; // between [ and M
const gap2 = 1; // between M and ]
const gap3 = 0; // between ] and S
const gap4 = 5; // between S and CLI
const totalCols = bracketW + gap1 + mW + gap2 + bracketW + gap3 + smallSW + gap4 + cliW;

console.log(`Total width: ${totalCols} cols`);

// Build each piece with proper gradient colors
const bracketLRows = makeBracket(BRACKET_L, 0, totalCols);
const gap1Rows = makeGap(gap1, NUM_ROWS);
// Recolor M 
const mColStart = bracketW + gap1;
const mRecolored = mRows.slice(0, NUM_ROWS).map((row, ri) =>
    row.map((c, ci) => {
        const t = (mColStart + ci) / totalCols;
        return { color: lerpColor(...MINT_START, ...MINT_END, t), char: c.char };
    })
);
const gap2Rows = makeGap(gap2, NUM_ROWS);
const bracketRStart = mColStart + mW + gap2;
const bracketRRows = makeBracket(BRACKET_R, bracketRStart, totalCols);
const gap3Rows = makeGap(gap3, NUM_ROWS);
const sColStart = bracketRStart + bracketW + gap3;
const superS = makeSmallS(sColStart, totalCols);
const gap4Rows = makeGap(gap4, NUM_ROWS);
// Recolor CLI
const cliColStart = sColStart + smallSW + gap4;
const cliRecolored = cliRows.slice(0, NUM_ROWS).map((row, ri) =>
    row.map((c, ci) => {
        const t = (cliColStart + ci) / totalCols;
        return { color: lerpColor(...MINT_START, ...MINT_END, t), char: c.char };
    })
);

// Combine all pieces row by row
const finalRows = [];
for (let r = 0; r < NUM_ROWS; r++) {
    const combined = [
        ...bracketLRows[r],
        ...gap1Rows[r],
        ...mRecolored[r],
        ...gap2Rows[r],
        ...bracketRRows[r],
        ...gap3Rows[r],
        ...superS[r],
        ...gap4Rows[r],
        ...cliRecolored[r],
    ];
    finalRows.push(combined);
}

const rendered = renderRows(finalRows);
const output = `\r\n\r\n${rendered}\r\n\r\n\r\n\x1b[0m\x1b[?25h\x1b[K`;

const base = path.join(__dirname, "..");
const outPath1 = path.join(base, "python/akg_agents/op/resources/logo.ans");
const outPath2 = path.join(base, "python/akg_agents/resources/logo.ans");

fs.writeFileSync(outPath1, output, "utf-8");
fs.writeFileSync(outPath2, output, "utf-8");
console.log(`Logo written to both locations (${output.length} bytes)`);

// Verify
const verify = fs.readFileSync(outPath1, "utf-8");
console.log(`Has ESC: ${verify.includes("\x1b")}`);
const plain = verify.replace(/\x1b\[[^m]*m/g, "").replace(/\r/g, "");
plain.split("\n").filter(l => l.trim()).forEach(l => console.log(l));
