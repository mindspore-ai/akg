const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const env = { ...process.env, FORCE_COLOR: "1" };

function generate(text) {
    return execSync(`npx oh-my-logo "${text}" mint --filled`, { env }).toString("utf-8");
}

function parseAnsi(raw) {
    const lines = raw.replace(/\r/g, "").split("\n");
    const artLines = lines.filter(l => /[█╗╔═╚╝║]/.test(l));
    const rows = [];
    for (const line of artLines) {
        const chars = [];
        let i = 0;
        while (i < line.length) {
            if (line[i] === "\x1b" && line[i + 1] === "[") {
                let j = i + 2;
                while (j < line.length && line[j] !== "m" && !"hHJK".includes(line[j])) j++;
                i = j + 1;
            } else {
                chars.push(line[i]);
                i++;
            }
        }
        rows.push(chars);
    }
    return rows;
}

function lerpColor(r1, g1, b1, r2, g2, b2, t) {
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return `\x1b[38;2;${r};${g};${b}m`;
}

const MINT_START = [0, 210, 255];
const MINT_END = [58, 123, 213];

console.log("Generating MIND...");
const a = parseAnsi(generate("MIND"));
console.log("Generating SPORE...");
const b = parseAnsi(generate("SPORE"));
console.log("Generating AKG...");
const c = parseAnsi(generate("AKG"));

const NUM_ROWS = Math.max(a.length, b.length, c.length);
const combined = [];
let totalCols = 0;

for (let r = 0; r < NUM_ROWS; r++) {
    const rowA = a[r] || [];
    const rowB = b[r] || [];
    const rowC = c[r] || [];
    // Space between 'MIND' and 'SPORE' ideally 1 col to match standard kerning. Space between 'SPORE' and 'AKG' 3 cols.
    const gap1 = [" "]; // 1 space
    const gap2 = [" ", " ", " ", " "]; // 4 spaces between words
    const joined = [...rowA, ...gap1, ...rowB, ...gap2, ...rowC];
    combined.push(joined);
    totalCols = Math.max(totalCols, joined.length);
}

console.log(`Total width: ${totalCols} cols`);

const finalLines = [];
for (const row of combined) {
    let line = "";
    let lastColor = "";
    for (let ci = 0; ci < row.length; ci++) {
        const char = row[ci];
        if (char !== " ") {
            const t = ci / totalCols;
            const color = lerpColor(...MINT_START, ...MINT_END, t);
            if (color !== lastColor) {
                line += color;
                lastColor = color;
            }
        }
        line += char;
    }
    line += "\x1b[39m"; // reset at end of line
    finalLines.push(line);
}

const rendered = finalLines.join("\r\n");
const output = `\r\n\r\n${rendered}\r\n\r\n\x1b[0m\x1b[?25h\x1b[K`;

const base = path.join(__dirname, "..");
const outPath = path.join(base, "python/akg_agents/op/resources/mindspore_akg_logo.ans");

fs.mkdirSync(path.dirname(outPath), { recursive: true });

fs.writeFileSync(outPath, output, "utf-8");
console.log(`Logo written to ${outPath} (${output.length} bytes)`);
