"""
make_fonts.py -- convert the Inter typeface (CFF-flavoured .otf) to TrueType
outlines under the family name "Inter Paper", so that matplotlib embeds it in
PDF figures as a clean Type 42 font.

    python paper/fonts/make_fonts.py [DIRECTORY_WITH_Inter-*.otf]

Inter is free under the SIL Open Font License: https://rsms.me/inter/
Default source directory: /usr/share/fonts/opentype/inter (Debian/Ubuntu
package fonts-inter). The converted .ttf files are not committed to the repo.
"""
import sys
from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

HERE = Path(__file__).resolve().parent
WEIGHTS = ["Regular", "Italic", "Medium", "SemiBold", "Bold"]


def convert(src, dst, max_err=1.0):
    font = TTFont(src)
    order = font.getGlyphOrder()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    gs = font.getGlyphSet()
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyf[name] = pen.glyph()
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for a in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs",
              "maxInstructionDefs", "maxStackElements", "maxSizeOfInstructions",
              "maxComponentElements", "maxComponentDepth"):
        setattr(maxp, a, 0)
    maxp.maxZones = 1
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    for t in ("CFF ", "VORG"):
        if t in font:
            del font[t]
    font["head"].indexToLocFormat = 0
    font.sfntVersion = "\000\001\000\000"
    # rename the family so it never clashes with an installed Inter
    n = font["name"]
    n.names = [r for r in n.names if r.nameID not in (21, 22, 25)]
    for r in n.names:
        if r.nameID in (1, 16):
            r.string = "Inter Paper"
    font.save(dst)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/usr/share/fonts/opentype/inter")
    for w in WEIGHTS:
        convert(src / f"Inter-{w}.otf", HERE / f"InterPaper-{w}.ttf")
        print("wrote", HERE / f"InterPaper-{w}.ttf")
