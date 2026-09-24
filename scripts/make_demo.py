# make_demo.py: render the README's demo GIF, a short terminal session showing
# oracle answering a question about its own codebase with cited sources. It is a
# faithful reproduction of real output (the answer and the source lines are what
# `oracle ask` actually prints), drawn rather than screen-recorded so it needs no
# recording tools and regenerates deterministically.
#
# Run:  python scripts/make_demo.py   (needs Pillow)  ->  assets/demo.gif

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "demo.gif")

W, H = 760, 430
PAD = 26
LINE = 23
BG = (15, 20, 18)
BAR = (26, 34, 30)
MINT = (127, 227, 196)
PALE = (222, 240, 228)
INK = (231, 242, 236)
DIM = (138, 163, 154)
WARN = (240, 190, 120)
DOTS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]

MONO = "/System/Library/Fonts/Menlo.ttc"
font = ImageFont.truetype(MONO, 15)
bold = ImageFont.truetype(MONO, 15, index=1)

PROMPT = "› "
CMD = 'oracle ask "how does the index reuse unchanged files?"'
# The answer and sources are real output from `oracle ask` on this repo.
ANSWER = [
    "The persistent index re-embeds only the files whose size",
    "or modification time has changed, so keeping it current",
    "after edits is fast.",
]
SOURCES = [
    "Sources:",
    "[1] scripts/retrieve.py:201-240",
    "[2] tests/index_test.py:1-40",
]


def base():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=BAR)
    for i, c in enumerate(DOTS):
        cx = 20 + i * 22
        d.ellipse([cx - 6, 11, cx + 6, 23], fill=c)
    d.text((W // 2 - 26, 9), "oracle", font=font, fill=DIM)
    return img, d


def frame(cmd_shown, answer_lines, source_lines, cursor):
    img, d = base()
    y = 34 + PAD
    d.text((PAD, y), PROMPT, font=bold, fill=MINT)
    px = PAD + int(d.textlength(PROMPT, font=bold))
    d.text((px, y), cmd_shown, font=font, fill=INK)
    if cursor:
        cx = px + int(d.textlength(cmd_shown, font=font))
        d.rectangle([cx + 1, y + 2, cx + 9, y + 18], fill=MINT)
    y += LINE * 2
    for ln in answer_lines:
        d.text((PAD, y), ln, font=font, fill=PALE)
        y += LINE
    if answer_lines and source_lines:
        y += LINE - 6
    for ln in source_lines:
        col = DIM
        d.text((PAD, y), ln, font=font, fill=col)
        y += LINE
    return img


def main():
    frames = []
    holds = []  # ms per frame

    def add(im, ms):
        frames.append(im)
        holds.append(ms)

    # 1. blink the cursor on the empty prompt
    for _ in range(2):
        add(frame("", [], [], True), 350)
        add(frame("", [], [], False), 350)
    # 2. type the command
    for i in range(1, len(CMD) + 1):
        add(frame(CMD[:i], [], [], True), 32)
    add(frame(CMD, [], [], True), 500)
    # 3. reveal the answer, a few words at a time
    words = " ".join(ANSWER).split(" ")
    shown = 0
    while shown < len(words):
        shown = min(len(words), shown + 3)
        partial = " ".join(words[:shown])
        # re-wrap to the original line breaks by width
        lines, cur = [], ""
        for w in partial.split(" "):
            trial = (cur + " " + w).strip()
            if len(trial) > 54:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        add(frame(CMD, lines, [], False), 60)
    add(frame(CMD, ANSWER, [], False), 350)
    # 4. reveal the sources
    for i in range(1, len(SOURCES) + 1):
        add(frame(CMD, ANSWER, SOURCES[:i], False), 220)
    # 5. hold, then loop
    add(frame(CMD, ANSWER, SOURCES, False), 2600)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:], duration=holds, loop=0, optimize=True
    )
    print("wrote %s (%d frames, %.0f KB)" % (OUT, len(frames), os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    main()
