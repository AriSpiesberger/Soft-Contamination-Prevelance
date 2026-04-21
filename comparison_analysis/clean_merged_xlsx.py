"""Clean LaTeX markers and unreadable characters in the merged xlsx in-place."""
import pandas as pd
import re
import unicodedata
import os

PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "human_annotation", "sd_samples_merged.xlsx",
)

# Common LaTeX -> unicode/ascii replacements for codeforces text.
LATEX_REPL = [
    (r"\\le\b", "<="), (r"\\leq\b", "<="),
    (r"\\ge\b", ">="), (r"\\geq\b", ">="),
    (r"\\ne\b", "!="), (r"\\neq\b", "!="),
    (r"\\cdot\b", "*"), (r"\\times\b", "*"),
    (r"\\ldots\b", "..."), (r"\\dots\b", "..."),
    (r"\\to\b", "->"), (r"\\rightarrow\b", "->"),
    (r"\\leftarrow\b", "<-"),
    (r"\\infty\b", "inf"),
    (r"\\bmod\b", "mod"), (r"\\mod\b", "mod"),
    (r"\\sum\b", "sum"), (r"\\prod\b", "prod"),
    (r"\\gcd\b", "gcd"), (r"\\lcm\b", "lcm"),
    (r"\\max\b", "max"), (r"\\min\b", "min"),
    (r"\\log\b", "log"),
    (r"\\oplus\b", "XOR"),
    (r"\\land\b", "AND"), (r"\\lor\b", "OR"),
    (r"\\mathit\b", ""), (r"\\mathbb\b", ""), (r"\\mathcal\b", ""),
    (r"\\operatorname\b", ""),
    (r"\\!+\!", "+"), (r"\\!-\!", "-"),
    (r"\\,", " "), (r"\\;", " "), (r"\\:", " "), (r"\\ ", " "),
    (r"\\quad\b", "  "), (r"\\qquad\b", "    "),
]


def clean(text):
    if not isinstance(text, str) or not text:
        return text
    s = text
    # Strip $$$ and $ math delimiters (keep inner content).
    s = s.replace("$$$", "")
    s = re.sub(r"(?<!\\)\$", "", s)
    # LaTeX token replacements.
    for pat, rep in LATEX_REPL:
        s = re.sub(pat, rep, s)
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    # \sqrt{a} -> sqrt(a)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    # ^{...} -> ^(...), _{...} -> _(...)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    # \text{...} -> ...
    s = re.sub(r"\\text(?:it|bf|tt|rm)?\s*\{([^{}]*)\}", r"\1", s)
    # Strip leftover backslash commands: \foo -> foo
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    # Collapse {x} -> x for remaining single-level braces.
    s = re.sub(r"\{([^{}]*)\}", r"\1", s)
    # Normalize unicode (NFKC) to collapse weird forms.
    s = unicodedata.normalize("NFKC", s)
    # Remove control chars except \n and \t.
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    # Collapse runs of whitespace within a line.
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def main():
    df = pd.read_excel(PATH)
    for col in ["test_text", "corpus_text", "reasoning"]:
        if col in df.columns:
            df[col] = df[col].map(clean)
    df.to_excel(PATH, index=False)
    print(f"cleaned in place: {PATH}")
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()
