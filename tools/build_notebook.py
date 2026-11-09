"""Convert a `# %%`-style script into a Jupyter notebook and (optionally) execute it,
embedding stdout, rich DataFrame output and matplotlib figures.

Usage: python tools/build_notebook.py notebooks_src/x.py x.ipynb [--execute]
"""
import ast
import base64
import contextlib
import io
import json
import re
import sys
import time
import traceback


def split_cells(src):
    parts = re.split(r"^# %%(.*)$", src, flags=re.M)
    cells = []
    for i in range(1, len(parts), 2):
        tag, body = parts[i], parts[i + 1].strip("\n")
        if "[markdown]" in tag:
            body = "\n".join(re.sub(r"^# ?", "", l) for l in body.split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": body, "outputs": [], "execution_count": None})
    return cells


def _lines(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def execute(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = {"__name__": "__main__"}
    figs = []

    def _show(*a, **k):
        for num in plt.get_fignums():
            f = plt.figure(num)
            buf = io.BytesIO()
            f.savefig(buf, format="png", dpi=90, bbox_inches="tight")
            figs.append(base64.b64encode(buf.getvalue()).decode())
        plt.close("all")

    plt.show = _show
    n = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        n += 1
        c["execution_count"] = n
        out, figs[:] = io.StringIO(), []
        tree = ast.parse(c["source"])
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        t0 = time.time()
        result = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(tree, f"<cell {n}>", "exec"), ns)
                if last is not None:
                    result = eval(compile(ast.Expression(last.value), f"<cell {n}>", "eval"), ns)
            _show()
        except Exception:
            sys.stdout.write(out.getvalue())
            traceback.print_exc()
            raise SystemExit(f"cell {n} failed")
        print(f"cell {n:2d} ok ({time.time() - t0:5.1f}s)", flush=True)
        outputs = []
        if out.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": _lines(out.getvalue())})
        for f in figs:
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": f, "text/plain": ["<Figure>"]}})
        if result is not None:
            data = {"text/plain": _lines(repr(result))}
            if hasattr(result, "to_html"):
                data["text/html"] = _lines(result.to_html())
            outputs.append({"output_type": "execute_result", "execution_count": n, "metadata": {}, "data": data})
        c["outputs"] = outputs
    return cells


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cells = split_cells(open(src, encoding="utf-8").read())
    if "--execute" in sys.argv:
        cells = execute(cells)
    for c in cells:
        c["source"] = _lines(c["source"])
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
