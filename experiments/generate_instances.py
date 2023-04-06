# This file generates concatenated trap function instances
# i.e. with all ones as optimum, and all zeroes as deceptive attractor.

from pathlib import Path

out_dir = Path("./instances")
l_and_ks = [(25, 5), (50, 5), (100, 5), (200, 5)]


def generate(path: Path, l: int, k: int):
    with path.open("w") as f:
        f.write(f"1\n{l} {k}\n")
        # Optimum: all ones
        for _ in range(l - 1):
            f.write("1 ")
        f.write("1\n")
        # Permutation: 0 - (l - 1) -- Tight encoding
        for i in range(l - 1):
            f.write(f"{i} ")
        f.write(f"{l - 1}\n")


for l, k in l_and_ks:
    generate(out_dir / f"trap__l_{l}__k_{k}.txt", l, k)
