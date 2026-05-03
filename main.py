import argparse
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt


def gue_matrix(n: int) -> np.ndarray:
    """
    Generate an n x n random Hermitian matrix from the GUE-like ensemble.
    Hermitian means H = H.conj().T, so all eigenvalues are real.
    """
    A = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))
    H = (A + A.conj().T) / 2
    return H / np.sqrt(n)


def normalized_spacings(eigs: np.ndarray, trim_fraction: float = 0.15) -> np.ndarray:
    """
    Sort eigenvalues, remove edge eigenvalues, compute neighboring gaps,
    then normalize gaps so the average spacing is 1.
    """
    eigs = np.sort(eigs.real)
    n = len(eigs)

    # Avoid edge effects where eigenvalue density changes a lot
    lo = int(trim_fraction * n)
    hi = int((1 - trim_fraction) * n)
    center = eigs[lo:hi]

    if len(center) < 2:
        raise ValueError(
            "Matrix is too small after trimming; reduce --trim-fraction or use a larger matrix."
        )

    gaps = np.diff(center)
    return gaps / np.mean(gaps)


def collect_gue_spacings(n: int = 120, trials: int = 80, trim_fraction: float = 0.15) -> np.ndarray:
    all_spacings = []
    for _ in range(trials):
        H = gue_matrix(n)
        eigs = np.linalg.eigvalsh(H)
        all_spacings.append(normalized_spacings(eigs, trim_fraction=trim_fraction))
    return np.concatenate(all_spacings)


def load_matrices_from_file(path: str) -> list[np.ndarray]:
    """Load one or more square matrices from a file.

    Supported formats:
    - .npy : single NumPy array
    - .npz : single array inside an archive
    - .txt, .csv, .dat : whitespace- or comma-delimited text
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if len(archive.files) == 1:
            data = archive[archive.files[0]]
        else:
            raise ValueError(".npz file must contain exactly one array")
    elif suffix in {".txt", ".csv", ".dat"}:
        delimiter = "," if suffix == ".csv" else None
        data = np.loadtxt(path, dtype=np.complex128, delimiter=delimiter)
    else:
        raise ValueError("Unsupported file type: choose .npy, .npz, .txt, .csv, or .dat")

    if data.ndim == 2:
        matrices = [data]
    elif data.ndim == 3:
        matrices = [data[i] for i in range(data.shape[0])]
    else:
        raise ValueError("Input file must contain a square matrix or a stack of square matrices.")

    for i, matrix in enumerate(matrices):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("All matrices must be square.")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-8):
            warnings.warn(
                f"Matrix {i} is not Hermitian; symmetrizing using (M + M^H)/2.",
                UserWarning,
            )
            matrices[i] = (matrix + matrix.conj().T) / 2

    return matrices


def spacings_from_matrices(matrices: list[np.ndarray], trim_fraction: float = 0.15) -> np.ndarray:
    all_spacings = []
    for matrix in matrices:
        eigs = np.linalg.eigvalsh(matrix)
        all_spacings.append(normalized_spacings(eigs, trim_fraction=trim_fraction))
    return np.concatenate(all_spacings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eigenvalue spacings from random Hermitian matrices or input matrices against the GUE level repulsion curve."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to a matrix file to load instead of generating random matrices.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=120,
        help="Matrix dimension for random GUE matrices.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=80,
        help="Number of random matrices to generate when not using --input.",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.15,
        help="Fraction of eigenvalues to trim from each edge before computing spacings.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=281,
        help="Random seed for the random matrix ensemble.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and only print summary statistics.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.input:
        matrices = load_matrices_from_file(args.input)
        spacings = spacings_from_matrices(matrices, trim_fraction=args.trim_fraction)
        print(f"Loaded {len(matrices)} matrix(ices) from {args.input}.")
    else:
        spacings = collect_gue_spacings(
            n=args.n,
            trials=args.trials,
            trim_fraction=args.trim_fraction,
        )
        print(f"Generated {args.trials} random GUE matrices of size {args.n} x {args.n}.")

    s = np.linspace(0, 4, 400)
    gue_curve = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

    if not args.no_plot:
        plt.figure(figsize=(8, 5))
        plt.hist(
            spacings,
            bins=args.bins,
            density=True,
            alpha=0.65,
            label="Eigenvalue spacings",
        )
        plt.plot(s, gue_curve, linewidth=2.5, label="GUE Wigner surmise")
        plt.xlabel("Normalized spacing between neighboring eigenvalues")
        plt.ylabel("Density")
        plt.title("Random Matrix Theory: Eigenvalue Spacing Repulsion")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    print(f"Collected {len(spacings)} normalized spacings.")
    print(f"Mean spacing: {np.mean(spacings):.4f}")
    print(f"Fraction of very tiny gaps, s < 0.1: {np.mean(spacings < 0.1):.4f}")


if __name__ == "__main__":
    main()