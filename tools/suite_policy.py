"""Reviewable acceptance policy; recorded reports cannot choose their own gates."""

RTOL = 1e-4
ATOL = 1e-5
REPEATS = 4
WARMUP = 50
ITERATIONS = 500
THREADS = 1
CASES = (("linear", 1), ("mlp", 1), ("mlp", 16), ("mlp", 64), ("residual", 16))


def providers_for(kind, batch):
    return ["CPUExecutionProvider", "CoreMLExecutionProvider"] if (kind, batch) == ("mlp", 16) else ["CPUExecutionProvider"]
