"""Concrete bipartite factor-graph object for a tiny instantiated model."""
from dataclasses import asdict, dataclass


@dataclass
class Variable:
    name: str
    domain: str
    observed: bool = False


@dataclass
class Factor:
    name: str
    kind: str
    variables: list[str]
    expression: str


def build_graph(n_users=2, n_items=2, k=2, observations=None):
    if observations is None:
        observations = [(0, 0), (0, 1), (1, 0)]
    variables, factors = {}, []
    for prefix, count in [("s", n_users), ("t", n_items)]:
        for j in range(count):
            for d in range(k):
                name = f"{prefix}_{j}_{d}"
                variables[name] = Variable(name, "real")
                factors.append(Factor("prior_" + name, "Gaussian prior", [name], "N(mean_prior, trait_var)"))
            name = f"b{prefix}_{j}"
            variables[name] = Variable(name, "real")
            factors.append(Factor("prior_" + name, "Gaussian prior", [name], "N(0, bias_var)"))
    for row, (u, i) in enumerate(observations):
        z = []
        for d in range(k):
            name = f"z_{row}_{d}"
            variables[name] = Variable(name, "real")
            factors.append(Factor("product_" + name, "deterministic product",
                                  [f"s_{u}_{d}", f"t_{i}_{d}", name], "delta(z-s*t)"))
            z.append(name)
        a, y, r = f"a_{row}", f"y_{row}", f"R_{row}"
        for name in [a, y]:
            variables[name] = Variable(name, "real")
        variables[r] = Variable(r, "{1,2,3,4,5}", True)
        factors.append(Factor(f"sum_{row}", "deterministic sum", z + [f"bs_{u}", f"bt_{i}", a],
                              "delta(a-offset-bs-bt-sum(z))"))
        factors.append(Factor(f"noise_{row}", "Gaussian observation", [a, y], "N(y;a,noise_var)"))
        factors.append(Factor(f"ordinal_{row}", "fixed ordinal threshold", [y, r],
                              "I(threshold[R-1] < y <= threshold[R])"))
    assert all(v in variables for f in factors for v in f.variables)
    return dict(variables=[asdict(v) for v in variables.values()],
                factors=[asdict(f) for f in factors],
                edges=[dict(factor=f.name, variable=v) for f in factors for v in f.variables],
                note="Tiny real instantiation; plates in the report describe repeated factors on MovieLens. Structure fixed, parameters learned.")
