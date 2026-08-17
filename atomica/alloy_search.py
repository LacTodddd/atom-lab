import numpy as np

def random_config(n_sites, n_au, rng):
    return tuple(sorted(int(x) for x in rng.choice(n_sites, n_au, replace=False)))

def mutate_swap(config, n_sites, rng):
    au = set(config)
    cu = [s for s in range(n_sites) if s not in au]
    out = int(rng.choice(list(au)))
    inn = int(rng.choice(cu))
    au.discard(out)
    au.add(inn)
    return tuple(sorted(au))

def random_search(evaluate, n_sites, n_au, budget, seed):
    rng = np.random.default_rng(seed)
    best_e, best_c = np.inf, None
    history = []
    for i in range(budget):
        c = random_config(n_sites, n_au, rng)
        e = evaluate(c)
        if e < best_e:
            best_e, best_c = e, c
        history.append((i + 1, best_e))
    return history, best_c
