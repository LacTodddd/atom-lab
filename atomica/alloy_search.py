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

def crossover(p1, p2, n_sites, n_au, rng):
    s1, s2 = set(p1), set(p2)
    shared = s1 & s2                      # Au in both parents -> always kept
    only_one = list(s1 ^ s2)             # Au in exactly one parent
    need = n_au - len(shared)
    chosen = rng.choice(only_one, need, replace=False) if need > 0 else []
    return tuple(sorted(shared | {int(x) for x in chosen}))

def genetic_search(evaluate, n_sites, n_au, budget, seed, pop_size=10):
    rng = np.random.default_rng(seed)
    used = 0
    history = []
    best_e, best_c = np.inf, None
    pop = []  # list of (energy, config)

    def record(e, c):
        nonlocal best_e, best_c, used
        used += 1
        if e < best_e:
            best_e, best_c = e, c
        history.append((used, best_e))

    for _ in range(min(pop_size, budget)):
        c = random_config(n_sites, n_au, rng)
        e = evaluate(c)
        record(e, c)
        pop.append((e, c))

    while used < budget:
        pop.sort(key=lambda t: t[0])
        parents = pop[: max(2, pop_size // 2)]
        pa = parents[int(rng.integers(len(parents)))][1]
        pb = parents[int(rng.integers(len(parents)))][1]
        child = mutate_swap(crossover(pa, pb, n_sites, n_au, rng), n_sites, rng)
        e = evaluate(child)
        record(e, child)
        pop.append((e, child))
        pop.sort(key=lambda t: t[0])
        pop = pop[:pop_size]

    return history, best_c
