"""Duplicate bridge score of a contract for the declaring side, as a table over tricks taken."""
import numpy as np

STRAINS = "SHDCN"          # same order as the DD table


def contract_score(level, strain, doubled, vul, tricks):
    """level 1..7, strain index into STRAINS, doubled 0/1/2, vul bool, tricks 0..13 -> declarer's score."""
    need = level + 6
    if tricks >= need:
        per = 20 if STRAINS[strain] in "DC" else 30
        base = (level * per + (10 if STRAINS[strain] == "N" else 0)) * (1, 2, 4)[doubled]
        s = base + (0, 50, 100)[doubled]
        s += (500 if vul else 300) if base >= 100 else 50
        if level == 6:
            s += 750 if vul else 500
        elif level == 7:
            s += 1500 if vul else 1000
        over = tricks - need
        s += over * (per if doubled == 0 else (200 if vul else 100) * doubled)
        return s
    down = need - tricks
    if doubled == 0:
        return -down * (100 if vul else 50)
    if vul:
        pen = 200 + 300 * (down - 1)
    else:
        pen = (100, 300, 500)[down - 1] if down <= 3 else 500 + 300 * (down - 3)
    return -pen * doubled


# SCORE[level-1, strain, doubled, vul, tricks]
SCORE = np.array([[[[[contract_score(l, s, d, v, t) for t in range(14)] for v in (False, True)]
                    for d in range(3)] for s in range(5)] for l in range(1, 8)], dtype=np.float32)

if __name__ == "__main__":
    checks = [((4, 0, 0, False, 10), 420), ((4, 0, 0, True, 11), 650), ((3, 4, 0, False, 9), 400), ((6, 4, 0, True, 12), 1440),
              ((7, 3, 0, False, 13), 1440), ((2, 1, 1, False, 8), 470), ((2, 1, 1, True, 9), 870), ((1, 4, 2, True, 7), 760),
              ((3, 0, 0, False, 7), -100), ((3, 0, 1, False, 5), -800), ((3, 0, 1, True, 6), -800), ((4, 1, 2, True, 9), -400),
              ((1, 3, 0, False, 9), 110), ((5, 2, 1, False, 12), 650)]
    for args, want in checks:
        assert contract_score(*args) == want, (args, contract_score(*args), want)
    print("scoring ok")
