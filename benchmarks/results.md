# Threshold validation

n=1000, 20 replicates per level. `fire` = share of runs where the check flagged the data; `harm` = how badly the protected method did.

## linearity  (harm = LiNGAM direction error)

| level | fire | harm |
|---|---|---|
| 0 | 0.00 | 0.00 |
| 0.1 | 0.00 | 0.00 |
| 0.2 | 0.10 | 0.00 |
| 0.3 | 0.05 | 0.00 |
| 0.4 | 0.50 | 0.00 |
| 0.6 | 1.00 | 0.00 |
| 0.8 | 1.00 | 0.90 |
| 1 | 1.00 | 1.00 |

## noise  (harm = LiNGAM direction error)

| level | fire | harm |
|---|---|---|
| 1 | 0.00 | 0.55 |
| 1.1 | 0.35 | 0.25 |
| 1.2 | 1.00 | 0.00 |
| 1.35 | 1.00 | 0.00 |
| 1.5 | 1.00 | 0.00 |
| 1.75 | 1.00 | 0.00 |
| 2 | 1.00 | 0.00 |
| 2.5 | 1.00 | 0.00 |

## determinism  (harm = PC skeleton edges lost)

| level | fire | harm |
|---|---|---|
| 1 | 0.00 | 0.00 |
| 2 | 0.00 | 0.04 |
| 3 | 1.00 | 0.09 |
| 4 | 1.00 | 0.18 |
| 6 | 1.00 | 0.25 |
| 8 | 1.00 | 0.39 |
| 12 | 1.00 | 0.41 |

## scale  (harm = varsortability (raw data))

| level | fire | harm |
|---|---|---|
| 0.2 | 0.00 | 0.63 | variance ratio 1.1
| 0.5 | 0.00 | 0.82 | variance ratio 1.4
| 1 | 0.00 | 1.00 | variance ratio 4.0
| 1.5 | 1.00 | 1.00 | variance ratio 19.9
| 2 | 1.00 | 1.00 | variance ratio 85.5
| 3 | 1.00 | 1.00 | variance ratio 820.7
