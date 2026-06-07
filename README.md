# PLDN Numerical Examples

This repository contains the three numerical examples used in Section 3 of
the paper "Probabilistic latent dynamics network for efficient and scalable
modeling".

## Case Map

| Folder | Paper section | Description | Main code structure |
| --- | --- | --- | --- |
| `PLD-1dacc` | 3.1 | 1D linear force-response prediction for a cantilever beam with 401 nodes and 1024 time steps. | `reduction/` for DeepONet/AE dimension reduction; `latent_pred/` for probabilistic latent prediction. |
| `PLD-1dnlacc` | 3.2 | 1D nonlinear response reconstruction for a simply supported beam with 501 nodes and 1024 stored time steps. | Same two-stage structure as `PLD-1dacc`, with four sparse input sensors. |
| `PLD-2dpv` | 3.3 | 2D force-response prediction for a rectangular plate over 201 time points and an 81 by 61 nodal grid. | Same two-stage structure, adapted to 2D displacement fields. |

## Notes

- The `reduction/` scripts train or evaluate the dimension reduction models.
- The `latent_pred/` scripts train or evaluate the probabilistic latent
  dynamics models.
- Console output is normalized with tags such as `[data]`, `[train]`,
  `[eval]`, and `[checkpoint]`.
- Large datasets, trained weights, and generated `.mat` results are ignored by
  `.gitignore`. Publish those files separately with Git LFS, GitHub Releases,
  or an external data repository if they are needed for reproducibility.
