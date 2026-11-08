# nba-ai

Learning to train models on NBA data. Two small PyTorch models (an MLP and a one-block
transformer) predict a player's points in their next game from their last 10 games,
then draw a chart of the training curves, predictions, and attention weights.

## Run

```sh
uv sync                                                   # installs CPU-only torch
uv run python nba_train.py                                # synthetic data, offline
uv run python nba_train.py --source nba --season 2024-25  # real games via nba_api
```

Writes `nba_training.png`. Other flags: `--epochs`, `--seed`, `--out`.

## Reading the chart

- **Top left:** error per epoch. Dashed = training games, solid = unseen future games.
  If dashed keeps falling while solid rises, that's overfitting.
- **Top right:** transformer predicted vs actual points on test games.
- **Bottom:** which of the last 10 games the transformer pays attention to.

## The part to learn

`fit()` is the whole of training: forward pass, compute loss, `loss.backward()`,
`opt.step()`. Every model, whatever the data, uses that same loop.
