# StarCraft Match Prediction with a Probabilistic Graphical Model

Project implementing a Bayesian Bradley-Terry probabilistic graphical model for predicting StarCraft match outcomes.

## Model

The model represents each player with a latent skill variable. A match outcome is modeled as a Bernoulli random variable depending on the differences between the two players' skills, plus optional race and match-context features.

```text
skill[player] ~ Normal(0, sigma_skill^2)
beta[feature] ~ Normal(0, sigma_beta^2)
y_match ~ Bernoulli(sigmoid(skill[player_A] - skill[player_B] + x_beta))
```

where:
- `skill[player]` represents a latent player skill parameter.
- `beta[feature]` represents race and contextual effects.
- `y_match` indicates whether Player A wins.

Model parameters are estimated using Maximum A Posterior (MAP) inference with Gaussian priors.

## Data

Expected directory structure:

```text
.
├── README.md
├── requirements.txt
├── starcraft_pgm_project.py
└── starcraft/
    ├── train.csv
    └── valid.csv
```

## Installation

Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the project

### Quick smoke test

```bash
python starcraft_pgm_project.py \
    --data_dir starcraft \
    --out_dir results_quick \
    --quick
```

## Full experiment

```bash
python starcraft_pgm_project.py \
    --data_dir starcraft \
    --out_dir results
```

## Evaluation metrics

- Accuracy
- Log Loss
- Brier Score
- Expected Calibration Error (ECE)
- Training Time