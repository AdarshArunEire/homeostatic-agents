# Prototype 1b - DQN port to PyTorch

Prototype 1b ports the original hand-written NumPy DQN into PyTorch while keeping the same compact homeostatic-control task. 

The goal was not to make the agent more advanced yet, but to check whether PyTorch could reproduce the same learned behaviour before using it as the base for larger spatial experiments. The port broadly mirrors Prototype 1’s learning curve and final hydration/satiation cluster.

It also preserves the same short-gap death-clustering failure mode, and slightly amplfies it. This suggests the core behaviour belongs to the DQN policy itself, not just the original NumPy implementation.

## Figures

<p>
  <img src="results/best_figures/phase_trajectory_wu500_buf5k_g95_e03_u50_vs_PyTorch_upper_median.gif" width="700">
  <br>
  <sub><em>Final greedy-policy evaluation phase. The PyTorch port succesfully mirrors the NumPy DQN's homeostatic cluster</em></sub>
</p>
<br>

<p>
  <img src="results/best_figures/LearningNumPyVsPyTorch.png" width="700">
  <br>
  <sub><em>The PyTorch port reproduces the same broad learning transition as the NumPy DQN.</em></sub>
</p>
<br>

<p>
  <img src="results/best_figures/Death_rateNumPyVSPyTorch.png" width="700">
  <br>
<sub><em>Death-gap distribution during eval. The PyTorch port shows the same short-gap death clustering, but with a lower median gap.</em></sub>
</p>
<br>
