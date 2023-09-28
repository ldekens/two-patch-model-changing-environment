# two-patch-model-changing-environment

This depository contains the codes to reproduce the figures of the paper "Sharp habitat shifts, evolutionary tipping points and rescue: quantifying the perilous path of a specialist species toward a refugium in a changing environment".

It contains three main Python files, and three tools Python files.
- 20230927_figures_2_to_5.py should be run alongside tools_figures_2_to_5.py. It produces the figures 2 to 5 (all the figures in the main text). WARNING: some functions within this code run parallel replicates by batch of 6, so check that you have enough CPU available. 
- 20230927_figures_6.py should be run alongside tools_fig_6.py. It produces the figure 6 (comparison with individual-based simulations). WARNING: some functions within this code run parallel replicates by batch of 15, so check that you have enough CPU available.
- 20230927_figures_7_to_9.py should be run alongside tools_fig_7_to_9.py. It produces the figures 7 to 9 (supplementaries). WARNING: some functions within this code run parallel replicates by batch of 6, so check that you have enough CPU available.

