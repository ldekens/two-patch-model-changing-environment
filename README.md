# two-patch-model-changing-environment

This depository contains the codes to reproduce the figures of the paper "Sharp habitat shifts, evolutionary tipping points and rescue: quantifying the perilous path of a specialist species toward a refugium in a changing environment" (https://doi.org/10.1016/j.tpb.2024.09.001)

It contains one main Python file and a corresponding tools Python file for each figures in the main text. Additionally, the three figures in the appendices can be generated through a single main Python file and the corresponding tools file.
- 20241001_figure_n.py should be run alongside tools_figure_n.py (replace n by an integer between 2 and 6 to generate the corresponding figure). WARNING: some functions within this code run parallel replicates by batch of 5 or 6, so check that you have enough CPU available. 

- 20230927_figures_7_to_9.py should be run alongside tools_fig_7_to_9.py. It produces the figures 7 to 9 (supplementaries). WARNING: some functions within this code run parallel replicates by batch of 6, so check that you have enough CPU available.

