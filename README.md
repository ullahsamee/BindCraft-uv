# uv BindCraft 
[Nature BindCraft](https://www.nature.com/articles/s41586-025-09429-6)

![alt text](https://github.com/ullahsamee/BindCraft/blob/main/uii.gif)

<samp>
Simple binder design pipeline using AlphaFold2 backpropagation, MPNN, and PyRosetta. Select your target and let the script do the rest of the work and finish once you have enough designs to order!
<samp>
  
![alt text](https://github.com/ullahsamee/BindCraft/blob/main/ui.png)

## Installation
clone this repo

```
git clone https://github.com/ullahsamee/BindCraft-uv
```

```
cd BindCraft
```

```
chmod +x install_bindcraft.sh
```

```
./install_bindcraft.sh
```

The script will activate the env for you, but in case if not then use this.
```
source ./activate_bindcraft.sh
```

Test your installation
```
python -u ./bindcraft.py --settings './settings_target/PDL1.json' --filters './settings_filters/default_filters.json' --advanced './settings_advanced/default_4stage_multimer.json'
```

Note: Using uv `./install_bindcraft.sh` script will setup BindCraft with Python3.12, CUDA 13.0 and jax 0.7.2 and PyRosetta with(Python3.12). PyRosetta, requires a license for commercial purposes. The code requires about 2 Mb of storage space, while the AlphaFold2 weights take up about 5.3 Gb.

## Google Colab
<a href="https://colab.research.google.com/github/martinpacesa/BindCraft/blob/main/notebooks/BindCraft.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> <br />

 <samp>
We prepared a convenient google colab notebook to test the bindcraft code functionalities. However, as the pipeline requires significant amount of GPU memory to run for larger target+binder complexes, we highly recommend to run it using a local installation and at least 32 Gb of GPU memory.
 </samp>

**Always try to trim the input target PDB to the smallest size possible! It will significantly speed up the binder generation and minimise the GPU memory requirements.**

**Be ready to run at least a few hundred trajectories to see some accepted binders, for difficult targets it might even be a few thousand.**


#  Running the Script Locally
To run the script locally, you need to configure your target `.json` file in the **`settings_target`** folder.

| 🏷️ **Parameter** | 📖 **Description** |
|:------------------|:-------------------|
| `design_path` | Path where to save designs and statistics |
| `binder_name` | What to prefix your designed binder files with |
| `starting_pdb` | The path to the PDB of your target protein |
| `chains` | Which chains to target in your protein, rest will be ignored |
| `target_hotspot_residues` | Which position to target for binder design, for example `1,2-10` or chain specific `A1-10,B1-20` or entire chains `A`, set to null if you want AF2 to select binding site; better to select multiple target residues or a small patch to reduce search space for binder |
| `lengths` | Range of binder lengths to design |
| `number_of_final_designs` | How many designs that pass all filters to aim for, script will stop if this many are reached |

---

<div align="center">

💡 *Configure these parameters in your JSON file before running the script*

</div>

Then run the binder design script:

`sbatch ./bindcraft.slurm --settings './settings_target/PDL1.json' --filters './settings_filters/default_filters.json' --advanced './settings_advanced/default_4stage_multimer.json'`

The *settings* flag should point to your target .json which you set above. The *filters* flag points to the json where the design filters are specified (default is ./filters/default_filters.json). The *advanced* flag points to your advanced settings (default is ./advanced_settings/default_4stage_multimer.json). If you leave out the filters and advanced settings flags it will automatically point to the defaults.

Alternatively, if your machine does not support SLURM, you can run the code directly by activating the environment in conda and running the python code:

```
conda activate BindCraft
cd /path/to/bindcraft/folder/
python -u ./bindcraft.py --settings './settings_target/PDL1.json' --filters './settings_filters/default_filters.json' --advanced './settings_advanced/default_4stage_multimer.json'
```

> [!IMPORTANT]
> We recommend to generate at least a 100 final designs passing all filters, then order the top 5-20 for expression.
If high affinity binders are required, it is better to screen more, as the ipTM metric used for ranking is not a good predictor for affinity, but has been shown to be a good binary predictor of binding. 

Below are explanations for individual filters and advanced settings.

## Advanced settings
Here are the advanced settings controlling the design process:

```
omit_AAs                        -> which amino acids to exclude from design (note: they can still occur if no other options are possible in the position)
force_reject_AA                 -> whether to force reject design if it contains any amino acids specified in omit_AAs
design_algorithm                -> which design algorithm for the trajecory to use, the currently implemented algorithms are below
use_multimer_design             -> whether to use AF2-ptm or AF2-multimer for binder design; the other model will be used for validation then
num_recycles_design             -> how many recycles of AF2 for design
num_recycles_validation         -> how many recycles of AF2 use for structure prediction and validation
sample_models = True            -> whether to randomly sample parameters from AF2 models, recommended to avoid overfitting
rm_template_seq_design          -> remove target template sequence for design (increases target flexibility)
rm_template_seq_predict         -> remove target template sequence for reprediction (increases target flexibility)
rm_template_sc_design           -> remove sidechains from target template for design
rm_template_sc_predict          -> remove sidechains from target template for reprediction
predict_initial_guess           -> Introduce bias by providing binder atom positions as a starting point for prediction. Recommended if designs fail after MPNN optimization.
predict_bigbang                 -> Introduce atom position bias into the structure module for atom initilisation. Recommended if target and design are large (more than 600 amino acids).

# Design iterations
soft_iterations                 -> number of soft iterations (all amino acids considered at all positions)
temporary_iterations            -> number of temporary iterations (softmax, most probable amino acids considered at all positions)
hard_iterations                 -> number of hard iterations (one hot encoding, single amino acids considered at all positions)
greedy_iterations               -> number of iterations to sample random mutations from PSSM that reduce loss
greedy_percentage               -> What percentage of protein length to mutate during each greedy iteration

# Design weights, higher value puts more weight on optimising the parameter.
weights_plddt                   -> Design weight - pLDDT of designed chain
weights_pae_intra               -> Design weight - PAE within designed chain
weights_pae_inter               -> Design weight - PAE between chains
weights_con_intra               -> Design weight - maximise number of contacts within designed chain
weights_con_inter               -> Design weight - maximise number of contacts between chains
intra_contact_distance          -> Cbeta-Cbeta cutoff distance for contacts within the binder
inter_contact_distance          -> Cbeta-Cbeta cutoff distance for contacts between binder and target
intra_contact_number            -> how many contacts each contact esidue should make within a chain, excluding immediate neighbours
inter_contact_number            -> how many contacts each contact residue should make between chains
weights_helicity                -> Design weight - helix propensity of the design, Default 0, negative values bias towards beta sheets
random_helicity                 -> whether to randomly sample helicity weights for trajectories, from -1 to 1

# Additional losses
use_i_ptm_loss                  -> Use i_ptm loss to optimise for interface pTM score?
weights_iptm                    -> Design weight - i_ptm between chains
use_rg_loss                     -> use radius of gyration loss?
weights_rg                      -> Design weight - radius of gyration weight for binder
use_termini_distance_loss       -> Try to minimise distance between N- and C-terminus of binder? Helpful for grafting
weights_termini_loss            -> Design weight - N- and C-terminus distance minimisation weight of binder

# MPNN settings
mpnn_fix_interface              -> whether to fix the interface designed in the starting trajectory
num_seqs                        -> number of MPNN generated sequences to sample and predict per binder
max_mpnn_sequences              -> how many maximum MPNN sequences per trajectory to save if several pass filters
sampling_temp = 0.1             -> sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sampling randomly.")

# MPNN settings - advanced
backbone_noise                  -> backbone noise during sampling, 0.00-0.02 are good values
model_path                      -> path to the MPNN model weights
mpnn_weights                    -> whether to use "original" mpnn weights or "soluble" weights
save_mpnn_fasta                 -> whether to save MPNN sequences as fasta files, normally not needed as the sequence is also in the CSV file

# AF2 design settings - advanced
num_recycles_design             -> how many recycles of AF2 for design
num_recycles_validation         -> how many recycles of AF2 use for structure prediction and validation
optimise_beta                   -> optimise predictions if beta sheeted trajectory detected?
optimise_beta_extra_soft        -> how many extra soft iterations to add if beta sheets detected
optimise_beta_extra_temp        -> how many extra temporary iterations to add if beta sheets detected
optimise_beta_recycles_design   -> how many recycles to do during design if beta sheets detected
optimise_beta_recycles_valid    -> how many recycles to do during reprediction if beta sheets detected

# Optimise script
remove_unrelaxed_trajectory     -> remove the PDB files of unrelaxed designed trajectories, relaxed PDBs are retained
remove_unrelaxed_complex        -> remove the PDB files of unrelaxed predicted MPNN-optimised complexes, relaxed PDBs are retained
remove_binder_monomer           -> remove the PDB files of predicted binder monomers after scoring to save space
zip_animations                  -> at the end, zip Animations trajectory folder to save space
zip_plots                       -> at the end, zip Plots trajectory folder to save space
save_trajectory_pickle          -> save pickle file of the generated trajectory, careful, takes up a lot of storage space!
max_trajectories                -> how many maximum trajectories to generate, for benchmarking
acceptance_rate                 -> what fraction of trajectories should yield designs passing the filters, if the proportion of successful designs is less than this fraction then the script will stop and you should adjust your design weights
start_monitoring                -> after what number of trajectories should we start monitoring acceptance_rate, do not set too low, could terminate prematurely

# debug settings
enable_mpnn = True              -> whether to enable MPNN design
enable_rejection_check          -> enable rejection rate check
```

#  Protein Design Metrics

<div align="center">

---

| 📊 **Metric** | 📝 **Description** |
|:--------------|:------------------|
| `MPNN_score` | MPNN sequence score, generally not recommended as it depends on protein |
| `MPNN_seq_recovery` | MPNN sequence recovery of original trajectory |
| `pLDDT` | pLDDT confidence score of AF2 complex prediction, normalised to 0-1 |
| `pTM` | pTM confidence score of AF2 complex prediction, normalised to 0-1 |
| `i_pTM` | Interface pTM confidence score of AF2 complex prediction, normalised to 0-1 |
| `pAE` | Predicted alignment error of AF2 complex prediction, normalised compared AF2 by n/31 to 0-1 |
| `i_pAE` | Predicted interface alignment error of AF2 complex prediction, normalised compared AF2 by n/31 to 0-1 |
| `i_pLDDT` | Interface pLDDT confidence score of AF2 complex prediction, normalised to 0-1 |
| `ss_pLDDT` | Secondary structure pLDDT confidence score of AF2 complex prediction, normalised to 0-1 |
| `Unrelaxed_Clashes` | Number of interface clashes before relaxation |
| `Relaxed_Clashes` | Number of interface clashes after relaxation |
| `Binder_Energy_Score` | Rosetta energy score for binder alone |
| `Surface_Hydrophobicity` | Surface hydrophobicity fraction for binder |
| `ShapeComplementarity` | Interface shape complementarity |
| `PackStat` | Interface packstat rosetta score |
| `dG` | Interface rosetta dG energy |
| `dSASA` | Interface delta SASA (size) |
| `dG/dSASA` | Interface energy divided by interface size |
| `Interface_SASA_%` | Fraction of binder surface covered by the interface |
| `Interface_Hydrophobicity` | Interface hydrophobicity fraction of binder interface |
| `n_InterfaceResidues` | Number of interface residues |
| `n_InterfaceHbonds` | Number of hydrogen bonds at the interface |
| `InterfaceHbondsPercentage` | Number of hydrogen bonds compared to interface size |
| `n_InterfaceUnsatHbonds` | Number of unsatisfied buried hydrogen bonds at the interface |
| `InterfaceUnsatHbondsPercentage` | Number of unsatisfied buried hydrogen bonds compared to interface size |
| `Interface_Helix%` | Proportion of alfa helices at the interface |
| `Interface_BetaSheet%` | Proportion of beta sheets at the interface |
| `Interface_Loop%` | Proportion of loops at the interface |
| `Binder_Helix%` | Proportion of alfa helices in the binder structure |
| `Binder_BetaSheet%` | Proportion of beta sheets in the binder structure |
| `Binder_Loop%` | Proportion of loops in the binder structure |
| `InterfaceAAs` | Number of amino acids of each type at the interface |
| `HotspotRMSD` | Unaligned RMSD of binder compared to original trajectory, in other words how far is binder in the repredicted complex from the original binding site |
| `Target_RMSD` | RMSD of target predicted in context of the designed binder compared to input PDB |
| `Binder_pLDDT` | pLDDT confidence score of binder predicted alone |
| `Binder_pTM` | pTM confidence score of binder predicted alone |
| `Binder_pAE` | Predicted alignment error of binder predicted alone |
| `Binder_RMSD` | RMSD of binder predicted alone compared to original trajectory |

---

<div align="center">

*This table provides a comprehensive overview of metrics used to evaluate protein design quality and binding interactions.*

</div>

##  Implemented Design Algorithms

|  **Algorithm** |  **Description** |
|:-----------------|:-------------------|
| `2stage` | Design with logits→pssm_semigreedy (faster) |
| `3stage` | Design with logits→softmax(logits)→one-hot (standard) |
| `4stage` | Design with logits→softmax(logits)→one-hot→pssm_semigreedy (default, extensive) |
| `greedy` | Design with random mutations that decrease loss (less memory intensive, slower, less efficient) |
| `mcmc` | Design with random mutations that decrease loss, similar to Wicky et al. (less memory intensive, slower, less efficient) |

---

## ⚠️ Known Limitations

**Important considerations when using the design algorithms**

|  **Limitation** |  **Details** |
|:------------------|:---------------|
| **Target Specificity** | Settings might not work for all targets! Number of iterations, design weights, and/or filters might have to be adjusted. Target site selection is also important, but AF2 is very good at detecting good binding sites if no hotspot is specified. |
| **Interface Type Bias** | AF2 is worse at predicting/designing hydrophilic then it is at hydrophobic interfaces. |
| **Trajectory Deformation** | Sometimes the trajectories can end up being deformed or 'squashed'. This is normal for AF2 multimer design, as it is very sensitive to the sequence input, this cannot be avoided without model retraining. However these trajectories are quickly detected and discarded. |

---

**Note: Before posting an issue, read the complete wiki <a href="https://github.com/martinpacesa/BindCraft/wiki/De-novo-binder-design-with-BindCraft">here</a>. Issues that are covered in the wiki will be closed without an answer.**

 <samp>
Thanks to Lennart Nickel, Yehlin Cho, Casper Goverde, and Sergey Ovchinnikov for help with coding and discussing ideas. This repository uses code from: Sergey Ovchinnikov's ColabDesign (https://github.com/sokrypton/ColabDesign), Justas Dauparas's ProteinMPNN (https://github.com/dauparas/ProteinMPNN), and PyRosetta (https://github.com/RosettaCommons/PyRosetta.notebooks)
</samp>
