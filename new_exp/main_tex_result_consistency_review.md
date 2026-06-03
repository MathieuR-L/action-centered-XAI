# Revisions recommandees pour `paper/main.tex`

Ce fichier traite les deux remarques de coherence sur les resultats numeriques.
Je n'ai pas utilise de recherche web: les incoherences concernent les sorties
experimentales locales, et les fichiers de resultats du depot suffisent pour
trancher.

## Sources locales consultees

- `new_exp/section6_benchmark_500_20seeds/section6_tables_500.md`
- `new_exp/section6_benchmark_500_20seeds/summary_by_lambda.csv`
- `new_exp/section6_benchmark_500_20seeds/summary.json`
- `new_exp/section6_explainer_500/section6_explainer_500_table.md`
- `new_exp/section6_explainer_500/explainer_500_table.csv`
- `new_exp/section6_explainer_500/raw_explainer_500_results.json`
- `new_exp/section6_explainer_500/explainer_500_by_variant.csv`
- `new_exp/section6_explainer_500/section6_explainer_500_causal_comparison.md`
- `experiments/results/ace_experiment_results.json`
- `experiments/make_paper_figures.py`
- `TRACEABILITY.md`

## Decision recommandee

Utiliser partout le nouveau benchmark "publication-oriented":

| Dimension | Source a utiliser | Pourquoi |
| --- | --- | --- |
| Robustesse ID/OOD | `new_exp/section6_benchmark_500_20seeds/` | 500 cas OOD, 20 seeds, grille de lambda, bootstrap CI, McNemar, Wilcoxon |
| Latences et mesures Captum | `new_exp/section6_explainer_500/` | 500 cas expliques, table finale deja alignee avec la Section 6 |
| Ancien run single-seed | `experiments/results/ace_experiment_results.json` | A ne garder que comme analyse preliminaire explicitement nommee, ou a retirer |

Le manuscrit melange actuellement l'ancien run single-seed et le nouveau run
500 cas / 20 seeds. C'est la source principale du probleme.

## 1. Incoherence performance ID/OOD

### Probleme

Le manuscrit utilise deux ensembles de resultats:

1. Nouveau benchmark 500 cas / 20 seeds:
   - Baseline ID accuracy: `0.854`
   - Baseline OOD accuracy: `0.604`
   - Causal ID accuracy: `0.843`
   - Causal OOD accuracy: `0.753`
   - Generalization gap: `0.252 -> 0.091`

2. Ancien run single-seed:
   - Baseline OOD accuracy: `66.1%`
   - Causal OOD accuracy: `82.9%`
   - ID cost: environ `4.0` points
   - Gap: `23.6 -> 2.9` points

Ces deux sets ne doivent pas coexister comme si c'etait le meme benchmark.

### Justification

Les chiffres `66.1% -> 82.9%`, `23.6 -> 2.9` et `4.0` viennent de
`experiments/results/ace_experiment_results.json`, qui indique:

- `seed = 7`
- `explain_samples = 12`
- baseline ID/OOD = `0.8978 / 0.6613`
- causal ID/OOD = `0.8579 / 0.8289`

Les chiffres `0.604 -> 0.753` viennent de
`new_exp/section6_benchmark_500_20seeds/`, qui est plus solide:

- `20` seeds
- `500` cas OOD
- lambda grid jusqu'a `lambda = 4.00`
- bootstrap `5000`
- McNemar `p < 1e-300`
- Wilcoxon `p = 0.000235`

Donc, pour un reviewer, il faut privilegier le nouveau benchmark et supprimer
ou qualifier tous les chiffres de l'ancien run.

### Remplacement principal recommande

Partout ou le texte dit:

```latex
improved OOD accuracy from 66.1\% to 82.9\%
```

remplacer par:

```latex
improved mean OOD accuracy from 0.604 to 0.753 across the 500-case OOD subset and 20 random seeds
```

Ou, en pourcentage:

```latex
improved mean OOD accuracy from 60.4\% to 75.3\% across the 500-case OOD subset and 20 random seeds
```

Je recommande la version decimale, car elle est coherente avec les tables.

### Remplacement ligne 1038

Texte actuel:

```latex
Table \ref{tab:controlled_robustness} shows that the standard multimodal model relied heavily on the spurious hospital feature: it lost 23.6 percentage points when that variable was randomized. The causal-invariance objective reduced this gap to 2.9 points, at the cost of a modest 4.0-point decrease in in-distribution accuracy.
```

Remplacer par:

```latex
Table \ref{tab:controlled_robustness} shows that causal-invariance training improved mean OOD accuracy from 0.604 to 0.753, a 14.9-point gain on the 500-case OOD subset. The mean generalization gap decreased from 0.252 to 0.091, while in-distribution accuracy changed from 0.854 to 0.843, corresponding to a 1.1-point ID accuracy cost.
```

### Justification du remplacement

- `0.753 - 0.604 = 0.149`, donc gain OOD = `14.9` points.
- `0.854 - 0.843 = 0.011`, donc cout ID = `1.1` point.
- `0.252 -> 0.091` vient de la table 500 cas / 20 seeds.
- Le texte actuel `23.6`, `2.9` et `4.0` appartient a l'ancien run, pas a la Table 6 actuelle.

### Remplacement conclusion ligne 1344

Texte actuel:

```latex
The strongest empirical signal in our experiments was not random explanation drift, but spurious multimodal reliance: a causal-invariance objective improved OOD accuracy from 66.1\% to 82.9\% while reducing hospital-feature importance and metadata-induced explanation leakage.
```

Remplacer par:

```latex
The strongest empirical signal in our experiments was not random explanation drift, but spurious multimodal reliance: in the 500-case, 20-seed benchmark, a causal-invariance objective improved mean OOD accuracy from 0.604 to 0.753 and reduced the mean generalization gap from 0.252 to 0.091.
```

### Justification du remplacement

La conclusion doit reprendre exactement le meme set que l'abstract et la Table
`controlled_robustness`. La phrase actuelle recycle l'ancien run single-seed.

### Remplacement ligne 1312

Texte actuel:

```latex
A non-causal hospital feature produced a 23.6-point OOD accuracy drop in the baseline model
```

Remplacer par:

```latex
The baseline model showed a 0.252 mean generalization gap under the OOD hospital-shift setting
```

Ou, si vous voulez garder l'expression en points:

```latex
The baseline model showed a 25.2-point mean generalization gap under the OOD hospital-shift setting
```

### Justification du remplacement

Le `23.6-point drop` est l'ancien ecart ID/OOD:

```text
0.8978 - 0.6613 = 0.2365
```

Le nouveau benchmark donne:

```text
0.85399 - 0.60152 = 0.25247
```

Donc la formulation correcte est `0.252` ou `25.2 points`.

### Remplacement ligne 1324

Texte actuel:

```latex
\item Causal training reduced OOD accuracy loss from 23.6 to 2.9 points, but no ground-truth causal labels exist to formally validate whether the resulting explanations are causally correct.
```

Remplacer par:

```latex
\item Causal training reduced the mean generalization gap from 0.252 to 0.091 in the 500-case, 20-seed benchmark, but no ground-truth causal labels exist to formally validate whether the resulting explanations are causally correct.
```

### Justification du remplacement

Meme raison: `23.6 -> 2.9` vient de l'ancien run. La phrase doit reprendre la
Table 6 actuelle.

## 2. Petite incoherence interne de la Table 6

### Probleme

La Table 6 affiche `OOD Acc. 500 = 0.604` et `Gen. gap = 0.252`. Or:

```text
0.854 - 0.604 = 0.250
```

Le gap `0.252` vient de `ood_accuracy_full_mean = 0.6015`, pas exactement de
la colonne `OOD Acc. 500`.

### Suggestion recommandee

Ajouter une note sous la table:

```latex
Generalization gaps are computed against the full OOD test split; bootstrap confidence intervals are estimated on the 500-case OOD evaluation subset.
```

### Alternative plus propre

Renommer la colonne:

```latex
\textbf{OOD Acc. 500 subset}
```

et ajouter une colonne:

```latex
\textbf{Full OOD Acc.}
```

avec:

| Model | Full OOD Acc. | OOD Acc. 500 subset | Gen. gap |
| --- | ---: | ---: | ---: |
| Baseline | 0.602 | 0.604 | 0.252 |
| Causal invariance | 0.752 | 0.753 | 0.091 |

### Justification

Ce n'est pas la contradiction principale signalee par le reviewer, mais c'est
le genre de detail qu'un reviewer peut recalculer mentalement. Une note suffit
si vous ne voulez pas agrandir la table.

## 3. Figures a mettre a jour

### Probleme

Le script `experiments/make_paper_figures.py` lit encore:

```python
RESULTS_PATH = ROOT / "experiments" / "results" / "ace_experiment_results.json"
```

Ce fichier contient l'ancien run single-seed. La figure
`paper/ace_benchmark_overview.png` risque donc d'afficher les anciens chiffres
`66.1` et `82.9`, meme si le texte/table utilisent `0.604` et `0.753`.

### Suggestion

Regenerer `ace_benchmark_overview.png` a partir de:

- `new_exp/section6_benchmark_500_20seeds/summary_by_lambda.csv`
- `new_exp/section6_explainer_500/explainer_500_table.csv`

La figure doit afficher:

- ID baseline `85.4`
- ID causal `84.3`
- OOD baseline `60.4`
- OOD causal `75.3`
- Attention `1.15 ms`
- Grad-CAM `2.30 ms`
- Retrieval proxy `2.35 ms`
- Modality ablation `3.98 ms`
- Integrated Gradients `5.62 ms`
- LIME `48.75 ms`
- KernelSHAP `142.66 ms`

### Justification

Le reviewer mentionne precisement une divergence entre table, texte et figure.
Mettre a jour le texte sans mettre a jour la figure laisserait l'incoherence
visible.

## 4. Incoherence des latences Table 2 / Table 7

### Probleme

La Table 2 utilise les anciennes latences:

| Methode | Table 2 actuelle | Source |
| --- | ---: | --- |
| Attention | 2.75 ms | ancien run single-seed |
| LIME | 111.54 ms | ancien run single-seed |
| KernelSHAP | 294.16 ms | ancien run single-seed |

La Table 7 utilise les nouvelles latences Captum 500 cas:

| Methode | Table 7 actuelle | Source |
| --- | ---: | --- |
| Attention | 1.15 ms | nouveau run 500 cas, lambda causal 4.00 |
| LIME | 48.75 ms | nouveau run 500 cas, lambda causal 4.00 |
| KernelSHAP | 142.66 ms | nouveau run 500 cas, lambda causal 4.00 |

### Decision recommandee

Mettre la Table 2 sur les memes valeurs que la Table 7. C'est l'option la plus
simple et la plus robuste.

### Remplacements Table 2 lignes 364-370

Remplacer les valeurs de la Table 2 par:

```latex
KernelSHAP & \textcolor{red}{\xmark} (142.66 ms) & \partialcmark ($\rho=1.00$) & \partialcmark (feature ranking) & \partialcmark (0.844 stability) & \textcolor{red}{\xmark} (no drift signal) & Slow perturbational attribution \\ \hline
LIME & \textcolor{red}{\xmark} (48.75 ms) & \partialcmark ($\rho=0.90$) & \partialcmark (local surrogate) & \partialcmark (0.826 stability) & \textcolor{red}{\xmark} (no lifecycle signal) & Hyperparameter sensitivity \\ \hline
Integrated Gradients & \partialcmark (5.62 ms) & \partialcmark ($\rho=1.00$) & \textcolor{red}{\xmark} (path attribution only) & \partialcmark (0.875 stability) & \textcolor{red}{\xmark} (no drift signal) & Baseline dependence \\ \hline
Grad-CAM & \textcolor{green}{\cmark} (2.30 ms) & \partialcmark (visual only) & \textcolor{red}{\xmark} (no CF) & \partialcmark (0.868 stability) & \textcolor{red}{\xmark} (leakage 0.056) & Metadata-induced heatmap shift \\ \hline
Attention & \textcolor{green}{\cmark} (1.15 ms) & \partialcmark (spurious token focus) & \textcolor{red}{\xmark} (no intervention) & \partialcmark (0.847 stability) & \textcolor{red}{\xmark} (no drift flag) & Spurious hospital-token focus\\ \hline
Retrieval proxy & \textcolor{green}{\cmark} (2.35 ms) & \partialcmark (purity 0.413/0.584) & \partialcmark (case support) & \textcolor{green}{\cmark} (explicit evidence) & \partialcmark (purity shift) & No generative grounding or report validation \\ \hline
Modality Ablation & \textcolor{green}{\cmark} (3.98 ms) & \textcolor{green}{\cmark} (direct intervention, $\rho=1.00$) & \textcolor{green}{\cmark} (counterfactual removal) & \partialcmark (0.827 stability) & \textcolor{green}{\cmark} (OOD masking reveals drift) & Intervention definition matters \\ \hline
```

### Remplacement de la note sous Table 2

Texte actuel:

```latex
Ratings are derived from the controlled benchmark described in Section 6 (Table 6). Absolute latencies are lower bounds measured on a lightweight architecture and should be rescaled for full-resolution clinical pipelines.
```

Remplacer par:

```latex
Ratings are derived from the final Captum-based measurements on the 500-case OOD subset reported in Table \ref{tab:controlled_methods}. Absolute latencies are lower bounds measured on a lightweight 28$\times$28 architecture and should be rescaled for full-resolution clinical pipelines.
```

### Justification

Cela supprime l'incoherence Table 2 / Table 7. Les deux tables parlent alors
du meme benchmark. C'est plus propre que d'expliquer deux benchmarks differents
dans le corps principal.

## 5. Alternative si vous voulez garder les anciennes latences en Table 2

Si vous preferez conserver Table 2 comme une table preliminaire, il faut le
dire explicitement dans la caption ou la note.

Ajouter sous Table 2:

```latex
Table \ref{tab:method_suitability} reports ACE suitability scores derived from the preliminary single-seed benchmark in `experiments/results/ace_experiment_results.json`. The final Captum-based measurements on the 500-case OOD subset are reported separately in Table \ref{tab:controlled_methods}; therefore, absolute latency values should not be compared across the two tables without considering this difference in evaluation protocol.
```

### Pourquoi je ne recommande pas cette option

Elle explique l'incoherence, mais elle ajoute de la complexite. Comme Table 2
est un tableau de synthese clinique, elle devrait reprendre les mesures finales
de Table 7 plutot qu'un ancien run preliminaire.

## 6. Mettre a jour la grande matrice de methodes

### Probleme

La grande matrice autour des lignes 810-864 contient aussi les anciennes
valeurs:

- KernelSHAP `294.16 ms`, stability `0.968`
- LIME `111.54 ms`, stability `0.951`
- IG `10.60 ms`, stability `0.959`
- Grad-CAM `3.76 ms`, stability `0.725`
- Attention `2.75 ms`, stability `0.929`

### Suggestion

Remplacer par les valeurs du run 500 cas:

| Methode | Latency | Stability |
| --- | ---: | ---: |
| KernelSHAP | 142.66 ms | 0.844 |
| LIME | 48.75 ms | 0.826 |
| Integrated Gradients | 5.62 ms | 0.875 |
| Grad-CAM | 2.30 ms | 0.868 |
| Cross-Modal Attention | 1.15 ms | 0.847 |
| Retrieval Proxy | 2.35 ms | -- |
| Modality Ablation | 3.98 ms | 0.827 |

### Justification

Un reviewer comparera probablement Table 2, la grande matrice et Table 7. Les
trois doivent utiliser les memes mesures, sauf si une difference de protocole
est annoncee explicitement.

## 7. Corriger les ratios de latence

### Probleme

Le texte dit:

```latex
KernelSHAP and LIME were respectively about 107$\times$ and 41$\times$ slower than attention
```

Avec les nouvelles valeurs:

```text
KernelSHAP / Attention = 142.66 / 1.15 = 124.1
LIME / Attention = 48.75 / 1.15 = 42.4
```

### Remplacement recommande

```latex
KernelSHAP and LIME were respectively about 124$\times$ and 42$\times$ slower than attention on the same model
```

### Autres lignes a harmoniser

Remplacer aussi les formulations du type:

```latex
41--107$\times$
41–107×
```

par:

```latex
42--124$\times$
```

### Justification

Les ratios doivent etre recalcules apres remplacement des latences. Sinon le
texte reste mathematiquement incoherent meme si les tables sont corrigees.

## 8. Claims "apres causal training" dans les explications

### Probleme

Ces affirmations ont maintenant ete relancees sur le protocole final 500 cas
avec `lambda=4.00` et des explications baseline + causal:

- hospital-feature importance `0.429 -> 0.279`
- hospital attention `0.410 -> 0.265`
- Grad-CAM leakage `0.056 -> 0.016`
- retrieval evidence purity `0.413 -> 0.584`

### Formulation recommandee

```latex
On the 500-case explanation subset with the final causal-invariance weight ($\lambda=4.00$), causal training reduced hospital-feature masking reliance from 0.429 to 0.279, hospital-token attention from 0.410 to 0.265, and Grad-CAM metadata leakage from 0.056 to 0.016, while retrieval evidence purity increased from 0.413 to 0.584.
```

### Justification

Les explications baseline et causal sont maintenant mesurees sur le meme
sous-ensemble OOD de 500 cas et avec le meme `lambda=4.00` que le meilleur
modele causal du benchmark de robustesse. La robustesse modele reste agregee
sur 20 seeds; l'analyse d'explication reste seed `7` avec stabilite mesuree
contre un modele seed `8` de la meme variante.

## 9. Mettre a jour `TRACEABILITY.md`

### Probleme

`TRACEABILITY.md` indique encore:

- Baseline OOD accuracy: `66.1%`
- Causal OOD accuracy: `82.9%`
- Attention latency: `2.75 ms/case`
- LIME latency: `111.54 ms/case`
- KernelSHAP latency: `294.16 ms/case`

### Suggestion

Remplacer par:

- Baseline OOD accuracy: `60.4%`
- Causal OOD accuracy: `75.3%`
- Baseline generalization gap: `0.252`
- Causal generalization gap: `0.091`
- Attention latency: `1.15 ms/case`
- LIME latency: `48.75 ms/case`
- KernelSHAP latency: `142.66 ms/case`

### Justification

Si le papier est audite avec les fichiers de reproductibilite, la traceability
matrix ne doit pas contredire le manuscrit.

## Checklist concrete pour `main.tex`

- [ ] Garder dans l'abstract: `0.604 -> 0.753` et `0.252 -> 0.091`.
- [ ] Remplacer dans la conclusion: `66.1\% -> 82.9\%` par `0.604 -> 0.753`.
- [ ] Remplacer `23.6 -> 2.9` par `0.252 -> 0.091`.
- [ ] Remplacer le cout ID `4.0 points` par `1.1 points`.
- [ ] Ajouter une note sous Table 6 sur le calcul du gap, ou ajouter la colonne `Full OOD Acc.`.
- [ ] Regenerer `ace_benchmark_overview.png` avec les nouveaux resultats.
- [ ] Mettre Table 2, Table 7 et la grande matrice sur les memes latences.
- [ ] Recalculer les ratios de latence: `124x` KernelSHAP, `42x` LIME.
- [ ] Remplacer les claims mecanistiques issus de l'ancien run par les valeurs finales 500 cas, `lambda=4.00`.
- [ ] Mettre a jour `TRACEABILITY.md`.
