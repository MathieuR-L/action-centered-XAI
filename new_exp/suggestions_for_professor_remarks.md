# Suggestions de révision pour `paper/main.tex`

Ce document liste, remarque par remarque, les corrections suggérées pour le brouillon journal `paper/main.tex`. Il ne modifie pas le manuscrit; il sert de plan de révision et de base pour rédiger les nouveaux paragraphes.

## Synthèse Prioritaire

1. Harmoniser partout les cinq actions ACE: **triage, diagnosis, treatment selection, audit, monitoring**.
2. Corriger l'incohérence majeure de la Table 1: la première ligne doit être **Triage**, pas **Intra-operative assistance**, car la question associée est "Is this case urgent?".
3. Ajouter une mini-section PRISMA/méthode de sélection avant ou au début de la section datasets/benchmarks.
4. Remplacer les résultats Section 6 fondés sur 12 cas OOD par les nouvelles expériences 500 cas, 20 seeds, grille de λ, bootstrap CI, McNemar/Wilcoxon.
5. Ajouter un paragraphe de reproductibilité: GitHub/Zenodo, versions, seeds, scripts, matériel.
6. Clarifier que les scénarios cliniques de Section 4 sont illustratifs sauf si une étude déployée est explicitement citée.
7. Ajouter fairness dans Table 1 et dans les métriques réglementaires.
8. Reconnaître explicitement l'absence d'évaluation clinicienne et proposer un protocole d'expert review.

## 1. Abstract

### Remarque

> Incohérence triage/intraoperative : corriger partout, fixer les 5 actions  
> Section 7.4 à remonter dans l'abstract : argument fort, modification légère

### Localisation dans `main.tex`

- Abstract: `paper/main.tex`, lignes 137-144.
- Les cinq actions sont déjà annoncées ligne 139 comme: triage, diagnosis, treatment selection, audit, monitoring.
- La conclusion et plusieurs sections reprennent aussi ces cinq actions, notamment lignes 1252-1259.

### Problème

L'abstract annonce cinq actions cohérentes, mais le corps du papier introduit parfois **intra-operative assistance** comme action ACE distincte. Cela crée une incohérence entre l'abstract, la Table 1 et les autres tableaux.

### Suggestion

Garder les cinq actions ACE stables partout:

- Triage
- Diagnosis
- Treatment selection
- Audit
- Monitoring

Mentionner l'intraoperative assistance uniquement comme **cas d'usage time-critical rattaché au triage**, pas comme sixième action ACE.

### Proposition de phrase pour l'abstract

Remplacer la phrase expérimentale actuelle, qui mentionne encore les anciens résultats 12 cas:

> Crucially, enforcing causal invariance reduces out-of-distribution accuracy loss from 23.6 to 2.9 percentage points...

par une version alignée avec les nouvelles expériences:

> In an expanded controlled benchmark with 500 OOD cases, 20 random seeds, and a grid of causal-invariance weights, causal training improved mean OOD accuracy from 0.604 to 0.753 and reduced the generalization gap from 0.252 to 0.091, with bootstrap confidence intervals and paired statistical tests supporting the robustness of the effect.

Ajouter ensuite une phrase courte issue de la Section 7.4 / discussion:

> These results reinforce the central ACE claim: MXAI methods should not be selected by explainer family alone, but by whether they satisfy the latency, causal validity, stability, fairness, and traceability requirements of the clinical action they support.

## 2. PRISMA et Méthodologie de Sélection

### Remarque

> PRISMA : ajouter section méthodologique de sélection  
> Section 5 (Datasets) PRISMA manquant : facile à ajouter, renforce la crédibilité du survey

### Localisation dans `main.tex`

- Introduction: lignes 171-197.
- Début de Section 5 Datasets and Benchmarks: lignes 885-893.
- Il n'y a pas actuellement de vraie méthode de sélection des articles/datasets.

### Problème

Le papier est présenté comme survey, mais la méthode de sélection de la littérature, des méthodes et des datasets n'est pas suffisamment explicite. Cela affaiblit la crédibilité du survey.

### Suggestion

Ajouter une courte section avant Section 5 ou juste après l'introduction, par exemple:

```latex
\subsection{Literature and Dataset Selection Protocol}
```

ou:

```latex
\subsection{Survey Selection Methodology}
```

### Contenu recommandé

Inclure:

- Bases consultées: PubMed, IEEE Xplore, ACM Digital Library, Scopus, arXiv, Google Scholar.
- Période: par exemple 2017-2026, avec priorité aux travaux 2022-2026.
- Requêtes: "multimodal explainable AI healthcare", "medical multimodal XAI", "clinical explainability", "foundation model explainability healthcare", "medical image text XAI".
- Critères d'inclusion:
  - méthodes MXAI en santé;
  - multimodalité explicite;
  - évaluation clinique, métrique d'explicabilité, benchmark, ou application médicale;
  - datasets et benchmarks utilisés pour des tâches multimodales.
- Critères d'exclusion:
  - XAI unimodal sans implication multimodale;
  - articles sans méthode ou évaluation exploitable;
  - travaux non médicaux sauf pertinence méthodologique directe.
- Résultat de sélection:
  - nombre initial d'articles;
  - nombre après déduplication;
  - nombre après screening titre/résumé;
  - nombre inclus.

### Proposition de formulation

> We followed a PRISMA-inspired selection protocol to identify MXAI methods, datasets, and evaluation frameworks relevant to healthcare. Searches were conducted in PubMed, IEEE Xplore, ACM Digital Library, Scopus, arXiv, and Google Scholar using combinations of "multimodal explainable AI", "medical XAI", "clinical decision support", "foundation model explainability", and "medical image-text learning". Studies were included when they addressed multimodal healthcare data, proposed or evaluated an explainability method, introduced a relevant benchmark, or provided clinical/regulatory evaluation criteria. Studies were excluded when they were purely unimodal, non-clinical without methodological relevance, or lacked sufficient methodological detail. The final corpus was organized by ACE action rather than by explainer family alone.

### Figure/table suggérée

Ajouter une petite table PRISMA au lieu d'une figure complexe si le temps manque:

| Stage | Count | Exclusion reason |
| --- | ---: | --- |
| Records identified | X | Database search |
| After duplicates removed | X | Duplicate records |
| Title/abstract screened | X | Not healthcare / not multimodal |
| Full-text assessed | X | No explainability or weak methodological detail |
| Included in survey | X | Final corpus |

## 3. Section 3 ACE: Harmonisation des 5 Actions

### Remarque

> 5 actions incohérentes : confirmé dans Table 1 → harmoniser systématiquement

### Localisation dans `main.tex`

- Section ACE: lignes 285-305.
- Table 1: lignes 291-305.
- Table 2: lignes 311-335.
- Clinical Action Suitability Matrix: lignes 740-850.
- Summary of gaps: lignes 867-880.

### Problème principal

La ligne 298 de Table 1 indique:

> Intra-operative assistance & "Is this case urgent?"

Cette association est incohérente. "Is this case urgent?" correspond à **triage**, pas à intraoperative assistance.

### Correction recommandée

Remplacer dans Table 1:

```latex
\textbf{Intra-operative assistance} & ``Is this case urgent?''
```

par:

```latex
\textbf{Triage} & ``Is this case urgent?''
```

Puis traiter l'intraoperative assistance comme un exemple dans la description:

> Time-critical workflows such as emergency triage or intraoperative assistance require low-latency explanations, but ACE uses "triage" as the canonical action category for urgent prioritization.

### Harmonisation recommandée

Utiliser ces labels partout:

- `Triage`
- `Diagnosis`
- `Treatment selection` ou `Treatment Choice`, mais choisir un seul terme.
- `Audit`
- `Monitoring`

Je recommande **Treatment selection** parce qu'il est déjà utilisé dans l'abstract et la conclusion.

### Table 1 suggérée

| Clinical Action | Explanation Question | Required Properties | Why Current XAI Fails |
| --- | --- | --- | --- |
| Triage | Is this case urgent? | Low latency, calibrated uncertainty, high sensitivity | Perturbational explainers are too slow; most methods lack calibrated uncertainty |
| Diagnosis | Why this disease? | Cross-modal causality, modality-specific fidelity, subgroup robustness | Attention can overweight spurious signals; saliency may leak across modalities |
| Treatment selection | Why this option rather than another? | Counterfactual validity, risk stratification, personalization | Standard saliency rarely answers intervention-style questions |
| Audit | Can this decision be defended? | Reproducibility, traceability, versioned evidence, stability | Most methods lack standardized audit trails and explanation CIs |
| Monitoring | Is the model still valid after deployment? | Drift detection, temporal consistency, recalibration triggers | Most explainers are static and lack lifecycle diagnostics |

## 4. Section 4 Méthodes: Clarifier Illustratif vs Déployé

### Remarque

> Confusion illustratif/déployé : les encadrés "Illustrative scenario, not deployed" sont déjà partiellement là, les généraliser

### Localisation dans `main.tex`

- Section Applications: lignes 641-727.
- Une clarification existe déjà lignes 649-651:
  > The following workflows are illustrative scenarios...
- Exemples à clarifier:
  - Breast cancer workflow: lignes 655-662.
  - Alzheimer's workflow: lignes 666-673.
  - SimonMed-Lunit example: lignes 675-682.
  - Personalized chemotherapy: lignes 693-700.
  - Laparoscopic surgery: lignes 704-711.

### Problème

Certains exemples ressemblent à des systèmes déployés alors qu'ils sont des scénarios illustratifs. Cela peut être critiqué comme surinterprétation clinique.

### Suggestion

Ajouter systématiquement un label au début de chaque workflow:

```latex
\textbf{Illustrative scenario, not deployed:}
```

Pour les cas réellement basés sur une annonce ou une étude:

```latex
\textbf{Publicly reported deployment example, not independently validated:}
```

### Application concrète

- Breast cancer: "Illustrative scenario, not deployed"
- Alzheimer's: "Illustrative scenario, not deployed"
- Personalized chemotherapy: "Illustrative scenario, not deployed"
- Laparoscopic surgery assistance: "Illustrative scenario, not deployed"
- SimonMed-Lunit: "Publicly reported deployment example, not independently validated"

### Phrase transversale à ajouter

> Unless explicitly described as a reported deployment or clinical study, the workflows in this section are illustrative scenarios used to show how MXAI methods could support clinical actions. They should not be interpreted as evidence of clinical deployment or validated patient benefit.

## 5. Section 5 Datasets: PRISMA et Tableau Datasets

### Remarque

> PRISMA manquant : facile à ajouter, renforce la crédibilité du survey

### Localisation dans `main.tex`

- Section Datasets: lignes 885-950.
- Table datasets: lignes 909-928.

### Problème

La section liste des datasets mais n'explique pas comment ils ont été sélectionnés. Certains ajouts récents sont aussi mélangés dans une ligne `\textcolor{re}{...}` aux lignes 903-904, qui devrait être nettoyée.

### Suggestions

1. Ajouter un paragraphe "Dataset selection criteria" avant la liste.
2. Expliquer pourquoi chaque dataset est pertinent pour ACE.
3. Nettoyer l'entrée SMMILE/PairDx lignes 903-904: actuellement SMMILE et PairDx sont fusionnés dans une seule puce.
4. Ajouter une colonne optionnelle dans Table `tab:datasets`: "Selection rationale" ou "ACE relevance".

### Proposition de formulation

> Datasets were selected when they satisfied at least one of three criteria: (i) they are widely used in multimodal medical AI/XAI research, (ii) they support evaluation of cross-modal reasoning or explanation fidelity, or (iii) they represent an ACE-relevant clinical action such as diagnosis, audit, or monitoring. This selection is not intended as an exhaustive catalogue of all medical datasets, but as a structured set of representative resources for evaluating action-centered MXAI.

## 6. Section 6 Benchmark: Remplacer les Anciennes Expériences

### Remarques

> 12 cas OOD : possible d'augmenter à 200-500 ?  
> Un seul λ : grille λ obligatoire  
> Une seule seed : minimum 5 seeds  
> Pas de CI : bootstrap faisable  
> Pas de tests statistiques : McNemar/Wilcoxon faisables  
> Chemin local : remplacer par GitHub/Zenodo

### Localisation dans `main.tex`

- Benchmark design: lignes 953-970.
- Ancienne mention des 12 OOD cases: lignes 968-969.
- Table robustness: lignes 974-984.
- Table methods: lignes 995-1015.
- Discussion ancienne: lignes 1017-1027.

### Nouveaux résultats disponibles

Les nouveaux résultats sont dans:

- `new_exp/section6_benchmark_500_20seeds/section6_tables_500.md`
- `new_exp/section6_benchmark_500_20seeds/summary_by_lambda.csv`
- `new_exp/section6_benchmark_500_20seeds/mcnemar_tests.csv`
- `new_exp/section6_benchmark_500_20seeds/wilcoxon_tests.csv`
- `new_exp/section6_benchmark_500_20seeds/lambda_grid_ood_accuracy.png`
- `new_exp/section6_benchmark_500_20seeds/lambda_grid_generalization_gap.png`
- `new_exp/section6_explainer_500/section6_explainer_500_table.md`

### Résultats clés à intégrer

- OOD cases: 500.
- Seeds: 20.
- λ grid: 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00.
- Bootstrap repetitions: 5000.
- Baseline OOD accuracy: 0.604.
- Best causal OOD accuracy: 0.753 at λ = 4.00.
- Baseline generalization gap: 0.252.
- Best causal generalization gap: 0.091.
- McNemar at λ = 4.00: p < 1e-300.
- Wilcoxon at λ = 4.00 across 20 seeds: p = 0.000235.

### Correction de texte recommandée

Remplacer lignes 968-970:

> measured explanations on 12 OOD test cases...

par:

> We evaluated robustness on a 500-case OOD subset across 20 random seeds and swept the causal-invariance weight λ over nine values (0.10 to 4.00). Confidence intervals were estimated with 5000 bootstrap resamples. Paired model comparisons used exact McNemar tests on pooled case-level correctness and Wilcoxon signed-rank tests across seeds.

### Nouvelle Table Robustness suggérée

| Model | λ | ID Acc. | OOD Acc. 500 | 95% Bootstrap CI | Generalization Gap |
| --- | ---: | ---: | ---: | --- | ---: |
| Baseline | 0.00 | 0.854 | 0.604 | [0.549, 0.662] | 0.252 |
| Causal invariance | 4.00 | 0.843 | 0.753 | [0.723, 0.777] | 0.091 |

### Nouvelle Table Lambda Grid suggérée

Utiliser la table complète de `section6_tables_500.md`.

### Nouvelle Table Explainers suggérée

Depuis `section6_explainer_500_table.md`:

| Method | Latency ms | Seed stability | Causal alignment | Grad-CAM leakage |
| --- | ---: | ---: | ---: | ---: |
| Attention | 1.06 | 0.936 | 0.700 | -- |
| Integrated Gradients | 5.31 | 0.933 | 0.800 | -- |
| LIME | 47.58 | 0.898 | 0.800 | -- |
| KernelSHAP | 126.61 | 0.904 | 0.900 | -- |
| Modality ablation | 3.69 | 0.931 | 0.900 | -- |
| Grad-CAM | 2.16 | 0.835 | -- | 0.080 |
| Retrieval proxy | 3.41 | -- | -- | -- |

### GitHub/Zenodo

Remplacer le chemin local:

> implemented in the workspace (`experiments/mxai_assessment.py`)

par:

> implemented in the public reproducibility repository [GitHub URL to be inserted] and archived on Zenodo [DOI to be inserted].

Ajouter une phrase:

> The archive includes scripts, seeds, environment versions, raw predictions, summary tables, and figure-generation code.

## 7. Section 7 Métriques: FDA → Métrique → Seuil

### Remarque

> Tableau FDA→métrique→seuil : renforce le positionnement réglementaire

### Localisation dans `main.tex`

- Regulatory compliance metrics: lignes 1053-1104.
- Table FDA actuelle: lignes 1062-1100.

### Problème

La table actuelle traduit FDA → question → métrique → action, mais elle ne propose pas de **seuils opérationnels**. Pour un positionnement réglementaire fort, il faut ajouter une colonne "Suggested threshold / reporting target".

### Suggestion

Modifier Table `tab:fda_operational_metrics` pour ajouter une colonne:

```latex
\textbf{Suggested Reporting Threshold}
```

### Exemple de seuils prudents

| FDA requirement | Metric | Suggested threshold/reporting target |
| --- | --- | --- |
| Explainability validation | Clinician comprehension, time-to-interpretation | Report median and IQR; predefine task-specific non-inferiority criterion |
| Bias documentation | Subgroup performance and explanation stability gap | Report subgroup gaps; flag any clinically meaningful disparity |
| Uncertainty quantification | Prediction CI, explanation variance | Report 95% CI and abstention behavior |
| Lifecycle monitoring | Update-to-update explanation drift | Predefine drift trigger and revalidation threshold |
| Traceability | Versioned explanation record | 100% reproducible decision logs for audited cases |

### Fairness à intégrer ici aussi

Ajouter:

> Fairness should be evaluated not only at the predictive level but also at the explanation level: the same clinical evidence should not receive systematically different attribution across demographic groups, acquisition sites, or protected subgroups without clinical justification.

## 8. Section 7.4 à Remonter dans l'Abstract

### Remarque

> Section 7.4 à remonter dans l'abstract : argument fort, modification légère

### Localisation probable

La section actuelle "ACE Cross-Dataset Stress-Test Benchmark" commence ligne 1124. L'argument fort apparaît surtout lignes 1154-1164: la performance des méthodes dépend de l'action clinique et du domaine, pas d'un classement global unique.

### Suggestion

Ajouter dans l'abstract une phrase synthétique:

> Across both the controlled OOD benchmark and the six-domain ACE stress-test suite, no explainer dominated all clinical actions: low-latency methods were strongest for triage-like workflows, while intervention-based or perturbational methods were more informative for diagnosis, treatment reasoning, and audit.

Cette phrase résume l'apport de Section 7.4 sans allonger excessivement l'abstract.

## 9. Section 8 Challenges: Fairness Insuffisante

### Remarque

> Fairness insuffisante : ajouter une question fairness par action dans Table 1

### Localisation dans `main.tex`

- Table ACE: lignes 291-305.
- Challenges and future directions: lignes 1166-1211.
- Regulatory table bias row: lignes 1078-1081.

### Problème

La fairness est mentionnée dans l'introduction et dans les challenges, mais elle n'est pas intégrée dans l'architecture ACE elle-même.

### Suggestion 1: Ajouter une colonne à Table 1

Ajouter:

```latex
\textbf{Fairness Question}
```

### Questions fairness par action

| Action | Fairness question |
| --- | --- |
| Triage | Are urgent cases prioritized consistently across demographic groups and acquisition sites? |
| Diagnosis | Are the same clinical findings explained similarly across subgroups? |
| Treatment selection | Are counterfactual treatment explanations equally reliable across patient groups? |
| Audit | Can subgroup-specific explanation failures be traced and justified? |
| Monitoring | Does explanation drift appear earlier or more strongly in specific subgroups or sites? |

### Suggestion 2: Ajouter un paragraphe Section 8

> A key limitation of current MXAI evaluation is that fairness is usually measured only at the prediction level. Under ACE, fairness must also be action-specific and explanation-specific. For triage, fairness concerns whether urgent cases are prioritized consistently across subgroups; for diagnosis, whether equivalent clinical evidence receives equivalent explanatory weight; for treatment selection, whether counterfactual recommendations remain reliable across patient groups; for audit, whether subgroup-specific failures are traceable; and for monitoring, whether explanation drift is detected across sites and populations.

## 10. Section 9 Discussion: Absence d'Évaluation Cliniciens

### Remarque

> Pas d'évaluation cliniciens : impossible à corriger maintenant, mais reconnaître explicitement comme future work : proposer un protocole d'expert review

### Localisation dans `main.tex`

- Discussion: lignes 1213-1248.
- Important open questions: lignes 1233-1247.s
- La ligne 1164 reconnaît déjà que le benchmark ne remplace pas reader studies, mais il faut le rendre plus explicite dans la discussion.

### Problème

Le benchmark est technique. Il ne valide pas que les explications sont réellement utiles pour les cliniciens.

### Suggestion

Ajouter une sous-section courte dans Discussion:

```latex
\subsection{Clinical Expert Review as Future Validation}
```

### Protocole proposé

Inclure un protocole réaliste:

1. Recruter 3-5 cliniciens par domaine ou au minimum deux experts pour une étude pilote.
2. Présenter des cas avec prédiction + explication, randomisés par méthode.
3. Évaluer:
   - compréhension de l'explication;
   - confiance calibrée;
   - temps d'interprétation;
   - capacité à détecter un artefact/spurious cue;
   - utilité pour l'action ACE ciblée.
4. Mesurer accord inter-rater.
5. Comparer les méthodes sur des scénarios triage, diagnosis, treatment, audit, monitoring.

### Proposition de paragraphe

> The present benchmark does not include clinician reader studies or prospective workflow evaluation. Therefore, the results should be interpreted as technical stress tests of ACE-relevant properties rather than evidence of clinical utility. A natural next step is an expert-review protocol in which clinicians assess randomized model explanations for representative ACE actions, rating interpretability, actionability, trust calibration, time-to-interpretation, and ability to detect spurious cues. Such a protocol would connect the technical metrics reported here to clinical usefulness and would help define action-specific acceptance thresholds.

## 11. Reproductibilité

### Remarque

> GitHub/Zenodo/seeds/versions : correction simple, indispensable pour publication

### Localisation dans `main.tex`

- Section 6 starts with local workspace reference: lignes 956-958.
- No dedicated reproducibility paragraph currently.

### Suggestion

Ajouter une sous-section dans Section 6:

```latex
\subsection{Reproducibility}
```

ou une phrase finale avant Section 7.

### Éléments à inclure

- GitHub repository URL.
- Zenodo DOI.
- Exact seeds:
  - `7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137, 147, 157, 167, 177, 187, 197`
- Lambda grid:
  - `0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00`
- OOD subset size:
  - `500`
- Bootstrap repetitions:
  - `5000`
- Hardware:
  - RTX 4050 Laptop GPU
- Python/PyTorch versions:
  - Python 3.12 environment created via `uv`
  - Torch 2.12.0+cu130
- Raw predictions and generated figures archived.

### Proposition de paragraphe

> All benchmark scripts, raw predictions, summary tables, figure-generation code, seeds, and environment specifications will be released in a public GitHub repository and archived on Zenodo. The expanded Section 6 benchmark uses 20 random seeds, a nine-value λ grid, 500 OOD cases, and 5000 bootstrap repetitions. The archive reports package versions, hardware details, and raw case-level predictions to support exact reproduction of the McNemar, Wilcoxon, and confidence-interval analyses.

## 12. Nettoyage Technique du Manuscrit

### Points à corriger pendant la révision

Ces points ne faisaient pas tous partie des remarques, mais ils apparaissent dans `main.tex` et risquent de gêner la soumission:

- Ligne 947: il y a une accolade fermante en trop avant `\newline`:
  ```latex
  benchmarks.}\newline
  ```
- Ligne 703: `\textcolor{For instance...}` est syntaxiquement incorrect car `\textcolor` exige une couleur.
- Plusieurs `\textcolor{color}{...}` ou `\textcolor{re}{...}` doivent être supprimés ou remplacés par du texte normal avant soumission.
- Ligne 865: "As conlusion" → "In conclusion".
- Ligne 1030: "failling" → "failing".
- Ligne 511: "on of the most promising current approach" → "one of the most promising current approaches".
- Les références à "Table 5" et "Table 6" lignes 972-973 devraient utiliser `\ref{...}` au lieu de numéros fixes.

## 13. Ordre de Révision Recommandé

1. Corriger l'ontologie ACE: cinq actions stables partout.
2. Mettre à jour Abstract avec les nouveaux résultats 500 OOD / 20 seeds.
3. Ajouter PRISMA / sélection méthodologique.
4. Remplacer Section 6 avec nouvelles tables, CIs et tests.
5. Ajouter reproductibilité GitHub/Zenodo.
6. Ajouter fairness dans Table 1 et Section 8.
7. Généraliser les labels "Illustrative scenario, not deployed".
8. Ajouter le protocole futur d'expert review dans Discussion.
9. Nettoyer les erreurs LaTeX et labels provisoires.
