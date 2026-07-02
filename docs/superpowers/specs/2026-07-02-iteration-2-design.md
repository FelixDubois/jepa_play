# JEPA Pinball — Itération 2 de la boucle world-model — Spécification de design

Date : 2026-07-02. État du projet au départ : tag v2.2, champion agent V2 à
40 % de victoires / 54 % des cibles (éval n=200 seeds appariées) vs V1 29 %
vs baselines 8-10 %. Poids du champion sur le Drive :
`checkpoints_targets_v2` (jepa.pt epoch 22 + 4 têtes).

## 1. Motivation et question posée

L'itération 1 (notebook 05) a montré que la boucle des world models —
*la politique améliore les données, les données améliorent la politique* —
rapporte +11 pts de victoires (~3σ). Les gains diminuent normalement à
chaque tour. L'itération 2 pose une question précise : **la boucle
tient-elle un deuxième tour ?**

Le résultat est publié quel que soit le verdict :

- **gain ≥ 8 pts** (≈2σ à n=200 apparié) → `git tag v3`, champion remplacé,
  notebook 06 repointable sur `checkpoints_targets_v3` ;
- **plateau ou régression** → documenté tel quel comme la leçon attendue
  des rendements décroissants ; pas de tag, le champion reste V2.

## 2. Décisions utilisateur (brainstorming 2026-07-02)

1. **Support** : nouveau notebook `07_iteration2` (`.py` + `.ipynb`
   jupytext, double-mode Colab/local). Le 05 reste intact — il raconte
   l'itération 1.
2. **Critère de succès** : documenter honnêtement (double sortie ci-dessus),
   pas de chasse au record.
3. **Évaluation** : n=200 seeds appariées, **V3 vs V2 uniquement** ; V1 et
   baselines cités des mesures n=200 du finisher au lieu d'être rejoués.
4. **Entraînement** : warm-start champion +6 epochs, avec **garde-fou
   d'érosion disqualifiant** (diagnostic n°4 avant toute éval).

## 3. Flux de données — le champion n'est jamais modifié

```
data/targets_v1  (100k, aléatoire)      checkpoints_targets_v2  (champion,
data/targets_v2  (50k, agent V1 mixte)                           LECTURE SEULE)
data/targets_v3  (50k, agent V2 mixte)   → copie jepa.pt →
                                         checkpoints_targets_v3  (écriture)
```

- Aucune suppression sur le Drive : l'itération 2 **ajoute** une génération
  (`data/targets_v3`, `checkpoints_targets_v3`), elle ne refait pas la
  précédente.
- Garde skip-if-exists sur la collecte ET la copie du checkpoint (pattern du
  05) ; message : « supprimer `data/targets_v3` ET `checkpoints_targets_v3`
  pour refaire l'itération ».

## 4. Les étapes du notebook 07

1. **Recharger l'agent V2 champion.** `load_jepa(CKPT_V2/"jepa.pt")` +
   4 têtes construites via `jepa.z_dim` (pattern V2.2, checkpoints
   auto-descriptifs), `MPCPlanner` 256 candidats.
2. **Collecte.** `MixedPolicy(agent_v2, rng)`, 50 000 transitions (symétrie
   avec l'itération 1), env et rng sur **seed neuf 456** (dispositions de
   cibles fraîches — 123 a servi à l'itération 1). Stat imprimée : durée
   moyenne des épisodes v1/v2/v3.
3. **Warm-start.** Copie de `jepa.pt` du champion vers `CKPT_V3` si absent ;
   `episodes_mixed = v1 + v2 + v3` (~200k transitions) ;
   `epochs = ckpt_epoch + 6` (22 → 28, reprise robuste du 05). Courbes
   pred/copy + variance latente, ligne verticale à l'epoch de reprise.
4. **Garde-fou érosion (diagnostic n°4).** Cellule reprise du notebook 03
   (sonde `PositionProbe` 400 pas AdamW sur latents gelés, validation =
   40 derniers épisodes du mélange — donc du jeu d'agent V2, là où la
   lisibilité compte). Verdict :
   - MAE ≤ 0.12 → on continue ;
   - MAE > 0.12 → **modèle disqualifié avant l'éval**, le notebook conclut
     sur l'érosion documentée (c'est un résultat, pas un échec du notebook).
   On imprime aussi `copy_mse` à h=8 pour suivre la glissade connue
   (0.0128 → 0.0096 au dernier réentraînement). Rappel calibrage : les runs
   de production lisent systématiquement plus haut que les seuils locaux
   (0.108 observé, sous l'alerte 0.12).
5. **Têtes V3 + éval.** `train_objective_heads(jepa_v3, episodes_mixed)`,
   sauvegarde des 4 `.pt` dans `CKPT_V3`, `MPCPlanner` V3. Puis
   `evaluate(env, pol, n_episodes=200)` sur **V3 et V2** — l'appariement des
   seeds est automatique (`seed0 + i`, même `seed0=1000` par défaut pour les
   deux). Instances `MPCPlanner` fraîches pour l'éval (le RNG d'agent_v2 a
   été consommé par la collecte — piège payé au 05). Graphique : barres de
   victoire V3/V2 + repères cités V1 29 % et baselines 8-10 %, annotation
   nudges.
6. **Lecture des résultats.** Double conclusion pré-écrite (§1). Seuil de
   significativité explicite dans le texte : à n=200 apparié, ~8 pts ≈ 2σ
   (l'itération 1 : +11 pts ≈ 3σ).

## 5. Code, tests, validation

- **Aucun code de bibliothèque nouveau.** `MixedPolicy`, `collect_dataset`,
  la reprise de `train_jepa`, `train_objective_heads`, `evaluate`, la sonde
  de lisibilité : tout existe et est couvert par les 90 tests.
- Donc **pas de nouveaux tests unitaires**. Vérifications à la livraison :
  suite complète verte, synchro jupytext `.py` ↔ `.ipynb`, AST propre,
  aucun constructeur de tête sans `z_dim` explicite.
- **Validation réelle = exécution du pipeline sur Colab** par l'utilisateur
  (comme pour les notebooks 01-06). Coût attendu : collecte ~20-40 min CPU,
  +6 epochs GPU courts, éval n=200 × 2 agents.

## 6. Hors périmètre

- Réentraînement from scratch (écarté au brainstorming : change deux
  variables, coûte plus, casse la narration warm-start ; reste l'issue de
  secours documentée si le garde-fou disqualifie).
- Score numérique par cible, curriculum `n_targets`, imagination
  contrefactuelle : autres suites possibles, non couvertes ici.
- Toute modification des notebooks 01-06 (le 06 se repointe sur v3 en
  changeant un chemin, sans édition de code).
