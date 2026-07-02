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

## 6bis. Amendement — run 1 Colab : garde-fou déclenché, issue de secours intégrée (2026-07-02 soir)

Le run 1 sur Colab a déclenché le garde-fou : **warm-start disqualifié**
(lisibilité 0.135 > 0.12 après 22 → 28 epochs), alors que le diag n°3
restait sain (pred sous copy ×4,1) et que toutes les courbes
d'entraînement « s'amélioraient » (baseline copie fondant 0.0092 → 0.0072 —
la signature epoch par epoch de l'érosion). La collecte est un succès
acquis : 832 épisodes champion, durée moyenne 4,0 s (vs 2,8 s agent V1,
2,1 s aléatoire), `data/targets_v3` réutilisable.

Décision utilisateur : activer l'issue de secours prévue au §4.4, enrichie
d'une contre-expertise. Amendements au notebook 07 :

1. **Contre-expertise** (nouvelle cellule) : le seuil 0.12 a été calibré sur
   du jeu aléatoire, la validation ici est du jeu de champion — l'encodeur
   du champion (epoch 22) lit donc les MÊMES 40 épisodes avec le même
   protocole. S'il lit nettement mieux que 0.135, l'érosion est confirmée ;
   s'il lit pareil, c'est le seuil qui est mal calibré pour cette
   distribution.
2. **Issue de secours intégrée** (nouvelle section) : si le warm-start est
   disqualifié, réentraînement from scratch 10 epochs (recette notebook 03)
   sur v1+v2+v3 dans un dossier NEUF `checkpoints_targets_v3_scratch`,
   re-vérifié au même garde-fou. Têtes et éval tournent sur `jepa_final`
   (le candidat survivant) ; l'assert final ne bloque que si AUCUN candidat
   ne passe.
3. **Reprise épinglée au champion** : `epochs=epoch_v2 + 6` au lieu de
   `ckpt_epoch + 6` — re-exécuter le notebook n'empile plus 6 epochs
   d'érosion à chaque passage (piège découvert en préparant le re-run).
4. `readability_mae(model, train, val)` factorisée (elle sert trois fois :
   warm-start, champion, from scratch).

## 6ter. Résultat partiel acquis (run 1)

Le warm-start répété N'EST PAS une recette d'itération viable sur cette
architecture : +6 epochs sur un checkpoint mûr (22) suffisent à pousser la
lisibilité au-delà du seuil, pendant que les métriques d'entraînement
s'améliorent — validation directe de la raison d'être du diagnostic n°4.

## 7. Hors périmètre

- ~~Réentraînement from scratch~~ — écarté au brainstorming comme voie
  principale, puis INTÉGRÉ comme issue de secours par l'amendement §6bis
  (le garde-fou a disqualifié le warm-start au run 1).
- Score numérique par cible, curriculum `n_targets`, imagination
  contrefactuelle : autres suites possibles, non couvertes ici.
- Toute modification des notebooks 01-06 (le 06 se repointe sur v3 en
  changeant un chemin, sans édition de code).
