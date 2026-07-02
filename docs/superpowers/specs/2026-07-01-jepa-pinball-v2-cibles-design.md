# JEPA Pinball V2 « Cibles » — Spécification de design

**Date** : 2026-07-01
**Statut** : validé par l'utilisateur (conversation du 2026-07-01, soir)
**Complète** : la spec V1 (`2026-07-01-jepa-pinball-design.md`) — les sections
non mentionnées ici restent valables.

## 1. Motivation

L'itération 1 (V1.5) a diagnostiqué un **reward hacking** : sous l'objectif
« éviter de perdre la balle », l'agent a appris à immobiliser la balle sur les
flippers levés (agent 16,0 s / 3,5 nudges ≈ « toujours appuyé » 14,4 s / 3,2).
La V2 change l'objectif du jeu et, par construction, tue cet exploit :

- **des cibles à toucher, placées aléatoirement à chaque partie** → la vision
  devient indispensable (aucune mémorisation de table possible), et une balle
  piégée ne touche rien ;
- **un objectif de hauteur continu** → être bas (= piégé) coûte, à chaque pas ;
- **victoire = toutes les cibles touchées** → l'épisode a une fin heureuse, la
  métrique reine devient le taux de victoire.

S'y ajoute la demande de **visualiser les prédictions** du world model, en
superposition avec le réel.

## 2. Décisions utilisateur

- Cible touchée → **elle disparaît** (physiquement et à l'image) ; quand toutes
  sont touchées → **fin d'épisode en victoire**.
- Hauteur : objectif **continu et proportionnel** (pas de seuil).
- 1 à 3 cibles par épisode, tirées aléatoirement, **jamais près des flippers**.
- Notebooks : **double-mode** Colab (recommandé, GPU T4) + local (CPU possible).

## 3. Mécanique des cibles (validée par prototype, 2026-07-01)

- Plots circulaires **pleins**, rayon 26, élasticité 1,2, dessinés en gris 110
  au rendu 64×64 (≈ 3 px : nettement distincts de la balle, blanche et plus
  petite — visibilité vérifiée à l'image).
- Placement à `env.reset()` : n ∈ {1, 2, 3} uniforme ; positions uniformes dans
  la zone sûre x ∈ [70, 470], y ∈ [420, 860] (les flippers/slingshots culminent
  à ~350) ; séparation minimale de 100 entre cibles ; tirage par le RNG de
  l'env (déterministe à seed fixée).
- **Détection de contact par distance** (robuste, indépendante de l'API de
  collision pymunk) : après chaque pas de contrôle, un contact est compté si
  dist(balle, cible) ≤ r_balle + r_cible + 2. La cible touchée est retirée de
  l'espace physique et du rendu.
- Équilibrage mesuré (table dure, politique aléatooire, 120 épisodes) : 10 %
  des cibles touchées par hasard, ≥1 contact dans 21 % des épisodes, 2 %
  de victoires chanceuses, aucun nouveau point de blocage.

- **Amendement saillance (2026-07-02, validé par expérience contrôlée)** : les
  cibles brillantes cannibalisaient l'encodage de la balle (sonde de position :
  MAE 0,124 avec cibles vs 0,051 sans ; prédicteur battu par la baseline
  copie ; s'aggrave avec la durée d'entraînement — MAE 0,137 à 2× plus
  d'epochs). Remède mesuré : balle rendue à 3 px (plancher) et cibles gris 110
  → MAE 0,058 et prédicteur à nouveau meilleur que la copie. Conséquence :
  l'observation change — datasets et checkpoints antérieurs obsolètes.
  Recommandation : rester à ~10 epochs (notebook 03) et surveiller le
  diagnostic n°3.

## 4. Environnement

- `info` gagne : `targets_total`, `targets_hit` (cumul), `hit_now` (nb de
  contacts ce pas), `completed` (bool).
- `done = ball_lost OU stuck OU steps ≥ max OU completed`.
- L'observation reste l'image seule (64×64×2) : les cibles n'y sont connues
  QUE par les pixels.

## 5. Têtes d'objectif et coût de planification

Toutes entraînées sur les latents `encode_target` (l'espace des prédictions),
encodeur gelé, labels auto-générés :

| Tête | Sortie | Label |
|---|---|---|
| danger (honnête) | P(fin PERDANTE dans k pas) | queue des épisodes finissant `ball_lost` OU `stuck` — les fins en victoire ne sont PAS dangereuses |
| hauteur | ŷ ∈ [0, 1] (régression) | `ball_pos.y / height` |
| cible | P(contact dans k pas) | fenêtres de k pas précédant un `hit` |
| sonde position | (x̂, ŷ) normalisés | `ball_pos` (pour la visualisation uniquement) |

**Coût MPC** (par pas imaginé, somme sur l'horizon 8) :
`w_d·σ(danger(ẑ)) − w_h·hauteur(ẑ) − w_t·σ(cible(ẑ))`, défauts
w_d = 1,0 ; w_h = 0,5 ; w_t = 2,0 — réglables dans le notebook 04.
`MPCPlanner` reste rétro-compatible (têtes hauteur/cible optionnelles).

## 6. Visualisation des prédictions (nouveau notebook 06)

1. **Superposition trajectoire** : sur une fenêtre réelle (o_t, actions
   exécutées), dérouler le prédicteur 8 pas, sonder chaque ẑ avec la tête
   position, dessiner les positions prédites (marqueurs colorés, dégradé
   d'horizon) SUR les frames réelles → GIF/planche « prédit vs réel ».
2. **Décodeur d'imagination** : petit déconv `z̄ → image 64×64` entraîné à part
   (encodeur gelé, MSE pixels, ~10 min GPU — le décodeur ne sert QU'À la
   visualisation, JEPA n'apprend jamais en pixels). Affichages : réel |
   décodé(ẑ) côte à côte, et superposition RGB (rouge = prédit, vert = réel,
   jaune = accord).

## 7. Données et évaluation

- Nouveaux dossiers : `data/targets_v1`, `checkpoints_targets` (Drive :
  `MyDrive/jepa_pinball/...`) — les datasets précédents sont obsolètes (les
  cibles doivent être dans les pixels).
- Shards : + `hits` (T,) uint8 par pas, + `targets_total`, `completed` par
  épisode (format additif).
- `evaluate` gagne : `completion_rate`, `mean_height`, `targets_hit_rate`
  (clés additives). Métrique reine : **taux de victoire**, puis temps moyen
  de complétion, hauteur moyenne, nudges.
- Baselines inchangées (aléatoire, toujours appuyé, flapper) + agent V1-style
  (danger seul) comme témoin de l'apport des nouvelles têtes.

## 8. Hors périmètre V2

Multi-balles, score numérique par cible (toutes valent 1), curriculum sur n,
transfert de table. La boucle d'itération (notebook 05) reste applicable telle
quelle après la V2 — elle enrichira notamment les labels de la tête cible.

## 9. Amendement V2.2 — architecture BIG+GAMMA (2026-07-02, expérience contrôlée)

Constat post-revalidation : trajectoires imaginées décalées, bras figés dans
l'imagination, lisibilité de la balle en production (0,13) inférieure au
potentiel mesuré (0,058) — l'érosion du contenu dynamique continue avec les
pas d'optimisation, même après le correctif de saillance.

Expérience (5 variantes, protocole local identique) : grossir seul ne change
rien ; grille spatiale 8×8 rejetée ; **la pondération des horizons courts
(γ=0,7 sur la perte de rollout) améliore tout** — balle 0,054→0,041, bras
0,114→0,059, trajectoire déroulée h=1 0,064→0,046, variance dynamique ×2,8.
Décision : **BIG+GAMMA** — encodeur (48, 96, 192, 384), z=384, prédicteur
768, a_dim 64, γ=0,7 par défaut. Checkpoints auto-descriptifs (hparams
embarqués, chargement reconstruisant l'architecture) ; anciens checkpoints
rejetés avec message clair ; datasets inchangés (l'observation ne bouge pas).
Nouveau diagnostic n°4 au notebook 03 : lisibilité de la balle par sonde
(alerte si > 0,12).
