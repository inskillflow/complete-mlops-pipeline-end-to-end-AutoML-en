# Antisèche - Wine Quality MLOps

Fiche de référence rapide. Pour tous les détails, voir
[01-guide-complet.md](01-guide-complet.md).

---

## Démarrage en 3 commandes

```bash
docker compose up -d --build mlflow     # 1. serveur de suivi MLflow
docker compose run --rm trainer         # 2. entraîne 9 modèles (3 x 3)
docker compose up -d --build api ui     # 3. API + interface web
```

Puis arrêter :

```bash
docker compose down        # garde base + modèles
docker compose down -v     # tout supprimer
```

---

## Adresses (ports)

| Service | URL | Rôle |
| --- | --- | --- |
| Streamlit (UI) | http://localhost:8501 | interface web à cliquer |
| FastAPI (docs) | http://localhost:8000/docs | documentation interactive |
| MLflow | http://localhost:5000 | suivi des expériences |

---

## Les 4 services

| Service | Vit-il en continu ? | Rôle |
| --- | --- | --- |
| `mlflow` | oui | mémoire : runs, métriques, modèles |
| `trainer` | non (une fois) | entraîne les modèles, puis s'arrête |
| `api` | oui | charge les modèles et prédit |
| `ui` | oui | page web ; ne parle qu'à l'API |

Règle d'or : **UI -> API -> MLflow**. L'UI ne touche jamais MLflow directement.

---

## Les 3 modèles

| Modèle | Pénalité | Effet clé |
| --- | --- | --- |
| Ridge | L2 (carrés) | rétrécit tous les coefficients, jamais à zéro |
| Lasso | L1 (valeurs absolues) | met des coefficients à zéro (sélection) |
| ElasticNet | L1 + L2 | mélange des deux (`l1_ratio` = mélange) |

`alpha` = force de la régularisation. Plus grand = modèle plus simple.
Sur ce jeu de données, **Ridge gagne** (RMSE ~0,66).

---

## Métriques

| Métrique | Sens | Bon score |
| --- | --- | --- |
| RMSE | erreur typique (points de qualité) | bas |
| MAE | erreur absolue moyenne | bas |
| R2 | part de variance expliquée (0 à 1) | proche de 1 |

Le **champion** = le run avec le plus petit RMSE.

---

## Endpoints de l'API

| Méthode + chemin | Rôle |
| --- | --- |
| `GET /health` | MLflow joignable ? nombre d'expériences |
| `GET /experiments` | liste des expériences + nombre de runs |
| `GET /runs?experiment_name=...` | runs d'une expérience (triés par RMSE) |
| `GET /features` | min/max/moyenne/écart-type par variable |
| `GET /presets` | profil médian par niveau de qualité |
| `GET /model/{run_id}/coefficients` | coefficients du modèle |
| `POST /predict` | prédit la qualité (run_id + 11 variables) |

Exemple de prédiction :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"run_id":"RUN_ID","features":{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11,"total_sulfur_dioxide":34,"density":0.9978,"ph":3.51,"sulphates":0.56,"alcohol":9.4}}'
```

---

## Les 5 onglets de l'interface

1. **Home** : vue d'ensemble + chiffres clés.
2. **Data exploration** : histogrammes, corrélations, boxplots.
3. **Theory** : formules Ridge/Lasso/ElasticNet + biais-variance.
4. **MLflow comparison** : tableau, histogramme, radar, champion.
5. **Prediction** : curseurs, presets, jauge, coefficients.

---

## Dépannage express

| Symptôme | Solution |
| --- | --- |
| « No run found » | lancer le trainer, puis « Refresh data » |
| Prédiction : `No such file ... my_new_model_1` | vérifier le montage `./mlruns` dans `trainer` et `api` |
| Build : `Read timed out` | relancer la commande (internet lent) |
| « API unreachable » | vérifier `docker compose ps` + `/health` |
| Port occupé (5000/8000/8501) | changer le mappage dans `docker-compose.yml` |
| Avertissement Git de MLflow | sans danger, à ignorer |

---

## Commandes Docker utiles

```bash
docker compose ps                 # état des services
docker compose logs api           # logs d'un service
docker compose logs -f ui         # logs en direct (-f = follow)
docker compose run --rm trainer --alpha 0.3 --l1_ratio 0.5   # autres hyperparamètres
docker compose build api          # (re)construire une image
```

---

## Où sont mes données ?

- `database/mlflow.db` : métadonnées (runs, params, métriques).
- `mlruns/` : fichiers de modèles (artefacts).
- Les deux survivent à `docker compose down` (supprimés par `down -v`).
