# 00 - Run it step by step (copy, paste, done)

> This is the **most literal** guide. Do the commands **one by one**, from top to
> bottom. After each command there is a short note telling you what should
> happen and, when needed, **what to open in your browser**.
>
> You only need **Docker Desktop** installed and running. You do NOT need Python.
>
> For explanations of *why*, read [01-complete-guide.md](01-complete-guide.md).

---

## Before you start

1. Open **Docker Desktop** and wait until it says it is running.
2. Open a terminal:
   - Windows: open **PowerShell**.
   - macOS / Linux: open **Terminal**.
3. Go into the project folder (the one that contains `docker-compose.yml`).

Type this (change the path to where the project is on your machine):

```powershell
cd "C:\Users\Haythem\Downloads\Compressed\processus-et-ecosystemes-ia-main\processus-et-ecosystemes-ia-main\solutions-lab\14-mlflow-step-by-step-recap-multiple-experiments-elasticnet-ridge-lasso-fastapi-streamlit"
```

Check you are in the right place:

```powershell
ls
```

> You should see `docker-compose.yml`, and the folders `api`, `ui`, `trainer`,
> `mlflow`, `data`. If you do NOT see them, you are in the wrong folder. Fix the
> `cd` path above before continuing.

---

## PART A - Destroy everything (clean slate)

Do this if you have run the project before and want to start 100% fresh. If this
is your very first time, you can still run these commands safely.

### A.1 Stop and remove all containers, networks, and volumes

```powershell
docker compose down -v
```

> This stops the 4 services and deletes their containers and named volumes.
> It is safe. If nothing was running, it just prints a few lines and finishes.

### A.2 Delete the old database (MLflow metadata)

```powershell
Remove-Item -Recurse -Force database -ErrorAction SilentlyContinue
```

> This removes the `database/` folder that holds `mlflow.db`. Deleting it erases
> the list of past experiments and runs. That is what "clean slate" means.

### A.3 Delete the old artifacts (saved models)

```powershell
Remove-Item -Recurse -Force mlruns -ErrorAction SilentlyContinue
```

> This removes the `mlruns/` folder that holds the saved model files.

### A.4 (Optional) Remove the built images to rebuild from zero

```powershell
docker image rm mlops/mlflow-recap:latest mlops/trainer-recap:latest mlops/api-recap:latest mlops/ui-recap:latest
```

> This deletes the 4 project images. The next build will recreate them from
> scratch. If an image does not exist, Docker prints an error for that line only;
> that is fine, keep going.

### A.5 Confirm nothing is left running

```powershell
docker compose ps
```

> You should see an empty list (only a header, no rows). Everything is now clean.

---

## PART B - Build and run, one command at a time

### B.1 Start the MLflow tracking server

```powershell
docker compose up -d --build mlflow
```

> The first time, this downloads and installs libraries. It can take a few
> minutes. If it fails with a network error like `Read timed out`, just run the
> exact same command again.

### B.2 Check that MLflow is healthy

```powershell
docker compose ps
```

> Wait until you see `mlflow-recap-11 ... Up ... (healthy)`. If it says
> `(health: starting)`, wait ~15 seconds and run the command again.

### B.3 Open MLflow in your browser

**In your browser, type this address and press Enter:**

```text
http://localhost:5000
```

> You should see the MLflow web page. It is **empty** for now (only a `Default`
> experiment). That is normal: we have not trained anything yet.

### B.4 Train the models (creates 9 runs)

```powershell
docker compose run --rm trainer
```

> The first time, this also installs libraries (a few minutes). Then it prints
> the results of 9 runs, ending with lines like:
>
> ```text
> ========== Experiment: exp_multi_Ridge ==========
>   >>> run3.1  Ridge({'alpha': 0.4})  RMSE=0.6612  MAE=0.5081  R2=0.3805
> ...
> Recent active run name : run3.1
> ```
>
> When it finishes, the trainer container stops by itself. That is expected.

### B.5 Refresh MLflow in your browser

**Go back to your browser tab and refresh (press F5) this address:**

```text
http://localhost:5000
```

> Now you should see **three experiments**: `exp_multi_EL`, `exp_multi_Ridge`,
> `exp_multi_Lasso`, each with **3 runs**. Click one to explore its metrics.

### B.6 Start the API and the UI

```powershell
docker compose up -d --build api ui
```

> This builds and starts the last two services. The first build installs
> libraries; later runs are fast because Docker reuses the cache.

### B.7 Check that all services are up

```powershell
docker compose ps
```

> You should now see **three** running services: `mlflow-recap-11`,
> `api-recap-14`, and `ui-recap-14`, all `Up`.

### B.8 Test the API from the terminal (optional but nice)

```powershell
curl.exe http://localhost:8000/health
```

> You should see something like:
> `{"status":"ok","tracking_uri":"http://mlflow:5000","experiment_count":4}`

### B.9 Open the API documentation in your browser

**In your browser, type this address and press Enter:**

```text
http://localhost:8000/docs
```

> You should see the interactive Swagger page listing all endpoints. You can
> click "Try it out" on any endpoint to test it.

### B.10 Open the Streamlit interface in your browser

**In your browser, type this address and press Enter:**

```text
http://localhost:8501
```

> This is the main app. Go through the 5 tabs at the top:
> Home, Data exploration, Theory, MLflow comparison, Prediction.

### B.11 Make your first prediction (in the browser)

1. Click the **Prediction** tab.
2. (Optional) Click a preset button such as **Quality 5 (681 wines)**.
3. Leave the model on the champion (already selected).
4. Click **Predict quality**.

> You should see a predicted number (for example `5.06`) and a gauge. Below it,
> a bar chart shows which features pushed the score up or down.

---

## PART C - Stop everything when you are done

### C.1 Stop the services but keep your data

```powershell
docker compose down
```

> This stops and removes the containers. Your `database/` and `mlruns/` folders
> stay, so next time you can skip training and go straight to `up`.

### C.2 OR stop and erase everything

```powershell
docker compose down -v
```

> Use this only if you want a full reset. Next time you must train again
> (Part B.4).

---

## Quick recap (the happy path, no destroy)

If your machine already built the images and you just want to run it again:

```powershell
docker compose up -d mlflow
docker compose run --rm trainer      # only if database/ was erased
docker compose up -d api ui
```

Then open in the browser, in this order:

```text
http://localhost:5000     (MLflow)
http://localhost:8000/docs (API docs)
http://localhost:8501     (the app)
```

---

## If something goes wrong

| What you see | What to do |
| --- | --- |
| Build error `Read timed out` / `Name or service not known` | Run the exact same command again (slow internet). |
| Browser: "can't reach this page" on 5000/8000/8501 | Wait a bit, then `docker compose ps` to check the service is `Up`. |
| App says "No run found" | You skipped training. Run `docker compose run --rm trainer`, then click "Refresh data" in the app sidebar. |
| Prediction error about `my_new_model_1` | Reset with Part A, then redo Part B in order. |
| Port already in use | Another program uses 5000/8000/8501. Close it, or edit the port in `docker-compose.yml`. |

For deeper explanations, open [01-complete-guide.md](01-complete-guide.md)
(English) or [01-guide-complet.md](01-guide-complet.md) (French).
