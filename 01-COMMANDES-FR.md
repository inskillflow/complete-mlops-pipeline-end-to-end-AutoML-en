# End-to-End AutoML Pipeline

## H2O + MLflow + FastAPI + Streamlit

Ce guide explique comment cloner, lancer, tester et arrêter le projet **End-to-End AutoML Pipeline**.

---

## STEP 1 — Cloning the Project

Open **PowerShell**, then run the following commands:

```powershell
git clone https://github.com/inskillflow/complete-mlops-pipeline-end-to-end-AutoML-en.git complete-pipeline-demo-1
cd complete-pipeline-demo-1
```

This command downloads the project from GitHub and moves into the project folder.

---

## STEP 2 — Running the Project

To build and start the full stack, run:

```powershell
.\start.ps1 -Rebuild
```

If you prefer using Docker Compose directly, you can also run:

```powershell
docker compose up -d --build
```

This command builds the required Docker images and starts all services in detached mode.

---

## STEP 3 — Checking the Services and Running the Experiments

Once the containers are running, open the following links in your browser:

| Service      | URL                        | Description                                        |
| ------------ | -------------------------- | -------------------------------------------------- |
| Streamlit UI | http://localhost:8501      | Web interface for interacting with the application |
| FastAPI Docs | http://localhost:8000/docs | API documentation and testing interface            |
| MLflow UI    | http://localhost:5000      | Experiment tracking and model registry interface   |

Use these interfaces to verify that the full pipeline is working correctly.

---

## STEP 4 — Stopping the Project

To stop the project, run:

```powershell
.\start.ps1 -Down
```

This command stops the running services cleanly.

---

## Troubleshooting

If something does not work correctly, check the Docker logs:

```powershell
docker compose logs -f
```

This command displays the logs of all running services in real time.

You can also check the status of the containers with:

```powershell
docker compose ps
```

---

## Summary of Useful Commands

| Action                    | Command                                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Clone the project         | `git clone https://github.com/inskillflow/complete-mlops-pipeline-end-to-end-AutoML-en.git complete-pipeline-demo-1` |
| Enter the project folder  | `cd complete-pipeline-demo-1`                                                                                        |
| Start with rebuild        | `.\start.ps1 -Rebuild`                                                                                               |
| Start with Docker Compose | `docker compose up -d --build`                                                                                       |
| Stop the project          | `.\start.ps1 -Down`                                                                                                  |
| View logs                 | `docker compose logs -f`                                                                                             |
| Check containers          | `docker compose ps`                                                                                                  |
