# End-to-End AutoML Pipeline

## H2O + MLflow + FastAPI + Streamlit

This guide explains how to clone, run, test, and stop the **End-to-End AutoML Pipeline** project.

---

## STEP 1 — Cloning the Project

Open **PowerShell**, then run the following commands:

```powershell
git clone https://github.com/inskillflow/complete-mlops-pipeline-end-to-end-AutoML-en.git complete-pipeline-demo-1
cd complete-pipeline-demo-1
```

These commands download the project from GitHub and move into the project folder.

---

## STEP 2 — Running the Project

To build and start the full stack, run:

```powershell
.\start.ps1 -Rebuild
```

Alternatively, you can run the project directly with Docker Compose:

```powershell
docker compose up -d --build
```

This command builds the Docker images and starts all services in detached mode.

---

## STEP 3 — Checking the Services and Running the Experiments

Once the containers are running, open the following links in your browser:

| Service      | URL                        | Description                                        |
| ------------ | -------------------------- | -------------------------------------------------- |
| Streamlit UI | http://localhost:8501      | Web interface for interacting with the application |
| FastAPI Docs | http://localhost:8000/docs | API documentation and testing interface            |
| MLflow UI    | http://localhost:5000      | Experiment tracking and model registry interface   |

Use these interfaces to verify that the complete pipeline is working correctly.

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

This command displays the logs of all services in real time.

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

---

## Expected Result

After completing these steps, the project should be running locally with:

* Streamlit available on port `8501`
* FastAPI available on port `8000`
* MLflow available on port `5000`

The full AutoML pipeline is now ready for testing and experimentation.
