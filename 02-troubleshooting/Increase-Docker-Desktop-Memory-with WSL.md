# Increase Docker Desktop Memory with WSL

## Objective

This procedure explains how to increase the amount of RAM available to Docker Desktop when Docker is using the WSL 2 backend on Windows.

This can be useful when Docker containers are slow, training jobs fail, or applications such as MLflow, FastAPI, Streamlit, or AutoML require more memory.

---

## Step 1 — Check the current Docker memory

Run the following command in PowerShell:

```powershell
docker info | findstr "Memory"
```

Example result:

```powershell
Total Memory: 3.744GiB
```

This means Docker currently has access to approximately **3.7 GB of RAM**.

---

## Step 2 — Open the WSL configuration file

Run this command in PowerShell:

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

This opens the WSL configuration file in Notepad.

If the file does not exist, Notepad may ask you to create it.

---

## Step 3 — Add or update the WSL configuration

Add the following configuration:

```ini
[wsl2]
memory=8GB
processors=4
swap=4GB
localhostForwarding=true
```

---

## Step 4 — Explanation of the configuration

```ini
memory=8GB
```

This allows WSL and Docker Desktop to use up to **8 GB of RAM**.

```ini
processors=4
```

This allows WSL and Docker Desktop to use **4 CPU cores**.

```ini
swap=4GB
```

This adds **4 GB of swap memory**, which can help when RAM is not enough.

```ini
localhostForwarding=true
```

This keeps access to Docker services through `localhost`, for example:

```text
http://localhost:8501
http://localhost:8000
http://localhost:5000
```

---

## Step 5 — Save the file

In Notepad, click:

```text
File > Save
```

Then close Notepad.

---

## Step 6 — Restart WSL

Run this command in PowerShell:

```powershell
wsl --shutdown
```

This completely stops WSL so the new configuration can be applied.

---

## Step 7 — Restart Docker Desktop

Close Docker Desktop and open it again.

Wait until Docker Desktop is fully running.

---

## Step 8 — Verify the new Docker memory

Run the command again:

```powershell
docker info | findstr "Memory"
```

Example expected result:

```powershell
Total Memory: 7.599GiB
```

This confirms that Docker now has access to approximately **8 GB of RAM**.

---

## Summary

Before the change, Docker may have had only around **3 or 4 GB of RAM**.

After updating the `.wslconfig` file and restarting WSL and Docker Desktop, Docker can use more memory, which improves stability for heavier applications such as AutoML, MLflow, FastAPI, and Streamlit.
