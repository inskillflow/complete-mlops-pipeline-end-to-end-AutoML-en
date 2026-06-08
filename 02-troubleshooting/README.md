
# Docker Cleanup Commands

## Objective

These commands are used to clean the local Docker environment by stopping containers, removing containers, and deleting Docker images.

> **Warning:** These commands are destructive. They may remove containers and images from your machine. Use them only when you are sure you do not need the existing containers or images anymore.

---

## 1. Stop all Docker containers

```powershell
docker stop $(docker ps -a -q)
```

### Explanation

This command stops all existing Docker containers.

* `docker ps -a -q` lists the IDs of all containers.
* `docker stop` stops the containers returned by the previous command.

Use this command when some containers are still running and you want to stop everything before cleaning Docker.

---

## 2. Remove all Docker containers

```powershell
docker rm $(docker ps -a -q)
```

### Explanation

This command removes all Docker containers.

* It removes containers that already exist on the machine.
* Containers must normally be stopped before they can be removed.
* This does not remove Docker images.

Use this command after stopping the containers.

---

## 3. Remove all Docker images

```powershell
docker rmi $(docker images -q)
```

### Explanation

This command removes all Docker images from the local machine.

* `docker images -q` lists the IDs of all Docker images.
* `docker rmi` removes the images returned by the previous command.

Use this command when you want to force Docker to rebuild images from scratch later.

---

# Recommended Order

Run the commands in this order:

```powershell
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
docker rmi $(docker images -q)
```

---

# When to Use These Commands

Use these commands when:

* You want to reset your Docker environment.
* You want to remove old containers.
* You want to remove old images.
* You want to rebuild your project from zero.
* Docker Compose is using old containers or old images.

---

# After Cleanup

After running these commands, you can rebuild and restart your project with:

```powershell
docker compose up -d --build
```

or:

```powershell
.\start.ps1 -Rebuild
```
