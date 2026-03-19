# Notes — Before You Run Example 4 (Kubernetes)

---

## Hard Requirements Before Starting

- [ ] Examples 1–3 work correctly on your local machine
- [ ] Docker Desktop (or equivalent) installed and running
- [ ] `kubectl` installed: `kubectl version --client`
- [ ] A running Kubernetes cluster (see options below)
- [ ] You understand what a Docker image, Pod, Deployment, and Service are

If you have never used Kubernetes before, read the
[Kubernetes Basics tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
first. It takes about 30 minutes and covers everything you need.

---

## Choose Your Cluster

You do not need a cloud account to run this example. Pick the option that matches
your setup:

| Option | Best For | Setup Time |
|---|---|---|
| **Minikube** | Local dev, single node | 5 min |
| **kind** (Kubernetes in Docker) | CI / Docker Desktop users | 5 min |
| **k3s** | Lightweight, Raspberry Pi, Linux VMs | 10 min |
| **Docker Desktop** | Mac/Windows, already have Docker | 2 min (enable in settings) |
| **EKS / GKE / AKS** | Production, cloud | 30+ min |

### Minikube (recommended for beginners)
```bash
# Install
brew install minikube          # macOS
choco install minikube         # Windows
# Linux: https://minikube.sigs.k8s.io/docs/start/

# Start
minikube start --memory=4096 --cpus=2

# Important: use Minikube's Docker daemon so images are available in-cluster
eval $(minikube docker-env)    # run this in every terminal session
```

### kind
```bash
kind create cluster --name rag-workshop
kubectl cluster-info --context kind-rag-workshop
```

---

## Concepts You Must Understand Before Running

### Pod
The smallest unit in Kubernetes. Think of it as a running container (or group
of tightly coupled containers). A Pod is **ephemeral** — if it dies, Kubernetes
replaces it, but its local storage is gone unless you use a PersistentVolume.

### Deployment
A controller that says "always keep N replicas of this Pod running."
If a Pod crashes, the Deployment creates a new one automatically.

```
Deployment (replicas: 2)
  ├── Pod 1  ← running
  └── Pod 2  ← running
              ← Pod 3 starts automatically if Pod 1 dies
```

### Service
A stable network address that routes traffic to Pods.
Pods get random IPs that change on restart. A Service's IP never changes.

```
Client → Service (stable IP: 10.96.x.x) → Pod 1 or Pod 2 (round-robin)
```

Types used in this example:
- `ClusterIP` — only accessible inside the cluster (default)
- `LoadBalancer` — exposes a public IP (cloud clusters only)

### ConfigMap vs Secret
Both inject configuration into Pods as environment variables or files.

```
ConfigMap  → non-sensitive data  (DB_NAME, port numbers, feature flags)
Secret     → sensitive data      (passwords, API keys, connection strings)
```

Secrets are base64-encoded, **not encrypted**. Use an external secrets manager
(AWS Secrets Manager, HashiCorp Vault) for real production workloads.

### PersistentVolume (PV) and PersistentVolumeClaim (PVC)
MongoDB needs disk storage that survives Pod restarts.

```
PersistentVolume (PV)     ← the actual storage (disk, cloud volume, hostPath)
PersistentVolumeClaim (PVC) ← a Pod's request for storage ("I need 5Gi")
```

The `pv.yaml` in this example uses `hostPath` (a directory on the node).
This only works on single-node clusters (Minikube, kind). For multi-node
clusters, use a cloud StorageClass (EBS, GCE PD, Azure Disk).

### Job
A one-shot task. Unlike a Deployment, a Job runs to completion and stops.
Used here to run the document ingestion script once after the cluster is set up.

```
kubectl apply -f job-ingest.yaml   ← starts the Job
kubectl wait --for=condition=complete job/rag-ingest  ← waits for it to finish
kubectl logs job/rag-ingest        ← see what it did
```

---

## Understanding the Two Deployment Scenarios

### Scenario A — MongoDB as a Pod (dev/test)

```
rag-workshop namespace
  ├── mongodb Deployment (1 replica)
  │     └── mongo:7.0 container
  │           └── /data/db → PersistentVolumeClaim → hostPath on node
  ├── mongodb-svc Service (ClusterIP)
  │     └── routes to MongoDB Pod on port 27017
  ├── rag-app Deployment (2 replicas)
  │     └── rag-app:latest container
  │           └── MONGODB_URI = mongodb://admin:pass@mongodb-svc:27017
  └── rag-app Service (ClusterIP)
```

The RAG app connects to MongoDB using the Service DNS name `mongodb-svc`.
Inside the cluster, Kubernetes DNS resolves `mongodb-svc` to the ClusterIP.

**Limitation:** This MongoDB setup has no replication, no Atlas Vector Search.
You cannot use Atlas Vector Search with an in-cluster MongoDB pod — it is an
Atlas-only feature. This scenario is useful for testing the app container and
Kubernetes setup, but use real Atlas for the vector search features.

### Scenario B — MongoDB Atlas (production)

```
rag-workshop namespace
  ├── rag-app Deployment (2 replicas)
  │     └── MONGODB_URI = mongodb+srv://user:pass@cluster.atlas.mongodb.net/...
  └── rag-app Service

MongoDB Atlas (external)
  └── Vector Search enabled
```

All vector search features work. The app pods connect to Atlas over the internet.
Make sure Atlas Network Access allows your cluster's egress IP.

---

## Setting Up Secrets (Do This Before Anything Else)

The `secrets/rag-secrets.yaml` file has placeholder values. You must replace them
with real base64-encoded values before applying.

```bash
# Encode your values
echo -n "admin" | base64                        # → YWRtaW4=
echo -n "mysecretpassword" | base64             # → bXlzZWNyZXRwYXNzd29yZA==
echo -n "mongodb+srv://..." | base64            # → long string
echo -n "sk-proj-..." | base64                  # → long string
```

Then edit `secrets/rag-secrets.yaml` and replace each `<base64-encoded-...>`:

```yaml
data:
  MONGODB_URI: bW9uZ29kYitzcnY6Ly8...    ← your encoded Atlas URI
  MONGO_USERNAME: YWRtaW4=               ← "admin" encoded
  MONGO_PASSWORD: bXlzZWNyZXRwYXNzd29yZA==
  OPENAI_API_KEY: c2stcHJvai0...         ← your encoded key
```

**Never commit this file with real values.** Add it to `.gitignore`.

---

## Building the Docker Image

```bash
# From the rag-mongodb/ directory
cd /path/to/rag-mongodb

# If using Minikube: build inside Minikube's Docker (so the cluster can find it)
eval $(minikube docker-env)

# Build
docker build -t rag-app:latest -f 04-kubernetes/rag-app/Dockerfile .

# Verify the image exists
docker images | grep rag-app
```

The `Dockerfile` is a **multi-stage build**:
- Stage 1 (`builder`): installs Python packages
- Stage 2 (final): copies only the installed packages + app code, not build tools

This keeps the final image smaller (~200MB vs ~800MB).

The container runs as a non-root user (`appuser`) — a security best practice.

---

## Step-by-Step Deployment Order

Order matters. Apply in this sequence:

```bash
# 1. Create namespace (isolation: all resources live here)
kubectl create namespace rag-workshop

# 2. Secrets first (other resources reference them)
kubectl apply -f secrets/rag-secrets.yaml -n rag-workshop

# 3. ConfigMap (non-sensitive config)
kubectl apply -f configmaps/rag-config.yaml -n rag-workshop

# 4a. MongoDB pod (skip if using Atlas)
kubectl apply -f mongodb/pv.yaml          # storage
kubectl apply -f mongodb/deployment.yaml -n rag-workshop
kubectl apply -f mongodb/service.yaml -n rag-workshop

# Wait for MongoDB to be ready
kubectl rollout status deployment/mongodb -n rag-workshop

# 4b. RAG app
kubectl apply -f rag-app/deployment.yaml -n rag-workshop
kubectl apply -f rag-app/service.yaml -n rag-workshop

# 5. Wait for RAG app to be ready
kubectl rollout status deployment/rag-app -n rag-workshop

# 6. Run the ingest job (one time only)
kubectl apply -f rag-app/job-ingest.yaml -n rag-workshop
kubectl wait --for=condition=complete job/rag-ingest -n rag-workshop --timeout=300s
kubectl logs job/rag-ingest -n rag-workshop  # verify it succeeded

# 7. Test the app
kubectl port-forward svc/rag-app 8080:8080 -n rag-workshop &
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is MongoDB Atlas Vector Search?"}'
```

Or use Kustomize to apply everything at once:
```bash
kubectl apply -k 04-kubernetes/ -n rag-workshop
```

---

## Useful Debugging Commands

```bash
# See all resources in the namespace
kubectl get all -n rag-workshop

# See why a Pod is not starting
kubectl describe pod <pod-name> -n rag-workshop

# See app logs
kubectl logs deployment/rag-app -n rag-workshop
kubectl logs deployment/rag-app -n rag-workshop --previous  # last crashed pod

# See MongoDB logs
kubectl logs deployment/mongodb -n rag-workshop

# Shell into the RAG app container
kubectl exec -it deployment/rag-app -n rag-workshop -- /bin/bash

# Shell into MongoDB
kubectl exec -it deployment/mongodb -n rag-workshop -- mongosh \
  -u admin -p mysecretpassword --authenticationDatabase admin

# Watch pods in real time
kubectl get pods -n rag-workshop -w
```

---

## Things That Can Go Wrong

| Symptom | Likely Cause | Fix |
|---|---|---|
| Pod stuck in `Pending` | No node has enough CPU/memory | Increase Minikube resources: `minikube start --memory=4096` |
| Pod stuck in `ImagePullBackOff` | Image not in cluster | Rebuild with `eval $(minikube docker-env)` then rebuild |
| Pod in `CrashLoopBackOff` | App is crashing on start | `kubectl logs <pod>` to see the error |
| `Secret "rag-secrets" not found` | Applied secret to wrong namespace | Re-apply with `-n rag-workshop` |
| `connection refused` on port 8080 | Port-forward not running | Re-run `kubectl port-forward ...` |
| `Authentication failed` (MongoDB) | Wrong credentials in Secret | Decode and verify: `echo "YWRtaW4=" \| base64 --decode` |
| Ingest Job fails | Atlas index not built yet | Wait 60s after atlas index creation, re-run the Job |
| `hostPath` PV not working | Multi-node cluster | Use a cloud StorageClass instead of hostPath |

---

## After This Example

You now have a production-like deployment. Next steps for a real project:

1. **Ingress controller** — expose the app on a proper domain with TLS
2. **Horizontal Pod Autoscaler** — scale replicas based on CPU/request load
3. **External Secrets Operator** — sync secrets from AWS/GCP/Vault automatically
4. **Liveness/readiness tuning** — adjust probe delays for your startup time
5. **MongoDB Atlas** instead of the in-pod MongoDB for real vector search
6. **Persistent session history** — replace in-memory `ChatMessageHistory` with a
   MongoDB-backed store so conversations survive pod restarts
