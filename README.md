# neural_scale-ai
# NeuralScale AI - Intelligent Cloud Resource Optimization 🚀

## 🎯 Vision
**A cloud‑native, multi‑cloud platform for predictive anomaly detection and cost‑aware auto‑scaling of AI/analytics workloads, with RL‑based optimization, edge extensions, and auditable billing.**

## 📈 Success Metrics (Initial)
- **MTTD/MTTR**: ≥50% reduction in mean time to detect and recover incidents
- **Cost/Throughput**: ≥25% reduction in $/1000 requests at same SLOs
- **Accuracy/Drift**: ≥10% improvement in anomaly precision@95% recall; ≤1% weekly model performance drift
- **Autoscaling QoS**: 99.9% of requests finish under p95 target with RL autoscaler enabled

## 📌 Overview
NeuralScale AI is an **intelligent cloud resource optimization system** powered by **Reinforcement Learning (RL), AI-based anomaly detection, and quantum-inspired optimization**. It dynamically scales cloud resources based on workload demands while optimizing for **cost, performance, and efficiency**.

🔹 **Key Features:**  
✅ Reinforcement Learning-based **Auto Scaling** (Ray RLlib + PPO)  
✅ **Predictive Anomaly Detection** (Isolation Forest)  
✅ **Quantum-Inspired Resource Optimization** (Qiskit COBYLA)  
✅ **Multi-Cloud Support** (AWS, GCP, Vultr)  
✅ **AI-Powered Resource Forecasting** (LSTM-based prediction)  
✅ **Cost-Aware Scaling** for optimized cloud spending  
✅ **Edge Computing Optimization** for IoT & latency-sensitive applications  
✅ **Security & Compliance Monitoring** (GDPR, SOC2, Intrusion Detection)  
✅ **Real-Time Visualization** via FastAPI + WebSockets  

---

## 🚀 Deployment on AWS App Runner
### Option A: Deploy from a container registry (recommended)
1. Build and push the image:
   ```sh
   docker build -t YOUR_ECR_REPO_URI:latest .
   docker push YOUR_ECR_REPO_URI:latest
   ```
2. In AWS Console → App Runner → Create service → From container registry.
3. Select the ECR image, set Runtime to container, and set the following:
   - Port: `8080`
   - Environment variables:
     - `ENABLE_TRAINING=0` (avoid heavy training on startup)
   - Auto scaling per your needs.
4. Create service. Health check path: `/`.

### Option B: Deploy from source (App Runner builds the image)
1. Connect GitHub repo and select this directory as the root.
2. Use the included `Dockerfile` so App Runner performs a container build.
3. Set Port `8080` and env var `ENABLE_TRAINING=0`.

### Local test
```sh
docker build -t neural-scale-ai .
docker run -p 8080:8080 -e ENABLE_TRAINING=0 neural-scale-ai
curl http://localhost:8080/
```

---

## 🚀 Deployment on Render.com
### **Step 1: Fork or Clone the Repository**
```sh
git clone https://github.com/your-username/neural-scale-ai.git
cd neural-scale-ai
```

### **Step 2: Create a `requirements.txt` File**
Ensure the repository has the following dependencies:
```txt
fastapi
uvicorn
ray[default]
gym
numpy
tensorflow
qiskit
boto3
google-cloud-compute
matplotlib
scikit-learn
pandas
```

### **Step 3: Push to GitHub**
```sh
git add .
git commit -m "Initial commit of NeuralScale AI"
git push origin main
```

### **Step 4: Deploy on Render.com**
1. **Go to** [Render.com](https://render.com/) and sign in.
2. Click **"New Web Service"** → Connect your **GitHub repository**.
3. Set the **Build Command**:
   ```sh
   pip install -r requirements.txt
   ```
4. Set the **Start Command**:
   ```sh
   uvicorn server:app --host 0.0.0.0 --port $PORT
   ```
5. Click **"Deploy"** and wait for the server to go live.
6. Your API will be available at:
   ```
   https://your-app-name.onrender.com
   ```

---

## 📡 API Endpoints & Usage
### **Check if API is Running**
```sh
curl https://your-app-name.onrender.com/
```
#### ✅ Response:
```json
{"message": "NeuralScale AI is running on Render!"}
```

### **Get Simulated Cloud Users**
```sh
curl https://your-app-name.onrender.com/test_users
```

### **Get Auto Scaling Decision (Reinforcement Learning Model)**
```sh
curl -X GET https://your-app-name.onrender.com/scale_decision
```
#### ✅ Example Response:
```json
{"scaling_decision": "Scale Up"}
```

### **Detect Anomalies in Cloud Metrics**
```sh
curl -X POST https://your-app-name.onrender.com/anomaly_detection -H "Content-Type: application/json" -d '[45, 10, 5]'
```
#### ✅ Example Response:
```json
{"anomaly_status": "Normal"}
```

### **Optimize Resources with Quantum-Inspired Algorithms**
```sh
curl -X POST https://your-app-name.onrender.com/quantum_optimization -H "Content-Type: application/json" -d '50'
```

### **AI-Powered Resource Forecasting**
```sh
curl -X POST https://your-app-name.onrender.com/forecast_utilization -H "Content-Type: application/json" -d '[50, 55, 60, 65, 70, 75, 80, 85, 90, 95]'
```

---

## 📚 Technologies Used
🔹 **Machine Learning**: TensorFlow, Scikit-learn, Reinforcement Learning (Ray RLlib)  
🔹 **Optimization**: Qiskit (Quantum-Inspired Optimization)  
🔹 **Cloud Integration**: AWS, GCP, Vultr APIs  
🔹 **Web Framework**: FastAPI + Uvicorn  
🔹 **Visualization**: WebSockets, Matplotlib  

---

## 🛠 Development & Contribution
### **Run Locally**
```sh
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### **Contribute to NeuralScale AI**
1. **Fork the Repository** on GitHub.
2. Create a **new branch**: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m "Added new feature"`
4. Push to GitHub: `git push origin feature-branch`
5. Create a **Pull Request** 🚀

---

## 🏆 Credits & Acknowledgments
Developed by Ayan Nagar | GitHub: aYaN2727 
Inspired by **cutting-edge cloud AI research & quantum computing advancements**.  
💡 **Feedback & Issues?** Open a [GitHub Issue](https://github.com/your-username/neural-scale-ai/issues)  

