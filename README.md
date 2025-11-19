# 🚀 US Visa Approval Prediction — End-to-End ML Pipeline  
FastAPI | Docker | MongoDB | AWS EC2/ECR | CI/CD | Evidently Monitoring

---

## 🧩 1. Problem Statement  
This project predicts whether a US visa application will be **approved or rejected** based on structured applicant features.  
It includes a **production-ready ML pipeline** with:

- Data ingestion → validation → transformation  
- Model training + evaluation  
- FastAPI inference API  
- Docker containerization  
- CI/CD with GitHub Actions  
- AWS EC2 + ECR deployment  
- Monitoring with Evidently  
- MongoDB as backend storage  

---

## 📁 2. Folder Structure  
```bash
.
├── .github/                     # CI/CD workflows (GitHub Actions)
│   └── workflows/
├── cloud_storage/               # Cloud/S3 helper functions
├── components/                  # Modular ML components
├── config/                      # YAML configuration files
│   ├── model.yaml
│   └── schema.yaml
├── constants/                   # Global constants
├── data_access/                 # Data access layer (DB/Storage)
├── entity/                      # Entity classes (config + artifacts)
├── exception/                   # Custom exception handling
├── flowcharts/                  # Architecture & pipeline diagrams
├── logger/                      # Logging module
├── notebook/                    # Jupyter notebooks (EDA/Training)
├── pipline/                     # Training + prediction pipelines
│   ├── training_pipeline.py
│   └── prediction_pipeline.py
├── static/                      # Static files (CSS, JS)
├── templates/                   # HTML templates (FastAPI/Jinja2)
├── us_visa/                     # Main package code
│   ├── components/
│   ├── configuration/
│   ├── constants/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── pipline/
│   └── utils/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── app.py                       # FastAPI application
├── demo.py
├── requirements.txt
├── setup.py
└── template.py

# ⚙️ 3. Workflow (High-Level)
constants → entity → components → pipeline → app.py → AWS deployment



# 🔧 5. How to Run Locally


Create Conda Environment
conda create -n visa python=3.8 -y
conda activate visa

## Install Dependencies
pip install -r requirements.txt

## Set Environment Variables

export MONGODB_URL="mongodb+srv://<username>:<password>..."
export AWS_ACCESS_KEY_ID=<KEY>
export AWS_SECRET_ACCESS_KEY=<SECRET>

## Run FastAPI
python app.py

## Swagger UI:
http://54.147.165.235:8080/

## 🐳 6. Docker Commands
# Build Image

docker build -t visa-app .

## ☁️ 7. AWS Deployment (EC2 + ECR + CI/CD)
# Required IAM Permissions

AmazonEC2FullAccess

AmazonEC2ContainerRegistryFullAccess

# Create ECR Repo 
315865595366.dkr.ecr.us-east-1.amazonaws.com/visarepo

## Install Docker on EC2
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

## Add EC2 as Self-Hosted Runner

GitHub → Settings → Actions → Runners → Add Runner

## Add GitHub Secrets
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO

## CI/CD Pipeline Does

Build Docker image

Push to ECR

SSH into EC2

Pull + Run container

# 🔄 8. Git Commands
git add .
git commit -m "Updated"
git push origin main

## 📊 9. Monitoring (Evidently)

# Evidently monitors:

- Data drift

- Model drift

- Feature distribution

Great for production ML monitoring.

🔗 10. Project Links
GitHub Repo: <https://github.com/MassYadav/End-to-End-US-Visa-Approval-System>
Live Demo: <http://54.147.165.235:8080/>

## project work will done 


