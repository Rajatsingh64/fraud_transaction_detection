<!-- Project Status -->
<p align="center">
  <img src="https://img.shields.io/badge/Project_Status-Completed-green?style=for-the-badge" />
</p>

<!-- Centered Title Banner -->
<p align="center">
  <img src="demo/assets/title.png" alt="Project Banner" width="100%" style="max-width:800px; display:block; margin:auto;">
</p>

<!-- Key Tech Stack Summary -->
<p align="center">
  <img src="https://img.shields.io/badge/Used-MLflow-black?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Operations-Apache%20Airflow-black?style=for-the-badge&logo=apache-airflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deployed%20On-AWS-black?style=for-the-badge&logo=amazon-aws&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model%20Storage-S3%20Bucket-black?style=for-the-badge&logo=amazon-s3&logoColor=white"/>
  <img src="https://img.shields.io/badge/Prediction%20Source-Latest%20Model%20from%20S3-black?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Continuous-Training-black?style=for-the-badge&logo=github-actions&logoColor=white"/>
</p>

<!-- App Design Documents -->
<h2 align="center">🛠️ App Design Documents</h2>

<p align="center">
  <a href="docs/HLD.pdf">
    <img src="https://img.shields.io/badge/View%20HLD-1E90FF?style=for-the-badge&logo=blueprint&logoColor=white" alt="View HLD">
  </a>
  <a href="docs/LLD.pdf">
    <img src="https://img.shields.io/badge/View%20LLD-28a745?style=for-the-badge&logo=markdown&logoColor=white" alt="View LLD">
  </a>
  <a href="docs/DPR.pdf">
    <img src="https://img.shields.io/badge/View%20DPR-FF8C00?style=for-the-badge&logo=processwire&logoColor=white" alt="View DPR">
  </a>
  <a href="docs/Architecture.pdf">
    <img src="https://img.shields.io/badge/View%20Architecture-8A2BE2?style=for-the-badge&logo=archlinux&logoColor=white" alt="View Architecture">
  </a>
</p>

<!-- License Badge -->
<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License: MIT">
  </a>
</p>

## 🧠 Flowchart

<p align="center">
  <img src="demo/assets/mlops_flowchart.png" alt="MLOPS Flowchart" width="45%" />
  <img src="demo/assets/deployment_flowchart.png" alt="Deployment Flowchart" width="45%" />
</p>

## 🧾 App Overview

<p align="center">
  <img src="demo/assets/gif_demo1.gif" alt="Fraud Detection App" width="80%" />
</p>

## 🧰 Technologies & Frameworks Used

<p align="center">
  <img src="https://img.shields.io/badge/MongoDB-%2347A248.svg?style=for-the-badge&logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-%23336791.svg?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Airflow-%23017CEE.svg?style=for-the-badge&logo=apache-airflow&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-%23232F3E.svg?style=for-the-badge&logo=amazonaws&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-%232496ED.svg?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/imblearn-%23990000.svg?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-%2311557C.svg?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-%233C6F9C.svg?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/VS%20Code-%23007ACC.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white" />
</p>

## 📂 Project Navigation  

📁 [**Notebooks**](notebook/) | 📁 [**Pipelines**](src/pipeline/) | 📁 [**Components**](src/components) | 📁 [**Project Details**](project_details/) | 📁 [**Documents**](project_details/documents/)

### 📌 Project Overview

- **Python Version**: 3.12  
- Detecting **fraudulent card transactions** using machine learning techniques to enhance the security of online and ATM transactions for banks.
- The system:
  - Logs experiments using **MLflow**
  - Automates training with **Apache Airflow**
  - Fetches the **latest trained model** from **AWS S3**
  - Supports **continuous learning**
  - Deployed via **AWS Cloud Services**

## 💁️ Project Structure

```

Fraud-Transaction-Detection/
│
|── dataset/                                   # 📂 Contains Project Dataset
|
|── templates/                                 # 📂 Contains HTML files for the app's structure
|   └── predict.html
|
|── static/                                    # 📂 Contains CSS files for styling the app
|   └── style.css
|
├── project_details/                           # 📂 Contains Project info and reports
|   └── documents/                             # 📖 Project Reports(HLD, LLD,DPR ,etc.)
|
├── .dockerignore                              # 🚫 Ignore files for Docker
├── .env                                       # 🔑 Environment variables
├── .gitignore                                 # 🚫 Ignore files for Git
│
|── demo/
|   ├── project_demo.mp4                       # Full Project Explaination
│   └── assets/                                # Gif and Images 
| 
├── .github/
│   └── workflows/
│       └── main.yaml                          # ⚙️ GitHub Actions CI/CD pipeline
│
├── airflow/                                   # 💨 Apache Airflow DAGs
│   └── dags/                                  # 📅 Workflow DAGs
│       ├── batch_prediction_pipeline.py       # 🔍 Airflow DAG for prediction
│       └── training_pipeline.py               # 🎯 Airflow DAG for model training
│
├── start.sh                                   #  Initialize the Airflow database
|                                 
├── artifacts/                                 # 🐂 Contains all intermediate and final outputs
├── predictions/                               # 📂 Predictions processed files
├── data_dump.py                               # 🛋️ Dumps data into MongoDB Atlas
├── docker-compose.yml                         # 🔧 Docker Compose for multi-container setup
├── Dockerfile                                 # 💪 Docker image setup
│
├── LICENSE                                    # 📚 MIT License file
├── main.py                                    # 🚀 Entry point for training and predictions
├── notebook/                                  # 📚 Jupyter notebooks
│   ├── research.ipynb                         # 🔄 Exploratory Data Analysis and Model Training
│   
│
├── README.md                                  # 📖 Project documentation
├── requirements.txt                           # 📌 Dependencies for the project
├── saved_models/                              # 🎯 Production-ready models and transformers
├── setup.py                                   # ⚙️ Package setup for `src`
|
|── app.py                                     # Flask Fraud Prediction App 
|
├── src/
│   ├── components/                            # 🏢 Core pipeline components
│   │   ├── data_ingestion.py                  # 📅 Handles data collection
│   │   ├── data_preprocessing.py              # 🔄 Prepares data for training
│   │   ├── feature_engineering.py             # 🔄 Generate new features for training
|   |   |── data_validation.py                 # ✅ Validates raw data
│   │   ├── model_evaluation.py                # 📊 Evaluates the model
│   │   ├── model_pusher.py                    # 🚀 Pushes the trained model to deployment
│   │   ├── model_training.py                  # 🎓 Trains the machine learning model
│   │
│   ├── config.py                              # ⚙️ Configuration management and environment variables
│   ├── entity/                                # 📆 Data structures for pipeline
│   │   ├── artifact_entity.py                 # 🐂 Artifacts generated by pipeline stages
│   │   └── config_entity.py                   # ⚙️ Configuration-related entities
│   │
│   ├── exceptions.py                          # ❗ Custom exception handling
│   ├── logger.py                              # 💜 Logging setup
│   ├── pipeline/                              # 🔄 Pipeline automation
│   │   ├── batch_prediction_pipeline.py       # 🔍 Handles batch predictions
│   │   └── training_pipeline.py               # 🎯 Automates training workflow
│   │
│   └── utils.py                               # 🛠️ Utility functions
```
---

## Deployment Guide

### **Airflow(CT) and App Deployment on EC2 using Docker and GitHub Actions**

This guide provides step-by-step commands to deploy a Flask app and Apache-Airflow Training Pipeline(CT) on an EC2 instance using Docker, with automatic deployment through GitHub Actions.

#### Commands for EC2 Setup and Deployment

1. **Launch an EC2 Instance** using the AWS Management Console with your preferred settings.

2. **Connect to Your EC2 Instance**:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

#### **GitHub Repo Secrets Setup**

- `AWS_ACCESS_KEY_ID`=
- `AWS_SECRET_ACCESS_KEY`=
- `AWS_REGION`=
- `AWS_ECR_LOGIN_URI`=
- `ECR_REPOSITORY_NAME`=
- `BUCKET_NAME`=
- `MONGO_URL`= `MongoDB Atlas database url`
- `DATABASE_NAME`= `MongoDB database name`
- `AIRFLOW_USERNAME`=
- `AIRFLOW_PASSWORD`=
- `AIRFLOW_EMAIL`=

#### **Run All GitHub Runner Commands in AWS CLI and Activate It**

1. Set Up GitHub Actions Runner on EC2
2. Navigate to **Settings > Actions > Runners** in your GitHub repository.
3. Follow the instructions provided by GitHub to download and configure the runner on your EC2 instance.

```bash
curl -o actions-runner-linux-x64-<version>.tar.gz -L https://github.com/actions/runner/releases/download/v<version>/actions-runner-linux-x64-<version>.tar.gz
tar xzf actions-runner-linux-x64-<version>.tar.gz
```

---

## 🚀 Connect with Me

<p align="center">

  <a href="https://github.com/Rajatsingh64" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
  &nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/rajat-singh-292124240" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-black?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  &nbsp;&nbsp;
  <a href="mailto:rajat.k.singh64@gmail.com">
    <img src="https://img.shields.io/badge/Email-black?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>

</p>

<p align="center">
  💻 Developed by <strong>Rajat Singh</strong> | ⚡ Powered by Passion & Code  
</p>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&center=true&width=435&lines=Thanks+for+visiting!;Happy+Coding!+🚀" alt="Typing SVG" />
</p>
