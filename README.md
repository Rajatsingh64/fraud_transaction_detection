<img src="https://img.shields.io/badge/Project_Status-Ongoing-blue?style=for-the-badge" />

<!-- Centered Title Banner -->
<p align="center">
  <img src="demo/assets/title.png" alt="Project Banner" width="100%" style="max-width:800px; display:block; margin:auto;">
</p>


<!-- Shield.io Badges Flow (Dark Theme, No Gaps) -->
<p align="center">
  <img src="https://img.shields.io/badge/Used-MLflow-black?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Operations-Apache%20Airflow-black?style=for-the-badge&logo=apache-airflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deployed%20On-AWS-black?style=for-the-badge&logo=amazon-aws&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model%20Storage-S3%20Bucket-black?style=for-the-badge&logo=amazon-s3&logoColor=white"/>
  <img src="https://img.shields.io/badge/Prediction%20Source-Latest%20Model%20from%20S3-black?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Continuous-Training-black?style=for-the-badge&logo=github-actions&logoColor=white"/>
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

---

![Fraud Detection App](demo/assets/gif_demo1.gif)

<div align="center" style="background-color:#1B1B1B; padding:20px; border-radius:10px;">

  <h2 style="color:#E50914;">🚀 Tools and Technologies Used</h2>

  <p>
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-learn" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" alt="Matplotlib" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png" alt="AWS" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Docker_%28container_engine%29_logo.svg" alt="Docker" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" alt="Seaborn" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" alt="Pandas" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="NumPy" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://www.mongodb.com/assets/images/global/favicon.ico" alt="MongoDB" height="40">&nbsp;&nbsp;&nbsp;
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFzCIuPsPokbP-V0RFFgCRJqcve5gpjJmTtg&s" alt="Apache Airflow" height="40">
  </p>

</div>

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
├── artifact/                                  # 🐂 Contains all intermediate and final outputs
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
├── src/
│   ├── components/                            # 🏢 Core pipeline components
│   │   ├── data_ingestion.py                  # 📅 Handles data collection
│   │   ├── data_transformation.py             # 🔄 Prepares data for training
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
