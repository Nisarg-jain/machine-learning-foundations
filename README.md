# ML Foundation: 100 Days of Machine Learning

Structured implementation code, mathematical foundations, production workflows, and practical experiments following CampusX's *100 Days of Machine Learning* roadmap. Built entirely within an isolated local virtual environment using VS Code and Git version control.

---

## 🛠️ Environment & Architecture

- **Runtime:** Python 3.14 (`.venv` virtual environment)
- **Primary IDE:** Visual Studio Code
- **Core Libraries:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `mlxtend`, `requests`, `beautifulsoup4`, `ipykernel`, `joblib`
- **Workflow Standards:** Isolated package management, modular scripts, Jupyter interactive kernels, and Git commit tracking for daily progress.

---

## 🗺️ Roadmap & Progress Tracker

| Day / Lecture | Topic | Core Focus / Deliverables | Status | Artifacts / Scripts |
| :---: | :--- | :--- | :---: | :--- |
| **01–08** | **ML Foundations & Taxonomy** | Instance vs. Model-based learning, Batch vs. Online, ML challenges & bottlenecks | Completed | `instance_vs_model.py` |
| **09** | **MLDLC & Production Monitoring** | Full lifecycle stages, covariate drift, concept drift simulation, model saving | Completed | `drift_detection_demo.py` |
| **10** | **Data Science Job Roles & Ecosystem** | Data Engineer vs. Analyst vs. Scientist vs. MLOps Engineer roles & tech stacks | Completed | Architectural notes |
| **11** | **What are Tensors?** | 0D scalars to 5D video tensors, axis manipulation, rank, shape, and raw memory footprint calculations | Completed | `tensor_fundamentals.py` |
| **12** | **Environment Configuration** | Native VS Code `.venv` setup with `ipykernel` integration | Completed | `.venv` configuration |
| **13** | **First End-to-End Toy Project** | Full placement prediction pipeline: EDA, train/test split, data leakage avoidance, scaling, Logistic Regression, decision boundaries, artifact serialization | Completed | `first_toy_project.ipynb`<br>`placement.csv`<br>`model.pkl`<br>`scaler.pkl`<br>`decision_boundary.png` |
| **14** | **Data Gathering: REST APIs** | HTTP protocols, JSON payload normalization, status handling, query pagination | Completed | `data_gathering_pipeline.py` |
| **15** | **Data Gathering: Web Scraping** | HTML DOM parsing, BeautifulSoup selectors, User-Agent header spoofing | Completed | `data_gathering_pipeline.py`<br>`scraped_quotes.csv` |
| **16** | **Ingestion with `pd.read_csv()`** | Custom delimiters, character encodings (`utf-8`, `latin-1`), malformed rows, downcasting, and memory-safe chunking | Completed | `csv_parameters_demo.py` |
| **17** | **Data Understanding & Profiling** | 8-step data audit: shape, cardinality, missingness reports, duplicate filtering, distribution skew, and target correlation | Completed | `data_understanding_audit.py` |
| **18** | **Univariate Analysis** | Histograms