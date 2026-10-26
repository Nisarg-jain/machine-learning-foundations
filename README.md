ML Foundation/
├── data/                          # Raw and processed tabular datasets
│   ├── placement.csv
│   ├── scraped_companies.csv
│   └── scraped_quotes.csv
│
├── notebooks/                     # Interactive prototypes and exploratory notebooks
│   └── first_toy_project.ipynb
│
├── artifacts/                     # Serialized models, scalers, and diagnostic reports
│   ├── model.pkl
│   ├── placement_model.joblib
│   ├── scaler.pkl
│   └── exploratory_data_profile.html
│
├── visualizations/                # Generated distribution curves, plots, and heatmaps
│   ├── bivariate_lineplot.png
│   ├── categorical_heatmap.png
│   ├── categorical_univariate_plots.png
│   ├── decision_boundary.png
│   ├── hierarchical_clustermap.png
│   ├── multivariate_pairplot.png
│   ├── multivariate_scatter.png
│   ├── numerical_categorical_analysis.png
│   ├── numerical_univariate_plots.png
│   └── placement_scatter.png
│
└── modules/                       # Production-oriented source modules
    ├── foundations/               # Core theory, tensors, drift detection, and environments
    │   ├── drift_detection_demo.py
    │   ├── env_diagnostics.py
    │   ├── instance_vs_model.py
    │   └── tensor_fundamentals.py
    │
    ├── ingestion/                 # Web scrapers, REST APIs, and structured file loaders
    │   ├── ambitionbox_scraper.py
    │   ├── csv_parameters_demo.py
    │   └── data_gathering_pipeline.py
    │
    ├── eda/                       # Univariate, bivariate, multivariate, and automated audits
    │   ├── automated_data_profiler.py
    │   ├── bivariate_multivariate_analysis.py
    │   ├── data_understanding.py
    │   ├── data_understanding_audit.py
    │   └── univariate_analysis.py
    │
    └── feature_engineering/       # Transformations, scalers, encodings, and selections
        ├── feature_engineering_taxonomy.py
        ├── normalization_scaling.py
        └── standardization_scaling.py