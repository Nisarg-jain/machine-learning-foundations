import os
import pandas as pd

print("=" * 65)
print("AUTOMATED EXPLORATORY DATA PROFILING PIPELINE")
print("=" * 65)

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
print("Loading dataset from remote repository...")
df = pd.read_csv(url)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.\n")

output_report = "exploratory_data_profile.html"

try:
    from ydata_profiling import ProfileReport

    print("Generating comprehensive exploratory profile report via YData Profiling...")
    # Initialize the profiling engine
    profile = ProfileReport(
        df,
        title="Titanic Dataset Diagnostic Profile",
        explorative=True,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "phi_k": {"calculate": True}
        }
    )

    # Export report as a standalone web artifact
    profile.to_file(output_report)
    print(f"\n[SUCCESS] Profile report generated: '{output_report}'")

except (ImportError, Exception) as exc:
    print(f"Note: ydata-profiling encountered: {exc}")
    print("Building lightweight native HTML exploratory audit report...")

    # Fallback: Generate clean, dependency-free HTML summary
    num_summary = df.describe().to_html(classes="table table-striped")
    missing_data = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing (%)": (df.isnull().sum() / len(df) * 100).round(2),
        "Dtype": df.dtypes
    }).to_html(classes="table table-bordered")

    corr_data = df.corr(numeric_only=True).round(3).to_html(classes="table table-hover")

    html_payload = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Exploratory Data Audit Report</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    </head>
    <body class="container py-4">
        <h1 class="mb-4">Automated Exploratory Data Audit</h1>
        <div class="card mb-4 p-3">
            <h3>1. Dataset Overview</h3>
            <p><strong>Rows:</strong> {df.shape[0]} | <strong>Columns:</strong> {df.shape[1]} | <strong>Duplicates:</strong> {df.duplicated().sum()}</p>
        </div>
        <div class="card mb-4 p-3">
            <h3>2. Missing Values & Schema Types</h3>
            {missing_data}
        </div>
        <div class="card mb-4 p-3">
            <h3>3. Numerical 5-Number Summaries</h3>
            {num_summary}
        </div>
        <div class="card mb-4 p-3">
            <h3>4. Pearson Correlation Matrix</h3>
            {corr_data}
        </div>
    </body>
    </html>
    """
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(html_payload)
    print(f"[SUCCESS] Native HTML exploratory profile saved: '{output_report}'")

print("\nOpen the generated HTML report in any web browser to explore.")