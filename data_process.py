import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats as stats
import time
import numpy as np
import streamlit as st
from io import StringIO
from scipy.stats.mstats import winsorize
import pandas as pd
import nbformat

# Function to import CSV from BytesIO object.
def import_notebook(uploaded_file):
    """Read a CSV file from the uploaded file."""
    try:
        df = pd.read_csv(uploaded_file)  # Read directly from the uploaded file
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")
    return df

def preprocess_data(df, notebook_cells, columns_to_drop):
    start_time = time.time()
    initial_rows = len(df)
    removed_rows_all = removed_rows_na = 0

    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"## Dropped Columns: {', '.join(columns_to_drop)}"))

    try:
        df.dropna(how='all', inplace=True)
        removed_rows_all = initial_rows - len(df)
    except Exception as e:
        st.error(f"Error removing rows with all missing values: {e}")

    try:
        df.replace('', np.nan, inplace=True)
        initial_rows_after_all = len(df)
        df.dropna(inplace=True)
        removed_rows_na = initial_rows_after_all - len(df)
    except Exception as e:
        st.error(f"Error removing rows with missing values: {e}")

    try:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        imputed_numerical = df[numerical_cols].isnull().sum().sum()
    except Exception as e:
        st.error(f"Error imputing missing numerical values: {e}")
        imputed_numerical = 0

    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        imputed_categorical = df[categorical_cols].isnull().sum().sum()
    except Exception as e:
        st.error(f"Error imputing missing categorical values: {e}")
        imputed_categorical = 0

    try:
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        removed_duplicates = initial_rows - len(df)
    except Exception as e:
        st.error(f"Error removing duplicate rows: {e}")
        removed_duplicates = 0

    try:
        for col in categorical_cols:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    except Exception as e:
        st.error(f"Error converting columns to category type: {e}")

    winsorized_rows = []
    winsorize_limits = [0.05, 0.05]
    try:
        for col in numerical_cols:
            original_data = df[col].copy()
            df[col] = winsorize(df[col], limits=winsorize_limits)
            winsorized_diff = (original_data != df[col]).sum()
            if winsorized_diff > 0:
                winsorized_rows.append(winsorized_diff)
    except Exception as e:
        st.error(f"Error winsorizing data: {e}")

    preprocess_time = time.time() - start_time
    st.write(f"Preprocessing took {preprocess_time:.2f} seconds")

    notebook_cells.append(nbformat.v4.new_markdown_cell("## Preprocessing Summary"))
    notebook_cells.append(nbformat.v4.new_code_cell(
        "initial_rows = len(df)\n"
        "df.dropna(how='all', inplace=True)\n"
        "df.replace('', np.nan, inplace=True)\n"
        "df.dropna(inplace=True)\n"
        "removed_rows_all = initial_rows - len(df)\n\n"
        "numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns\n"
        "df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())\n"
        "imputed_numerical = df[numerical_cols].isnull().sum().sum()\n\n"
        "categorical_cols = df.select_dtypes(include=['object']).columns\n"
        "for col in categorical_cols:\n"
        "    df[col] = df[col].fillna(df[col].mode()[0])\n"
        "imputed_categorical = df[categorical_cols].isnull().sum().sum()\n\n"
        "initial_rows = len(df)\n"
        "df.drop_duplicates(inplace=True)\n"
        "removed_duplicates = initial_rows - len(df)\n\n"
        "for col in categorical_cols:\n"
        "    if df[col].nunique() / len(df) < 0.5:\n"
        "        df[col] = df[col].astype('category')\n\n"
        "from scipy.stats.mstats import winsorize\n"
        "for col in numerical_cols:\n"
        "    df[col] = winsorize(df[col], limits=[0.05, 0.05])"
    ))

    notebook_cells.append(nbformat.v4.new_markdown_cell(f"- Removed {removed_rows_all} rows with all missing values."))
    notebook_cells.append(nbformat.v4.new_markdown_cell(f"- Removed {removed_rows_na} rows with missing values."))
    notebook_cells.append(nbformat.v4.new_markdown_cell(
        f"- Imputed {imputed_numerical} missing numerical values." if imputed_numerical > 0 else "- No missing numerical values imputed."))
    notebook_cells.append(nbformat.v4.new_markdown_cell(
        f"- Imputed {imputed_categorical} missing categorical values." if imputed_categorical > 0 else "- No missing categorical values imputed."))
    notebook_cells.append(nbformat.v4.new_markdown_cell(
        f"- Removed {removed_duplicates} duplicate rows." if removed_duplicates > 0 else "- No duplicate rows removed."))
    notebook_cells.append(nbformat.v4.new_markdown_cell(
        f"- Winsorized: {len(winsorized_rows)} rows, {len(numerical_cols)} cols using limits {winsorize_limits}."))

    st.write("**Preprocessing Summary**")
    st.markdown(f"- Removed {removed_rows_all} rows with all missing values.")
    st.markdown(f"- Removed {removed_rows_na} rows with missing values.")
    imputation_summary = [
        f"- Imputed {imputed_numerical} missing numerical values." if imputed_numerical > 0 else "- No missing numerical values imputed.",
        f"- Imputed {imputed_categorical} missing categorical values." if imputed_categorical > 0 else "- No missing categorical values imputed.",
        f"- Removed {removed_duplicates} duplicate rows." if removed_duplicates > 0 else "- No duplicate rows removed.",
        f"- Winsorized: {len(winsorized_rows)} rows, {len(numerical_cols)} cols using limits {winsorize_limits}."
    ]
    for summary in imputation_summary:
        st.markdown(summary)

    st.write("**Data Information**")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    notebook_cells.append(nbformat.v4.new_markdown_cell("## Data Information"))
    notebook_cells.append(nbformat.v4.new_code_cell("df.info()"))

    return df, categorical_cols, numerical_cols


# Function for univariate analysis
def univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Univariate Analysis**")

    # Plot for numerical columns
    for col in numerical_cols:
        st.write(f"### Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"sns.histplot(df['{col}'], kde=True)"))

    # Plot for categorical columns
    for col in categorical_cols:
        st.write(f"### Count plot of {col}")
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"sns.countplot(x=df['{col}'])"))


# Function for multivariate analysis
def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Multivariate Analysis**")

    # Pairplot for numerical columns
    if len(numerical_cols) > 1:
        st.write("### Pairplot of numerical columns")
        fig = sns.pairplot(df[numerical_cols])
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_code_cell("sns.pairplot(df[numerical_cols])"))

    # Heatmap of correlations for numerical columns
    st.write("### Correlation heatmap")
    correlation_matrix = df[numerical_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    notebook_cells.append(nbformat.v4.new_code_cell("sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')"))

    # Clustering Analysis
    st.write("### Clustering Analysis")
    n_clusters = st.number_input("Number of Clusters:", min_value=1, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[numerical_cols])

    # Visualizing clusters
    fig, ax = plt.subplots()
    ax.scatter(df[numerical_cols[0]], df[numerical_cols[1]], c=df['cluster'], cmap='viridis', s=50)
    ax.set_xlabel(numerical_cols[0])
    ax.set_ylabel(numerical_cols[1])
    ax.set_title(f'KMeans Clustering with {n_clusters} clusters')
    st.pyplot(fig)

    notebook_cells.append(nbformat.v4.new_code_cell(
        f"kmeans = KMeans(n_clusters={n_clusters}, random_state=42)\n"
        f"df['cluster'] = kmeans.fit_predict(df[numerical_cols])\n"
        f"fig, ax = plt.subplots()\n"
        f"ax.scatter(df['{numerical_cols[0]}'], df['{numerical_cols[1]}'], c=df['cluster'], cmap='viridis', s=50)\n"
        f"ax.set_xlabel('{numerical_cols[0]}')\n"
        f"ax.set_ylabel('{numerical_cols[1]}')\n"
        f"ax.set_title('KMeans Clustering')"
    ))
# Function for statistical analysis
def statistical_analysis(df, numerical_cols, categorical_cols, notebook_cells):
    """Perform statistical analysis and add results to notebook cells."""
    for num_col in numerical_cols:
        for cat_col in categorical_cols:
            # Example statistical test (t-test)
            groups = [df[num_col][df[cat_col] == cat_val] for cat_val in df[cat_col].unique()]
            if len(groups) == 2:  # Ensure there are two groups for t-test
                t_stat, p_value = stats.ttest_ind(*groups)
                # Log the results in the notebook cells
                notebook_cells.append(nbformat.v4.new_code_cell(f"""
# Statistical Analysis Results for {num_col} vs {cat_col}
t-statistic: {t_stat}, p-value: {p_value}
"""))

# Function for exporting notebook cells
def export_notebook_cells(notebook_cells, filepath):
    """Export notebook cells to a specified file."""
    nb = nbformat.v4.new_notebook()
    nb.cells = notebook_cells
    try:
        with open(filepath, 'w') as f:
            nbformat.write(nb, f)
    except Exception as e:
        raise IOError(f"Could not save notebook to {filepath}: {e}")

# Main function for the Streamlit app
def main():
    # Embed the updated banner image
    banner_path = "EDA App Banner.png"  # Update to the correct path
    st.image(banner_path, use_column_width=True)

    st.title("Data Preprocessing and Analysis App")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        notebook_cells = []

        # Add dataset reading cell
        notebook_cells.append(nbformat.v4.new_code_cell("df = pd.read_csv('uploaded_file.csv')"))

        df = import_notebook(uploaded_file)  # Ensure you have this function defined
        columns_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist())

        df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop)  # Ensure this function is defined

        # Checkboxes for analysis functions
        if st.checkbox("Perform Univariate Analysis"):
            univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)  # Ensure this is defined

        if st.checkbox("Perform Multivariate Analysis"):
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)  # Ensure this is defined

        if st.checkbox("Perform T-tests and ANOVA"):
            statistical_analysis(df, numerical_cols, categorical_cols, notebook_cells)

        # Exporting notebook
        export_option = st.checkbox("Export analysis as notebook")
        if export_option:
            # Add text input for user to specify the file path
            file_path = st.text_input("Specify file path to save the notebook (including .ipynb):", 'analysis_notebook.ipynb')
            if st.button("Save Notebook"):
                if not file_path.endswith('.ipynb'):
                    st.error("File path must end with '.ipynb'")
                else:
                    try:
                        export_notebook_cells(notebook_cells, file_path)
                        st.success(f"Notebook exported successfully to {file_path}!")
                    except Exception as e:
                        st.error(f"Error saving notebook: {e}")

if __name__ == "__main__":
    main()
