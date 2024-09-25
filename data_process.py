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
import os
import nbformat


def import_notebook(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


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
    fig, ax = plt.subplots()
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    notebook_cells.append(
        nbformat.v4.new_code_cell("sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')"))


# Function for KMeans clustering
def kmeans_clustering(df, numerical_cols, notebook_cells, n_clusters=3):
    st.write(f"**KMeans Clustering with {n_clusters} Clusters**")

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

    # Show cluster centers
    st.write("Cluster Centers:")
    st.write(pd.DataFrame(kmeans.cluster_centers_, columns=numerical_cols))

    # Plot clusters
    st.write("### Cluster Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[numerical_cols[0]], y=df[numerical_cols[1]], hue='Cluster', data=df, palette="viridis", ax=ax)
    st.pyplot(fig)
    notebook_cells.append(nbformat.v4.new_code_cell(f"kmeans = KMeans(n_clusters={n_clusters})\n"
                                                    f"df['Cluster'] = kmeans.fit_predict(df[numerical_cols])"))


# Function for statistical testing
def perform_statistical_testing(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Statistical Testing**")

    # Example: ANOVA test for numerical vs categorical columns
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            st.write(f"ANOVA test for {num_col} by {cat_col}")
            groups = [group[num_col].values for name, group in df.groupby(cat_col)]
            f_val, p_val = stats.f_oneway(*groups)
            st.write(f"F-statistic: {f_val}, p-value: {p_val}")
            notebook_cells.append(nbformat.v4.new_code_cell(
                f"stats.f_oneway(*[group['{num_col}'].values for name, group in df.groupby('{cat_col}')])"))


# Main function to orchestrate the flow
def main():
    # Embed the updated banner image
    banner_path = "EDA App Banner.png"  # Update to the correct path
    st.image(banner_path, use_column_width=True)

    # Update the title to "EDA App"
    st.title("Exploratory Data Analysis App")
    notebook_cells = []

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        filename = uploaded_file.name  # Get the file name
        st.success(f"File '{filename}' uploaded successfully.")
        st.write("First five rows of the dataset:")
        st.write(df.head())
        notebook_cells.append(nbformat.v4.new_markdown_cell("## Initial Data Preview"))
        notebook_cells.append(nbformat.v4.new_code_cell("df.head()"))

        # Add code to automatically load the dataset in the generated notebook
        notebook_cells.insert(0, nbformat.v4.new_code_cell(f"import pandas as pd\ndf = pd.read_csv('{filename}')"))

        # Select columns to drop
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)

        # Preprocess data
        df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop)

        # Univariate analysis
        if st.checkbox("Perform Univariate Analysis"):
            univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # Multivariate analysis
        if st.checkbox("Perform Multivariate Analysis"):
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # KMeans clustering
        if st.checkbox("Perform KMeans Clustering"):
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)
            kmeans_clustering(df, numerical_cols, notebook_cells, n_clusters=n_clusters)

        # Statistical testing
        if st.checkbox("Perform Statistical Testing"):
            perform_statistical_testing(df, categorical_cols, numerical_cols, notebook_cells)

        # Option to download notebook
        if st.button("Download Notebook"):
            notebook_content = nbformat.v4.new_notebook()
            notebook_content['cells'] = notebook_cells
            notebook_filename = "analysis_notebook.ipynb"
            with open(notebook_filename, 'w', encoding='utf-8') as f:
                nbformat.write(notebook_content, f)
            st.success(f"Notebook saved as {notebook_filename}")


if __name__ == "__main__":
    main()
