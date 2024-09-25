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


# Function for clustering analysis
def clustering_analysis(df, numerical_cols, notebook_cells):
    st.write("**Clustering Analysis**")

    # Selecting the number of clusters
    k = st.slider("Select number of clusters (k)", min_value=1, max_value=10, value=3)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df[numerical_cols])
    st.write(f"### KMeans Clustering Results (k={k})")
    st.write(df[['Cluster'] + list(numerical_cols)].head())
    notebook_cells.append(nbformat.v4.new_code_cell(f"kmeans = KMeans(n_clusters={k})\n"
                                                      "df['Cluster'] = kmeans.fit_predict(df[numerical_cols])"))


# Function to run statistical tests
def statistical_tests(df, numerical_cols, notebook_cells):
    st.write("**Statistical Tests**")
    
    test_options = ['T-test', 'ANOVA', 'Chi-squared']
    test_choice = st.selectbox("Select statistical test", test_options)

    if test_choice == 'T-test':
        col1 = st.selectbox("Select first numerical column", numerical_cols)
        col2 = st.selectbox("Select second numerical column", numerical_cols)

        stat, p = stats.ttest_ind(df[col1], df[col2], nan_policy='omit')
        st.write(f"T-test results: statistic={stat}, p-value={p}")
        notebook_cells.append(nbformat.v4.new_code_cell(f"stat, p = stats.ttest_ind(df['{col1}'], df['{col2}'], nan_policy='omit')"))

    elif test_choice == 'ANOVA':
        col1 = st.selectbox("Select numerical column", numerical_cols)
        col2 = st.selectbox("Select categorical column", categorical_cols)

        model = stats.f_oneway(*(df[df[col2] == group][col1] for group in df[col2].unique()))
        st.write(f"ANOVA results: F-statistic={model.statistic}, p-value={model.pvalue}")
        notebook_cells.append(nbformat.v4.new_code_cell(f"model = stats.f_oneway(*(df[df['{col2}] == group][{col1}] for group in df['{col2}'].unique()))"))

    elif test_choice == 'Chi-squared':
        col1 = st.selectbox("Select first categorical column", categorical_cols)
        col2 = st.selectbox("Select second categorical column", categorical_cols)

        contingency_table = pd.crosstab(df[col1], df[col2])
        stat, p, dof, expected = stats.chi2_contingency(contingency_table)
        st.write(f"Chi-squared test results: statistic={stat}, p-value={p}")
        notebook_cells.append(nbformat.v4.new_code_cell(f"contingency_table = pd.crosstab(df['{col1}'], df['{col2}'])\n"
                                                         "stat, p, dof, expected = stats.chi2_contingency(contingency_table)"))


# Main function for the Streamlit app
def main():
    st.title("Exploratory Data Analysis App")

    # Select CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview")
        st.write(df.head())

        # Initialize notebook cells
        notebook_cells = []

        # Preprocess Data
        df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop=[])

        # Univariate Analysis
        univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # Multivariate Analysis
        multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # Clustering Analysis
        clustering_analysis(df, numerical_cols, notebook_cells)

        # Statistical Tests
        statistical_tests(df, numerical_cols, notebook_cells)

        # Save notebook cells to a .ipynb file
        with open("EDA_Notebook.ipynb", "w") as f:
            nb = nbformat.v4.new_notebook()
            nb.cells = notebook_cells
            nbformat.write(nb, f)

        st.success("Notebook saved as EDA_Notebook.ipynb")


if __name__ == "__main__":
    main()
