import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats as stats
import time
import numpy as np
import streamlit as st
from io import StringIO, BytesIO
from scipy.stats.mstats import winsorize
import pandas as pd
import os
import nbformat


def import_notebook(uploaded_file):
    """Function to import CSV from BytesIO object."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        return None


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
        "    df[col].fillna(df[col].mode()[0], inplace=True)\n"
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


def univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.header("Univariate Analysis")
    for col in numerical_cols:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Distribution of {col}"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"fig, ax = plt.subplots()\n"
            f"sns.histplot(df['{col}'], kde=True, ax=ax)\n"
            f"plt.xticks(rotation=45)\n"
            f"fig.show()"
        ))

    for col in categorical_cols:
        st.subheader(f"Count Plot of {col}")
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Count Plot of {col}"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"fig, ax = plt.subplots()\n"
            f"sns.countplot(x='{col}', data=df, ax=ax)\n"
            f"plt.xticks(rotation=45)\n"
            f"fig.show()"
        ))


def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.header("Multivariate Analysis")
    pair_plot_cols = st.multiselect("Select columns for pair plot", numerical_cols.tolist())
    if pair_plot_cols:
        st.subheader("Pair Plot")
        fig = sns.pairplot(df[pair_plot_cols])
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_markdown_cell("### Pair Plot"))
        notebook_cells.append(nbformat.v4.new_code_cell(f"sns.pairplot(df[{pair_plot_cols}])"))

    if len(numerical_cols) >= 2:
        x_col = st.selectbox("Select X column for scatter plot", numerical_cols.tolist())
        y_col = st.selectbox("Select Y column for scatter plot", numerical_cols.tolist())
        st.subheader(f"Scatter Plot of {x_col} vs {y_col}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Scatter Plot of {x_col} vs {y_col}"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"fig, ax = plt.subplots()\n"
            f"sns.scatterplot(x='{x_col}', y='{y_col}', data=df, ax=ax)\n"
            f"fig.show()"
        ))


def bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.header("Bivariate Analysis")
    if numerical_cols and categorical_cols:
        cat_col = st.selectbox("Select categorical column for box plot", categorical_cols.tolist())
        num_col = st.selectbox("Select numerical column for box plot", numerical_cols.tolist())
        st.subheader(f"Box Plot of {num_col} by {cat_col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Box Plot of {num_col} by {cat_col}"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"fig, ax = plt.subplots()\n"
            f"sns.boxplot(x='{cat_col}', y='{num_col}', data=df, ax=ax)\n"
            f"fig.show()"
        ))


def main():
    # Embed the updated banner image
    banner_path = "EDA App Banner.png"  # Update to the correct path
    st.image(banner_path, use_column_width=True)
    
    st.title("Data Analysis App")
    notebook_cells = []
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = import_notebook(uploaded_file)
        if df is not None:  # Proceed only if the DataFrame is valid
            # Data Preprocessing
            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop)
            
            # Univariate Analysis
            univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)
            
            # Bivariate Analysis
            bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)
            
            # Multivariate Analysis
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)


if __name__ == "__main__":
    main()
