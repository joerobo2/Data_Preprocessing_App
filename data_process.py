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
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Pair Plot"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"import seaborn as sns\n"
            f"fig = sns.pairplot(df[{pair_plot_cols}])\n"
            f"fig.show()"
        ))

    st.subheader("Correlation Matrix")
    corr = df[numerical_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    notebook_cells.append(nbformat.v4.new_markdown_cell("### Correlation Matrix"))
    notebook_cells.append(nbformat.v4.new_code_cell(
        f"corr = df[{numerical_cols}].corr()\n"
        f"fig, ax = plt.subplots()\n"
        f"sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)\n"
        f"plt.xticks(rotation=45, ha='right')\n"
        f"plt.yticks(rotation=0)\n"
        f"fig.show()"
    ))


def kmeans_clustering(df, numerical_cols, notebook_cells, max_clusters=10):
    st.header("KMeans Clustering")
    for col in numerical_cols:
        st.subheader(f"KMeans Clustering for {col}")
        sse = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(df[[col]])
            sse.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, max_clusters + 1), sse)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('SSE')
        ax.set_title(f'Elbow Method for {col}')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### KMeans Clustering for {col}"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"sse = []\n"
            f"for k in range(1, {max_clusters + 1}):\n"
            f"    kmeans = KMeans(n_clusters=k)\n"
            f"    kmeans.fit(df[['{col}']])\n"
            f"    sse.append(kmeans.inertia_)\n"
            f"fig, ax = plt.subplots()\n"
            f"ax.plot(range(1, {max_clusters + 1}), sse)\n"
            f"ax.set_xlabel('Number of Clusters')\n"
            f"ax.set_ylabel('SSE')\n"
            f"ax.set_title('Elbow Method for {col}')\n"
            f"plt.xticks(rotation=45)\n"
            f"fig.show()"
        ))


def perform_statistical_testing(df, categorical_cols, numerical_cols, notebook_cells):
    st.header("Statistical Testing (ANOVA, T-tests)")

    # ANOVA tests
    anova_results = []
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            if df[cat_col].nunique() > 1:
                anova_result = stats.f_oneway(*(df[df[cat_col] == group][num_col] for group in df[cat_col].unique()))
                anova_results.append((cat_col, num_col, anova_result.pvalue))

    st.subheader("ANOVA Test Results")
    if anova_results:
        anova_df = pd.DataFrame(anova_results, columns=["Categorical Variable", "Numerical Variable", "p-value"])
        st.dataframe(anova_df)
        notebook_cells.append(nbformat.v4.new_markdown_cell("### ANOVA Test Results"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"anova_results = []\n"
            f"for cat_col in {categorical_cols}:\n"
            f"    for num_col in {numerical_cols}:\n"
            f"        if df[cat_col].nunique() > 1:\n"
            f"            anova_result = stats.f_oneway(*(df[df[cat_col] == group][num_col] for group in df[cat_col].unique()))\n"
            f"            anova_results.append((cat_col, num_col, anova_result.pvalue))\n"
            f"anova_df = pd.DataFrame(anova_results, columns=['Categorical Variable', 'Numerical Variable', 'p-value'])\n"
            f"anova_df"
        ))

    # T-tests
    ttest_results = []
    for num_col1 in numerical_cols:
        for num_col2 in numerical_cols:
            if num_col1 != num_col2:
                ttest_result = stats.ttest_ind(df[num_col1], df[num_col2])
                ttest_results.append((num_col1, num_col2, ttest_result.pvalue))

    st.subheader("T-test Results")
    if ttest_results:
        ttest_df = pd.DataFrame(ttest_results, columns=["Numerical Variable 1", "Numerical Variable 2", "p-value"])
        st.dataframe(ttest_df)
        notebook_cells.append(nbformat.v4.new_markdown_cell("### T-test Results"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"ttest_results = []\n"
            f"for num_col1 in {numerical_cols}:\n"
            f"    for num_col2 in {numerical_cols}:\n"
            f"        if num_col1 != num_col2:\n"
            f"            ttest_result = stats.ttest_ind(df[num_col1], df[num_col2])\n"
            f"            ttest_results.append((num_col1, num_col2, ttest_result.pvalue))\n"
            f"ttest_df = pd.DataFrame(ttest_results, columns=['Numerical Variable 1', 'Numerical Variable 2', 'p-value'])\n"
            f"ttest_df"
        ))


def save_results(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)
    st.success(f"Results saved to {file_path}")


def main():
    # Embed the updated banner image
    banner_path = "EDA_App_Banner.png"  # Update to the correct path
    st.image(banner_path, use_column_width=True)

    # Update the title to "EDA App"
    st.title("EDA App")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        columns_to_drop = st.multiselect("Select columns to drop (if any)", df.columns.tolist())

        st.write("Columns in the dataset:", df.columns.tolist())

        # Notebook cells container
        notebook_cells = []

        # Preprocess the data
        df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop)

        # Univariate Analysis
        if st.checkbox("Perform Univariate Analysis", value=True):
            univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # Multivariate Analysis
        if st.checkbox("Perform Multivariate Analysis", value=True):
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # KMeans Clustering
        if st.checkbox("Perform KMeans Clustering", value=True):
            max_clusters = st.slider("Select the maximum number of clusters", 2, 10, 3)
            kmeans_clustering(df, numerical_cols, notebook_cells, max_clusters=max_clusters)

        # Statistical Testing
        if st.checkbox("Perform Statistical Testing (ANOVA, T-tests)", value=True):
            perform_statistical_testing(df, categorical_cols, numerical_cols, notebook_cells)

        # Save Results
        save_path = st.text_input("Enter file path to save results", "results.csv")
        if st.button("Save Results"):
            save_results(df, save_path)

        # Generate and Download Notebook
        notebook = nbformat.v4.new_notebook(cells=notebook_cells)
        notebook_io = StringIO()
        nbformat.write(notebook, notebook_io)
        st.download_button(
            label="Download Jupyter Notebook",
            data=BytesIO(notebook_io.getvalue().encode('utf-8')),
            file_name="analysis_notebook.ipynb",
            mime="application/octet-stream"
        )


if __name__ == "__main__":
    main()
