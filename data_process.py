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

# Function to import CSV from BytesIO object.
def import_notebook(uploaded_file):
    """Read a CSV file from the uploaded file."""
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


# Function for bivariate analysis
def bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Bivariate Analysis**")

    # Scatter plots for numerical vs numerical
    if len(numerical_cols) > 1:
        x_col = st.selectbox("Select X numerical column", numerical_cols)
        y_col = st.selectbox("Select Y numerical column", numerical_cols)

        st.write(f"### Scatter plot of {x_col} vs {y_col}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col)
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"sns.scatterplot(data=df, x='{x_col}', y='{y_col}')"))

    # Box plots for numerical vs categorical
    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        cat_col = st.selectbox("Select categorical column for box plot", categorical_cols)

        st.write(f"### Box plot of {cat_col} vs numerical columns")
        fig, ax = plt.subplots()
        for num_col in numerical_cols:
            sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
            st.pyplot(fig)
            notebook_cells.append(nbformat.v4.new_code_cell(f"sns.boxplot(data=df, x='{cat_col}', y='{num_col}')"))


# Function for multivariate analysis
def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Multivariate Analysis**")

    if len(numerical_cols) > 1:
        target_col = st.selectbox("Select target numerical column", numerical_cols)
        features = st.multiselect("Select feature numerical columns", numerical_cols)
        if features:
            X = df[features]
            y = df[target_col]

            # KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(X)
            df['Cluster'] = kmeans.labels_

            st.write(f"### KMeans Clustering on {target_col} with features {features}")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Cluster'], palette='viridis')
            st.pyplot(fig)
            notebook_cells.append(nbformat.v4.new_code_cell(f"sns.scatterplot(x=df['{features[0]}'], y=df['{features[1]}'], hue=df['Cluster'], palette='viridis')"))


# Function for ANOVA test
def anova_test(df, categorical_cols, numerical_cols):
    st.write("**ANOVA Test**")

    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        cat_col = st.selectbox("Select categorical column for ANOVA", categorical_cols)
        num_col = st.selectbox("Select numerical column for ANOVA", numerical_cols)

        if st.button("Run ANOVA"):
            groups = [group[num_col].values for name, group in df.groupby(cat_col)]
            f_stat, p_val = stats.f_oneway(*groups)

            st.write(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                st.success("Reject the null hypothesis: at least one group mean is different.")
            else:
                st.warning("Fail to reject the null hypothesis: no significant difference in means.")


# Function for T-tests
def t_test(df, categorical_cols, numerical_cols):
    st.write("**T-Test**")

    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        cat_col = st.selectbox("Select categorical column for T-test", categorical_cols)
        num_col = st.selectbox("Select numerical column for T-test", numerical_cols)

        if st.button("Run T-test"):
            unique_values = df[cat_col].unique()
            if len(unique_values) != 2:
                st.error("T-test requires exactly two groups.")
                return

            group1 = df[df[cat_col] == unique_values[0]][num_col]
            group2 = df[df[cat_col] == unique_values[1]][num_col]

            t_stat, p_val = stats.ttest_ind(group1, group2)

            st.write(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                st.success("Reject the null hypothesis: means are significantly different.")
            else:
                st.warning("Fail to reject the null hypothesis: no significant difference in means.")


# Streamlit App
def main():
    st.title("Exploratory Data Analysis (EDA) App")

    notebook_cells = []

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        df = import_notebook(uploaded_file)
        if df is not None:  # Only proceed if df is successfully read
            st.write("Data Preview")
            st.dataframe(df.head())

            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop)

            univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)
            bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)
            anova_test(df, categorical_cols, numerical_cols)
            t_test(df, categorical_cols, numerical_cols)

            with open("notebook_summary.ipynb", "w") as f:
                nbformat.write(nbformat.v4.new_notebook(cells=notebook_cells), f)

if __name__ == "__main__":
    main()
