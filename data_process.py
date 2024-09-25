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
        if removed_rows_all > 0:
            st.success(f"Removed {removed_rows_all} rows with all missing values.")
    except Exception as e:
        st.error(f"Error removing rows with all missing values: {e}")

    try:
        df.replace('', np.nan, inplace=True)
        initial_rows_after_all = len(df)
        df.dropna(inplace=True)
        removed_rows_na = initial_rows_after_all - len(df)
        if removed_rows_na > 0:
            st.success(f"Removed {removed_rows_na} rows with missing values.")
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
        if removed_duplicates > 0:
            st.success(f"Removed {removed_duplicates} duplicate rows.")
    except Exception as e:
        st.error(f"Error removing duplicate rows: {e}")
        removed_duplicates = 0

    # Convert object columns with few unique values to categorical
    try:
        for col in categorical_cols:
            if df[col].nunique() / len(df) < 0.5:  # Change threshold as needed
                df[col] = df[col].astype('category')
        st.success("Converted object columns to categorical types where applicable.")
    except Exception as e:
        st.error(f"Error converting columns to category type: {e}")

    return df, categorical_cols.tolist(), numerical_cols.tolist()

# Function for ANOVA across all numerical columns
def anova(df, categorical_cols, numerical_cols):
    st.write("**ANOVA Summary**")

    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        cat_col = st.selectbox("Select categorical column for ANOVA", categorical_cols)

        # Initialize a summary table
        summary_data = {
            "Numerical Column": [],
            "F-statistic": [],
            "p-value": [],
            "Significance": []
        }

        # Add a Run button to execute the ANOVA
        if st.button("Run ANOVA"):
            # Ensure the categorical column has more than two groups for ANOVA
            if df[cat_col].nunique() > 1:
                groups = [df[df[cat_col] == group][numerical_cols].dropna() for group in df[cat_col].unique()]

                for num_col in numerical_cols:
                    f_stat, p_val = stats.f_oneway(*[group[num_col] for group in groups if num_col in group.columns])
                    
                    summary_data["Numerical Column"].append(num_col)
                    summary_data["F-statistic"].append(f_stat)
                    summary_data["p-value"].append(p_val)
                    summary_data["Significance"].append("Reject H0" if p_val < 0.05 else "Fail to Reject H0")

                # Create a DataFrame from the summary data
                summary_df = pd.DataFrame(summary_data)

                st.write(summary_df)
            else:
                st.warning("ANOVA requires more than one group in the selected categorical column.")

# Function for T-test across all numerical columns
def t_test(df, categorical_cols, numerical_cols):
    st.write("**T-test Summary**")

    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        cat_col = st.selectbox("Select categorical column for T-tests", categorical_cols)

        # Initialize a summary table
        summary_data = {
            "Numerical Column": [],
            "T-statistic": [],
            "p-value": [],
            "Significance": []
        }

        # Add a Run button to execute the T-test
        if st.button("Run T-test"):
            # Ensure the categorical column has exactly two groups for T-test
            if df[cat_col].nunique() == 2:
                for num_col in numerical_cols:
                    group1 = df[df[cat_col] == df[cat_col].unique()[0]][num_col]
                    group2 = df[df[cat_col] == df[cat_col].unique()[1]][num_col]
                    
                    t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')

                    summary_data["Numerical Column"].append(num_col)
                    summary_data["T-statistic"].append(t_stat)
                    summary_data["p-value"].append(p_val)
                    summary_data["Significance"].append("Reject H0" if p_val < 0.05 else "Fail to Reject H0")

                # Create a DataFrame from the summary data
                summary_df = pd.DataFrame(summary_data)

                st.write(summary_df)
            else:
                st.warning("T-test requires exactly two groups in the selected categorical column.")

# Function for univariate analysis
def univariate_analysis(df, numerical_cols, notebook_cells):
    st.write("**Univariate Analysis**")

    if len(numerical_cols) > 0:
        selected_col = st.selectbox("Select numerical column for Univariate Analysis", numerical_cols)
        
        if st.button("Run Univariate Analysis"):
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            ax.set_title(f'Univariate Analysis of {selected_col}')
            st.pyplot(fig)

            # Append notebook cell for univariate analysis
            notebook_cells.append(nbformat.v4.new_code_cell(
                f"sns.histplot(df['{selected_col}'], kde=True)\n"
                f"plt.title('Univariate Analysis of {selected_col}')\n"
                "plt.show()"
            ))

# Function for bivariate analysis
def bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Bivariate Analysis**")

    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        selected_cat = st.selectbox("Select categorical column for Bivariate Analysis", categorical_cols)
        selected_num = st.selectbox("Select numerical column for Bivariate Analysis", numerical_cols)
        
        if st.button("Run Bivariate Analysis"):
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_cat], y=df[selected_num], ax=ax)
            ax.set_title(f'Bivariate Analysis of {selected_num} by {selected_cat}')
            st.pyplot(fig)

            # Append notebook cell for bivariate analysis
            notebook_cells.append(nbformat.v4.new_code_cell(
                f"sns.boxplot(x=df['{selected_cat}'], y=df['{selected_num}'])\n"
                f"plt.title('Bivariate Analysis of {selected_num} by {selected_cat}')\n"
                "plt.show()"
            ))

# Function for multivariate analysis
def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Multivariate Analysis**")

    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        selected_cat = st.selectbox("Select categorical column for Multivariate Analysis", categorical_cols)
        
        if st.button("Run Multivariate Analysis"):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(x=df[selected_cat], y=df[numerical_cols[0]], data=df, ax=ax)
            ax.set_title(f'Multivariate Analysis of {numerical_cols[0]} by {selected_cat}')
            st.pyplot(fig)

            # Append notebook cell for multivariate analysis
            notebook_cells.append(nbformat.v4.new_code_cell(
                f"sns.violinplot(x=df['{selected_cat}'], y=df['{numerical_cols[0]}'])\n"
                f"plt.title('Multivariate Analysis of {numerical_cols[0]} by {selected_cat}')\n"
                "plt.show()"
            ))

# Clustering Analysis
def clustering_analysis(df, numerical_cols, notebook_cells):
    st.write("**Clustering Analysis**")

    if len(numerical_cols) > 0:
        k = st.slider("Select number of clusters (k)", min_value=1, max_value=10, value=3)
        
        if st.button("Run Clustering Analysis"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df[numerical_cols])
            inertia = kmeans.inertia_
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(range(1, 11), inertia, marker='o')
            ax1.set_title('Elbow Method for Optimal k')
            ax1.set_xlabel('Number of clusters (k)')
            ax1.set_ylabel('Inertia')
            ax2.scatter(df[numerical_cols[0]], df[numerical_cols[1]], c=df['Cluster'], cmap='viridis')
            ax2.set_title(f'KMeans Clustering Results (k={k})')
            ax2.set_xlabel(numerical_cols[0])
            ax2.set_ylabel(numerical_cols[1])
            st.pyplot(fig)

            # Append notebook cell for clustering analysis
            notebook_cells.append(nbformat.v4.new_code_cell(
                f"kmeans = KMeans(n_clusters={k}, random_state=42)\n"
                f"df['Cluster'] = kmeans.fit_predict(df[{numerical_cols}])\n"
                f"inertia = kmeans.inertia_\n"
                f"plt.figure(figsize=(12, 5))\n"
                f"plt.subplot(1, 2, 1)\n"
                f"plt.plot(range(1, 11), inertia, marker='o')\n"
                f"plt.title('Elbow Method for Optimal k')\n"
                f"plt.xlabel('Number of clusters (k)')\n"
                f"plt.ylabel('Inertia')\n"
                f"plt.subplot(1, 2, 2)\n"
                f"plt.scatter(df[{numerical_cols[0]}], df[{numerical_cols[1]}], c=df['Cluster'], cmap='viridis')\n"
                f"plt.title('KMeans Clustering Results (k={k})')\n"
                f"plt.xlabel('{numerical_cols[0]}')\n"
                f"plt.ylabel('{numerical_cols[1]}')\n"
                "plt.show()"
            ))

def main():
    st.title("Exploratory Data Analysis App")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Import notebook cells
        notebook_cells = []

        df = import_notebook(uploaded_file)
        if df is not None:
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Preprocessing
            st.write("## Data Preprocessing")
            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop)

            # Univariate Analysis
            univariate_analysis(df, numerical_cols, notebook_cells)

            # Bivariate Analysis
            bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

            # Multivariate Analysis
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

            # Clustering Analysis
            clustering_analysis(df, numerical_cols, notebook_cells)

            # T-test Analysis
            t_test(df, categorical_cols, numerical_cols)

            # ANOVA Analysis
            anova(df, categorical_cols, numerical_cols)

            # Analysis
            st.write("## Data Analysis")
            if st.button("Show Data Summary"):
                st.write("### Summary Statistics")
                st.write(df.describe(include='all'))

            # Button to download notebook
            if st.button("Download Notebook"):
                notebook_filename = "EDA_notebook.ipynb"
                with open(notebook_filename, "w") as f:
                    nb = nbformat.v4.new_notebook(cells=notebook_cells)
                    nbformat.write(nb, f)
                with open(notebook_filename, "rb") as f:
                    st.download_button("Download EDA Notebook", f, file_name=notebook_filename)

if __name__ == "__main__":
    main()
