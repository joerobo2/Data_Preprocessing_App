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

    # Check if there are enough numerical columns for analysis
    if len(numerical_cols) > 1:
        st.write("### Pairplot of numerical columns")
        
        # Create a pairplot
        pairplot_fig = sns.pairplot(df[numerical_cols])
        st.pyplot(pairplot_fig)
        notebook_cells.append(nbformat.v4.new_code_cell("sns.pairplot(df[numerical_cols])"))

    # Create a subplot for the correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))  # Adjusted figure size for better visibility
    
    # Heatmap of correlations for numerical columns
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                ax=ax, square=True, cbar_kws={"shrink": .8}, 
                annot_kws={"size": 10}, linewidths=.5)  # Added annotation size and linewidths

    ax.set_title('Correlation Heatmap', fontsize=16)  # Increased title font size
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)  # Adjusted x-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)  # Adjusted y-axis labels
    
    st.pyplot(fig)
    notebook_cells.append(
        nbformat.v4.new_code_cell("fig, ax = plt.subplots(figsize=(12, 10))\n"
                                   "correlation_matrix = df[numerical_cols].corr()\n"
                                   "sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', "
                                   "ax=ax, square=True, cbar_kws={'shrink': .8}, "
                                   "annot_kws={'size': 10}, linewidths=.5)\n"
                                   "ax.set_title('Correlation Heatmap', fontsize=16)\n"
                                   "ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)\n"
                                   "ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)"))

# Function for clustering analysis with elbow method
def clustering_analysis(df, numerical_cols, notebook_cells):
    st.write("**Clustering Analysis**")

    # Selecting the number of clusters for KMeans
    max_k = st.slider("Select maximum number of clusters (k)", min_value=1, max_value=10, value=3)

    # Elbow method to find the optimal number of clusters
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[numerical_cols])
        inertia.append(kmeans.inertia_)

    # Create subplots for elbow chart and cluster scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow Chart
    ax1.plot(range(1, max_k + 1), inertia, marker='o')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')

    # KMeans clustering and scatter plot
    k = st.slider("Select number of clusters for KMeans", min_value=1, max_value=max_k, value=3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

    ax2.scatter(df[numerical_cols[0]], df[numerical_cols[1]], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k')
    ax2.set_title(f'KMeans Clustering Results (k={k})')
    ax2.set_xlabel(numerical_cols[0])
    ax2.set_ylabel(numerical_cols[1])

    st.pyplot(fig)
    notebook_cells.append(nbformat.v4.new_code_cell(f"for k in range(1, {max_k + 1}):\n"
                                                    f"    kmeans = KMeans(n_clusters=k)\n"
                                                    f"    kmeans.fit(df[numerical_cols])\n"
                                                    f"    inertia.append(kmeans.inertia_)\n"
                                                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n"
                                                    "ax1.plot(range(1, max_k + 1), inertia, marker='o')\n"
                                                    "ax1.set_title('Elbow Method for Optimal k')\n"
                                                    "ax1.set_xlabel('Number of clusters (k)')\n"
                                                    "ax1.set_ylabel('Inertia')\n"
                                                    "ax2.scatter(df[numerical_cols[0]], df[numerical_cols[1]], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k')\n"
                                                    "ax2.set_title(f'KMeans Clustering Results (k={k})')\n"
                                                    "ax2.set_xlabel(numerical_cols[0])\n"
                                                    "ax2.set_ylabel(numerical_cols[1])"))


# Function for ANOVA test across all numerical columns
def anova_test(df, categorical_cols, numerical_cols):
    st.write("**ANOVA Test Summary**")

    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        cat_col = st.selectbox("Select categorical column for ANOVA", categorical_cols)
        
        # Initialize a summary table
        summary_data = {
            "Numerical Column": [],
            "F-statistic": [],
            "p-value": [],
            "Significance": []
        }

        for num_col in numerical_cols:
            groups = [group[num_col].values for name, group in df.groupby(cat_col)]
            f_stat, p_val = stats.f_oneway(*groups)
            
            summary_data["Numerical Column"].append(num_col)
            summary_data["F-statistic"].append(f_stat)
            summary_data["p-value"].append(p_val)
            summary_data["Significance"].append("Reject H0" if p_val < 0.05 else "Fail to Reject H0")

        # Create a DataFrame from the summary data
        summary_df = pd.DataFrame(summary_data)

        st.write(summary_df)


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
