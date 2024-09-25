import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats as stats
import time
import numpy as np
import streamlit as st
import pandas as pd
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

    return df, categorical_cols, numerical_cols

# Function for univariate analysis
def univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Univariate Analysis**")
    if numerical_cols:
        num_col = st.selectbox("Select numerical column for Univariate Analysis", numerical_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[num_col], kde=True, ax=ax)
        ax.set_title(f'Histogram of {num_col}')
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"sns.histplot(df['{num_col}'], kde=True)\nplt.title('Histogram of {num_col}')\nplt.show()"))

    if categorical_cols:
        cat_col = st.selectbox("Select categorical column for Univariate Analysis", categorical_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=cat_col, data=df, ax=ax)
        ax.set_title(f'Count Plot of {cat_col}')
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"sns.countplot(x='{cat_col}', data=df)\nplt.title('Count Plot of {cat_col}')\nplt.show()"))

# Function for bivariate analysis
def bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Bivariate Analysis**")
    if numerical_cols and categorical_cols:
        num_col = st.selectbox("Select numerical column for Bivariate Analysis", numerical_cols)
        cat_col = st.selectbox("Select categorical column for Bivariate Analysis", categorical_cols)

        fig, ax = plt.subplots()
        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
        ax.set_title(f'Box Plot of {num_col} by {cat_col}')
        st.pyplot(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"sns.boxplot(x='{cat_col}', y='{num_col}', data=df)\nplt.title('Box Plot of {num_col} by {cat_col}')\nplt.show()"))

# Function for multivariate analysis
def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Multivariate Analysis**")
    if len(numerical_cols) >= 2:
        num_cols = st.multiselect("Select numerical columns for Multivariate Analysis", numerical_cols)
        if len(num_cols) == 2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=num_cols[0], y=num_cols[1], data=df, hue=df[categorical_cols[0]])
            ax.set_title(f'Scatter Plot of {num_cols[0]} vs {num_cols[1]}')
            st.pyplot(fig)
            notebook_cells.append(nbformat.v4.new_code_cell(f"sns.scatterplot(x='{num_cols[0]}', y='{num_cols[1]}', data=df, hue=df['{categorical_cols[0]}'])\nplt.title('Scatter Plot of {num_cols[0]} vs {num_cols[1]}')\nplt.show()"))

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

    ax2.scatter(df[numerical_cols[0]], df[numerical_cols[1]], c=df['Cluster'], cmap='viridis', marker='o')
    ax2.set_title(f'KMeans Clustering Results (k={k})')
    ax2.set_xlabel(numerical_cols[0])
    ax2.set_ylabel(numerical_cols[1])

    st.pyplot(fig)

    # Append notebook cell for clustering analysis
    notebook_cells.append(nbformat.v4.new_code_cell(
        f"for k in range(1, {max_k + 1}):\n"
        f"    kmeans = KMeans(n_clusters=k)\n"
        f"    kmeans.fit(df[numerical_cols])\n"
        f"    inertia.append(kmeans.inertia_)\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n"
        "ax1.plot(range(1, max_k + 1), inertia, marker='o')\n"
        "ax1.set_title('Elbow Method for Optimal k')\n"
        "ax1.set_xlabel('Number of clusters (k)')\n"
        "ax1.set_ylabel('Inertia')\n"
        "ax2.scatter(df[numerical_cols[0]], df[numerical_cols[1]], c=df['Cluster'], cmap='viridis', marker='o')\n"
        "ax2.set_title(f'KMeans Clustering Results (k={k})')\n"
        "ax2.set_xlabel(numerical_cols[0])\n"
        "ax2.set_ylabel(numerical_cols[1])"
    ))

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

        # Add a Run button to execute ANOVA
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

# Streamlit App
def main():
    # Embed the updated banner image
    banner_path = "EDA App Banner.png"  # Update to the correct path
    st.image(banner_path, use_column_width=True)
    
    # Title
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

        # Univariate Analysis Button
        if st.button("Run Univariate Analysis"):
            univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # Bivariate Analysis Button
        if st.button("Run Bivariate Analysis"):
            bivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # Multivariate Analysis Button
        if st.button("Run Multivariate Analysis"):
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

        # KMeans Clustering Button
        if st.button("Run Clustering Analysis"):
            clustering_analysis(df, numerical_cols, notebook_cells)

        # Statistical Tests
        anova(df, categorical_cols, numerical_cols)
        t_test(df, categorical_cols, numerical_cols)

        # Input fields for file paths
        notebook_path = st.text_input("Path to save the notebook (.ipynb)", "EDA_Notebook.ipynb")
        csv_path = st.text_input("Path to save the transformed CSV file", "Transformed_Data.csv")

        # Save notebook cells to a .ipynb file
        if st.button("Save Notebook and Transformed Data"):
            # Save the notebook
            with open(notebook_path, "w") as f:
                nb = nbformat.v4.new_notebook()
                nb.cells = notebook_cells
                nbformat.write(nb, f)

            # Save the transformed data
            df.to_csv(csv_path, index=False)
            st.success(f"Notebook saved as {notebook_path} and transformed data saved as {csv_path}")

if __name__ == "__main__":
    main()
