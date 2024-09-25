import pandas as pd
import numpy as np
import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy import stats
from io import StringIO
import nbformat
from io import BytesIO

# Function to import CSV from BytesIO object.
def import_notebook(uploaded_file):
    """Read a CSV file from the uploaded file."""
    try:
        df = pd.read_csv(uploaded_file)  # Read directly from the uploaded file
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        return None
    return df

def preprocess_data(df, notebook_cells, columns_to_drop, drop_na_all, drop_na_any, impute_numerical, impute_categorical, remove_duplicates, convert_to_category, winsorize_data):
    start_time = time.time()
    initial_rows = len(df)
    removed_rows_all = removed_rows_na = 0

    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"## Dropped Columns: {', '.join(columns_to_drop)}"))

    if drop_na_all:
        try:
            df.dropna(how='all', inplace=True)
            removed_rows_all = initial_rows - len(df)
        except Exception as e:
            st.error(f"Error removing rows with all missing values: {e}")

    if drop_na_any:
        try:
            df.replace('', np.nan, inplace=True)
            initial_rows_after_all = len(df)
            df.dropna(inplace=True)
            removed_rows_na = initial_rows_after_all - len(df)
        except Exception as e:
            st.error(f"Error removing rows with missing values: {e}")

    if impute_numerical:
        try:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
            imputed_numerical = df[numerical_cols].isnull().sum().sum()
        except Exception as e:
            st.error(f"Error imputing missing numerical values: {e}")
            imputed_numerical = 0
    else:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    if impute_categorical:
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            imputed_categorical = df[categorical_cols].isnull().sum().sum()
        except Exception as e:
            st.error(f"Error imputing missing categorical values: {e}")
            imputed_categorical = 0
    else:
        categorical_cols = df.select_dtypes(include=['object']).columns

    if remove_duplicates:
        try:
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            removed_duplicates = initial_rows - len(df)
        except Exception as e:
            st.error(f"Error removing duplicate rows: {e}")
            removed_duplicates = 0
    else:
        removed_duplicates = 0

    if convert_to_category:
        try:
            for col in categorical_cols:
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        except Exception as e:
            st.error(f"Error converting columns to category type: {e}")

    winsorized_rows = []
    winsorize_limits = [0.05, 0.05]
    if winsorize_data:
        try:
            for col in numerical_cols:
                original_data = df[col].copy()
                df[col] = stats.mstats.winsorize(df[col], limits=winsorize_limits)
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

    notebook_cells.append(nbformat.v4.new_markdown_cell(f"- Removed {removed_rows_all} rows with all missing values." if drop_na_all else "- Did not remove rows with all missing values."))
    notebook_cells.append(nbformat.v4.new_markdown_cell(f"- Removed {removed_rows_na} rows with missing values." if drop_na_any else "- Did not remove rows with any missing values."))
    imputation_summary = [
        f"- Imputed {imputed_numerical} missing numerical values." if imputed_numerical > 0 else "- No missing numerical values imputed.",
        f"- Imputed {imputed_categorical} missing categorical values." if imputed_categorical > 0 else "- No missing categorical values imputed.",
        f"- Removed {removed_duplicates} duplicate rows." if removed_duplicates > 0 else "- No duplicate rows removed.",
        f"- Winsorized: {len(winsorized_rows)} rows, {len(numerical_cols)} cols using limits {winsorize_limits}." if winsorize_data else "- Did not winsorize data."
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


# Function for univariate analysis using Plotly
def univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Univariate Analysis**")

    # Plot for numerical columns
    for col in numerical_cols:
        st.write(f"### Distribution of {col}")
        fig = px.histogram(df, x=col, nbins=30, title=f'Distribution of {col}', marginal='box')
        st.plotly_chart(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"px.histogram(df, x='{col}', nbins=30)"))

    # Plot for categorical columns
    for col in categorical_cols:
        st.write(f"### Count plot of {col}")
        fig = px.histogram(df, x=col, title=f'Count plot of {col}', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"px.histogram(df, x='{col}')"))


# Function for multivariate analysis using Plotly
def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Multivariate Analysis**")

    # Pairplot for numerical columns
    if len(numerical_cols) > 1:
        st.write("### Pairplot of numerical columns")
        fig = px.scatter_matrix(df[numerical_cols], title='Scatter Matrix of Numerical Columns')
        st.plotly_chart(fig)
        notebook_cells.append(nbformat.v4.new_code_cell("px.scatter_matrix(df[numerical_cols])"))

    # Correlation heatmap using Plotly
    st.write("### Correlation Heatmap")
    correlation_matrix = df[numerical_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(title='Correlation Heatmap', xaxis_title='Variables', yaxis_title='Variables')
    st.plotly_chart(fig)
    notebook_cells.append(nbformat.v4.new_code_cell("go.Figure(data=go.Heatmap(...))"))


def clustering_analysis(df, numerical_cols, notebook_cells):
    st.write("### Clustering Analysis")
    
    # KMeans Clustering
    num_clusters = st.slider("Select number of clusters for KMeans", min_value=1, max_value=10, value=3)
    kmeans = KMeans(n_clusters=num_clusters)
    
    # Fitting the model
    kmeans.fit(df[numerical_cols])
    
    # Adding cluster column to the dataframe
    df['Cluster'] = kmeans.labels_
    st.success(f"Clustered data into {num_clusters} clusters.")
    
    # Plotting the clusters
    st.write("### Cluster Visualization")
    fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1], color='Cluster', title='Cluster Visualization')
    st.plotly_chart(fig)

    # Append code to notebook cells
    notebook_cells.append(nbformat.v4.new_code_cell(f"""
import pandas as pd
from sklearn.cluster import KMeans

num_clusters = {num_clusters}
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(df[{numerical_cols}])
df['Cluster'] = kmeans.labels_
""""))
    
def create_notebook(notebook_cells):
    """Creates a Jupyter Notebook from cells."""
    nb = nbformat.v4.new_notebook()
    nb.cells = notebook_cells
    return nb


def save_notebook(notebook):
    """Saves the Jupyter notebook to BytesIO and returns the bytes."""
    buffer = BytesIO()
    nbformat.write(notebook, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def main():
    st.title("Data Preprocessing and Analysis App")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = import_notebook(uploaded_file)
        
        if df is not None:
            notebook_cells = []
            st.write("### Initial Data Preview")
            st.write(df.head())
            notebook_cells.append(nbformat.v4.new_markdown_cell("## Initial Data Preview"))
            notebook_cells.append(nbformat.v4.new_code_cell("df.head()"))
            
            # Data Preprocessing
            st.write("## Data Preprocessing")
            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            drop_na_all = st.checkbox("Drop rows with all missing values", value=False)
            drop_na_any = st.checkbox("Drop rows with any missing values", value=False)
            impute_numerical = st.checkbox("Impute missing numerical values with mean", value=True)
            impute_categorical = st.checkbox("Impute missing categorical values with mode", value=True)
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            convert_to_category = st.checkbox("Convert categorical columns to category dtype", value=True)
            winsorize_data = st.checkbox("Winsorize numerical columns", value=True)

            df, categorical_cols, numerical_cols = preprocess_data(df, notebook_cells, columns_to_drop, drop_na_all, drop_na_any, impute_numerical, impute_categorical, remove_duplicates, convert_to_category, winsorize_data)

            # Provide option to download cleaned dataset
            st.download_button(
                label="Download Cleaned Dataset",
                data=df.to_csv(index=False),
                file_name='cleaned_dataset.csv',
                mime='text/csv'
            )

            # Univariate Analysis
            univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

            # Multivariate Analysis
            multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

            # Clustering Analysis
            clustering_analysis(df, numerical_cols, notebook_cells)

            # Create and download notebook
            notebook = create_notebook(notebook_cells)
            notebook_bytes = save_notebook(notebook)
            st.download_button(
                label="Download Jupyter Notebook",
                data=notebook_bytes,
                file_name="data_analysis_notebook.ipynb",
                mime="application/x-ipynb+json"
            )


if __name__ == "__main__":
    main()
