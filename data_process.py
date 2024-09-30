import pandas as pd
import numpy as np
import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy import stats
from io import StringIO, BytesIO
import nbformat


# Function to import CSV from BytesIO object.
def import_notebook(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        return None
    return df


# Function to preprocess data
def preprocess_data(df, notebook_cells, columns_to_drop, drop_na_all, drop_na_any, impute_numerical, impute_categorical,
                    remove_duplicates, convert_to_category, winsorize_data):
    initial_rows = len(df)
    removed_rows_all = removed_rows_na = 0

    # Columns to drop
    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"## Dropped Columns: {', '.join(columns_to_drop)}"))

    # Drop rows with all NaN
    if drop_na_all:
        try:
            df.dropna(how='all', inplace=True)
            removed_rows_all = initial_rows - len(df)
        except Exception as e:
            st.error(f"Error removing rows with all missing values: {e}")

    # Drop rows with any NaN
    if drop_na_any:
        try:
            df.replace('', np.nan, inplace=True)
            initial_rows_after_all = len(df)
            df.dropna(inplace=True)
            removed_rows_na = initial_rows_after_all - len(df)
        except Exception as e:
            st.error(f"Error removing rows with missing values: {e}")

    # Impute numerical
    if impute_numerical:
        try:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        except Exception as e:
            st.error(f"Error imputing missing numerical values: {e}")

    # Impute categorical
    if impute_categorical:
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        except Exception as e:
            st.error(f"Error imputing missing categorical values: {e}")

    # Remove duplicates
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

    # Convert to category
    if convert_to_category:
        try:
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        except Exception as e:
            st.error(f"Error converting columns to category type: {e}")

    # Winsorize
    if winsorize_data:
        try:
            winsorize_limits = [0.05, 0.05]
            for col in numerical_cols:
                df[col] = stats.mstats.winsorize(df[col], limits=winsorize_limits)
        except Exception as e:
            st.error(f"Error winsorizing data: {e}")

    st.write("**Data Information**")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    notebook_cells.append(nbformat.v4.new_markdown_cell("## Data Information"))
    notebook_cells.append(nbformat.v4.new_code_cell("df.info()"))

    return df, numerical_cols, categorical_cols


# Function for univariate analysis
def univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Univariate Analysis**")

    for col in numerical_cols:
        st.write(f"### Distribution of {col}")
        fig = px.histogram(df, x=col, nbins=30)
        st.plotly_chart(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"px.histogram(df, x='{col}', nbins=30)"))

    for col in categorical_cols:
        st.write(f"### Count plot of {col}")
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"px.histogram(df, x='{col}')"))


# Function for multivariate analysis
def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    st.write("**Multivariate Analysis**")

    if len(numerical_cols) > 1:
        st.write("### Pairplot of numerical columns")
        fig = px.scatter_matrix(df[numerical_cols])
        st.plotly_chart(fig)
        notebook_cells.append(nbformat.v4.new_code_cell(f"px.scatter_matrix(df[{list(numerical_cols)}])"))

    st.write("### Correlation Heatmap")
    correlation_matrix = df[numerical_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis'
    ))
    st.plotly_chart(fig)
    notebook_cells.append(nbformat.v4.new_code_cell(f"correlation_matrix = df[{list(numerical_cols)}].corr()\n"
                                                     f"fig = go.Figure(data=go.Heatmap(\n"
                                                     f"    z=correlation_matrix.values,\n"
                                                     f"    x=correlation_matrix.columns,\n"
                                                     f"    y=correlation_matrix.index,\n"
                                                     f"    colorscale='Viridis'\n"
                                                     f"))"))


# Function for clustering analysis
def clustering_analysis(df, numerical_cols, notebook_cells):
    st.write("### Clustering Analysis")

    num_clusters = st.slider("Select number of clusters for KMeans", 1, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df[numerical_cols])
    df['Cluster'] = kmeans.labels_

    st.write("### Cluster Visualization")
    fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1], color='Cluster')
    st.plotly_chart(fig)

    notebook_cells.append(nbformat.v4.new_code_cell(f"""
from sklearn.cluster import KMeans
num_clusters = {num_clusters}
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(df[{list(numerical_cols)}])
df['Cluster'] = kmeans.labels_
fig = px.scatter(df, x='{numerical_cols[0]}', y='{numerical_cols[1]}', color='Cluster')
"""))


# Function to create a Jupyter notebook
def create_notebook(notebook_cells, uploaded_filename):
    nb = nbformat.v4.new_notebook()

    # Add metadata to the notebook
    nb['metadata'] = {
        'kernelspec': {
            'name': 'python3',  # Specify the kernel name, e.g., python3
            'display_name': 'Python 3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.8',  # Specify the Python version
            'mimetype': 'text/x-python',
            'file_extension': '.py',
            'pygments_lexer': 'ipython3',
        },
    }

    # Add first cell with necessary imports
    nb.cells.append(nbformat.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy import stats
from io import StringIO
import time
"""))

    # Add cell to read the dataset
    nb.cells.append(nbformat.v4.new_code_cell(f"df = pd.read_csv('{uploaded_filename}')"))

    # Append other notebook cells (analysis steps)
    nb.cells.extend(notebook_cells)

    return nb


# Function to save the notebook as a byte stream
def save_notebook(notebook):
    buffer = BytesIO()
    notebook_content = nbformat.writes(notebook).encode('utf-8')
    buffer.write(notebook_content)
    buffer.seek(0)
    return buffer.getvalue()


def main():
    # Banner or header
    banner_path = "EDA App Banner.png"  # Update to the correct path
    st.image(banner_path, use_column_width=True)
    st.title("DataLytica")
    st.write(
        "DataLytica: A comprehensive tool for seamless data preprocessing, univariate/multivariate analysis, clustering, and Jupyter notebook generation."
    )

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = import_notebook(uploaded_file)
        uploaded_filename = uploaded_file.name

        if df is not None:
            notebook_cells = []

            # Tabs layout for each section
            tabs = st.tabs(["Data Preprocessing", "Univariate Analysis", "Multivariate Analysis", "Clustering Analysis",
                            "Jupyter Notebook"])

            # Preprocessing Tab
            with tabs[0]:
                st.header("Data Preprocessing")
                columns_to_drop = st.multiselect("Select columns to drop", df.columns)
                drop_na_all = st.checkbox("Drop rows with all NaN values", False)
                drop_na_any = st.checkbox("Drop rows with any NaN values", False)
                impute_numerical = st.checkbox("Impute missing numerical values with mean", True)
                impute_categorical = st.checkbox("Impute missing categorical values with mode", True)
                remove_duplicates = st.checkbox("Remove duplicate rows", True)
                convert_to_category = st.checkbox("Convert object columns to category type", True)
                winsorize_data = st.checkbox("Winsorize numerical data", False)

                df, numerical_cols, categorical_cols = preprocess_data(
                    df, notebook_cells, columns_to_drop, drop_na_all, drop_na_any, impute_numerical,
                    impute_categorical, remove_duplicates, convert_to_category, winsorize_data
                )

                st.download_button(
                    label="Download Cleaned Dataset",
                    data=df.to_csv(index=False),
                    file_name='cleaned_dataset.csv',
                    mime='text/csv'
                )

            # Univariate Analysis Tab
            with tabs[1]:
                st.header("Univariate Analysis")
                univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

            # Multivariate Analysis Tab
            with tabs[2]:
                st.header("Multivariate Analysis")
                multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells)

            # Clustering Analysis Tab
            with tabs[3]:
                st.header("Clustering Analysis")
                clustering_analysis(df, numerical_cols, notebook_cells)

            # Jupyter Notebook Tab
            with tabs[4]:
                st.header("Download Jupyter Notebook")
                notebook = create_notebook(notebook_cells, uploaded_filename)
                notebook_bytes = save_notebook(notebook)

                st.download_button(
                    label="Download Jupyter Notebook",
                    data=notebook_bytes,
                    file_name="analysis_notebook.ipynb",
                    mime="application/x-ipynb+json"
                )

    # Developer credit
    st.write("Developed by: Joseph Robinson")


if __name__ == "__main__":
    main()
