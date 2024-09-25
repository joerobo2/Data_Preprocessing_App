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
    start_time = time.time()  # Start timer
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

    preprocess_time = time.time() - start_time  # End timer
    st.write(f"Preprocessing took {preprocess_time:.2f} seconds")

    # Adding preprocessing steps to the notebook
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

    # Summary logs
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

    # Streamlit display logs
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

    # Display data information
    st.write("**Data Information**")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    notebook_cells.append(nbformat.v4.new_markdown_cell("## Data Information"))
    notebook_cells.append(nbformat.v4.new_code_cell("df.info()"))

    return df, categorical_cols, numerical_cols


def univariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    for col in categorical_cols:
        st.header(f"Univariate Analysis: {col}")
        st.bar_chart(df[col].value_counts())
        st.markdown(f"**Mode:** {df[col].mode().values[0]}")
        notebook_cells.append(
            nbformat.v4.new_markdown_cell(f"### Univariate Analysis: {col}\n**Mode:** {df[col].mode().values[0]}\n"))
        notebook_cells.append(nbformat.v4.new_code_cell(f"display(df['{col}'].value_counts().plot(kind='bar'))"))

    for col in numerical_cols:
        st.header(f"Univariate Analysis: {col}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), bins=30, alpha=0.5, ax=ax[0], color='skyblue')
        sns.boxplot(x=df[col].dropna(), ax=ax[1], color='lightgreen')
        st.pyplot(fig)

        summary_stats = df[col].describe().to_frame().T
        st.table(summary_stats)
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Univariate Analysis: {col}\n**Summary Statistics:**"))
        notebook_cells.append(nbformat.v4.new_code_cell("df['{}'].describe().to_frame().T".format(col)))


def multivariate_analysis(df, categorical_cols, numerical_cols, notebook_cells):
    # Categorical vs Categorical
    for col1, col2 in zip(categorical_cols, categorical_cols[1:]):
        st.header(f"Multivariate Analysis: {col1} vs. {col2}")
        crosstab = pd.crosstab(df[col1], df[col2])
        st.table(crosstab)
        chi2, p, _, _ = stats.chi2_contingency(crosstab)
        notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Multivariate Analysis: {col1} vs. {col2}"))
        notebook_cells.append(nbformat.v4.new_code_cell(f"display(pd.crosstab(df['{col1}'], df['{col2}']))"))

    # Numerical vs Numerical
    st.header("Correlation Matrix")
    corr_matrix = df[numerical_cols].corr()
    st.dataframe(corr_matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", cbar=True, ax=ax, fmt='.2f', annot_kws={"size": 8})
    st.pyplot(fig)
    notebook_cells.append(nbformat.v4.new_markdown_cell("### Correlation Matrix"))
    notebook_cells.append(nbformat.v4.new_code_cell(
        "sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 8});"))

    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.reset_index()
    corr_pairs.columns = ['Variable1', 'Variable2', 'Correlation']
    corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False).reset_index(drop=True)

    top_corr = corr_pairs.head(10)
    st.header("Top 10 Correlation Coefficients")
    st.table(top_corr)

    bottom_corr = corr_pairs.tail(10).iloc[::-1]
    st.header("Bottom 10 Correlation Coefficients (Reversed Order)")
    st.table(bottom_corr)

    notebook_cells.append(nbformat.v4.new_markdown_cell("### Correlation Coefficients Summary"))
    notebook_cells.append(nbformat.v4.new_code_cell(
        "corr_matrix = df.corr()\n"
        "corr_pairs = corr_matrix.unstack()\n"
        "corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]\n"
        "corr_pairs = corr_pairs.reset_index()\n"
        "corr_pairs.columns = ['Variable1', 'Variable2', 'Correlation']\n"
        "corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False).reset_index(drop=True)\n"
        "display(corr_pairs.head(10))\n"
        "display(corr_pairs.tail(10).iloc[::-1])"))

    # Categorical vs Numerical
    for col1 in categorical_cols:
        for col2 in numerical_cols:
            st.header(f"Multivariate Analysis: {col1} vs {col2}")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[col1], y=df[col2], ax=ax, palette="Set2")
            st.pyplot(fig)
            f_stat, p_val = stats.f_oneway(*[df[col2][df[col1] == group] for group in df[col1].unique()])
            notebook_cells.append(nbformat.v4.new_markdown_cell(f"### Multivariate Analysis: {col1} vs {col2}"))
            notebook_cells.append(
                nbformat.v4.new_code_cell(f"display(sns.boxplot(x=df['{col1}'], y=df['{col2}'], palette='Set2'));"))


def kmeans_clustering(df, numerical_cols, notebook_cells, n_clusters=3):
    st.header("KMeans Clustering")
    if len(numerical_cols) >= 2:
        kmeans = KMeans(n_clusters=n_clusters)
        df['kmeans_cluster'] = kmeans.fit_predict(df[numerical_cols].dropna())

        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numerical_cols)
        st.table(cluster_centers)

        # Plot KMeans Clusters
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df, x=numerical_cols[i], y=numerical_cols[j], hue='kmeans_cluster',
                                palette='viridis', ax=ax)
                ax.set_title(f'KMeans Clustering: {numerical_cols[i]} vs {numerical_cols[j]}')
                st.pyplot(fig)

                notebook_cells.append(nbformat.v4.new_markdown_cell(
                    f"### KMeans Clustering: {numerical_cols[i]} vs {numerical_cols[j]}"))
                notebook_cells.append(nbformat.v4.new_code_cell(
                    f"sns.scatterplot(data=df, x='{numerical_cols[i]}', y='{numerical_cols[j]}', hue='kmeans_cluster', palette='viridis');"
                ))

        notebook_cells.append(nbformat.v4.new_markdown_cell("### KMeans Clustering"))
        notebook_cells.append(nbformat.v4.new_code_cell(
            f"kmeans = KMeans(n_clusters={n_clusters})\ndf['kmeans_cluster'] = kmeans.fit_predict(df[{numerical_cols}].dropna())\npd.DataFrame(kmeans.cluster_centers_, columns={numerical_cols})"
        ))


def statistical_testing(df, categorical_cols, numerical_cols, notebook_cells):
    anova_results = []
    ttest_results = []

    # Categorical vs Numerical
    for col1 in categorical_cols:
        for col2 in numerical_cols:
            # ANOVA Test
            try:
                f_stat, p_val = stats.f_oneway(*[df[col2][df[col1] == group].dropna() for group in df[col1].unique()])
                anova_results.append(
                    {"Variable1": col1, "Variable2": col2, "Test": "ANOVA", "Statistic": f_stat, "P-value": p_val})
            except Exception as e:
                st.error(f"ANOVA test failed for {col1} vs {col2}: {e}")

            # T-Test (only for binary categorical variables)
            if df[col1].nunique() == 2:
                try:
                    group1 = df[col2][df[col1] == df[col1].unique()[0]].dropna()
                    group2 = df[col2][df[col1] == df[col1].unique()[1]].dropna()
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    ttest_results.append(
                        {"Variable1": col1, "Variable2": col2, "Test": "T-test", "Statistic": t_stat, "P-value": p_val})
                except Exception as e:
                    st.error(f"T-test failed for {col1} vs {col2}: {e}")

    # Numerical vs Numerical
    for i, col1 in enumerate(numerical_cols):
        for col2 in numerical_cols[i + 1:]:
            # ANOVA Test
            try:
                f_stat, p_val = stats.f_oneway(df[col1].dropna(), df[col2].dropna())
                anova_results.append(
                    {"Variable1": col1, "Variable2": col2, "Test": "ANOVA", "Statistic": f_stat, "P-value": p_val})
            except Exception as e:
                st.error(f"ANOVA test failed for {col1} vs {col2}: {e}")

            # T-Test
            try:
                t_stat, p_val = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
                ttest_results.append(
                    {"Variable1": col1, "Variable2": col2, "Test": "T-test", "Statistic": t_stat, "P-value": p_val})
            except Exception as e:
                st.error(f"T-test failed for {col1} vs {col2}: {e}")

    # Display ANOVA Results
    if anova_results:
        anova_df = pd.DataFrame(anova_results)
        st.header("ANOVA Results Summary")
        st.dataframe(anova_df)
        notebook_cells.append(nbformat.v4.new_markdown_cell("## ANOVA Results Summary"))
        notebook_cells.append(nbformat.v4.new_code_cell("anova_df = pd.DataFrame(anova_results)\nanova_df"))

    # Display T-Test Results
    if ttest_results:
        ttest_df = pd.DataFrame(ttest_results)
        st.header("T-Test Results Summary")
        st.dataframe(ttest_df)
        notebook_cells.append(nbformat.v4.new_markdown_cell("## T-Test Results Summary"))
        notebook_cells.append(nbformat.v4.new_code_cell("ttest_df = pd.DataFrame(ttest_results)\nttest_df"))


def save_results(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)
    st.success(f"Results saved to {file_path}")


def main():
    st.title("Data Preprocessing and Analysis App")

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
        if
