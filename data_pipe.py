import pandas as pd
import matplotlib.pyplot as plt
from prefect import task, flow


@task
def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    path (str): The path to the CSV data file.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_csv(path)


@task
def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by dropping duplicates and NaN values, and resetting the index.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
    pd.DataFrame: Cleaned data.
    """
    data = data.drop_duplicates()
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data


@task
def convert_dtypes(data: pd.DataFrame, types_dict: dict = None) -> pd.DataFrame:
    """
    Convert data types of the DataFrame columns.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.
    types_dict (dict): A dictionary specifying the desired data types for columns.

    Returns:
    pd.DataFrame: DataFrame with converted data types.
    """
    data = data.astype(dtype=types_dict)
    data["Date"] = pd.to_datetime(data["Date"])
    return data


@task
def data_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data analysis to calculate the average units sold by month.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing the average units sold by month.
    """
    data["month"] = data["Date"].dt.month
    new_df = data.groupby("month")["Units Sold"].mean()
    return new_df


@task
def data_visualization(new_df: pd.DataFrame, vis_type: str = "bar") -> pd.DataFrame:
    """
    Visualize the data using a specified plot type.

    Parameters:
    new_df (pd.DataFrame): The data to be visualized.
    vis_type (str): The type of plot to create (default is "bar").

    Returns:
    pd.DataFrame: The input DataFrame (for possible further use).
    """
    new_df.plot(kind=vis_type, figsize=(10, 5), title="Average Units Sold by Month")
    plt.savefig("average_units_sold_by_month.png")
    return new_df


@task
def save_to_csv(df: pd.DataFrame, filename: str):
    """
    Save the DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    filename (str): The name of the output CSV file.
    """
    df.to_csv(filename, index=False)
    return filename


@flow(name="Data Pipeline")
def run_pipeline(path: str):
    """
    Run the complete data pipeline.

    Parameters:
    path (str): The path to the CSV data file.
    """
    df = load_data(path)
    df_cleaned = data_cleaning(df)
    df_converted = convert_dtypes(
        df_cleaned, {"Product Category": "str", "Product Name": "str"}
    )
    analysis_result = data_analysis(df_converted)
    data_visualization(analysis_result, "line")
    save_to_csv(analysis_result, "average_units_sold_by_month.csv")


# Run the flow
if __name__ == "__main__":
    run_pipeline.serve(
        name="pass-params-deployment",
        parameters=dict(path="Online Sales Data.csv"),
    )
