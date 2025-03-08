
import pandas as pd

if __name__ == "__main__":

    # Load CSV into a DataFrame
    df = pd.read_csv("../data/search_trends_raw.csv")
    df['Woche'] = pd.to_datetime(df['Woche'], format='%d/%m/%Y')
    df = df.rename(columns={"Woche": "Day"})
    df = df.set_index('Day')
    df = df.resample('D').interpolate(method="polynomial", order=2)
    # Display the first few rows
    df = df.reset_index()



    df = df.melt(id_vars=['Day'], var_name='keyword', value_name='freq')
    df_sorted = df.sort_values(by="Day")

    df_sorted.to_csv("../data/preprocessed.csv", index=False)