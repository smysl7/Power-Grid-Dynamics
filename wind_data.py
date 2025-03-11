import pandas as pd

wind_raw_data_path = "WIND.xlsx"
xls_wind = pd.ExcelFile(wind_raw_data_path)

wind_data = pd.read_excel(xls_wind, sheet_name="Sheet1")
wind_data["DateTime"] = pd.to_datetime(wind_data["Date"].astype(str) + " " + wind_data["Time"].astype(str))

start_date = "2014-01-08"
end_date = "2014-08-05"

filtered_data = wind_data[(wind_data["DateTime"] >= start_date) & (wind_data["DateTime"] <= end_date)]

daily_total_power = filtered_data.groupby("Date")["P(MW)"].sum().reset_index()

daily_total_power["P(MW)"] = daily_total_power["P(MW)"].interpolate(method="linear").fillna(method="ffill").fillna(method="bfill")

daily_total_power["Date"] = pd.to_datetime(daily_total_power["Date"])

output_path = "Total_Daily_Wind_Power.csv"
daily_total_power.to_csv(output_path, index=False)

print("✅ Processing complete! First few rows of processed data:")
print(daily_total_power.head())

print(f"Data saved as: {output_path}")
