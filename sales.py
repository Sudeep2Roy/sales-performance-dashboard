# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Step 2: Load the Dataset
df = pd.read_csv("Walmart_Sales.csv")  # Make sure file is in the same directory
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Step 3: Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['IsHoliday'] = df['Holiday_Flag'].map({1: 'Yes', 0: 'No'})

# Step 4: Sales Trend Over Time
sales_trend = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

plt.figure()
sns.lineplot(data=sales_trend, x='Date', y='Weekly_Sales')
plt.title('Weekly Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig("sales_trend_over_time.png")
plt.show()

# Step 5: Total Sales by Store
store_sales = df.groupby('Store')['Weekly_Sales'].sum().reset_index().sort_values(by='Weekly_Sales', ascending=False)

plt.figure()
sns.barplot(data=store_sales, x='Store', y='Weekly_Sales', palette='viridis')
plt.title('Total Sales by Store')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("total_sales_by_store.png")
plt.show()

# Step 6: Monthly Sales Trend
monthly_sales = df.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
monthly_sales['Year-Month'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)

plt.figure()
sns.lineplot(data=monthly_sales, x='Year-Month', y='Weekly_Sales', marker="o")
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")
plt.show()

# Step 7: Holiday vs Non-Holiday Sales
holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()

plt.figure()
sns.barplot(data=holiday_sales, x='IsHoliday', y='Weekly_Sales', palette='coolwarm')
plt.title('Average Sales: Holiday vs Non-Holiday')
plt.xlabel('Holiday Week')
plt.ylabel('Average Weekly Sales')
plt.tight_layout()
plt.savefig("holiday_vs_nonholiday_sales.png")
plt.show()

# Step 8: Correlation Heatmap
corr_features = df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
corr_matrix = corr_features.corr()

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Step 9: Save Reports (Optional)
monthly_sales.to_csv("monthly_sales_report.csv", index=False)
store_sales.to_csv("store_sales_report.csv", index=False)
holiday_sales.to_csv("holiday_sales_report.csv", index=False)

print("✅ Analysis Complete.")
print("📁 All plots saved as PNG images.")
