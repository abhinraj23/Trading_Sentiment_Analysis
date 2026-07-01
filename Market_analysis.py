
import pandas as pd
import matplotlib.pyplot as plt

# Loading Datasets
trade_df = pd.read_csv("/storage/emulated/0/pydroid/files/historical_data.csv")
fear_df = pd.read_csv("/storage/emulated/0/pydroid/files/fear_greed_index.csv")

# Exploring Data

print(trade_df.head())
print(trade_df.info())
print(trade_df.isnull().sum())

print(fear_df.head())
print(fear_df.info())
print(fear_df.isnull().sum())

# Data Preprocessing

trade_df["Date"] = pd.to_datetime(
    trade_df["Timestamp IST"],
    format="%d-%m-%Y %H:%M"
).dt.date

fear_df["date"] = pd.to_datetime(fear_df["date"]).dt.date

# Merging process

merged_df = pd.merge(
    trade_df,
    fear_df,
    left_on="Date",
    right_on="date",
    how="left"
)

print(merged_df.head())
print(merged_df["classification"].value_counts(dropna=False))

# Analysis 1

avg_pnl = merged_df.groupby("classification")["Closed PnL"].agg(["count","mean","median"])
print(avg_pnl)

# Analysis 2
# Top Coins

print(merged_df["Coin"].value_counts().head())

top5 = merged_df["Coin"].value_counts().head(5).index

top5_df = merged_df[merged_df["Coin"].isin(top5)]

coin_sentiment = top5_df.groupby(
    ["classification","Coin"]
)["Closed PnL"].agg(["count","mean"])

print(coin_sentiment)

# Trade Result

def classify_trade(pnl):
    if pnl > 0:
        return "Winning"
    elif pnl < 0:
        return "Losing"
    else:
        return "Break-even"

merged_df["Trade Result"] = merged_df["Closed PnL"].apply(classify_trade)

print(merged_df["Trade Result"].value_counts())

trade_result = pd.crosstab(
    merged_df["classification"],
    merged_df["Trade Result"],
    normalize="index"
) * 100

print(trade_result)

# Direction Analysis

direction = pd.crosstab(
    merged_df["Direction"],
    merged_df["Trade Result"],
    normalize="index"
) * 100

print(direction)


# Graph 1

avg_graph = merged_df.groupby("classification")["Closed PnL"].mean()

avg_graph.plot(kind="bar", figsize=(8,5))
plt.title("Average Closed PnL by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Closed PnL")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph1.png", dpi=300)
plt.show()

# Graph 2

trade_result.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Trade Outcome Distribution by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Trade Distribution (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph2.png", dpi=300)
plt.show()


# Graph 3


coin_graph = top5_df.groupby(
    ["classification","Coin"]
)["Closed PnL"].mean().unstack()

coin_graph.plot(kind="bar", figsize=(10,6))
plt.title("Average Closed PnL of Top 5 Coins by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Closed PnL")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph3.png", dpi=300)
plt.show()

# Graph 4

trade_count = merged_df["classification"].value_counts().sort_index()

trade_count.plot(kind="bar", figsize=(8,5))
plt.title("Number of Trades by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Number of Trades")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph4.png", dpi=300)
plt.show()
