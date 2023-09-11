
#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr


# Load the dataset
df = pd.read_csv("/Users/susoeresia-eke/Downloads/nba_data_processed.csv")

#Data Cleaning!
# Check for missing values
print(df.isnull().sum())

# Fill missing values with zeros for numerical columns
df.fillna(0, inplace=True)

# Check for duplicate rows
duplicates = df[df.duplicated()]
# Remove duplicate rows, if any
df.drop_duplicates(inplace=True)


#Explore
# Display basic information about the dataset
print(df.info())

# Summary statistics of numerical columns
print(df.describe())


# Data Analysis

# Use Case 1 : To get started, calculate and visualize the distribution of player ages.
#The histogram displays the distribution of player ages, with the x-axis representing age and the y-axis representing the frequency (number of players).
# Calculate the distribution of player ages
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Player Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#Some meaningful Deductions
#Age Concentration: The highest concentration of players appears to be in the early to mid-20s, as indicated by the peak in the histogram. This suggests that a significant portion of NBA players in the 2022-23 season falls within this age range.
#Age Distribution: We can observe a gradual decline in the number of players as age increases beyond the early to mid-20s. This implies that there are fewer players in their late 20s, 30s, and 40s, highlighting the typical age range of NBA players.
#Youthful League: The concentration of players in the early to mid-20s may indicate that the NBA is a league with a significant presence of young talent, and teams may be investing in developing and nurturing emerging players.

#Use Case 2: Find the top 10 players with the highest points (PTS) per game.
#This list represents the top 10 players with the highest average Points Per Game (PPG) in the 2022-23 NBA season. PPG is a key metric for assessing a player's scoring ability and contribution to their team's offense.

# Calculate PTS per game (PPG) and display the top 10 players
df['PPG'] = df['PTS'] / df['G']
top_10_ppg = df[['Player', 'PPG']].sort_values(by='PPG', ascending=False).head(10)
print(top_10_ppg)

#Meaningful Deductions
#Scoring Leaders: The list highlights the leading scorers in the league for the specified season. Players like Louis King and RaiQuan Gray stand out with impressive PPG averages of 20.0 and 16.0, respectively.
#Diverse Scoring Levels: The list includes a range of players with varying scoring abilities. While some players average double-digit points (e.g., Louis King, RaiQuan Gray), others contribute fewer points per game (e.g., Jeenathan Williams, Skylar Mays).
#Emerging Talent: This list may include emerging or lesser-known players who are making their mark in terms of scoring. These players can be valuable assets to their respective teams and might be worth watching for future growth.


#Use Case 3: Explore the relationship between the number of games started (GS) and assists (AST).
#This scatter plot explores the relationship between the number of games started (GS) and the number of assists (AST) made by players. It helps us understand whether players who start more games tend to have more assists.

# Calculate the Pearson correlation coefficient between GS and AST
correlation_coefficient = df['GS'].corr(df['AST'])

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GS', y='AST', data=df, color='lightblue')
plt.title('Relationship Between Games Started (GS) and Assists (AST)')
plt.xlabel('Games Started (GS)')
plt.ylabel('Assists (AST)')

# Add correlation coefficient to the plot
plt.text(10, 100, f'Correlation Coefficient: {correlation_coefficient:.2f}', fontsize=12, color='red')

plt.show()

# Interpretation
if correlation_coefficient > 0:
    print("There is a positive correlation between Games Started (GS) and Assists (AST).")
elif correlation_coefficient < 0:
    print("There is a negative correlation between Games Started (GS) and Assists (AST).")
else:
    print("There is no significant linear correlation between Games Started (GS) and Assists (AST).")

#There is a positive correlation between Games Started (GS) and Assists (AST).

#Use Case 4: Calculate and visualize the correlation matrix of numerical attributes.

# Calculate the correlation matrix
# Select only numeric columns for correlation analysis
numeric_columns = df.select_dtypes(include='number')

# Calculate the correlation matrix
corr_matrix = numeric_columns.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Scattered throughout the heatmap, we observed pockets of blue, red, and orange. These pockets represented different correlation patterns between specific pairs of variables. Blue pockets indicated strong negative correlations, red pockets signified strong positive correlations, and orange pockets represented moderate positive correlations.

# Use Case 5: Team-Level Analysis - Average Points per Game (PPG) by Team
#In this analysis, we are examining the average Points Per Game (PPG) for each NBA team during the 2022-23 season. Each bar in the bar plot represents a team, and the height of the bar indicates the team's average PPG.

# Group the data by 'tm' (Team) and calculate the mean of 'PTS' for each team
team_stats = df.groupby('Tm')['PTS'].mean().reset_index()
team_stats.rename(columns={'PTS': 'Average_PPG'}, inplace=True)

# Sort the teams by average PPG in descending order
team_stats = team_stats.sort_values(by='Average_PPG', ascending=False)

# Create a bar plot to visualize average PPG by team
plt.figure(figsize=(12, 6))
sns.barplot(x='Average_PPG', y='Tm', data=team_stats, orient='h', palette='viridis')
plt.title('Average Points Per Game (PPG) by Team')
plt.xlabel('Average PPG')
plt.ylabel('Team')

# Display the interpretation
print('Interpretation:')
print('The bar plot displays the average Points Per Game (PPG) for each NBA team in the 2022-23 season.')
print('Each bar represents a team, and the height of the bar indicates their average PPG.')
print('Teams with higher average PPG have taller bars, indicating more scoring in games.')

plt.show()


# Interpretation of the histogram
plt.text(15, 7, 'Interpretation:', fontsize=12, fontweight='bold', color='blue')
plt.text(15, 6.5, 'The histogram displays the distribution of', fontsize=10, color='black')
plt.text(15, 6, 'average PPG for NBA teams in the 2022-23 season.', fontsize=10, color='black')
plt.text(15, 5.5, 'Each bar represents a range of average PPG', fontsize=10, color='black')
plt.text(15, 5, 'values for different teams.', fontsize=10, color='black')
plt.text(15, 4.5, 'Teams with higher average PPG are', fontsize=10, color='black')
plt.text(15, 4, 'located to the right on the histogram.', fontsize=10, color='black')


# Use Case 6: Position Analysis - Points Per Game (PPG) by Position
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pos', y='PTS', data=df, palette='Set2')
plt.title('Points Per Game (PPG) by Position')
plt.xlabel('Position')
plt.ylabel('PPG')
plt.show()

# Interpretation of the Box Plot
positions = df['Pos'].unique()
for position in positions:
    subset = df[df['Pos'] == position]
    median_ppg = subset['PTS'].median()
    q1 = subset['PTS'].quantile(0.25)
    q3 = subset['PTS'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = subset[(subset['PTS'] < lower_bound) | (subset['PTS'] > upper_bound)]

    print(f"Position: {position}")
    print(f"Median PPG: {median_ppg:.2f}")
    print(f"Interquartile Range (IQR): {iqr:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Number of Outliers: {len(outliers)}")
    print(f"Outlier Players: {', '.join(outliers['Player'])}\n")

#some meaningful Deductions**:
#Centers (C) tend to have a lower median PPG, suggesting their primary role may be focused on defense and rebounds rather than scoring.
#Shooting guards (SG) and point guards (PG) show higher variability in scoring, with some players being prolific scorers (outliers).
#Power forwards (PF) have a relatively high number of outliers, indicating that some PFs have scoring roles similar to small forwards (SF)

# Use Case 7: Player Efficiency Rating (PER) Analysis

# Calculate Player Efficiency Rating (PER) for each player
df['PER'] = (df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK'] - df['TOV'] - df['PF']) / df['MP']

# Plot the distribution of PER
plt.figure(figsize=(10, 6))
sns.histplot(df['PER'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Player Efficiency Rating (PER)')
plt.xlabel('PER')
plt.ylabel('Frequency')
plt.show()

# Interpretation of PER Distribution
mean_per = df['PER'].mean()
median_per = df['PER'].median()
std_per = df['PER'].std()

print(f"Mean PER: {mean_per:.2f}")
print(f"Median PER: {median_per:.2f}")
print(f"Standard Deviation of PER: {std_per:.2f}")

# Explanation
print("\nPlayer Efficiency Rating (PER) is a metric used to evaluate a player's overall contribution to their team's success. "
      "It takes into account various statistics such as points scored, rebounds, assists, steals, blocks, turnovers, and personal fouls, "
      "normalized per minute played (MP).")

print("\nMean PER represents the average PER value across all players in the dataset, providing an indication of the average "
      "efficiency level. Median PER represents the middle value of the PER distribution, which is less affected by outliers.")

print("\nStandard Deviation of PER measures the spread or variability in PER values. A higher standard deviation suggests greater "
      "variability in player efficiency within the dataset.")

# Top 10 Players with Highest PER
top_10_per = df[['Player', 'PER']].sort_values(by='PER', ascending=False).head(10)
print("\nTop 10 Players with Highest Player Efficiency Rating (PER):")
print(top_10_per)


#The histogram plot shows a bell-shaped curve, indicating that the distribution of PER is approximately normal. Most players have PER values concentrated around the mean PER, with fewer extreme outliers on both ends of the distribution.
#The top 10 players with the highest PER values include exceptional performers such as Stanley Umude, Donovan Williams, Giannis Antetokounmpo, and Joel Embiid. These players stand out for their efficiency and overall contributions to their teams.
#The mean PER, median PER, and standard deviation of PER provide summary statistics for the distribution, helping us understand the central tendency and spread of player efficiency.