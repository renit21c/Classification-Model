import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("tips.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

data['sex'].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
axes[0].set_title('Gender Distribution')
axes[0].set_ylabel('')

data.groupby('sex')['tip'].mean().plot.bar(ax=axes[1], color=['#4ECDC4', '#FF6B6B'])
axes[1].set_title('Average Tip by Gender')
axes[1].set_ylabel('Tip ($)')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('visualization.png')

