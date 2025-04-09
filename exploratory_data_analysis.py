import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

df = pd.read_csv("../data/processed/breast_cancer_data_processed.csv")

# Survival Status Counts
# Figure 1: Count of Survival Status
plt.figure(figsize=(8,4)) # Set up figure for plotting
sns.countplot(x="Status OS", data=df) # Plotting "Status OS" counts
plt.title('Survival Status Count') # Adding title and labels to improve clarity
plt.xlabel('Status OS')
plt.ylabel ('Count')
plt.savefig("../figures/survival_status_count.png")
plt.close()


# Figure 2: Tumor Grade Distribution
# Custom labels for the x-axis
custom_labels = ['Grade 1 or 2', 'Grade 3']

# Tumor Grade Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Tumor grade_3.0', data=df)
plt.title('Tumor Grade Distribution')
plt.xticks(ticks=[0, 1], labels=custom_labels)
plt.savefig("../figures/tumor_grade_distribution.png")
plt.close()






# Figure 3: Kaplan-Meier Survival Curve Overall
kmf = KaplanMeierFitter()
kmf.fit(df["OS at baseline"], df["Status OS"])
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve Overall')
plt.xlabel('Time in Weeks')
plt.ylabel('Survival Probability')
plt.savefig("../figures/km_curve_overall.png")
plt.close()


# Figure 4: Kaplan-Meier Survival Curves for "No Visceral" and "Visceral" Groups

# Filter the DataFrame to include only rows where metastatic site is "No visceral"
no_visceral_df = df[df['metastatic site'] == 'No visceral']

# Filter the DataFrame to include only rows where metastatic site is "Visceral"
visceral_df = df[df['metastatic site'] == 'Visceral']

# Initialize KaplanMeierFitter model
kmf_no_visceral = KaplanMeierFitter()
kmf_visceral = KaplanMeierFitter()

# Plotting
plt.figure(figsize=(10, 6))

# Fit the model for the "No visceral" group
kmf_no_visceral.fit(durations=no_visceral_df["OS from weeks 4"], event_observed=no_visceral_df["Status OS"], label='No visceral')
kmf_no_visceral.plot_survival_function()

# Fit the model for the "Visceral" group
kmf_visceral.fit(durations=visceral_df["OS from weeks 4"], event_observed=visceral_df["Status OS"], label='Visceral')
kmf_visceral.plot_survival_function()

# Add title and labels
plt.title('Kaplan-Meier Survival Curves for "No Visceral" and "Visceral" Groups')
plt.xlabel('Time in weeks')
plt.ylabel('Survival Probability')

# Show legend
plt.legend(title='Metastatic Site')

# Show plot
plt.show()


print("Exploratory Data Analysis completed, figures saved.")
