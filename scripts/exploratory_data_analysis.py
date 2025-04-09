import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Load the raw clinical dataset
df = pd.read_csv("../data/raw/breast_cancer_data_raw.csv")

# ---------------------------------------------
# Figure 1: Count Plot of Survival Status
# ---------------------------------------------
plt.figure(figsize=(8,4)) # Set up figure for plotting
sns.countplot(x="Status OS", data=df) # Plotting "Status OS" counts
plt.title('Survival Status Count') # Adding title and labels to improve clarity
plt.xlabel('Status OS')
plt.ylabel ('Count')
plt.savefig("../figures/survival_status_count.png")
plt.close()

# ---------------------------------------------
# Figure 2: Tumor Grade Distribution
# ---------------------------------------------
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Tumor grade', data=df)
ax.set_title('Tumor Grade Distribution')
ax.set_xlabel('Tumor Grade')
ax.set_ylabel('Count')
ax.set_xticklabels(['Grade 3', 'Grade 1 or 2'])  # Set custom labels
plt.savefig("../figures/tumor_grade_distribution.png")
plt.close()

# ---------------------------------------------
# Figure 3: Kaplan-Meier Curve (Overall Survival)
# ---------------------------------------------
kmf = KaplanMeierFitter()
kmf.fit(df["OS at baseline"], df["Status OS"])
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve Overall')
plt.xlabel('Time in Weeks')
plt.ylabel('Survival Probability')
plt.savefig("../figures/km_curve_overall.png")
plt.close()

# ---------------------------------------------
# Figure 4: Kaplan-Meier Curve (Mutation vs. No Mutation)
# ---------------------------------------------
# Filter the DataFrame to include only rows where Mutation is "No mutated"
no_mutated_df = df[df['Mutation'] == 'No mutated']

# Filter the DataFrame to include only rows where Mutation is not "No mutated"
mutated_df = df[df['Mutation'] != 'No mutated']

# Initialize KaplanMeierFitter model
kmf_no_mutated = KaplanMeierFitter()
kmf_mutated = KaplanMeierFitter()

# Plotting
plt.figure(figsize=(10, 6))

# Fit the model for the "No mutated" group
kmf_no_mutated.fit(durations=no_mutated_df["OS from weeks 4"], event_observed=no_mutated_df["Status OS"], label='No mutation')
kmf_no_mutated.plot_survival_function()

# Fit the model for the "mutated" group
kmf_mutated.fit(durations=mutated_df["OS from weeks 4"], event_observed=mutated_df["Status OS"], label='Mutation')
kmf_mutated.plot_survival_function()

# Add title and labels
plt.title('Kaplan-Meier Survival Curves for "No mutation" and "Mutation" Groups')
plt.xlabel('Time in weeks')
plt.ylabel('Survival Probability')

# Show legend
plt.legend(title='Mutation')

# Save plot
plt.savefig("../figures/km_curve_mutation_vs_no_mutation.png")
plt.close()

# ---------------------------------------------
# Figure 5: Kaplan-Meier Curves by Specific Mutations (No Mutation, TP53, and PIK3CA Mutation)
# ---------------------------------------------
no_mutation = df[df['Mutation'] == 'No mutated']
TP53 = df[df['Mutation'] == 'TP53']
PIK3CA = df[df['Mutation'] == 'PIK3CA']

# Inititative KaplanMeierFitter model
kmf_no_mutation = KaplanMeierFitter()
kmf_TP53 = KaplanMeierFitter()
kmf_PIK3CA = KaplanMeierFitter()

# Plotting
plt.figure(figsize=(12, 8))

# Fit the model for the "No mutation" group
kmf_no_mutation.fit(durations=no_mutation["OS from weeks 4"], event_observed=no_mutation["Status OS"], label='No mutation')
kmf_no_mutation.plot_survival_function()

# Fit the model for the "TP53" group
kmf_TP53.fit(durations=TP53["OS from weeks 4"], event_observed=TP53["Status OS"], label=' TP53 Mutation')
kmf_TP53.plot_survival_function()

# Fit the model for the "PIK3CA" group
kmf_PIK3CA.fit(durations=PIK3CA["OS from weeks 4"], event_observed=PIK3CA["Status OS"], label='PIK3CA Mutation')
kmf_PIK3CA.plot_survival_function()

# Add title and labels
plt.title('Kaplan-Meier Survival Curves for "No mutation" and "TP53 and PIK3CA Mutation" Groups')
plt.xlabel('Time in weeks')
plt.ylabel('Survival Probability')

# Show legend
plt.legend(title='Mutation Type')

# Save plot
plt.savefig("../figures/km_curve_by_mutation.png")
plt.close()

# ---------------------------------------------
# Figure 6: Kaplan-Meier Curve by Metastatic Site (Visceral Vs. Non-visceral)
# ---------------------------------------------
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

# Save plot
plt.savefig("../figures/km_curve_by_metastatic_site.png")
plt.close()

# ---------------------------------------------
# Figure 7: Kaplan-Meier Curve by Tumor Subtype (Triple Negative and RH+ Groups)
# ---------------------------------------------
# Filter the DataFrame to include only rows where metastatic site is "No visceral"
triple_negative_df = df[df['subtype'] == 'Triple negative']

# Filter the DataFrame to include only rows where metastatic site is "Visceral"
rh_plus_df = df[df['subtype'] == 'RH+']

# Initialize KaplanMeierFitter model
kmf_triple_negative = KaplanMeierFitter()
kmf_rh_plus = KaplanMeierFitter()

# Plotting
plt.figure(figsize=(10, 6))

# Fit the model for the "No visceral" group
kmf_triple_negative.fit(durations=triple_negative_df["OS from weeks 4"], event_observed=triple_negative_df["Status OS"], label='Triple Negative')
kmf_triple_negative.plot_survival_function()

# Fit the model for the "Visceral" group
kmf_rh_plus.fit(durations=rh_plus_df["OS from weeks 4"], event_observed=rh_plus_df["Status OS"], label='RH+')
kmf_rh_plus.plot_survival_function()

# Add title and labels
plt.title('Kaplan-Meier Survival Curves for "Triple Negative" and "RH+" Groups')
plt.xlabel('Time in weeks')
plt.ylabel('Survival Probability')

# Show legend
plt.legend(title='Tumour Subtype')

# Show plot
plt.savefig("../figures/km_curve_by_tumor_subtype.png")
plt.close()


print("Exploratory Data Analysis completed, figures saved.")


