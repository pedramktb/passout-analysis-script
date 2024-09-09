import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pre_test_data = pd.read_csv('Pre-Test.csv')
post_test_data = pd.read_csv('Post-Test.csv')

password_manager_count = pre_test_data['What password manager do you currently use, if any?'].value_counts(dropna=False)
pre_test_data['Used_Password_Manager_Before'] = pre_test_data['What password manager do you currently use, if any?'].apply(lambda x: 'No' if pd.isna(x) else 'Yes')

duration_summary = pre_test_data['How long have you been using your current password manager?'].value_counts()

common_likes = pre_test_data['What do you like most about your current password manager?'].value_counts()

common_challenges = pre_test_data['What do you find most challenging or frustrating about your current password manager?'].value_counts()

chart_size = (7, 5)

# Chart 1: Distribution of Password Managers in Use
plt.figure(figsize=chart_size)
password_manager_count.plot(kind='bar', color='skyblue')
plt.title('Distribution of Password Managers in Use')
plt.xlabel('Password Manager')
plt.ylabel('Number of Users')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# Chart 2: Duration of Use of Current Password Manager
plt.figure(figsize=chart_size)
duration_summary.plot(kind='bar', color='lightgreen')
plt.title('Duration of Use of Current Password Manager')
plt.xlabel('Duration')
plt.ylabel('Number of Users')
plt.xticks(rotation=0)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()

plt.show()

sus_columns = post_test_data.columns[1:11]
sus_data = post_test_data[sus_columns]

sus_scores = []
for index, row in sus_data.iterrows():
    sus_score = 0
    for i, value in enumerate(row):
        if i % 2 == 0:  # Odd-indexed
            sus_score += (value - 1)
        else:  # Even-indexed
            sus_score += (5 - value)
    sus_score *= 2.5
    sus_scores.append(sus_score)

sus_scores_df = pd.DataFrame(sus_scores, columns=["SUS Score"])
sus_summary_stats = sus_scores_df.describe()
sus_summary_stats.to_csv('SUS_Summary.csv')

# Creating a box plot for the SUS scores
plt.figure(figsize=(8, 6))
plt.boxplot(sus_scores_df['SUS Score'], vert=True, patch_artist=True, notch=True, 
            boxprops=dict(facecolor="lightblue"))
plt.title('Box Plot of SUS Scores')
plt.ylabel('SUS Score')
plt.xticks([1], ['SUS Scores'])
plt.show()


likert_columns = post_test_data.columns[-6:-1]
post_test_data['Mean'] = post_test_data[likert_columns].mean(axis=1)
likert_summary = post_test_data[['Mean'] + list(likert_columns)].describe()
likert_summary.to_csv('Likert_Scale_Summary.csv')

# Creating a box plot for the Likert scale data
plt.figure(figsize=(8, 6))
plt.boxplot(post_test_data['Mean'], vert=True, patch_artist=True, notch=True, 
            boxprops=dict(facecolor="lightgreen"))
plt.title('Box Plot of Likert Scale Data')
plt.ylabel('Mean Score')
plt.xticks([1], ['Mean Score'])
plt.show()

# Performing a one-sample t-test on SUS scores against the benchmark (68)
sus_scores = sus_scores_df['SUS Score']
benchmark_sus_score = 68
t_statistic, p_value = stats.ttest_1samp(sus_scores, benchmark_sus_score)
print(f'SUS Score T-Test - T-Statistic: {t_statistic} | P-Value: {p_value}')

sus_scores = pd.DataFrame(sus_scores, columns=["SUS Score"])
sus_scores['Used_Password_Manager_Before'] = pre_test_data['Used_Password_Manager_Before']


sus_used_password_manager = sus_scores[sus_scores['Used_Password_Manager_Before'] == 'Yes']['SUS Score']
sus_not_used_password_manager = sus_scores[sus_scores['Used_Password_Manager_Before'] == 'No']['SUS Score']

# Calculate the mean SUS score for users who have and have not used a password manager before
mean_sus_used = sus_used_password_manager.mean()
mean_sus_not_used = sus_not_used_password_manager.mean()
print(f'Mean SUS Score (Used Password Manager): {mean_sus_used}')
print(f'Mean SUS Score (Not Used Password Manager): {mean_sus_not_used}')

# Performing a one-sample t-test on SUS scores against the benchmark (68) for users who have and have not used a password manager before
t_stat_used, p_val_used = stats.ttest_1samp(sus_used_password_manager, benchmark_sus_score)
t_stat_not_used, p_val_not_used = stats.ttest_1samp(sus_not_used_password_manager, benchmark_sus_score)
print(f'SUS T-Test (Used Password Manager) - T-Statistic: {t_stat_used} | P-Value: {p_val_used}')
print(f'SUS T-Test (Not Used Password Manager) - T-Statistic: {t_stat_not_used} | P-Value: {p_val_not_used}')

# Performing a one-sample t-test on Likert scale data against the benchmark (3)
likert_mean = post_test_data['Mean']
benchmark_likert_score = 3
t_statistic, p_value = stats.ttest_1samp(likert_mean, benchmark_likert_score)
print(f'Likert Mean T-Test - T-Statistic: {t_statistic} | P-Value: {p_value}')

# Performing a one-sample t-test on Likert scale security questions against the benchmark (3)
likert_s = post_test_data['How do you feel about the security of the prototype compared to your current password manager?']
benchmark_likert_score = 3
t_statistic, p_value = stats.ttest_1samp(likert_s, benchmark_likert_score)
print(f'Likert Security T-Test - T-Statistic: {t_statistic} | P-Value: {p_value}')

# Performing a one-sample t-test on Likert scale data against the benchmark (3) for users who have and have not used a password manager before
likert_mean = post_test_data['Mean']
likert_used_password_manager = likert_mean[sus_scores['Used_Password_Manager_Before'] == 'Yes']
likert_not_used_password_manager = likert_mean[sus_scores['Used_Password_Manager_Before'] == 'No']
t_stat_used, p_val_used = stats.ttest_1samp(likert_used_password_manager, benchmark_likert_score)
t_stat_not_used, p_val_not_used = stats.ttest_1samp(likert_not_used_password_manager, benchmark_likert_score)
print(f'Likert Mean T-Test (Used Password Manager) - T-Statistic: {t_stat_used} | P-Value: {p_val_used}')
print(f'Likert Mean T-Test (Not Used Password Manager) - T-Statistic: {t_stat_not_used} | P-Value: {p_val_not_used}')

# Performing a one-sample t-test on Likert scale security questions against the benchmark (3) for users who have and have not used a password manager before
likert_s = post_test_data['How do you feel about the security of the prototype compared to your current password manager?']
likert_used_password_manager = likert_s[sus_scores['Used_Password_Manager_Before'] == 'Yes']
likert_not_used_password_manager = likert_s[sus_scores['Used_Password_Manager_Before'] == 'No']
t_stat_used, p_val_used = stats.ttest_1samp(likert_used_password_manager, benchmark_likert_score)
t_stat_not_used, p_val_not_used = stats.ttest_1samp(likert_not_used_password_manager, benchmark_likert_score)
print(f'Likert Security T-Test (Used Password Manager) - T-Statistic: {t_stat_used} | P-Value: {p_val_used}')
print(f'Likert Security T-Test (Not Used Password Manager) - T-Statistic: {t_stat_not_used} | P-Value: {p_val_not_used}')

# # ANOVA: Combining scores with duration of use from pre-test data
# combined_data = pd.DataFrame({
#     'SUS Score': sus_scores,
#     'Likert Mean': post_test_data['Mean'],
#     'Likert Security': post_test_data['How do you feel about the security of the prototype compared to your current password manager?'],
#     'Experience': pre_test_data['How long have you been using your current password manager?']
# })

# anova_results = stats.f_oneway(
#     combined_data[combined_data['Experience'] == 'n.a']['SUS Score'],
#     combined_data[combined_data['Experience'] == '<1 Year']['SUS Score'],
#     combined_data[combined_data['Experience'] == '3-4 Years']['SUS Score'],
#     combined_data[combined_data['Experience'] == '4-5 Years']['SUS Score'],
#     combined_data[combined_data['Experience'] == '>5 Years']['SUS Score']
# )
# with open('SUS_Summary.csv', 'a') as f:
#     f.write(f'ANOVA F-Statistic: {anova_results.statistic} | ANOVA P-Value: {anova_results.pvalue}\n')

# anova_results = stats.f_oneway(
#     combined_data[combined_data['Experience'] == 'n.a']['Likert Mean'],
#     combined_data[combined_data['Experience'] == '<1 Year']['Likert Mean'],
#     combined_data[combined_data['Experience'] == '3-4 Years']['Likert Mean'],
#     combined_data[combined_data['Experience'] == '4-5 Years']['Likert Mean'],
#     combined_data[combined_data['Experience'] == '>5 Years']['Likert Mean']
# )
# with open('Likert_Scale_Summary.csv', 'a') as f:
#     f.write(f'ANOVA F-Statistic: {anova_results.statistic} | ANOVA P-Value: {anova_results.pvalue}\n')

# anova_results = stats.f_oneway(
#     combined_data[combined_data['Experience'] == 'n.a']['Likert Security'],
#     combined_data[combined_data['Experience'] == '<1 Year']['Likert Security'],
#     combined_data[combined_data['Experience'] == '3-4 Years']['Likert Security'],
#     combined_data[combined_data['Experience'] == '4-5 Years']['Likert Security'],
#     combined_data[combined_data['Experience'] == '>5 Years']['Likert Security']
# )
# with open('Likert_Scale_Summary.csv', 'a') as f:
#     f.write(f'Security ANOVA F-Statistic: {anova_results.statistic} | Security ANOVA P-Value: {anova_results.pvalue}\n')

# duration_map = {
#     '<1 Year': 0.5,
#     '1-2 Years': 1.5,
#     '2-3 Years': 2.5,
#     '3-4 Years': 3.5,
#     '4-5 Years': 4.5,
#     '>5 Years': 6.0
# }

# from statsmodels.stats.multicomp import pairwise_tukeyhsd

# # Run Tukey HSD test
# tukey_results = pairwise_tukeyhsd(combined_data['Likert Mean'], combined_data['Experience'])
# print(tukey_results)

# pre_test_data['Duration (Years)'] = pre_test_data['How long have you been using your current password manager?'].map(duration_map)
# combined_data = pd.concat([pre_test_data['Duration (Years)'].reset_index(drop=True),sus_scores,likert_mean], axis=1)
# correlation_matrix = combined_data.corr()
# correlation_matrix.to_csv('Correlation_Matrix.csv')
