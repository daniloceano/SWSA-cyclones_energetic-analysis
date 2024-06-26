import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_df(df, percentage_threshold=1, filter_df=False, exclude_residual=False, contain_mature_phase=False):

    processed_df = df.copy()

    # Filter rows based on percentage and number of phases
    if filter_df:
        processed_df = df[df['Percentage'] > percentage_threshold]
        processed_df['Num Phases'] = processed_df['Type of System'].apply(lambda x: len(x.split(', ')))
    
    if exclude_residual:
        # Exclude systems with the 'residual' stage
        processed_df['Type of System'] = processed_df['Type of System'].str.replace(', residual', '')
        processed_df = processed_df.groupby('Type of System', as_index=False).agg({
                    'Total Count': 'sum',
                    'Percentage': 'sum'
                    }).sort_values(by='Total Count', ascending=False)
    
    if contain_mature_phase:
        processed_df = processed_df[processed_df['Type of System'].str.contains('mature')]
    
    return processed_df

def plot_barplot(df, label, ax, suffix):
    # Mapping labels to the desired scheme
    label_mapping = {
        'incipient': 'Ic',
        'incipient 2': 'Ic2',
        'intensification': 'It',
        'intensification 2': 'It2',
        'mature': 'M',
        'mature 2': 'M2',
        'decay': 'D',
        'decay 2': 'D2',
        'residual': 'R'
    }

    # Convert 'Type of System' column to string to handle NaN values
    df['Type of System'] = df['Type of System'].astype(str)

    # Replace labels with the desired scheme and join them back
    df['Type of System'] = df['Type of System'].apply(lambda x: ', '.join([label_mapping.get(word, word) for word in x.split(', ')]))

    # Determine the orientation based on the filter flag
    orient = 'h'

    if suffix == 'filtered':
        unique_sequences = df['Type of System'].str.replace(', R', '').str.replace('R, ', '').str.replace('R', '').unique()
        palette = sns.color_palette("pastel", n_colors=len(unique_sequences))
        color_mapping = dict(zip(unique_sequences, palette))
        # Modify the sequence column to exclude the residual stage
        df['Sequence Without R'] = df['Type of System'].str.replace(', R', '').str.replace('R, ', '').str.replace('R', '')

        # Create a new color column using the mapping
        df['Color'] = df['Sequence Without R'].map(color_mapping)

        # Plot using the color column
        sns.barplot(x='Total Count', y='Type of System', data=df, orient=orient, ci=None, 
                    palette=df.set_index('Type of System')['Color'].to_dict(), edgecolor='grey', ax=ax)

    else:
        # Create a bar plot using Seaborn on the provided axes
        sns.barplot(x='Total Count', y='Type of System', data=df, orient=orient, ci=None, palette='pastel', edgecolor='grey', ax=ax)
    
    ax.text(0.95, 0.05, label, ha='right', fontsize=14, va='bottom', transform=ax.transAxes, fontweight='bold')
    
    # Add text annotations for total count and percentage on the right side of each bar
    for index, value in enumerate(df['Total Count']):
        percentage = df.loc[df.index[index], 'Percentage']
        if orient == 'h':
            ax.text(value + 100, index, f"{value} ({percentage:.2f}%)", va='center', color='black',
                    fontsize=14)
        else:
            ax.text(index, value + 100, f"{value} ({percentage:.2f}%)", ha='center', color='black')

    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.tick_params(axis='both', labelsize=14)

    # Add text annotations for total count and percentage
    total_count = df['Total Count'].sum()
    total_percentage = df['Percentage'].sum()
    ax.title.set_text(f'Total Count: {total_count} - Total Percentage: {total_percentage:.2f}%')
    ax.title.set_fontsize(14)

    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    
def plot_combined_barplots(df1, df2, output_directory, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot bar plots for df1 and df2 on the provided axes
    plot_barplot(df1, '(A)', axes[0], suffix)
    plot_barplot(df2, '(B)', axes[1], suffix)

    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the combined figure
    combined_output_file = os.path.join(output_directory, f'combined_barplots_{suffix}.png')
    plt.savefig(combined_output_file)
    print(f'{combined_output_file} saved.')

output_directory = '../figures/manuscript_life-cycle/'
os.makedirs(output_directory, exist_ok=True)

# Read data from CSV file
df = pd.read_csv('../periods_species_statistics/70W-no-continental/count_systems_raw/total_count_of_systems_SAt.csv')

# df with all possible phases
df = df.sort_values(by='Total Count', ascending=False)

# df excluding residual phases
df_excluded_residual = process_df(df, filter_df=False, exclude_residual=True)

# df for phases with more than 1%
filtered_df = process_df(df, filter_df=True, exclude_residual=False)

# df for phases with more than 1% and exluding residual phases
filtered_df_exclude_residual_contain_mature = process_df(df, filter_df=True, exclude_residual=True, contain_mature_phase=True)
# export csv for further analysis
filtered_df_exclude_residual_contain_mature.to_csv('../periods_species_statistics/70W-no-continental/total_count_of_systems_filtered.csv')

# Combine the four bar plots into one figure
plot_combined_barplots(df, df_excluded_residual, output_directory, 'total')
plot_combined_barplots(filtered_df, filtered_df_exclude_residual_contain_mature, output_directory, 'filtered')