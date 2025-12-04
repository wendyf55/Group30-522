import click
import os
import pandas as pd
from ydata_profiling import ProfileReport
import altair as alt

DEFAULT_TRAIN_PATH = '../data/processed/abalone_train.csv'
DEFAULT_SAVE_DIR = '../results/eda'
DEFAULT_TABLE_FILE = 'descriptive_stats.csv'
DEFAULT_REPORT_FILE = 'pandas_profiling.html'
DEFAULT_INTERACTION_FILE = 'interaction_plot.png'


def save_descriptive_table(train_df, descriptive_path):
    train_df.describe().round(2).to_csv(descriptive_path)


def save_eda_report(train_df, report_path):
    report = ProfileReport(train_df)
    report.to_file(report_path)


def save_interaction_plot(train_df, interaction_path):
    # Create a copy with human-readable Sex labels for visualization
    train_df_viz = train_df.copy()
    sex_label_map = {'M': 'Male', 'F': 'Female', 'I': 'Infant'}
    train_df_viz['Sex Category'] = train_df_viz['Sex'].map(sex_label_map)

    # Scatter plot with regression lines
    base = alt.Chart(train_df_viz).mark_circle(opacity=0.4, size=60).encode(
        x=alt.X('Shell_weight:Q',
                title='Shell Weight (grams)',
                scale=alt.Scale(zero=False)),
        y=alt.Y('Rings:Q',
                title='Number of Rings (Age Proxy)'),
        color=alt.Color('Sex Category:N',
                        title='Sex Category',
                        scale=alt.Scale(domain=['Male', 'Female', 'Infant'],
                                        range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(orient='top-right'))
    )

    lines = base.transform_regression(
        'Shell_weight', 'Rings', groupby=['Sex Category']
    ).mark_line(strokeWidth=3).encode(
        color=alt.Color('Sex Category:N',
                        title='Sex Category',
                        legend=None)
    )

    (base + lines).properties(
        title=alt.TitleParams(
            text='Relationship Between Shell Weight and Ring Count by Sex',
            fontSize=14,
            fontWeight='bold'
        ),
        width=550,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=13
    ).configure_legend(
        titleFontSize=12,
        labelFontSize=11
    ).save(interaction_path)


@click.command()
@click.option('--train-path',
              type=str,
              default=DEFAULT_TRAIN_PATH,
              help='Path to train data'
              )
@click.option('--save-dir',
              type=str,
              default=DEFAULT_SAVE_DIR,
              help='Path to directory to save EDA outputs')
@click.option('--table-file',
              type=str,
              default=DEFAULT_TABLE_FILE,
              help='Filename to save descriptive stats including extension')
@click.option('--report-file',
              type=str,
              default=DEFAULT_REPORT_FILE,
              help='Filename to save profile report including extension')
@click.option('--interaction-file',
              type=str,
              default=DEFAULT_INTERACTION_FILE,
              help='Filename to save interaction plot including extension')
def main(train_path, save_dir, table_file, report_file, interaction_file):
    os.makedirs(save_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)

    save_descriptive_table(train_df, os.path.join(save_dir, table_file))
    save_eda_report(train_df, os.path.join(save_dir, report_file))
    save_interaction_plot(train_df, os.path.join(save_dir, interaction_file))


if __name__ == "__main__":
    main()
