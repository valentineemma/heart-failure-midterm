import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # Display our project title, authors, and research question.
    mo.md("""
    # Heart Failure Clinical Records Project

    **Emma Valentine and Olivia Barcela**

    CAI5735: Applied Data Science in Health

    ## Research question

    How is ejection fraction related to serum creatinine,
    and does that relationship differ between patients who died
    during follow-up and patients who did not?
    """)
    return (mo,)


@app.cell
def _(mo):
    # Explain the dataset's source and the variables used in our question.
    mo.md("""
    ## Dataset description and source

    We use the instructor-provided Heart Failure Clinical Records CSV.
    It contains **299 patient records and 13 columns**.

    According to the accompanying course handout, the records were
    collected in **2015** at the Faisalabad Institute of Cardiology
    and Allied Hospital in **Faisalabad, Pakistan**.
    All patients in this dataset already had heart failure.

    ### Columns used in our research question

    - **ejection_fraction:** Percentage of blood leaving the heart
      with each contraction, measured in percent.
    - **serum_creatinine:** Blood creatinine level, measured in mg/dL.
    - **DEATH_EVENT:** Whether a death occurred during follow-up.
      A value of 1 means death recorded; 0 means no death recorded.

    We also use **time**, measured in days, to describe the
    recorded follow-up/event time.

    Our supplied CSV includes `time` and uppercase `DEATH_EVENT`,
    although the course handout describes a simplified version.

    ### Original dataset citation

    Heart Failure Clinical Records [Dataset]. (2020).
    UCI Machine Learning Repository.

    [Original dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

    [Dataset DOI](https://doi.org/10.24432/C5Z89R)
    """)
    return


@app.cell
def _():
    # GitHub repository: https://github.com/valentineemma/heart-failure-midterm

    #Import pandas so we can read and work with the CSV data.
    import pandas as pd
    from pathlib import Path

    # Find the CSV in the same folder as this notebook.
    data_path = (
        Path(__file__).resolve().parent
        / "heart_failure_clinical_records_dataset.csv"
    )

    # Read the CSV into a table named df.
    df = pd.read_csv(data_path)

    # Display the first five records to check that loading worked.
    df.head()
    return (df,)


@app.cell
def _(df, mo):
    # Check the dataset's size and look for missing or duplicated records.
    number_of_rows = df.shape[0]
    number_of_columns = df.shape[1]
    missing_values = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()

    # Display the checks so we can decide whether cleaning is needed.
    mo.md(f"""
    ## Data checks

    - Patient records: **{number_of_rows}**
    - Columns: **{number_of_columns}**
    - Missing values: **{missing_values}**
    - Exact duplicate rows: **{duplicate_rows}**
    """)
    return


@app.cell
def _(df):
    # Create readable outcome labels from the numeric death indicator.
    # Keep df unchanged and store the added column in a new table.
    heart_data = df.assign(
        outcome_label=df["DEATH_EVENT"].map({
            0: "No death recorded",
            1: "Death recorded"
        })
    )

    # Preview the original indicator beside our computed labels.
    heart_data[["DEATH_EVENT", "outcome_label"]].head()
    return (heart_data,)


@app.cell
def _(heart_data, mo):
    # Filter the data to select patients with a recorded death.
    death_records = heart_data.loc[
        heart_data["DEATH_EVENT"] == 1
    ]

    # Display how many records matched our filter.
    mo.md(f"""
    ## Filtering by outcome

    There are **{len(death_records)} patients**
    with a death recorded during follow-up.

    This filtered table is a separate view.
    We retain both outcome groups in heart_data for later comparisons.
    """)
    return (death_records,)


@app.cell
def _(death_records):
    # Show the relevant measurements for five patients in the filtered group.
    death_records[
        ["ejection_fraction", "serum_creatinine", "outcome_label"]
    ].head()
    return


@app.cell
def _(heart_data):
    # Sort all patients from highest to lowest serum creatinine.
    # Keep heart_data unchanged and store the sorted table separately.
    sorted_heart_data = heart_data.sort_values(
        by="serum_creatinine",
        ascending=False
    )

    # Display the five highest values with the relevant columns.
    sorted_heart_data[
        ["ejection_fraction", "serum_creatinine", "outcome_label"]
    ].head()
    return


@app.cell
def _(heart_data):
    # Group patients by recorded outcome and summarize their measurements.
    outcome_summary = heart_data.groupby("outcome_label").agg(
        patient_count=("DEATH_EVENT", "size"),
        mean_ejection_fraction=("ejection_fraction", "mean"),
        median_ejection_fraction=("ejection_fraction", "median"),
        mean_serum_creatinine=("serum_creatinine", "mean"),
        median_serum_creatinine=("serum_creatinine", "median")
    )

    # Display the summary rounded to two decimal places.
    outcome_summary.round(2)
    return (outcome_summary,)


@app.cell
def _():
    # Import the plotting tools needed to create our charts.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(heart_data, plt):
    # Create a histogram to examine ejection fraction across all patients.
    histogram_fig, histogram_ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained"
    )

    # Use equal-width bins of five percentage points.
    histogram_ax.hist(
        heart_data["ejection_fraction"],
        bins=range(10, 86, 5),
        color="teal",
        edgecolor="white"
    )

    # Add the title and axis labels required by the assignment.
    histogram_ax.set_title("Distribution of Ejection Fraction")
    histogram_ax.set_xlabel("Ejection fraction (%)")
    histogram_ax.set_ylabel("Number of patients")

    # Display the completed figure.
    histogram_fig
    return


@app.cell
def _(heart_data, plt):
    # Select creatinine values for each outcome group.
    creatinine_no_death = heart_data.loc[
        heart_data["DEATH_EVENT"] == 0,
        "serum_creatinine"
    ]

    creatinine_death = heart_data.loc[
        heart_data["DEATH_EVENT"] == 1,
        "serum_creatinine"
    ]

    # Create a boxplot to compare the groups' medians and spread.
    boxplot_fig, boxplot_ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained"
    )

    boxplot_ax.boxplot(
        [creatinine_no_death, creatinine_death],
        tick_labels=["No death recorded", "Death recorded"],
        showfliers=True
    )

    # Add a title and labeled axes, including the measurement unit.
    boxplot_ax.set_title("Serum Creatinine by Recorded Outcome")
    boxplot_ax.set_xlabel("Outcome during follow-up")
    boxplot_ax.set_ylabel("Serum creatinine (mg/dL)")

    # Display the completed figure.
    boxplot_fig
    return


@app.cell
def _(heart_data, plt):
    # Separate the two groups so each can have a different color.
    scatter_no_death = heart_data.loc[
        heart_data["DEATH_EVENT"] == 0
    ]

    scatter_death = heart_data.loc[
        heart_data["DEATH_EVENT"] == 1
    ]

    # Create a scatter plot of the relationship between the measurements.
    scatter_fig, scatter_ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained"
    )

    # Plot patients without a recorded death in teal.
    scatter_ax.scatter(
        scatter_no_death["ejection_fraction"],
        scatter_no_death["serum_creatinine"],
        color="teal",
        alpha=0.6,
        label="No death recorded (n=203)"
    )

    # Plot patients with a recorded death in coral.
    scatter_ax.scatter(
        scatter_death["ejection_fraction"],
        scatter_death["serum_creatinine"],
        color="coral",
        alpha=0.6,
        label="Death recorded (n=96)"
    )

    # Add the required title, axis labels, and a color legend.
    scatter_ax.set_title(
        "Ejection Fraction and Serum Creatinine by Outcome"
    )
    scatter_ax.set_xlabel("Ejection fraction (%)")
    scatter_ax.set_ylabel("Serum creatinine (mg/dL)")
    scatter_ax.legend()

    # Display the completed figure.
    scatter_fig
    return scatter_death, scatter_no_death


@app.cell
def _(outcome_summary, plt):
    # Retrieve the mean ejection fraction for each outcome group.
    comparison_means = [
        outcome_summary.loc[
            "No death recorded", "mean_ejection_fraction"
        ],
        outcome_summary.loc[
            "Death recorded", "mean_ejection_fraction"
        ]
    ]

    # Create a bar chart to show the difference between group averages.
    comparison_fig, comparison_ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained"
    )

    comparison_ax.bar(
        ["No death recorded\n(n=203)", "Death recorded\n(n=96)"],
        comparison_means,
        color=["teal", "coral"]
    )

    # Label each bar with its calculated mean.
    for comparison_position, comparison_mean in enumerate(comparison_means):
        comparison_ax.text(
            comparison_position,
            comparison_mean + 0.8,
            f"{comparison_mean:.2f}%",
            ha="center"
        )

    # Add a title, axis labels, and a zero baseline for comparison.
    comparison_ax.set_title("Mean Ejection Fraction by Recorded Outcome")
    comparison_ax.set_xlabel("Outcome during follow-up")
    comparison_ax.set_ylabel("Mean ejection fraction (%)")
    comparison_ax.set_ylim(0, 50)

    # Display the completed figure.
    comparison_fig
    return


@app.cell
def _(heart_data, mo, scatter_death, scatter_no_death):
    # Calculate the linear relationship between ejection fraction
    # and serum creatinine across all patients.
    overall_correlation = heart_data[
        "ejection_fraction"
    ].corr(
        heart_data["serum_creatinine"]
    )

    # Calculate the same relationship within each outcome group.
    no_death_correlation = scatter_no_death[
        "ejection_fraction"
    ].corr(
        scatter_no_death["serum_creatinine"]
    )

    death_correlation = scatter_death[
        "ejection_fraction"
    ].corr(
        scatter_death["serum_creatinine"]
    )

    # Display the correlations to three decimal places.
    mo.md(f"""
    ## Correlation results

    Pearson correlation between ejection fraction and serum creatinine:

    - All patients: **{overall_correlation:.3f}**
    - No death recorded: **{no_death_correlation:.3f}**
    - Death recorded: **{death_correlation:.3f}**

    These describe the observed relationships in this dataset.
    We have not tested whether the group correlations differ
    statistically significantly.
    """)
    return death_correlation, no_death_correlation, overall_correlation


@app.cell
def _(heart_data, mo, scatter_death, scatter_no_death):
    # Use rank correlation to check the relationship without relying
    # on the exact magnitudes of extreme creatinine values.
    overall_spearman = heart_data["ejection_fraction"].corr(
        heart_data["serum_creatinine"],
        method="spearman"
    )

    no_death_spearman = scatter_no_death["ejection_fraction"].corr(
        scatter_no_death["serum_creatinine"],
        method="spearman"
    )

    death_spearman = scatter_death["ejection_fraction"].corr(
        scatter_death["serum_creatinine"],
        method="spearman"
    )

    # Display the rank correlations alongside an explanation.
    mo.md(f"""
    ## Checking the relationship with ranks

    Spearman correlation between ejection fraction and serum creatinine:

    - All patients: **{overall_spearman:.3f}**
    - No death recorded: **{no_death_spearman:.3f}**
    - Death recorded: **{death_spearman:.3f}**

    Using ranks reduces the influence of extreme measurement sizes.
    We retain every patient in this analysis.
    """)
    return (death_spearman,)


@app.cell
def _(heart_data):
    # Compare recorded follow-up/event times between outcome groups.
    # This helps explain a limitation of comparing the death indicators.
    followup_summary = heart_data.groupby("outcome_label").agg(
        patient_count=("DEATH_EVENT", "size"),
        median_followup_days=("time", "median"),
        minimum_followup_days=("time", "min"),
        maximum_followup_days=("time", "max")
    )

    # Display the summary for both groups.
    followup_summary
    return


@app.cell
def _(
    death_correlation,
    death_spearman,
    mo,
    no_death_correlation,
    overall_correlation,
):
    # Answer our research question using the calculated results,
    # and explain the limitations of this descriptive analysis.
    mo.md(f"""
    ## Conclusion

    ### Answer to our research question

    Ejection fraction and serum creatinine have little overall
    linear association in this dataset.
    The overall Pearson correlation is **{overall_correlation:.3f}**.

    The observed Pearson correlations differ between outcome groups:

    - No death recorded: **{no_death_correlation:.3f}**
    - Death recorded: **{death_correlation:.3f}**

    Both group correlations are weak. We have not performed a
    statistical test to determine whether their difference is significant.

    ### Comparing the two outcome groups

    Patients with a recorded death have a lower mean ejection fraction:
    **33.47%**, compared with **40.27%** for patients without a recorded death.

    They also have higher mean serum creatinine:
    **1.84 mg/dL**, compared with **1.18 mg/dL**.

    These group differences can exist even when the two numeric
    measurements have little overall linear association.

    ### Checking the influence of extreme values

    The Spearman correlation in the recorded-death group is
    **{death_spearman:.3f}**, compared with its Pearson correlation of
    **{death_correlation:.3f}**.

    The relationship appears much weaker when we compare ranks.
    This suggests that the positive Pearson result is sensitive
    to the measurement sizes and extreme observations.

    ### Limitations

    - These observational records cannot establish cause and effect.
    - All patients already had heart failure, so the results do not
      represent a comparison with healthy people.
    - The outcome groups have different sizes: 96 and 203 patients.
    - Median recorded follow-up/event time differs between groups:
      44.5 days for recorded deaths and 172 days for no death recorded.
    - We have not adjusted for other factors or performed survival analysis.
    - No death recorded during follow-up does not guarantee long-term survival.
    """)
    return


if __name__ == "__main__":
    app.run()
