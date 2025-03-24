# GITH Dataset Statistics
Generated on: 2025-03-24 12:22:20 EDT

## Overview

| Metric | Value |
|--------|-------|
| Total Teams | 726,697 |
| Total Skills | 30 |
| Total Experts | 1,100,779 |

## Team Composition

### Teams Breakdown

| Metric | Value |
|--------|-------|
| Unique Teams | 452,411 (62.3%) |
| Duplicate Teams | 19,628 (2.7%) |
| Teams with No Skills | 0 (0.0%) |
| Teams with No Experts | 0 (0.0%) |

### Skills per Team

| Metric | Value |
|--------|-------|
| Minimum Skills | 1 (254,729 teams, 35.1%) |
| Maximum Skills | 16 (3 teams, 0.0%) |
| Average Skills | 2.4536 |
| Standard Deviation | 1.4676 |

### Experts per Team

| Metric | Value |
|--------|-------|
| Minimum Experts | 1 (9 teams, 0.0%) |
| Maximum Experts | 30 (24,212 teams, 3.3%) |
| Average Experts | 4.7668 |
| Standard Deviation | 5.9272 |

### Expert Participation

| Metric | Value |
|--------|-------|
| Minimum Teams per Expert | 1 (466,919 experts) |
| Maximum Teams per Expert | 48,116 (1 experts) |
| Average Skills per Expert | 3.3076 |
| Average Teams per Expert | 3.1469 |

## Visualizations

The following visualizations are available in the reports directory:

### Skills Distribution

- `gith_skills_histogram.png`: Histogram of skills per team
- `gith_skills_compact.png`: Log-log plot of skills distribution
- `gith_skills_heatmap.png`: Heatmap of skill-team relationships

### Experts Distribution

- `gith_experts_histogram.png`: Histogram of experts per team
- `gith_experts_compact.png`: Log-log plot of experts distribution
- `gith_experts_heatmap.png`: Heatmap of expert-team relationships

## Notes on Statistics

- **Average Skills per Team**: This is calculated as the mean number of skills across all teams. Each team contributes one count to this average, regardless of team size.
- **Average Experts per Team**: Similarly, this is the mean number of experts across all teams.
- **Total Skills/Experts vs. Averages**: The total number of unique skills/experts represents distinct entities in the dataset, while averages show how many are typically associated with each team.
- **Standard Deviation**: Measures the variation in the distribution. Higher values indicate more diversity in team compositions.