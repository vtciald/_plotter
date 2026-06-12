# ezdata

A custom Python library intended to help with processing and visualizing survey data.

## Status: Work in progress
This is an early work in progress. I'm currently working on basic pre-processing and statistics before building out the clustering, dimensionality reduction, or plotting capabilities. I'm not attempting to reinvent the wheel. Many fantastic and well-tested packages already exist for each task I'm concerned with. My aim is to just make them more accessible and provide a more consistent API across them.

This is also intended primarily for my personal use. As such, it will be highly opinionated and make many assumptions about what a prospective user (i.e., yours truly) would like. :)

## Why
In order to explore survey data and create data visualizations to share with others, I've found myself repeatedly doing the same pre-processing methods or writing very similar expressions over and over (e.g., creating lists of desired column names, standardizing strings in string columns or column labels, filtering, creating confidence intervals, evaluating differences between subgroups, etc.).

While the availability of many high-quality packages for a variety of tasks has been great, it also can be clunky when I'm just trying to get something done. Remembering which package I need for what task and the differences in the APIs can be (minor) barriers to realizing my intent.

Years ago, I started what I'll admit to be a very messy and monolithic file containing custom classes that I used to simplify graph creation. While it works, it was in desperate need of refactoring. It also was scoped specifically to creating visualizations and very limited in its capabilities (e.g., only included standard Z or Wald confidence intervals).

In addition to that, I wanted to take this opportunity to build good (or better?) coding habits and develop familiarity with version control and Git.