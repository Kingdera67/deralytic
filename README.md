# Deralytic

A web app that cleans, explores and reports on data in a few clicks. Built with Python, pandas, scikit-learn and Streamlit.

**Live app:** deralytic.streamlit.app

## What it does
- **Load data** from a CSV or Excel file, a SQLite, MySQL or PostgreSQL database, or built-in sample data
- **Clean automatically** (column names, spaces, data types, duplicates, missing values, outliers) or **one column at a time** with undo
- **Analyse and visualise** with summary statistics, correlation heatmap, group analysis and interactive charts
- **Get plain-language insights** about missing data, skew and strong relationships
- **Predict a column** with a random forest model, with honest test scores and the columns that matter most
- **Find segments** with k-means clustering
- **Download** cleaned data (CSV, Excel, database table) and a one-click **PDF report**

## Run it on your own computer
```
pip install -r requirements.txt
streamlit run app.py
```

## Tools
Python, pandas, NumPy, scikit-learn, Plotly, Matplotlib, ReportLab, SQLAlchemy, Streamlit

## Notes
MySQL and PostgreSQL connections work only with databases the app can reach, so the public web version is best used with uploads, the sample data or SQLite.

Built by Shalom Chidera Reuben.
