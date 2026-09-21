import io
import re
import pandas as pd
import streamlit as st


def parse_and_render_markdown(mixed_text):
    # Regex pattern to match typical Markdown tables:
    # Looks for sequential lines starting and ending with '|'
    table_pattern = re.compile(
        r"((?:^\|.*\|$(?:\n)?)+)", re.MULTILINE | re.IGNORECASE
    )

    # Split the text into alternating table segments and text segments
    parts = table_pattern.split(mixed_text)

    for part in parts:
        if not part.strip():
            continue

        # Check if this part is a table (contains column separator row like |---| )
        if "|" in part and "-|-" in part or "---" in part:
            try:
                # Read the markdown table into a DataFrame, skipping the separator row (index 1)
                df = pd.read_csv(
                    io.StringIO(part.strip()), sep="|", skiprows=[1]
                ).dropna(axis=1, how="all")
            
                # Clean whitespaces from column headers and cell values
                df.columns = df.columns.str.strip()
                df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            
                # Render with the styled st.table component
                st.table(df)
            except Exception as e:
                st.markdown(part)

        else:
            # It's standard text/paragraphs, render natively
            st.markdown(part)


# --- Example Usage ---
mixed_data = """
### User Report Summary
Here is the text introduction explaining the metrics gathered over the last quarter.

| Name | Age | City | Role |
|---|---|---|---|
| Alice | 30 | New York | Engineer |
| Bob | 25 | London | Designer |

Please review the metrics above before proceeding to the final project phase. 
Below are the server details:

| Server ID | Status | Uptime |
|---|---|---|
| SRV-01 | Active | 99.9% |
| SRV-02 | Down | 45.2% |
"""

st.title("Dynamic Markdown & Table Parser")
parse_and_render_markdown(mixed_data)
