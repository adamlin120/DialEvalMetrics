"""
Make markdown table to csv
"""

import pandas as pd
import gspread

data_file2benchmark = {
    "convai2_grade.json": "GRADE-ConvAI2",
    "dailydialog_grade.json": "GRADE-ConvAI2",
    "empatheticdialogues_grade.json": "GRADE-ConvAI2",
    "dstc9.json": "DSTC9",
    "engage.json": "PredictiveEngage-DailyDialog",
    "fed.json": "FED",
    "fed_dialog.json": "FED",
    "holistic.json": "HolisticEval-DailyDialog",
    "personachat_usr.json": "USR-TopicalChat",
    "topicalchat_usr.json": "USR-TopicalChat",
}

# read csv "llm/llm_results.csv"
llm_df = pd.read_csv("llm/llm_results.csv")
llm_df["Sheet"] = llm_df["data_file"].map(data_file2benchmark)
# llm_sheet_df.columns
# Index(['data_file', 'prompt_style', 'turn_pearson_corr_coefficient',
#        'turn_spearman_corr_coefficient', 'dialog_pearson_corr_coefficient',
#        'dialog_spearman_corr_coefficient', 'Sheet'],
#       dtype='object')

# read from results.txt
with open("results.txt", "r") as f:
    makrdown_tabels_in_text = f.read()

# split the text into a list of markdown tables
makrdown_tabels_in_text = makrdown_tabels_in_text.split("\n\n")

for mt in makrdown_tabels_in_text:
    # parse the markdown table into a pandas dataframe
    df = pd.read_html(mt)[0]

    sheet_name = df.loc[0][1]

    # replace nan with empty string
    df = df.replace({pd.np.nan: ""})

    llm_sheet_df = llm_df[llm_df["Sheet"] == sheet_name]

    # merge the two dataframes

    # # assuming df is the DataFrame containing the data
    # header = df.iloc[:3, :].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=0)
    #
    # # drop the first three rows from the original DataFrame
    # df = df.iloc[3:, :]
    #
    # # set the new header as the column names
    # df.columns = header

    # clear "*" in the content
    df = df.replace(r"\*", "", regex=True)

    # write to google sheet - llm-dial-eval
    gc = gspread.service_account()
    gc.login()
    sh = gc.open("llm-dial-eval")
    # add the worksheet if it does not existed
    if sheet_name not in [sheet.title for sheet in sh.worksheets()]:
        sh.add_worksheet(title=sheet_name, rows="100", cols="20")
    worksheet = sh.worksheet(sheet_name)
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
