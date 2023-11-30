import pandas as pd

og_file = '/Users/zahir/Documents/MATLAB/data_output/output_val_backup_tests.xlsx'
sheets = ["DLDP_081", 'DLDP_082', 'DLDP_088', 'DLDP_090']


# Dictionary to store data frames from each sheet
dfs = {}

# Read data
for sheet in sheets:
    try:
        dfs[sheet] = pd.read_excel(og_file, sheet_name=sheet)
    except FileNotFoundError:
        # If the sheet doesn't exist, create an empty DataFrame
        dfs[sheet] = pd.DataFrame(columns=['Metric', 'Pert Type', 'Pert Size', 'Organ', 'Max', 'Min', 'Mean', 'Corr', 'Restriction', 'Message'])


for sheet in sheets:

    # store rows with conditions
    rows = {}
    for sheet, df in dfs.items():
        # cond = (df['Metric'] == 'Max') & (df['Pert Type'] == 'Erosion')
        cond = (df['Metric'] == 'Max')
        # rows[sheet] = df[cond]
        rows[sheet] = df[cond].drop(columns=['Metric'])

    # # Save the original sheets with 'Max' rows removed back to the original Excel file
    # with pd.ExcelWriter(og_file) as writer:
    #     for sheet, df in dfs.items():
    #         df.to_excel(writer, sheet_name=sheet, index=False)


    # Save rows with metric 'Max' in new sheets of the new Excel file
    out_file = '/Users/zahir/Documents/MATLAB/data_output/output_val_test1.xlsx'
    with pd.ExcelWriter(out_file) as writer:
        for sheet, df in rows.items():
            df.to_excel(writer, sheet_name=f'{sheet}_MaxE', index = False, encoding='utf-8')
            sheet = writer.sheets[f'{sheet}_MaxE']

    print("Rows with metric 'Max' have been saved into separate sheets in a new Excel file.")

