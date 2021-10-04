import pandas as pd 

while True:
	tsv_file='title_basic.tsv' #replace name of the file with file you want to convert
	out_name = tsv_file.replace(".tsv", ".csv")
	
	print("Please wait file is being converted....")
	csv_table=pd.read_table(tsv_file,sep='\t', dtype={"isAdult": object, "startYear": object, "isOriginalTitle": object})
	csv_table.to_csv(out_name,index=False)
	print("Done. File is converted successfully. \nOff to the next file.")
	break

