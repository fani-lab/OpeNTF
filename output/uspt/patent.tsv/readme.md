Public link to download: https://patentsview.org/download/data-download-tables

Metadata (data dictionary) that explains the schema, columns, data types can be found at https://patentsview.org/download/data-download-dictionary

The main table is ``patent`` whose ``type`` column distinguishes different types of patents including: 
- utility
- design
- ...

Also, ``subgroup_id`` is the foreign key to ``cbc_subgroup`` that shows the category hierarchy of a patent like ``"Y10S706/XX"	"Data processing: artificial intelligence"``

**Stats:**

|``Utility`` Patents Stat| Value|
|-----|------|
|#Patents (teams)| 7,068,508|
|#Unique Inventors (members) |3,508,807|
|#Unique Subgroups (skills)|241,961|
|Avg #Inventors per Patent| 2.51|
|Avg #Subgroup per Patent |6.29|
|Avg #Patent per Inventor |5.05|
|Avg #Subgroup per Inventor |19.49|
|#Patent w/ Single Inventor|2,578,898|
|#Patent w/ Single Subgroup|939,955|
|#Unique Inventor's Locations|261|
|Avg Inventors' Locations per Patent|2.50|

