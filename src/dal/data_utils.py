
import json
from cmn.member import Member
from cmn.author import Author
from cmn.team import Team
from cmn.document import Document

def read_data(data_path):
    counter = 0
    docs = []
    all_authors = {}  
    training_input = []
    training_output = []

    with open(data_path, "r") as jf:
        # Skip the first line
        jf.readline() 
        while counter < 50:
            # Read line by line to not overload the memory
            line = jf.readline().lower().lstrip(",")
            jsonline = json.loads(line)

            # Retrieve the desired attributes
            doc_id = jsonline['id']
            doc_title = jsonline['title']
            doc_year = jsonline['year']
            doc_type = jsonline['doc_type']
            doc_venue = jsonline['venue']
            
            if 'references' in jsonline.keys():
                doc_references = jsonline['references']
            else:
                doc_references = []
                
            doc_fos = jsonline['fos']
            
            if 'keywords' not in jsonline.keys():
                doc_keywords = []
            else:
                doc_keywords = jsonline['keywords']
                
            authors = []
            for auth in jsonline['authors']:
                
                # Retrieve the desired attributes
                auth_id = auth['id']
                auth_name = auth['name']
                
                if 'org' in auth.keys():
                    auth_org = auth['org']
                else:
                    auth_org = ""
                
                author = Author(auth_id, auth_name, auth_org)
                authors.append(author)
                
                
                if auth_id not in all_authors.keys():
                    all_authors[auth_id] = author
                
            doc = Document(doc_id, authors, doc_title, doc_year,doc_type, doc_venue, doc_references, doc_fos, doc_keywords)
            docs.append(doc)
            
            # training_input.append(", ".join(doc.get_fields()))
            training_input.append(doc.get_fields())

            # training_output.append(", ".join(doc.get_members_names()))
            training_output.append(doc.get_members_names())

            counter += 1
    return all_authors, docs, training_input, training_output