import json
from cmn.member import Member
from cmn.author import Author
from cmn.team import Team
from cmn.document import Document

def read_data(data_path):
    counter = 0
    all_docs = {}
    all_authors = {}  
    input_data = []
    output_data = []

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
            if doc.get_uid() not in all_docs.keys():
                all_docs[doc.get_uid()] = doc
            
            # input_data.append(", ".join(doc.get_fields()))
            input_data.append(doc.get_fields())

            # output_data.append(", ".join(doc.get_members_names()))
            output_data.append(doc.get_members_names())

            counter += 1
    return all_authors, all_docs, input_data, output_data

def build_index_authors(all_authors):
    idx = 0
    author_to_index = {}
    index_to_author = {}
    for auth in all_authors.values():
        index_to_author[idx] = auth.get_name()
        author_to_index[auth.get_name()] = idx
        idx += 1
    return index_to_author, author_to_index

def build_index_skills(all_docs):
    idx = 0
    skill_to_index = {}
    index_to_skill = {}

    for doc in all_docs.values():
        for field in doc.get_fields():
            if field not in skill_to_index.keys():
                skill_to_index[field] = idx
                index_to_skill[idx] = field
                idx += 1

    return index_to_skill, skill_to_index
