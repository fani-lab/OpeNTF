import json
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import time
import traceback


class Skills:

    @staticmethod
    def get_skills(data_path):
        skills_pre = set()
        with open(data_path, "r", encoding='utf-8') as jf:
            jf.readline()
            t = time.time()
            count = 0

            while True:
                try:
                    line = jf.readline()
                    count += 1
                    print(f'{count} Rows')
                    if not line:
                        print('Breaking the While Loop')
                        break
                    elif line == ']':
                        break
                    else:
                        jsonline = json.loads(line.lower().lstrip(","))

                        if 'fos' in jsonline.keys():
                            fos = jsonline['fos']

                            for item in fos:
                                if item["w"] != 0.0:
                                    #skills_pre.extend(item['name'].replace(" ", "_"))
                                    skills_pre.add(item['name'].replace(" ", "_"))
                except:
                    continue
            print(f'Time taken to complete {(time.time() - t) / 60} minutes')
            print(len(skills_pre))
            print(skills_pre)
            exit()
            return set(skills_pre)

    @staticmethod
    def text_preprocess(ab_dict):
        abstract_dict = {}
        fault_ids = []
        for key, text in ab_dict.items():
            try:
                str1 = ''.join(text)
                text_l = str1.lower()

            # hello my name is karan the and. hi how are you data?
                text_p = ["".join([char for char in text_l if char not in string.punctuation])] # hello, my, name, is, karan, web, the, and, hi, how, are, you, data
            # stop_words = stopwords.words('english')
            # filtered_words = [word for word in words if word not in stop_words]
                abstract_dict[key] = set(text_p)
            except TypeError:
                fault_ids.append(key)
                continue
        print(f'Fault IDs length  {len(fault_ids)} and they are {fault_ids}')
        return abstract_dict

