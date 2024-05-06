#!/usr/bin/env python
# coding: utf-8

# In[111]:


import requests
import xmltodict
from bs4 import BeautifulSoup as bs
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[112]:


def hasActionCode(dictionary):
    if 'actionCode' in dictionary:
        return True
    else:
        return False


# In[113]:


from IPython.display import clear_output

def getData():
    
    hr_con_list = [
    '116',
    '115',
    '114',
    '113',
    '112',
    '111',
    '110',
    '109',
    '108',
    ]

    hr_con_urls = ["https://www.govinfo.gov/sitemap/bulkdata/BILLSTATUS/" + x + "hr/sitemap.xml" for x in hr_con_list]

    data_list = []
    i = 1
    for congress in hr_con_urls:
        r = requests.get(congress)
        data = xmltodict.parse(r.content)

        data = data['urlset']['url']

        hr_urls = [x['loc'] for x in data]

        for hr_u in hr_urls:

            hr = requests.get(hr_u)

            hr_data = xmltodict.parse(hr.content)

            hr_data = hr_data['billStatus']['bill']

            actions = hr_data['actions']['item']

            action_codes = [x['actionCode'] for x in actions if hasActionCode(x)]

            #We are guaranteed at least one action code - Intro-H/1000

            if '8000' in action_codes: #bill passed
                passed = 1
            elif '9000' in action_codes: #bill failed
                passed = 0
            else: #bill not voted on
                continue

            print(f'{i/len(hr_urls)/9:.4%}', end='\r')

            #Number of committees

            committees = hr_data['committees']['item']

            if type(committees) is dict:
                num_committees = 1
            else:
                num_committees = len(committees)

            #Number of cosponsors

            if 'cosponsors' not in hr_data.keys():
                num_cosponsors = 0
                cosponsor_parties = []
            else:
                cosponsors = hr_data['cosponsors']['item']

                if type(cosponsors) is dict:
                    num_cosponsors = 1
                    cosponsor_parties = [cosponsors['party']]
                else:
                    num_cosponsors = len(cosponsors)
                    cosponsor_parties = [x['party'] for x in cosponsors]

            #Number of Sponsors - We are guaranteed at least one sponsor

            sponsors = hr_data['sponsors']['item']

            if type(sponsors) is dict:
                num_sponsors = 1
                sponsor_parties = [sponsors['party']]
            else:
                num_sponsors = len(sponsors)
                sponsor_parties = [x['party'] for x in sponsors]

            #Bipartisanship

            all_parties = sponsor_parties + cosponsor_parties

            if 'D' in all_parties and 'R' in all_parties:
                bipartisan = 1
            else:
                bipartisan = 0

            #Is first sponsor in the majority or minority?

            #First, must get majority party of current congress:

            cur_congress = hr_data['congress']

            congress_controls = {
                '116': 'D',
                '115': 'R',
                '114': 'R',
                '113': 'R',
                '112': 'R',
                '111': 'D',
                '110': 'D',
                '109': 'R',
                '108': 'R',
            }

            maj = congress_controls[cur_congress]

            if sponsor_parties[0] == maj:
                majority = 1
            else:
                majority = 0

            textVersions = hr_data['textVersions']['item']

            if type(textVersions) is dict:
                text_url = textVersions['formats']['item']['url']
            else:
                text_url = textVersions[0]['formats']['item']['url']

            text_request = requests.get(text_url)

            b = bs(text_request.text)
            textlist = b.find_all('text')
            full_text = ' '.join([x.text for x in textlist])

            if bipartisan == 1:
                partisan_lean = 0
            else:
                if sponsor_parties[0] == 'R':
                    partisan_lean = -1
                else:
                    partisan_lean = 1

            data_list.append([int(num_committees), int(num_cosponsors), int(num_sponsors), int(majority), full_text, int(passed), int(partisan_lean)])


            i+=1
        


# In[114]:


class ClassModel(Dataset):
    
    def __init__(self, data_list):
        
        self.labels = torch.from_numpy(np.array([x[5] for x in data_list]))
        self.inputs = torch.from_numpy(np.array([x[0:4] + [x[6]] for x in data_list]))
        
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        label = self.labels[idx]
        info = self.inputs[idx, :]
        
        return info, label


# In[115]:


class TextModel(Dataset):
    
    def __init__(self, data_list):
        
        
        text_data = [x[4] for x in data_list]
        
        count_vect = CountVectorizer()
        csr = count_vect.fit_transform(text_data)

        self.inputs = torch.sparse_coo_tensor(csr.nonzero(), csr.data, csr.shape)
        
        self.labels = torch.from_numpy(np.array([x[6] for x in data_list]))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        label = self.labels[idx]
        info = self.inputs[idx]
        
        return info, label


# In[116]:


class GovLoader():
    
    def __init__(self, train_size=0.7):
        
        data_list = getData()
        
        self.train, self.test = train_test_split(data_list, train_size=train_size)
    
    def getTextSet(self):
        
        return TextModel(self.train), TextModel(self.test)
    
    def getClassSet(self):
        
        return ClassModel(self.train), ClassModel(self.test)
    

