'''
Author: JasonYU
Date: 2020-10-02 09:17:33
LastEditTime: 2020-10-04 10:00:43
FilePath: \SE\flask_se\backend\se_sample.py
'''

import json
import re
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import copy

inverted_file = {}
df_table = {}
tf_table = {}
docs_keywords = {}



def tokenize(text):
    text = re.sub(r'[^\w\s]',' ',text)
    return word_tokenize(text)


def get_inverted_file(papers):

    
    for i in range(len(papers)):
        ps = PorterStemmer()
        paper_tokens = tokenize(papers[i]['abstract'])
        lower_tokens = [word.lower() for word in paper_tokens]
        filtered_tokens = [token for token in lower_tokens if token not in set(stopwords.words('english'))]        
        key_words = list(set(ps.stem(filtered_token) for filtered_token in filtered_tokens))
        
        docs_keywords[i] = key_words
        
        cur_dict = {}
        
        for index in range(len(lower_tokens)):
            if lower_tokens[index] in set(stopwords.words('english')):
                continue
            word = ps.stem(lower_tokens[index])
            
            if cur_dict.get(word) == None:
                cur_dict[word] = [index]
            else:
                cur_dict[word].append(index)
        
        for key in cur_dict.keys():
            if inverted_file.get(key) == None:
                inverted_file[key] = {i:cur_dict[key]}
            else:
                inverted_file[key][i] = cur_dict[key]
        
    for key_word in inverted_file.keys():
        pos_dict = inverted_file[key_word]
        df_table[key_word] = len(pos_dict)
        a_dict = {}
        for key_doc_pos in pos_dict.keys():
            a_dict[key_doc_pos] = len(pos_dict[key_doc_pos])
        tf_table[key_word] = a_dict
        
        
def process_string(string):
    processed_string = []
    ps = PorterStemmer()
    
    for t in tokenize(string):
        if t.lower() not in set(stopwords.words('english')):
            processed_string.append(ps.stem(t.lower()))
    
    processed_string = list(set(processed_string))
    
    return processed_string


# get data
with open('./paper.json','r') as f: 
    papers = json.load(f)

get_inverted_file(papers)


def get_weighted_d(doc_index, doc_keywords):
    weighted_d = {}
    for doc_keyword in doc_keywords:
        if tf_table.get(doc_keyword).get(doc_index) != None:
            weighted_d[doc_keyword] = tf_table.get(doc_keyword).get(doc_index) * math.log2(778.0 / df_table.get(doc_keyword))
    return weighted_d
    
def get_weighted_q(doc_index, processed_query):
    weighted_q = {}
    for q in processed_query:
        weighted_q[q] = 1.0
        
    return weighted_q

def get_mag(weighted_vec):
    mag = 0
    for key in weighted_vec.keys():
        mag += weighted_vec.get(key) * weighted_vec.get(key)
    mag = math.sqrt(mag)
    return mag

    
    
def get_cos_sim(weighted_d, weighted_q):
    inner_product = 0
    
    mag_d = get_mag(weighted_d)
    mag_q = get_mag(weighted_q)
    
    for key in weighted_q.keys():
        if key in weighted_d.keys():
            inner_product += weighted_d.get(key)

    cos_sim = inner_product / (mag_d * mag_q)
    
    return cos_sim

def show_top_info(top_documents_list):
    
    for top_document in top_documents_list:
        doc_weighted_vec = get_weighted_d(top_document[0], docs_keywords[top_document[0]])
        doc_mag = get_mag(doc_weighted_vec)
        
        top_words_list = sorted(doc_weighted_vec.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        
        num = min(5, len(top_words_list))
    
        top_words_list = top_words_list[0:num]
        
        
        unique_keys_num = []
        
        for doc_keyword in docs_keywords[top_document[0]]:
            if len(inverted_file.get(doc_keyword)) == 1:
                unique_keys_num.append(doc_keyword)
        
        print("DID: D",top_document[0])
        print("5 highest weighted words:", end = '')
        for top_word in top_words_list:
            print('')
            print('{}:'.format(top_word[0])) 
            for key, value in inverted_file.get(top_word[0]).items():
                print('|D{}:{} '.format(key, value), end = '')
        print('')
        print("number of unique keywords-> ", len(docs_keywords[top_document[0]]))
        print("number of unique keywords(the keywords other documents don't have')->", len(unique_keys_num))
        print("magnitude-> ", doc_mag)
        print("sim_score->", top_document[1])


def get_top_documents(papers, processed_query):
    #doc_id : cos_sim
    top_documents = {}
    
    #docs : total doc contains term of q
    docs_index = []
    for q in processed_query:
        if(inverted_file.get(q) != None):
            docs_index += inverted_file.get(q).keys()
    
    docs_index = list(set(docs_index))
    
    for doc_index in docs_index:
        doc_keywords = docs_keywords.get(doc_index)
        weighted_d = get_weighted_d(doc_index, doc_keywords)
        weighted_q = get_weighted_q(doc_index, processed_query)
        cos_sim = get_cos_sim(weighted_d, weighted_q)
        top_documents[doc_index] = cos_sim
        

    top_documents_list = sorted(top_documents.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    num = min(5, len(top_documents_list))
    
    top_documents_list = top_documents_list[0:num]
    
    show_top_info(top_documents_list)

    
    return top_documents_list      


out_dict = {}


def process_phase(phase):
    flag = True
    out_dict[" ".join(phase)] = copy.deepcopy(inverted_file.get(phase[0]))
    for i in range(len(phase) - 1):
        flag &= search_join(phase[i], phase[i+1], phase)
        if flag == False:
            del out_dict[" ".join(phase)]
            return False
    return True
    

def search_join(key_word1, key_word2, phase):
    cur_dict = inverted_file.get(key_word2)
    both_keys = []
    for cur_key in cur_dict.keys():
        if cur_key in out_dict[" ".join(phase)].keys():
            both_keys.append(cur_key)

        
    if len(both_keys) == 0:
        return False
    
    for out_key in list(out_dict[" ".join(phase)].keys()):
        if out_key not in both_keys:
            del out_dict[" ".join(phase)][out_key]
    
    result = False
    
    for both_key in both_keys:
        both_pos_list = []
        for pos in out_dict[" ".join(phase)][both_key]:
            if (pos + 1) in cur_dict[both_key]:
                both_pos_list.append(pos + 1)
        if len(both_pos_list) != 0:
            out_dict[" ".join(phase)][both_key] = both_pos_list
            result = True            
        else:
            del out_dict[" ".join(phase)][both_key]
    return result
                
                
                
def search_api(query):
    """
    query:[string] 
    return: list of dict, each dict is a paper record of the original dataset
    """
    
    total_quotation_num = 0
    for q in query:
        if q == '"':
            total_quotation_num += 1
    
    if total_quotation_num == 0 or total_quotation_num % 2 != 0 or len(process_string(query)) == 1:
        processed_query = process_string(query)
    
        top_documents_list = get_top_documents(papers, processed_query)
    
        result = []
    
        for top_document in top_documents_list:
            result.append(papers[top_document[0]])
                
        return result
    
    quotation_acc = 0
    
    
    single_word_query = ""
    phrase_string = ""
    
    
    for q in query:
        if q == '"':
            quotation_acc += 1
            phrase_string += '"'
            continue
        if quotation_acc % 2 == 0:
            single_word_query += q
        else:
            phrase_string += q
    #single words after tokenize
    if single_word_query != "":
        single_query_keys = process_string(single_word_query)
   
    cur_phrase_list = phrase_string.split('"')
    cur_phrase_list = [i for i in cur_phrase_list if i != '']
    

    
    query_phrase_list = []
    
    
    for cur_phrase in cur_phrase_list:
        if process_string(cur_phrase) in query_phrase_list:
            continue
        query_phrase_list.append(process_string(cur_phrase))
    


    
    query_keys = []
    if single_word_query != "":
        query_keys += single_query_keys
    for query_phrase in query_phrase_list:
        if process_phase(query_phrase):
            query_keys.append(query_phrase)
    
    #print('first out_dict:',out_dict)

    docs_index = []
    
    for query_key in query_keys:
        if isinstance(query_key, list):
            query_key_string = " ".join(query_key)
            docs_index += out_dict[query_key_string].keys()
        else:
            docs_index += inverted_file.get(query_key).keys()
    
    docs_index = list(set(docs_index))

    

    top_documents_ph = {}
    for doc_index in docs_index:
        weighted_d = [0 for _ in range(len(query_keys))]
        for i in range(len(query_keys)):
            if isinstance(query_keys[i], list):
                query_key_string = " ".join(query_keys[i])
                if out_dict[query_key_string].get(doc_index) != None:  
                    weighted_d[i] = len(out_dict[query_key_string].get(doc_index)) * math.log2(778.0 / len(out_dict[query_key_string]))
            else:
                if tf_table.get(query_keys[i]).get(doc_index) != None:
                    weighted_d[i] = tf_table.get(query_keys[i]).get(doc_index) * math.log2(778.0 / df_table.get(query_keys[i]))
        mag_d = 0
        for weight in weighted_d:
            mag_d += weight * weight
       
        weighted_dict = get_weighted_d(doc_index, docs_keywords[doc_index])
        for weighted_dict_key in weighted_dict.keys():
            mag_d += weighted_dict.get(weighted_dict_key) * weighted_dict.get(weighted_dict_key)
        
        mag_d = math.sqrt(mag_d)
        top_documents_ph[doc_index] = sum(weighted_d) / ( math.sqrt(len(query_keys)) * mag_d )
    


    top_documents_list_ph = sorted(top_documents_ph.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    
    #print('top_documents_list_ph:',top_documents_list_ph)

    num = min(5, len(top_documents_list_ph))
    
    top_documents_list_ph = top_documents_list_ph[0:num] 
    
    result_ph = []
    

    for top_document in top_documents_list_ph:
        result_ph.append(papers[top_document[0]])
        
    #print('final out_dict:',out_dict)
    print(result_ph)
    
    return result_ph
    

if __name__ == "__main__":
    search_api("Natural Language Processing")
