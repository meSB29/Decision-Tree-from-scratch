import re
from copy import deepcopy
import math
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.util import ngrams
import nltk, re, string, collections
import pandas as pd
import numpy as np

# LOAD THE DATA IN PROPER FORMAT
def load_data(data):
  data_string=""
  with open(data,'r',encoding="ISO-8859-1") as file_data:
    data_string+=file_data.read()
  print('DONE')
  data_string=data_string.lower()
  data_string = re.sub(r"[,.;@#?!&$]+\ *", " ", data_string)
  data_string = re.sub(r" 's ", " is ",data_string)
  data_string = re.sub(r" 't ", " is ",data_string)
  data_string = re.sub(r"['`]+", " " , data_string)
  data_list=list(data_string.split('\n'))
  data_set=[]
  for i in data_list:
    x=i.strip().split(':')
    data_set.append([t.strip() for t in x])
  if [''] in data_set:
    data_set.remove([''])
  return data_set

#MODELLING N GRAM
def n_gram_model(n, docs,Slice):
  lexical_feature={}
  vocab=[]
  for d in docs:
    tokenized=d.split()[1:]
    vocab+=tokenized
    
    esBigrams=ngrams(tokenized,n)
    esBigramFreq= collections.Counter(esBigrams)
    
    for i in esBigramFreq.keys():
      bg=""
      for x in range(n):
        bg+=i[x]
        if x!=n-1:
          bg+=" "
      
      if bg not in lexical_feature.keys():
        lexical_feature[bg]= 0
      lexical_feature[bg]+=esBigramFreq[i]
  lexical_feature=dict(sorted(lexical_feature.items(), key=lambda item: item[1],reverse=True))
  sum_lex=sum([lexical_feature[k] for k in lexical_feature.keys()])
  # print(sum_lex)
  for k in lexical_feature.keys():
    lexical_feature[k]=lexical_feature[k]/sum_lex
  lexical_feature = dict(list(lexical_feature.items())[:Slice]) 
  
  vocab_len=len(list(set(vocab)))
  return lexical_feature,vocab_len

#EXTRACT LEXICAL FEATURES (UNIGRAM, BIGRAM AND TRIGRAM) 
def lexical_feature(n,docs,Slice):
  lexical_feature_list=[]
  lexical_feature,vocab_len=n_gram_model(n,docs,Slice)
  for doc in docs:
    lex_str=""
    prob=1
    prob_def=1/vocab_len
    doc_list=doc.split()[1:] 
    for x in range(len(doc_list)-(n-1)):
      
      t=""
      for ind in range(n):
        t+=doc_list[x+ind]
        if(ind!=n-1):
          t+=" "
      
      if t in lexical_feature.keys():
        prob*=lexical_feature[t]
      else:
        prob*=prob_def
    lexical_feature_list.append(math.log(prob))

  return lexical_feature_list

#MODELLING UNIGRAM FOR POS TAGS
def postag_model(docs,Slice):
  synctactical_feature={}
  vocab=[]
  for doc in docs:
    vocab+=doc.split()[1:]
    text= doc.split()[1:]
    postag=nltk.pos_tag(text)
    
    postagFreq= collections.Counter(postag)
    
    for i in postagFreq.keys():
      bg=i
      
      if bg not in synctactical_feature.keys():
        synctactical_feature[bg]= 0
      synctactical_feature[bg]+=postagFreq[i]
  synctactical_feature=dict(sorted(synctactical_feature.items(), key=lambda item: item[1],reverse=True))
  sum_syn=sum([synctactical_feature[k] for k in synctactical_feature.keys()])
  for k in synctactical_feature.keys():
    synctactical_feature[k]=synctactical_feature[k]/sum_syn
  synctactical_feature = dict(list(synctactical_feature.items())[:Slice]) 
  vocab_len=len(list(set(vocab)))
  return synctactical_feature,vocab_len

# EXTRACT SYNTACTICAL FEATURES (UNIGRAM)
def syntactical_feature(docs,Slice):
  syntactical_feature_list=[]
  syn_feature,vocab_len=postag_model(docs,Slice)
  
  for doc in docs:
    syn_str=[]
    text=list(set(doc.split()[1:]))
    for txt in text:
      max_pos=[]
      for synk in list(syn_feature.keys()):
        if synk[0]==txt:
          max_pos.append([synk[1],syn_feature[tuple(synk)]])
      if len(max_pos)>0:
        max_pos=sorted(max_pos, key=lambda item: item[1],reverse=True)
        syn_str.append(max_pos[0][0])
    prob=1
    prob_def=1/vocab_len
    for s in syn_str:
      if s in syn_feature.keys():
        prob*=syn_feature[s]
      else:
        prob*=prob_def
    syntactical_feature_list.append(math.log(prob))
  return syntactical_feature_list

#STORING ALL THE REQUIRED FEATURES INTO DATAFRAME
def get_dataframe(data_set):
  mega_doc=""
  for st in data_set:
    mega_doc+=st[1]+" . "
  mega_doc=mega_doc.strip()
  docs=mega_doc.split('.')
  docs=docs[:-1]
  docs=[doc.strip() for doc in docs]
  # print(train_mega_doc)
  data=pd.DataFrame.from_dict({'question': [' '.join(doc.split()[1:]) for doc in docs] ,'Class':[i[0] for i in data_set] ,'length': [len(doc.split()) for doc in docs] , 'lexical_unigram': lexical_feature(1,docs,500), 'lexical_bigram': lexical_feature(2,docs,300), 'lexical_trigram': lexical_feature(3,docs,200), 'syntactical_unigram' : syntactical_feature(docs,500)})

  class_set=list(set(list(data['Class'])))
  
  return data

#FUNCTION TO CALCULATE GINI INDEX
def gini(dataset,attr_list,classes):
  print('USING GINI INDEX FOR SPLITTING...')
  gini_split={}
  for var in attr_list[2:]:
    gini_split[var]=0

    sum_temp_df=len(dataset)
    best_split=[]
    temp_df=np.linspace(min(list(dataset[var]))+1,max(list(dataset[var]))-1,num=int(np.round(max(list(dataset[var]))-min(list(dataset[var]))+1,0)))
    for td in temp_df:
      #for <= td
      gini_list=[]
      temp_data=dataset[dataset[var]<=td]
      for tc in classes:
        gini_list.append(len(temp_data[temp_data['Class']==tc]))
      sum_gini=sum(gini_list)
      gini=1
      for g in gini_list:
        gini-=(g**2)/(sum_gini**2)
      
      best_split_less=(len(temp_data)/sum_temp_df)*gini
      # for>td
      gini_list=[]
      temp_data=dataset[dataset[var]>td]
      for tc in classes:
        gini_list.append(len(temp_data[temp_data['Class']==tc]))
      sum_gini=sum(gini_list)
      gini=1
      for g in gini_list:
        gini-=(g**2)/(sum_gini**2)
      best_split_more=(len(temp_data)/sum_temp_df)*gini
      best_split.append([best_split_less+best_split_more,td])

    best_split=sorted(best_split,key=lambda item : item[0])
    # print('best split : ', best_split[0][1])
    gini_split[var]=[best_split[0][0],best_split[0][1]]
  gini_split=sorted(gini_split.items(),key=lambda kv : kv[1][0])
  return gini_split

#FUNCTION TO CALCULATE ENTROPY
def entropy(dataset,attr_list,classes):
  print('USING ENTROPY FOR SPLITTING...')
  entropy_split={}
  for var in attr_list[2:]:
    entropy_split[var]=0
    
    sum_temp_df=len(dataset)
    entropy_list=[]
    for tc in classes:
      entropy_list.append(len(dataset[dataset['Class']==tc]))
    sum_entropy=sum(entropy_list)
    parent_entropy=0
    for g in entropy_list:
      parent_entropy-=(g/sum_entropy)*(math.log(g/sum_entropy))
    best_split=[]
    temp_df=np.linspace(min(list(dataset[var]))+1,max(list(dataset[var]))-1,num=int(np.round(max(list(dataset[var]))-min(list(dataset[var]))+1,0)))
    for td in temp_df:
      #for <= td
      entropy_list=[]
      temp_data=dataset[dataset[var]<=td]
      for tc in classes:
        entropy_list.append(len(temp_data[temp_data['Class']==tc]))
      sum_entropy=sum(entropy_list)
      entropy=0
      for g in entropy_list:
        if g/sum_entropy > 0:
          entropy-=(g/sum_entropy)*(math.log(g/sum_entropy))
      
      best_split_less=(len(temp_data)/sum_temp_df)*entropy
      # for>td
      entropy_list=[]
      temp_data=dataset[dataset[var]>td]
      for tc in classes:
        entropy_list.append(len(temp_data[temp_data['Class']==tc]))
      sum_entropy=sum(entropy_list)
      entropy=0
      for g in entropy_list:
        if g/sum_entropy > 0:
          entropy-=(g/sum_entropy)*(math.log(g/sum_entropy))
      best_split_more=(len(temp_data)/sum_temp_df)*entropy
      #appending data for all splits
      best_split.append([best_split_less+best_split_more,td])

    best_split=sorted(best_split,key=lambda item : item[0])
    # print('best split : ', best_split[0][1])
    entropy_split[var]=[best_split[0][0],best_split[0][1]]
  entropy_split=sorted(entropy_split.items(),key=lambda kv : kv[1][0], reverse=True)
  return entropy_split

# FUNCTION TO CALCULATE MISCLASSIFICATION ERROR
def misclass(dataset,attr_list,classes):
  print('USING MISCLASS FOR SPLITTING...')
  misclass_split={}
  for var in attr_list[2:]:
    misclass_split[var]=0
    # print(var)
    
    sum_temp_df=len(dataset)
    best_split=[]
    temp_df=np.linspace(min(list(dataset[var]))+1,max(list(dataset[var]))-1,num=int(np.round(max(list(dataset[var]))-min(list(dataset[var]))+1,0)))
    for td in temp_df:
      #for <= td
      misclass_list=[]
      temp_data=dataset[dataset[var]<=td]
      for tc in classes:
        misclass_list.append(len(temp_data[temp_data['Class']==tc]))
      sum_misclass=sum(misclass_list)
      misclass=1
      max_misclass=0
      for g in misclass_list:
        if (g/sum_misclass) > max_misclass:
          max_misclass=(g/sum_misclass) 
      
      best_split_less=(len(temp_data)/sum_temp_df)*(misclass-max_misclass)
      # for>td
      misclass_list=[]
      temp_data=dataset[dataset[var]>td]
      for tc in classes:
        misclass_list.append(len(temp_data[temp_data['Class']==tc]))
      sum_misclass=sum(misclass_list)
      misclass=1
      max_misclass=0
      for g in misclass_list:
        if (g/sum_misclass) > max_misclass:
          max_misclass=(g/sum_misclass) 
      best_split_more=(len(temp_data)/sum_temp_df)*(misclass-max_misclass)
      best_split.append([best_split_less+best_split_more,td])
        
    best_split=sorted(best_split,key=lambda item : item[0])
    # print('best split : ', best_split[0][1])
    misclass_split[var]=[best_split[0][0],best_split[0][1]]
  misclass_split=sorted(misclass_split.items(),key=lambda kv : kv[1][0])
  return misclass_split

#FINDING BEST SPLITS USING COST FUNCTION "func" 
# "func" CAN HAVE VALUES : gini, entropy, misclass
def decision_tree_splits(dataset,func):
  attr_list=list(dataset.columns)
  classes=list(set(list(dataset.Class)))
  cost_split=func(dataset,attr_list,classes)
  
  cost_split=dict(cost_split)
  return cost_split

# RECURSIVE FUNCTION TO BUILD DECISION TREE
def decision_tree(cost_dict,dataset,d):
  if d==len(list(cost_dict.keys())):
    if len(list(set(list(dataset['Class']))))>1:
      maxclass=[]
      for c in list(set(list(dataset['Class']))):
        maxclass.append(len(dataset[dataset['Class']==c]))
      ind=maxclass.index(max(maxclass))
      return list(set(list(dataset['Class'])))[ind]
    elif len(list(set(list(dataset['Class']))))==1:
      return list(set(list(dataset['Class'])))[0]
    else:
      return 'desc'
  
  node={}
  atr=list(cost_dict.keys())[d]
  split_val=cost_dict[atr][1]
  
  if len(list(set(list(dataset[dataset[atr]<=split_val]['Class']))))==1:
    node['left']=list(set(list(dataset[dataset[atr]<=split_val]['Class'])))[0]
  else:
    node['left']=decision_tree(cost_dict,dataset[dataset[atr]<=split_val],d+1)

  if len(list(set(list(dataset[dataset[atr]>split_val]['Class']))))==1:
    node['right']=list(set(list(dataset[dataset[atr]>split_val]['Class'])))[0]
  else:
    node['right']=decision_tree(cost_dict,dataset[dataset[atr]>split_val],d+1)

  return node

# BUILD DECISION TREE ACCORDING TO COST FUNCTION "func"
# "func" CAN HAVE VALUES : gini, entropy, misclass 
def build_tree(dataset,func):
  cost_dict=decision_tree_splits(dataset,func)
  print('BUILDING TREE...')
  final_tree=decision_tree(cost_dict,dataset,0)
  return final_tree,cost_dict

# FUNCTION FOR CLASSIFICATION
def classify(test_tree,tst,g,test_data):
  for var in g.keys():
    
    if test_data.loc[tst][var]<=g[var][1]:
      test_tree=test_tree['left']
    else:
      test_tree=test_tree['right']
    
    if str(type(test_tree))=="<class 'str'>":
      return test_tree

# FUNCTION TO PERFORM K FOLD CROSS VALIDATION
def k_fold(data,k,class_set,metric):
  print(k,' FOLD SCORES...')
  scores=[]
  
  class_dict={class_set[i]:i for i in range(len(class_set))}
  for i in range(k):
    test_data=data.iloc[i*len(data)//k:min(((i+1)*len(data))//k,len(data)),:]
    train_data=data[~data.index.isin(test_data.index)]
    test_data.reset_index(inplace = True, drop = True)
    train_data.reset_index(inplace = True, drop = True)
    
    train_data.index=[i for i in range(len(train_data))]
    test_data.index=[i for i in range(len(test_data))]
    predict=[]
    acc=0
    
    ft,g=build_tree(train_data.copy(),metric)
    for tst in test_data.index:
      test_tree=deepcopy(ft)
      val=classify(test_tree,tst,g,test_data)
      
      predict.append(val)
    confusion_matrix=[[0 for l in range(len(class_dict.keys()))] for x in range(len(class_dict.keys()))]

    test_data['prediction']=predict
    for p in class_dict.keys():
      for t in class_dict.keys():
        confusion_matrix[class_dict[p]][class_dict[t]]=len(test_data[(test_data['Class']==t)&(test_data['prediction']==p)])


    acc=len(test_data[test_data['Class']==test_data['prediction']])/len(test_data)
    print('iteration ',i+1,'accuracy :', acc)
    

    scores.append([acc,confusion_matrix])
  return scores

if __name__ == '__main__':
  for metric in [gini,entropy,misclass]:
    train_set=load_data('train_5500.label.txt')
    test_set=load_data('test.label.txt')

    train_data=get_dataframe(deepcopy(train_set))
    test_data=get_dataframe(deepcopy(test_set))
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    class_set=sorted(list(set(list(train_data['Class']))))
    k_fold_scores=k_fold(deepcopy(train_data),10,class_set,metric)

    # UNCOMMENT FOLLOWING LINES TO PRINT THE CONFUSION MATRIX FOR K FOLD
    # print('-------CM FOR K FOLD------------')
    # for sc in k_fold_scores:
    #   for x in sc[1]:
    #       print(x)
    #   print('-------------------')

    # class_dict={class_set[i]:i for i in range(len(class_set))}

    print('APPLYING DECISION TREE FOR TEST SET...')
    predict=[]
    acc=0
    
    
    ft,g=build_tree(train_data.copy(),metric)
    for tst in range(len(test_data)):
      test_tree=deepcopy(ft)
      val=classify(test_tree,tst,g,test_data)
      
      predict.append(val)
    test_data['prediction']=predict
    confusion_matrix=[[0 for l in range(len(class_dict.keys()))] for x in range(len(class_dict.keys()))]
    for p in class_dict.keys():
      for t in class_dict.keys():
        confusion_matrix[class_dict[p]][class_dict[t]]=len(test_data[(test_data['Class']==t)&(test_data['prediction']==p)])
    
    print('----TEST DATA CM---------')
    for x in confusion_matrix:
      print(x)
    print('-------------------------')

    print('accuracy :', len(test_data[test_data['Class']==test_data['prediction']])/len(test_data)*100)

