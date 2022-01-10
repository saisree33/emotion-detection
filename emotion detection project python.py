#!/usr/bin/env python
# coding: utf-8

# In[165]:


from __future__ import division
import pandas as pd
import nltk 
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
from textblob.classifiers import NaiveBayesClassifier


# In[50]:


Dataset = pd.read_csv('ISEAR.csv',header=None)


# In[51]:


Dataset.info()


# In[52]:


emotion_files= ['joy.txt', 'fear.txt', 'anger.txt', 'sadness.txt', 'disgust.txt', 'shame.txt', 'guilt.txt']
emotion_labels=['joy','fear','anger','sadness','disguist','shame','guilt']


# In[53]:


def tokenization(sentence):
    tokenized = nltk.word_tokenize(sentence)
    specialcharacters = ["á", "\xc3", "\xa1", "\n", ",", ".", "[", "]", ""]
    q = []
    for i in tokenized:
        if i not in specialcharacters:
            q.append(i)
    return q


# In[54]:


def pos_tagging(sentence):
   
    taggedsen = nltk.pos_tag(sentence)

    nava_tags = []
    nava_words = []
    for i in taggedsen:
        if i[1].startswith('VB') or i[1].startswith('NN') or i[1].startswith('RB') or i[1].startswith('JJ'):
            nava_tags.append(i)
            nava_words.append(i[0])
    
    return nava_tags, nava_words


# In[55]:


def stemming(sentence):
    sentence_list = []
    sentence_string = []
    sentence_token = []
    porter_stemmer = PorterStemmer()
  
    string = ""
    for w in sentence:
        word_lower = w.lower()
        if len(word_lower) >= 3:
            string += porter_stemmer.stem(word_lower) + " "
    #sentence_string.append(string)
    w_tokenset = nltk.word_tokenize(string)
    #sentence_token.append(w_tokenset)
    w_Text = nltk.Text(w_tokenset)
    #sentence_list.append(w_Text)
    
    return w_Text, string, w_tokenset


# In[56]:


def write_to_file(fname, textdata):
    f = open(fname,'w')
    f.write(str(textdata))
    f.close()


# In[59]:


def convert_to_dataframe(Dataset):
    labels = []

    sen = []
    senstr = []
    sentoken = []
    labelset = []
    for p in range(len(Dataset)):
        if p >= 0:

            emotion = Dataset[0][p]
            statement = Dataset[1][p]

            labels.append(emotion)
            labelset.append([emotion])
            sent = tokenization(statement)
            nava_tags, nava_words = pos_tagging(sent)
            sentences, sentence_string, sentence_token = stemming(nava_words)
            sen.append(sentences)
            senstr.append(sentence_string)
            sentoken.append(sentence_token)

    df = pd.DataFrame({0 : labels, 1 : sen,2 : senstr,3 : sentoken,4 : labelset})
    return df, sentoken, labels, senstr


# In[60]:


df, sentoken, labels, senstr = convert_to_dataframe(Dataset)


# In[128]:


df


# In[83]:


#sentoken


# In[84]:


#senstr


# In[85]:


#labels


# In[11]:


def file_reading(fname):
    f = open(fname,'r')
    emotion_file_words = []
    for l in f.readlines():
        special_characters = ["\n", " ", "\r", "\t"]
        new_word = ''.join([i for i in l if not [c for c in special_characters if c in i]])
        emotion_file_words.append(new_word)
        
    return emotion_file_words


# In[12]:


def emotionfiles_wordlist(words):
    emotion_words = []
    porterstemmer = PorterStemmer()
    for word in words:
        word_lower = word.lower()
        word_stem = porterstemmer.stem(word_lower)
        if word_stem not in emotion_words:
            emotion_words.append(word_stem)
    return emotion_words


# In[168]:


def emotion_wordset(emotion_labels):
    emotion_wordset = {}
    for emotion in emotion_labels:
        index=emotion_labels.index(emotion)
        a=emotion_files[index]
        emotion_file_words = file_reading(a)
        emotion_words = emotionfiles_wordlist(emotion_file_words)
        emotion_wordset[emotion] = emotion_words
        
        #print("printing representative words or synonyms of emotion :"),
        #print(emotion)
        #print(emotion_wordset[emotion])    
    #print emotion_wordset
    return emotion_wordset
#emotion_wordset is dict


# In[14]:


def textvector(sentences, emotion_wordset):
    txt_vect = []
    for s in sentences:
        s_vect = []
        for w in s:
            w_vect = {}
            for emotion in emotion_wordset:
                if w in emotion_wordset[emotion]:

                    try:
                        if emotion not in w_vect[w]:
                            w_vect[w].append(emotion)
                    except KeyError:
                        w_vect[w] = [emotion]
               
            if w_vect:
                
                s_vect.append(w_vect)
        if not s_vect:
            txt_vect.append(s_vect)
        else:
            txt_vect.append(s_vect)
    #print("printing text vector")
    #print(txt_vect)

    return txt_vect
#lexicon_based


# In[15]:


def lexicon_based_classification(txt_vect, labels, emotion_labels):
    count = 0
    total_sent = 0
    for i in range(len(txt_vect)):
        sent_vector = txt_vect[i]
        sen_emo = np.empty(len(emotion_labels))
        sen_emo.fill(0)
        if sent_vector:
            total_sent += 1
            word_emotions = []
            for word in sent_vector:
                emotions =  word.values()[0][0]
            
                word_emotions.append(emotions)
                j = emotion_labels.index(emotions)
                sen_emo[j] += 1
            #print(sen_emo)
            
            highest = np.argwhere(sen_emo == np.amax(sen_emo))
            #print(np.amax(sen_emo))

            indices = highest.flatten().tolist()
            #print(indices)
            for j in indices:
                if emotion_labels[j] == labels[i]:
                    count += 1
                    break

    accuracy = count/total_sent
    
    return accuracy
#classify_lexicon


# In[167]:


e = emotion_wordset(emotion_labels)
#print e
txtvect = textvector(df[1],e)

accuracy = lexicon_based_classification(txtvect, df[0], emotion_labels)
print(accuracy)


# In[17]:


print("Accuracy of Lexicon based classification is : "),
print accuracy*100 ,"%"


# In[99]:



from nltk.classify.naivebayes import NaiveBayesClassifier


# In[75]:


def datacreation(sentence, emotion):
    data_se = []
    for j in range(len(sentence)):
        senten = []
        for sn in sentence[j]:
            senten.append(str(sn))
        emot = emotion[j]
        data_se.append((senten, emot))
    
    return data_se


# In[76]:


def word_features(dataset):
    total_words = []
    for word in sentoken:
            total_words.extend(word)
    #print(total_words)
    word_freq_list = nltk.FreqDist(total_words)
    #wordlist is dict with frequencies
    word_features = word_freq_list.keys()
    return word_features


# In[162]:


a='my name is saic saic'
c=nltk.tokenize.word_tokenize(a)
b=nltk.FreqDist(c)
keys=b.keys()
values=b.values()
print(keys),
print(values)


# In[78]:


def testdata_creation(sent, emot):
    testdata = []
    sen_list = []
    emo_list = []
    for sn in sent:
        sen_list.append(str(sn))
    for em in emot:
        emo_list.append(em)
    for i in range(len(sen_list)):
        t = []
        t.append(sen_list[i])
        t.append(emo_list[i])
        testdata.append(t)
    return testdata


# In[79]:


def data_classification(textdata):
    return         classifier.classify(features_extraction(nltk.word_tokenize(textdata)))


# In[80]:


def accuracy(testdata, classifier):
    testdata_len = float(len(testdata))
    c=0
    for data in testdata:
        if data_classification(data[0]) == data[1]:
            c=c+1
    print(c / testdata_len * 100)
    accuracy = c / testdata_len * 100
    return accuracy


# In[68]:


d=nltk.word_tokenize("am happy kidney ford")


# In[40]:


d


# In[91]:


def feature_extract(d):
    p=set(d)

    features_ = {}
    for word in word_features:
        features_['contains(%s)' % word] = (word in p)
    return features_


# In[101]:


feature_extract(d)


# In[83]:


word_features = word_features(Dataset)


# In[81]:


sentoken = df[3]
emo_labels = df[0]
#l = len(df[3])
size = ((9*(len(df[3])))//10)
senstr = df[2]
data_se = datacreation(sentoken[:size], emo_labels[:size])
testdata = testdata_creation(senstr[size:], emo_labels[size:])


# In[148]:


print("printing train data")
print(data_se)


# In[147]:


print("printing test data")
print(testdata)


# In[100]:


train_set = nltk.classify.util.apply_features(features_extraction, data_se)
#print(train_set)
classifier = NaiveBayesClassifier.train(train_set)
data_classification("am fed up with this")


# In[102]:


data_classification("am fed up with this")


# In[103]:


data_classification("i love sweets")


# In[104]:


data_classification("i don't like to dance public")


# In[105]:


data_classification("i ran away from that dark place")


# In[106]:


data_classification("i wouldn't have beat her")


# In[107]:


print("printing predicted labels of test data using naive bayes")
for i in range(len(testdata)):
    print(testdata[i][0]),
    print(" -> "),
    print(data_classification(testdata[i][0]))


# In[110]:


Naive_accuracy = accuracy(testdata, classifier)


# In[109]:


from gensim import corpora, models, similarities


# In[113]:



training_data = sentoken[:size]
testing_data = senstr[size:]
training_labels = labels[:size]
testing_labels = labels[size:]


# In[114]:


gensim_dictionary = corpora.Dictionary(training_data)
#print(gensim_dictionary.keys())


# In[115]:


gensim_dictionary.save('mycorpus.dict')


# In[116]:


gensim_token = gensim_dictionary.token2id


# In[117]:


mycorpusmm = [gensim_dictionary.doc2bow(textdata) for textdata in training_data]
#print(mycorpusmm)
#matrixmarketfile


# In[118]:


corpora.MmCorpus.serialize('mycorpus.mm', mycorpusmm)


# In[119]:


gensim_c = corpora.MmCorpus('mycorpus.mm')
#print(gensim_c)
#to view
#print(list(gensim_c))
#or
#for document in gensim_c:
 #   print(document)


# In[120]:


tfidf_model = models.TfidfModel(gensim_c)


# In[121]:



mycorpus_tfidf = tfidf_model[gensim_c]


# In[122]:


my_dictionary = corpora.Dictionary.load('mycorpus.dict')


# In[123]:


lsi_model = models.LsiModel(mycorpus_tfidf, id2word=gensim_dictionary, num_topics=7) 


# In[163]:


mycorpus_lsi = lsi_model[mycorpus_tfidf]


# In[164]:


myindex = similarities.MatrixSimilarity(lsi_model[gensim_c])


# In[143]:


lsi_model.save('mycorpus.lsi') 
myindex.save('mycorpus.index')


# In[161]:


texts = [['human', 'interface', 'computer','sing'],['humanbeing','saic','system'],['shravya','harshini','computer']]

dct = corpora.Dictionary(texts)
dcttoken=dct.token2id
c=[dct.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('c.mm',c)
m=corpora.MmCorpus('c.mm')
ttdf=models.TfidfModel(m)
ttdfi=ttdf[m]
ls = models.LsiModel(ttdfi, id2word=dct) # initialize an LSI transformation
corpus_ls = ls[ttdfi]
inde = similarities.MatrixSimilarity(ls[m])
ls.save('m.lsi') 
inde.save('m.index')
ls = models.LsiModel.load('m.lsi')
p=inde[corpus_ls]
n=sorted(enumerate(p))
print(n[0])
print("printing dct keys"),
print(dct.keys())
print("printing dct values"),
print(dct.values())
print("printing dcttoken"),
print(dcttoken)
print("printing c bow model"),
print(c)
print("printing corpus that can be accessed in random,bow")
for i in m:
    print(i)
print("printing ttidf model")
print(ttdf)
print("printing ttidf weights of ttdfi in random access")
for i in ttdfi:
    print(i)
print("printing topic models using lsi model"),
for i in corpus_ls:
    print(i)
print("printing p")
print(p)
print(n)


# In[157]:


def accuracy_of_semantic_similarity(testing_data):
    m = 0
    for cnt, document in enumerate(testing_data):
        vector_bow = my_dictionary.doc2bow(document.lower().split())
        vector_lsi = lsi_model[vector_bow]
        simils = myindex[vector_lsi]
        #print(simils)
        p = sorted(enumerate(simils), key=lambda item: -item[1])[0]
        #print(p)
        i = p[0] 
        print(cnt),
        print(document),
        print("->"),
        print(training_labels[i])
        #print(i),
        #print(training_labels[i])
        
        if training_labels[i] == testing_labels[cnt]:
            m += 2
            
            
    #print(cnt)  
    #print(m)
    acc = m/cnt
    return acc


# In[159]:


print("printing predictions of test data using semantic similarity")
semantic_accuracy = accuracy_of_semantic_similarity(testing_data)


# In[160]:


print semantic_accuracy*100,'%'


# In[ ]:




