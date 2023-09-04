import re
import nltk

def feature_cleaner(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None, remove_yr=True):
    
    '''
    
    NLP CLEANING CHECK LIST: 
    
    --- Lowecasing the data ## Essential 
    --- Removing Puncuatations ## Essential 
    --- Removing Numbers ## Question 100% fruit juice or contains a certain percentage might matter 
    --- Removing extra space ## Essential 
    --- Replacing the repetitions of punctations ## Essential
    --- Removing Emojis ## We have none 
    --- Removing emoticons ## We have none 
    --- Removing Contractions ## Essential replace them with full words 
    --- Removing unnecessary words ## Essential 
    
    '''

    # lowercasing and removes s from the end of words and white space
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    
     ## remove Year
    if remove_yr == True:
        text = re.sub(r'(?:(?:19|20)\d\d)', '', str(text).lower().strip())
            
    ## convert from string to list using stopwords in lst_stopwords
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word) 
    ## Might not work because it changes the essence of the word 
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

def cleaningPreprocess(df,columnName):
    # Apply feature_cleaner to columns 
    lst_stopwords = nltk.corpus.stopwords.words("english")
    df[columnName] = df[columnName].apply(lambda x: feature_cleaner(x, flg_stemm=False, flg_lemm=False, lst_stopwords=lst_stopwords, remove_yr=True))
    return df
