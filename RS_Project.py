from scipy.stats import chi2
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import nltk
import string
import os
import csv
import copy
import operator
import matplotlib.pyplot as plt


def load_data_csv(data_path):
    """
    load data of papre citation
    :param data_path:
    :return:
    """
    papers_dic = {}
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ', quotechar='"')
        for row in reader:
            p1 = row[0]
            if p1 not in papers_dic:
                papers_dic[p1]=[]
            p2=row[1]
            papers_dic[p1].append(p2)
    return papers_dic

def removeWeakPapers(citation_dic):
    """
    remove the  documents cited  less than 20 papers and The documents cited
    by less than 5 documentes according to the artical
    :param citation_dic:
    :return:
    """
    dic_strength = {}
    for p, v in citation_dic.items():
        for cit in v:

            if cit not in dic_strength:
                dic_strength[cit] = 1
            else:
                dic_strength[cit] += 1
    sorted_p = sorted(dic_strength.items(), key=operator.itemgetter(1))
    list_removePapers = []
    for p_tup in sorted_p:
        if p_tup[1] <5:
            citation_dic.pop(p_tup[0], None)
            list_removePapers.append(p_tup[0])
        else:
            break

    temp_dic= copy.deepcopy(citation_dic)
    for p, cit in temp_dic.items():

        if len(cit) < 20:
            citation_dic.pop(p, None)
            list_removePapers.append(p)

    temp_dic= copy.deepcopy(citation_dic)
    for p in temp_dic:
        set1 = set(citation_dic[p]) - set(list_removePapers)
        list1 = list(set1)
        citation_dic[p] = list1

        if (len(citation_dic[p])< 1):
            citation_dic.pop(p, None)
    return citation_dic

def split_test_train(citation_dic):
    """
    split_test_train
    :param citation_dic:
    :return:
    """
    dic_test={}
    dic_train = {}
    for key, value in citation_dic.items():
        X_train, x_test = train_test_split(value, test_size=0.2, random_state=42)

        dic_train[key]=X_train

        dic_test[key]=x_test
    return dic_train, dic_test

def read_files_abs(path):
    """
    get the name if the files of papers abstract
    :param path:
    :return:files_list
    """
    dirs = os.listdir(path)
    files_list=[]
    for dir in dirs:
        path__years=os.path.join(path, dir)
        dirs_files = os.listdir(path__years)
        for file in dirs_files:
            file_fall=os.path.join(path__years, file)
            files_list.append(file_fall)
            file_txt = file_fall[:-8]
            file_txt= file_txt+'.txt'
            files_list.append(file_fall)
    return files_list

def calc_popularity(papers_dic):
    """
    calaculate the popularity for each paper
    :param papers_dic:
    :return: popularity dic
    """
    dic_popular={}
    for p, v in papers_dic.items():
        for cit in v:
            if cit not in dic_popular:
                dic_popular[cit] = 1

            else:

                dic_popular[cit] += 1
    return dic_popular



def tokenize(text):
    """
    word_tokenize for calculation of tfidf
    """
    tokens = nltk.word_tokenize(text)
    return tokens

def get_file_name(path_file):
    """
    get short file name from the path
    :param path_file:
    :return: filename_short
    """
    dirname, filename = os.path.split(path_file)
    filename_short=filename[:-4]
    return filename_short



def tridf(files_list):
    """
    :param files_list:
    :return:
    """
    for file in files_list:
        shakes = open(file, 'r')
        text = shakes.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        cleanData = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", no_punctuation)
        doc_name=str(get_file_name(file))
        token_dict[doc_name] = cleanData
    print("get clean data")
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    counter = 0
    #connection between the tfidf vector and the paper name
    for k,v in token_dict.items():
        values.append(v)
        key_dics_tf[k] = counter
        counter+=1
    tfs = tfidf.fit_transform(values)
    return tfs


def run_tdidf_full():
    """
    run tfidf- get the files names and run tfidf function
    :return:
    """
    path_abs="C:\Users\lera\Desktop\data_recommandation\hep-th-abs.tar"
    files_list=read_files_abs(path_abs)
    print("get files_list")
    tfs=tridf(files_list)
    print("finish tfidf")
    return tfs

def calac_dfidf_similarity(d1, d2,tfs):
    """
    calc similaroty of two papers
    :param d1:
    :param d2:
    :return: cos_sim
    """
    d1_row=get_doc_row(d1,tfs)
    d2_row = get_doc_row(d2,tfs)
    cos_sim=cosine_similarity(d1_row, d2_row)
    return cos_sim

def calac_cos_din_neighbored_papers(papers_similarity_dic,dic_train,path_sim,tfs):
    """
        calc neighbored_papers similarity and saved it in csv file
        every threashold get it own file of similarity
    :param papers_similarity_dic:
    :param dic_train:
    :param path: the path to save the file in
    :return:
    """
    counter=0
    cosine_similarity_dic={}
    with open(path_sim, 'w') as csvfile:
        fieldnames = ['paper_i', 'paper_j', "score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p1,value in papers_similarity_dic.items():
            for p2 in value:
                if p2 not in dic_train[p1]:
                    if p1!=p2:
                        d1_row = get_doc_row(p1,tfs)
                        d2_row = get_doc_row(p2,tfs)
                        simillar=cosine_similarity(d1_row, d2_row)
                        tup=(p1, p2)
                        tup2 = (p2, p1)
                        cosine_similarity_dic[tup]=simillar[0][0]
                        cosine_similarity_dic[tup2] = simillar[0][0]
                        writer.writerow({'paper_i': p1, 'paper_j': p2, 'score':simillar[0][0]})
                        counter+=1
                        if (counter%5000==0):
                            print(counter)
    return cosine_similarity_dic


def sum_wight(p1_id, recommandation_dic, popular_dic_all, n, calac_cos_dic, alpha_pop=0.2, beta_similarity=0.5,
                  gama_score=0.3):
    """
    get the recommande list of citation to paper with our improve algoretem- use artical score, popularity of citation
    paprt and similarity tdidf between the paper and the the citation paper that candidates for recommendation
    :param p1_id:
    :param recommandation_dic:
    recommandation_dic is dictianry with scores for all papars according to artical algoretem
    :param popular_dic_all:
    :param n: number of papers to return in recommandation list
    :param calac_cos_dic:
    :param alpha_pop:
    :param beta_similarity:
    :param gama_score:
    :return:
    """
    total_score={}
    similarity_dic={}
    sum_pop=0.0
    for paper in recommandation_dic:
        tup=(p1_id,paper)
        similarity_dic[paper]=calac_cos_dic[tup]
        sum_pop += popular_dic_all[paper]
    for key,value in recommandation_dic.items():
        pop_p=popular_dic_all[key]/sum_pop
        total_score[key]=alpha_pop*pop_p+beta_similarity*similarity_dic[key]+gama_score*value
    sorted_x = sorted(total_score.items(), key=operator.itemgetter(1), reverse=True)
    topn=sorted_x[:n]
    first_element= [i[0] for i in topn]
    return first_element

def get_doc_row(docid,tfs):
    """
   return the tfidf vector according to dic id
    :param docid:
    :return:
    """
    rowid = key_dics_tf[docid]
    row = tfs[rowid,:]
    return row


def load_clean_data(path):
    """
    load the data of paper citation after cleaning the The documents cited  less than 20 papers and The documents cited
    by less than 5 documentes
    i saved only the clean documentes in file to Reduce running time
    :param path:
    :return:
    """
    papers_dic = {}
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            if (row[0] == "paper_i" or row[0] == ""):
                continue
            p1 = row[0]
            if p1 not in papers_dic:
                papers_dic[p1]=[]
            pj=row[1]
            papers_dic[p1].append(pj)
    return papers_dic

def load_tdidf_sim_dic(data_path):
    """
    after the sim_dic was saved load the similarity dictinart and use it
    :param data_path:
    :return:
    """
    papers_tfIdf_scores = {}
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            if (row[0] == "paper_i" or row[0] == ""):
                continue
            p1 = row[0]
            p2 = row[1]
            score = float(row[2])
            tup=(p1,p2)
            tup2 = (p2,p1)
            papers_tfIdf_scores[tup]=score
            papers_tfIdf_scores[tup2] = score
    return papers_tfIdf_scores


def coputeChiSqure ( list_p1, list_p2, number_papers):
    """
    coputeChiSqure for calculate the similarity between papers according to the artical
    :param list_p1:
    :param list_p2:
    :param number_papers:
    :return:
    """
    # cited by both
    N11 = len(set(list_p1) & set(list_p2))
    # cited by 1
    N12 = len(set(list_p1) - set(list_p2))

    # cited by 2
    N21 = len(set(list_p2) - set(list_p1))
    # cited by none
    N22 = number_papers - N11 - N12 - N21

    R1= N11+N12
    R2= N21 + N22
    C1= N11 + N21
    C2= N12 + N22
    N = C1 + C2
    if ((R1*R2*C1*C2) == 0):
        return 0
    chi_score = math.pow(( math.fabs( N11*N22-N12*N21 ) - N/2),2)/ (R1*R2*C1*C2)

    return chi_score


def func(chi_score,df):
    """
    calc The cumulative function of chi_score
    :param chi_score: calc by coputeChiSqure function
    :param df: degrees of freedom
    :return:
    """
    return chi2.cdf(chi_score, df)

def cala_ccd(papers_dic, ts,number_papers):
    """
    calac co_occur papers according to artical algorithm and save tham to file
    for our work we didnt used this files
    :param papers_dic:
    :param ts: threshold
    :param number_papers:number_papers in all the curpos
    :return:dic_co_occur, for each paper its co_occur papers
    """
    dic_co_occur={}
    for pair in itertools.combinations(papers_dic.keys(),2):
        chi_score=coputeChiSqure(papers_dic[pair[0]],papers_dic[pair[1]],number_papers )
        chi_cumulative=func(chi_score, 1)
        if pair[0] not in dic_co_occur:
            dic_co_occur[pair[0]] = []
        if pair[1] not in dic_co_occur:
            dic_co_occur[pair[1]] = []
        #if the chi_cumulative is bigger the threshold, check differernt threshold
        if chi_cumulative>ts:
            dic_co_occur[pair[0]].append(pair[1])
            dic_co_occur[pair[1]].append(pair[0])
    with open('calc_co_occur_0.5.csv', 'w') as csvfile:
        fieldnames = ['paper_i', 'list_co_occur_i']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key,value in dic_co_occur.items():
            writer.writerow({'paper_i': key, 'list_co_occur_i': value})
    return dic_co_occur

def calac_sim(dic_co_occur):
    """
    calc sim_cosine between all the papars in dic_co_occur
    :param dic_co_occur:
    :return:
    """
    dic_similarity={}
    for key,value in dic_co_occur.items():
        for item in value:
            if (key,item) not in dic_similarity:
                similar=sim_cosine(dic_co_occur[key], dic_co_occur[item])
                dic_similarity[(key,item)]=similar
                dic_similarity[(item, key)] = similar
    return dic_similarity

def sim_cosine(vec_p1, vec_p2):
    """
    calc sim_cosine
    :param vec_p1:
    :param vec_p2:
    :return:
    """
    up=0.0
    down=0.0
    up+=len(set(vec_p1).intersection(set(vec_p2)))
    down+=(len(vec_p1) * len(vec_p2))
    return up / down

def get_Score(sim_dic,dic_nighb, papers_dic, p1,pj):
    """
    get the score of p1 and pj, calac the scores that was given to pj by all the papers that similar to p1, similar
    according similarity calculation by function calac_sim
    :param sim_dic:
    :param dic_nighb:
    :param papers_dic:
    :param p1:
    :param pj:
    :return:
    """
    sum_som=0.0
    sum_sim_j_up=0.0
    for item in dic_nighb[p1]:
        sim_item=sim_dic[(p1, item)]
        sum_som+=sim_item
        if pj in papers_dic[item]:
            sum_sim_j_up+=sim_item
    if sum_som==0:
        return 0
    return sum_sim_j_up/sum_som


def calc_scores_dic(x_train,sim_dic,dic_nighb):
    """
    calc the scores sictinary for each paper according to artical algorithm
    :param x_train:
    :param sim_dic:
    :param dic_nighb:
    :return:
    """
    scores_dic={}
    for p1 in x_train:
        rec_papers=[]
        #find all the Neighbors of p1
        for Neighbo in dic_nighb[p1]:
            #all the papers thar can be recommanded
            for paper_neg_red in x_train[Neighbo]:
                if paper_neg_red not in x_train[p1]:
                    rec_papers.append(paper_neg_red)
        set_list_papers=set(rec_papers)
        for paper_j in set_list_papers:
            if p1 !=paper_j:
                score=get_Score(sim_dic,dic_nighb, x_train, p1,paper_j)
                writer.writerow({'paper_i': p1, 'paper_j': paper_j, 'score': str(score)})
                scores_dic[p1]=(paper_j, score)
    return scores_dic

def calc_claen_dic(path_read,path_write):
    """
    write clean data to file
    :param path_read:
    :param path_write:
    :return:
    """
    #create citation dic from citation releshinship file
    papers_dic=load_data_csv(path_read)
    clean_dic=removeWeakPapers(papers_dic)
    with open(path_write, 'w') as csvfile:
        fieldnames = ['paper_i', 'paper_j']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key,value in clean_dic.items():
            for item in value:
                writer.writerow({'paper_i': key, 'paper_j': item})


def read_file_scores(path):
    """
    get from file papers scores dic and sort it
    :param path:
    :return: sorted_doc,papers_similarity_dic
    """
    papers_scores = {}
    sorted_doc={}
    papers_similarity_dic={}
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            if (row[0]=="paper_i" or row[0]==""):
                continue
            p1 = row[0]
            p2=row[1]
            score=float(row[2])
            if p1 not in papers_scores:
                papers_scores[p1]=[]
                papers_similarity_dic[p1]=[]
            tup=(p2,score)
            papers_scores[p1].append(tup)
            papers_similarity_dic[p1].append(p2)
    #sort the dic
    for key,vaule in papers_scores.items():
        sorted_doc[key]=sorted(vaule, reverse=True, key=lambda s: s[1])
    return sorted_doc,papers_similarity_dic

def count_hit_func(recommanded_list, true_list):
    """

    :param recommanded_list:
    :param true_list:
    :return: count_hit,first- first place of the relevant document for mrr
    """
    count_hit=0.0
    first=0
    counter=1
    for item in recommanded_list:
        if item in true_list:
            count_hit+=1
            if first==0:
                first=counter
        else:
            counter+=1
    return count_hit,first

def precision(recommanded_list, count_hit):
    """
    calc_precision
    :param recommanded_list:
    :param count_hit:
    :return:
    """
    n=0.0
    n+=len(recommanded_list)
    if n==0:
        return 0
    return (count_hit/n)*100

def calc_mrr(first):
    """
    :param first: first- the palce of first relevant paper in recommandation list
    :return:
    """
    if first==0:
        return 0
    item=1.0
    return (item/first)*100


def recall(p_id, recommanded_list, true_list, count_hit ):
    """
    :param p_id:
    :param recommanded_list:
    :param true_list:
    :param count_hit:
    :return:
    """
    return (count_hit/len(true_list))*100


def f1(precision_score, recall_score):
    """
    :param precision_score:
    :param recall_score:
    :return:
    """
    if precision_score==0 and recall_score==0:
        return 0
    return (2*precision_score*recall_score)/(precision_score+recall_score)


def fumc_get_rec_list(key, N,sort_dic,dic_train):
    """
    get the recommandation list according to the artical algorithm
    :param key:
    :param N:
    :param sort_dic:
    :param dic_train:
    :return:
    """
    top=[]
    all_scores={}
    if key not in sort_dic:
        return top, all_scores
    tup_list=sort_dic[key]
    i=0
    for item in tup_list:
        if item[0] not in dic_train[key]:
            all_scores[item[0]]=item[1]
            if i<N:
                top.append(item[0])
                i+=1
    return top,all_scores


def Calc_Map(recommanded_list, true_list):
    """
    Calc_Map metric
    :param recommanded_list:
    :param true_list:
    :return:
    """
    count_hit = 0.0
    counter = 1
    sum_precision=0.0
    for item in recommanded_list:
        if item in true_list:
            count_hit += 1
            sum_precision+=(count_hit/counter)
        counter += 1
    if counter==1:
        return 0
    return (sum_precision / (counter - 1))*100


def plot_article_improve(dic_train, dic_test, N,sort_dic,pop_dic,calac_cos_dic):
    """
    plot_article_improve
    :param dic_train:
    :param dic_test:
    :param N:
    :param sort_dic:
    :param pop_dic:
    :param calac_cos_dic:
    :return:
    """
    N_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    recall_list__improve = []
    precision_list_improve = []
    f1_list__improve = []
    mrr_list=[]
    mrr_list_improve=[]
    map_list=[]
    map_list_improve=[]

    for i in range(1, N):
        calc_metrics(N_list, calac_cos_dic, dic_test, dic_train, f1_list, f1_list__improve, i, map_list,
                     map_list_improve, mrr_list, mrr_list_improve, pop_dic, precision_list, precision_list_improve,
                     recall_list, recall_list__improve, sort_dic)

    plt.subplot(3, 3, 1)
    plt.title('Precision@N')
    plt.xlabel('N')
    plt.ylabel('Precision(%)')
    plt.plot(N_list, precision_list,label="Article_BaseLine")
    plt.plot(N_list, precision_list_improve,label="Improvement")

    avrge_precision_reg=sum(precision_list) / float(N)
    avrge_precision_improve=sum(precision_list_improve)/float(N)
    print("avrge_precision_reg",avrge_precision_reg)
    print("avrge_precision_improve",avrge_precision_improve)
    print("precision_improvement(%)", (avrge_precision_improve - avrge_precision_reg) / avrge_precision_reg * 100 )
    plt.legend(loc=1, prop={'size': 6})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 3)
    plt.xlabel('N')
    plt.ylabel('Recall(%)')
    plt.title("Recall@N")

    plt.plot(N_list, recall_list,label="Article_BaseLine")
    plt.plot(N_list, recall_list__improve, label="Improvement")
    avrge_recall_reg=sum(recall_list)/float(N)
    avrge_recall_improve=sum(recall_list__improve)/float(N)
    print("avrge_recall_reg",avrge_recall_reg)
    print("avrge_recall_improve",avrge_recall_improve)
    print("recall_improvement(%)", (avrge_recall_improve - avrge_recall_reg) / avrge_recall_reg * 100 )

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(loc=4, prop={'size': 6})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 5)
    plt.xlabel('N')
    plt.ylabel('F1(%)')
    plt.title("F1@N")
    plt.plot(N_list, f1_list,label="Article_BaseLine")
    plt.plot(N_list, f1_list__improve,label="Improvement")

    avrge_f1_reg=sum(f1_list)/float(N)
    avrge_f1_improve=sum(f1_list__improve)/float(N)

    print("avrge_f1_reg",avrge_f1_reg)
    print("avrge_f1_improve",avrge_f1_improve)
    print("f1_improvement(%)", (avrge_f1_improve - avrge_f1_reg) / avrge_f1_reg * 100 )

    plt.legend(loc=4, prop={'size': 6})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 7)
    plt.plot(N_list, mrr_list,label="Article_BaseLine")
    plt.plot(N_list, mrr_list_improve,label="Improvement")
    avrge_mrr_reg=sum(mrr_list)/float(N)
    avrge_mrr_improve=sum(mrr_list_improve)/float(N)
    print("avrge_mrr_reg",avrge_mrr_reg)
    print("avrge_mrr_improve",avrge_mrr_improve)
    print("mrr_improvement(%)", (avrge_mrr_improve - avrge_mrr_reg) / avrge_mrr_reg * 100 )

    plt.xlabel('N')
    plt.ylabel('Mrr(%)')
    plt.title("Mrr@N")
    plt.legend(loc=4, prop={'size': 6})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 9)
    plt.plot(N_list, map_list,label="Article_BaseLine")
    plt.plot(N_list, map_list_improve,label="Improvement")
    avrge_map_reg=sum(map_list)/float(N)
    avrge_map_improve=sum(map_list_improve)/float(N)
    print("avrge_map_reg",avrge_map_reg)
    print("avrge_map_improve",avrge_map_improve)
    print("map_improvement(%)", (avrge_map_improve - avrge_map_reg) / avrge_map_reg * 100 )

    plt.xlabel('N')
    plt.ylabel('Map(%)')
    plt.title("Map@N")
    plt.legend(loc=1, prop={'size': 6})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.show()


def calc_metrics(N_list, calac_cos_dic, dic_test, dic_train, f1_list, f1_list__improve, i, map_list, map_list_improve,
                 mrr_list, mrr_list_improve, pop_dic, precision_list, precision_list_improve, recall_list,
                 recall_list__improve, sort_dic, artical=True):
    print(i)
    N_list.append(i)
    sum_precision = 0.0
    sum_precision_improve = 0.0
    sum_recall = 0.0
    sum_recall_improve = 0.0
    sum_f1 = 0.0
    sum_f1_improve = 0.0
    sum_mrr = 0.0
    sum_mrr_improve = 0.0
    sum_map = 0.0
    sum_map_improve = 0.0
    counter = 0
    # count all the precision for each paper with N number
    for key in sort_dic:
        recommanded_list, all_scores = fumc_get_rec_list(key, i, sort_dic, dic_train)
        count_hit, first = count_hit_func(recommanded_list, dic_test[key])
        # recommandation list according actical
        if artical:
            precision_score = precision(recommanded_list, count_hit)
            sum_precision += precision_score
            recall_score = recall(key, recommanded_list, dic_test[key], count_hit)

            sum_recall += recall_score
            mrr_score = calc_mrr(first)
            sum_f1 += f1(precision_score, recall_score)
            map_score = Calc_Map(recommanded_list, dic_test[key])
            sum_mrr += mrr_score
            sum_map += map_score

        # recommandation list with improve
        rec_list_impove = sum_wight(key, all_scores, pop_dic, i, calac_cos_dic)
        map_score_improve = Calc_Map(rec_list_impove, dic_test[key])
        count_hit_improve, first_improve = count_hit_func(rec_list_impove, dic_test[key])
        precision_score_improve = precision(rec_list_impove, count_hit_improve)
        mrr_score_improve = calc_mrr(first_improve)
        sum_precision_improve += precision_score_improve
        recall_score_improve = recall(key, rec_list_impove, dic_test[key], count_hit_improve)
        sum_recall_improve += recall_score_improve
        sum_f1_improve += f1(precision_score_improve, recall_score_improve)
        sum_mrr_improve += mrr_score_improve
        sum_map_improve += map_score_improve
        counter += 1
    if artical:
        precision_list.append(sum_precision / counter)
        recall_list.append(sum_recall / counter)
        f1_list.append(sum_f1 / counter)
        mrr_list.append(sum_mrr / counter)
        map_list.append(sum_map / counter)

    precision_list_improve.append(sum_precision_improve / counter)
    recall_list__improve.append(sum_recall_improve / counter)
    f1_list__improve.append(sum_f1_improve / counter)
    mrr_list_improve.append(sum_mrr_improve / counter)
    map_list_improve.append(sum_map_improve / counter)


def different_ts(N,dic_train,dic_test, pop_dic):
    """
    plot different threashold
    :param N:
    :param dic_train:
    :param dic_test:
    :param pop_dic:
    :return:
    """
    ts=[0.1,0.3,0.5]
    for t in ts:
        if(t==0.1):
            path="score_0.1.csv"
            path_tfIdf = 'calc_cosSim_neg_rec_0.1.csv'
        if (t==0.3):
            path = "score_0.3.csv"
            path_tfIdf = 'calc_cosSim_neg_rec_0.3.csv'
        if (t==0.5):
            path = "score_0.5.csv"
            path_tfIdf = 'calc_cosSim_neg_rec_0.5.csv'
        sort_dic, papers_similarity_dic = read_file_scores(path)
        calac_cos_dic = load_tdidf_sim_dic(path_tfIdf)
        N_list = []
        recall_list = []
        precision_list = []
        f1_list = []
        recall_list__improve = []
        precision_list_improve = []
        f1_list__improve = []
        mrr_list = []
        mrr_list_improve = []
        map_list = []
        map_list_improve = []

        for i in range(1, N):
            calc_metrics(N_list, calac_cos_dic, dic_test, dic_train, f1_list, f1_list__improve, i, map_list,
                         map_list_improve, mrr_list, mrr_list_improve, pop_dic, precision_list, precision_list_improve,
                         recall_list, recall_list__improve, sort_dic)
        plt.subplot(3, 3, 1)
        label_Article="Article_BaseLine-"+str(t)
        label_improve="Improvement-"+str(t)
        if (t==0.1):
            color1 = 'b'
            color2 = 'r'
        if (t==0.3):
            color1 = 'm'
            color2 = 'k'
        if (t==0.5):
            color1 = 'y'
            color2 = 'g'
        marker_simble_reg = None
        marker_simble_improve=None
        plt.plot(N_list, precision_list, label=label_Article,marker=marker_simble_reg,c=color1)
        plt.plot(N_list, precision_list_improve, label=label_improve,marker=marker_simble_improve,c=color2)
        plt.subplot(3, 3, 3)
        plt.plot(N_list, recall_list, label=label_Article,marker=marker_simble_reg,c=color1)
        plt.plot(N_list, recall_list__improve, label=label_improve,marker=marker_simble_improve,c=color2)
        plt.subplot(3, 3, 5)
        plt.plot(N_list, f1_list, label=label_Article,marker=marker_simble_reg,c=color1)
        plt.plot(N_list, f1_list__improve, label=label_improve,marker=marker_simble_improve,c=color2)
        plt.subplot(3, 3, 7)
        plt.plot(N_list, mrr_list, label=label_Article,marker=marker_simble_reg,c=color1)
        plt.plot(N_list, mrr_list_improve, label=label_improve,marker=marker_simble_improve,c=color2)
        plt.subplot(3, 3, 9)
        plt.plot(N_list, map_list, label=label_Article,marker=marker_simble_reg,c=color1)
        plt.plot(N_list, map_list_improve, label=label_improve,marker=marker_simble_improve,c=color2)

    plt.subplot(3, 3, 1)
    plt.title('Precision@N')
    plt.xlabel('N')
    plt.ylabel('Precision(%)')

    plt.legend(loc=1, prop={'size': 5})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 3)
    plt.xlabel('N')
    plt.ylabel('Recall(%)')
    plt.title("Recall@N")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.legend(loc=4, prop={'size': 5})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 5)
    plt.xlabel('N')
    plt.ylabel('F1(%)')
    plt.title("F1@N")
    plt.legend(loc=4, prop={'size': 5})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 7)
    plt.xlabel('N')
    plt.ylabel('Mrr(%)')
    plt.title("Mrr@N")
    plt.legend(loc=4, prop={'size': 5})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 9)
    plt.xlabel('N')
    plt.ylabel('Map(%)')
    plt.title("Map@N")
    plt.legend(loc=1, prop={'size': 5})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.show()



def plot_different_improve(dic_train, dic_test, N,sort_dic,pop_dic,calac_cos_dic):
    """
    plot for different Weights of improve, different Weights for popularity and tfidf score
    :param dic_train:
    :param dic_test:
    :param N:
    :param sort_dic:
    :param pop_dic:
    :param calac_cos_dic:
    :return:
    """
    list1=[(0.3,0.2,0.5),(0.5,0.2,0.3),(0.2,0.5,0.3),(0.8,0.1,0.1),(0.1,0.8,0.1),(0.1,0.1,0.8),(0.35,0.35,0.3)]
    list3 = [ (0, 0.5, 0.5), (0.5,0,0.5),(0.5,0.5,0.0), (0.0,0.0,1.0), (0.0,1.0,0.0),(1.0,0.0,0.0)]
    list14 = [  (0.2, 0.4, 0.4),(0.2, 0.5, 0.3), (0, 0.2, 0.8),(0, 0.1, 0.9),(0.1, 0.45, 0.45),(0.1, 0.55, 0.35),]
    for item in list1:
        N_list = []
        recall_list__improve = []
        precision_list_improve = []
        f1_list__improve = []
        mrr_list_improve = []
        map_list_improve = []
        pop_beta=item[0]
        similarity_gama=item[1]
        alphe_score=item[2]
        for i in range(1, N):
            print(i)
            N_list.append(i)
            sum_precision_improve=0.0
            sum_recall_improve=0.0
            sum_f1_improve=0.0
            sum_mrr_improve=0.0
            sum_map_improve=0.0
            counter = 0
            # count all the precision for each paper with N number
            for key in sort_dic:
                recommanded_list,all_scores = fumc_get_rec_list(key, i,sort_dic, dic_train)

                rec_list_impove=sum_wight(key, all_scores, pop_dic,i,calac_cos_dic,pop_beta,similarity_gama,alphe_score)
                map_score_improve = Calc_Map(rec_list_impove, dic_test[key])
                count_hit_improve, first_improve = count_hit_func( rec_list_impove, dic_test[key])
                precision_score_improve=precision( rec_list_impove, count_hit_improve)
                mrr_score_improve=calc_mrr(first_improve)
                sum_precision_improve+=precision_score_improve
                recall_score_improve = recall(key, rec_list_impove, dic_test[key],count_hit_improve)
                sum_recall_improve+=recall_score_improve
                sum_f1_improve += f1(precision_score_improve, recall_score_improve)
                sum_mrr_improve+=mrr_score_improve
                sum_map_improve+=map_score_improve

                counter += 1
            avg_precision_improve = sum_precision_improve / counter
            avg_recall_improve = sum_recall_improve / counter
            avg_f1_improve = sum_f1_improve / counter
            avg_mrr_improve=sum_mrr_improve/counter
            avg_map_improve=sum_map_improve/counter
            precision_list_improve.append(avg_precision_improve)
            recall_list__improve.append(avg_recall_improve)

            f1_list__improve.append(avg_f1_improve)
            mrr_list_improve.append(avg_mrr_improve)

            map_list_improve.append(avg_map_improve)

        tup=(alphe_score,pop_beta,similarity_gama)
        label_item=str(tup)
        plt.subplot(3, 3, 1)
        plt.plot(N_list, precision_list_improve,label=label_item)
        plt.subplot(3, 3, 3)
        plt.plot(N_list, recall_list__improve, label=label_item)
        plt.subplot(3, 3, 5)
        plt.plot(N_list, f1_list__improve,label=label_item)
        plt.subplot(3, 3, 7)
        plt.plot(N_list, mrr_list_improve,label=label_item)
        plt.subplot(3, 3, 9)
        plt.plot(N_list, map_list_improve,label=label_item)

    plt.subplot(3, 3, 1)
    plt.title('Precision@N')
    plt.xlabel('N')
    plt.ylabel('Precision(%)')
    plt.legend(loc=1, prop={'size': 5})

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.subplot(3, 3, 3)
    plt.xlabel('N')
    plt.ylabel('Recall(%)')
    plt.title("Recall@N")
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    plt.legend(loc=4, prop={'size': 5})

    plt.subplot(3, 3, 5)
    plt.xlabel('N')
    plt.ylabel('F1(%)')
    plt.title("F1@N")
    plt.legend(loc=4, prop={'size':5})

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 7)
    plt.xlabel('N')
    plt.ylabel('Mrr(%)')
    plt.title("Mrr@N")
    plt.legend(loc=4, prop={'size': 5})

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplot(3, 3, 9)
    plt.xlabel('N')
    plt.ylabel('Map(%)')
    plt.title("Map@N")
    plt.legend(loc=1, prop={'size': 5})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.show()

def calac_artical_scores(path_save,ts):
    """
    calc the papers score according to artical
    :param path_save:
    :param ts:
    :return:
    """
    clean_path = "claen_data_file.csv"
    clean_dic = load_clean_data(clean_path)
    dic_train, dic_test = split_test_train(clean_dic)
    print(len(dic_train))
    number_papers=len(dic_train)
    print("number pap")
    print(number_papers)
    dic_nighb=cala_ccd(dic_train, ts,number_papers)
    print("complete calc nighbored")
    print("dic_nighb",dic_nighb)
    sim_dic=calac_sim(dic_nighb)
    print("comolete calac similarity")
    print("sim_dic",sim_dic)
    with open(path_save, 'w') as csvfile:
        fieldnames = ['paper_i', 'paper_j', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        scores_dic=calc_scores_dic(dic_train,sim_dic,dic_nighb)
    print("complete scores")
    print("scores_dic",scores_dic)
    print("Writing complete")

token_dict = {}
tf_dic={}
values = []
key_dics_tf = {}

def calc_tfidf_similarity(papers_similarity_dic, dic_train):
    """
    :param papers_similarity_dic:
    :param dic_train:
    :return:
    """
    tfs = run_tdidf_full()
    cosinose_dic = calac_cos_din_neighbored_papers(papers_similarity_dic, dic_train, tfs)

def create_clean_dic():
    """
    create clean dic and write it to file
    :return:
    """
    #create dic of papers without weak papers and save to file
    path_read = "hep-th-citations"
    path_write = 'claen_data_file.csv'
    #write to flle papers clean dic
    calc_claen_dic(path_read,path_write )

def main(clean_path, score_path, path_tfIdf,n):
    sort_dic, papers_similarity_dic = read_file_scores(score_path)
    clean_dic = load_clean_data(clean_path)
    dic_train, dic_test = split_test_train(clean_dic)
    tfIdf_dic = load_tdidf_sim_dic(path_tfIdf)
    pop_dic = calc_popularity(clean_dic)
    #show the improvemnt of our algorithm
    plot_article_improve(dic_train, dic_test, n, sort_dic, pop_dic, tfIdf_dic)
    #plot with different Weights to the improvment params
    plot_different_improve(dic_train, dic_test,n,sort_dic,pop_dic,tfIdf_dic)
    #plot the check different threashold
    different_ts(n,dic_train, dic_test,pop_dic)
    #calc tfidf similarity and save to file
    #calc_tfidf_similarity(papers_similarity_dic, dic_train)
    #create_clean_dic()

score_path="score_0.3.csv"
clean_path="claen_data_file.csv"
path_tfIdf = 'calc_cosSim_neg_rec_0.3.csv'
N=20
main(clean_path, score_path, path_tfIdf,N)