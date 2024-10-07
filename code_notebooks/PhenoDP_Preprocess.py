from pyhpo import Ontology
import pickle
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class PhenoDP_Initial:
    def __init__(self, Ontology, ic_type='omim', sim_method='jc'):
        self.disease_dict_int = dict()
        self.disease_dict_str = dict()
        self.ic_type = ic_type
        if self.ic_type == 'omim':
            self.disease_list = np.unique([t.id for t in Ontology.omim_diseases])
            print('generate disease dict...')
            for d in list(Ontology.omim_diseases):
                self.disease_dict_int[d.id] = list(d.hpo)
                self.disease_dict_str[d.id] = [Ontology.get_hpo_object(t).id for t in d.hpo]
                
        if self.ic_type == 'orpha':
            self.disease_list = np.unique([t.id for t in Ontology.orpha_diseases])
            print('generate disease dict...')
            for d in list(Ontology.orpha_diseases):
                self.disease_dict_int[d.id] = list(d.hpo)
                self.disease_dict_str[d.id] = [Ontology.get_hpo_object(t).id for t in d.hpo]
                
        self.hpo_list = list(Ontology.to_dataframe().index)
        self.ontology = Ontology
        related_list = []
        for j in self.disease_dict_str.keys():
            related_list.extend(self.disease_dict_int[j])
        self.hpo_len = len(np.unique(related_list))
        print('related hpo num:', self.hpo_len)
        self.disease_len = len(self.disease_list)
        self.sim_method = sim_method
        self.disease_ic_dict = dict()
        print('generate disease ic dict... ')
        self.get_disease_ic_dict()
        self.hpo2disease_scores = []
        self.hpo2normal_sim = []
        self.processed_hpos = []
        self.processed_hpos_sim = []
        self.hp2disease_dict = dict()
        for hp in self.hpo_list:
            self.hp2disease_dict[hp] = self.get_disease(hp)
        self.hp_weight_dict = dict()
        print('calculating hp weights')
        for hp in self.hpo_list:
            self.hp_weight_dict[hp] = self.get_hpo_weight_2(hp)

    def get_hpo_weight_2(self, hp):
        ic = self.get_hpo_ic(hp)
        ds = self.get_disease(hp)
        dic_list = []
        if len(ds) == 0:
            return ic
        for d in ds:
            dic_list.append(self.disease_ic_dict[d])
        dic = np.mean(dic_list)
        return ic + dic
    
    def get_disease(self,hp):
        if self.ic_type == 'omim':
            return [t.id for t in self.ontology.get_hpo_object(hp).omim_diseases]
        elif self.ic_type == 'orpha':
            return [t.id for t in self.ontology.get_hpo_object(hp).orpha_diseases]
    
        
    def get_hpo_ic(self, hp):
        return self.ontology.get_hpo_object(hp).information_content[self.ic_type]
    
    def get_disease_ic_values(self, disease):
        local_len = len(self.disease_dict_str[disease])
        total_len = self.hpo_len
        frequence = abs(local_len) / abs(total_len)
        disease_ic = - np.log(frequence)
        return disease_ic
    
    def get_disease_ic_dict(self):
        for i in self.disease_list:
            self.disease_ic_dict[i] = self.get_disease_ic_values(i)
    
    def get_disease_ic(self, disease):
        return self.disease_ic_dict[disease]
    
    def get_hpo_sim(self, hp1, hp2):
        o1 = self.ontology.get_hpo_object(hp1)
        o2 = self.ontology.get_hpo_object(hp2)
        sim = o1.similarity_score(o2, method=self.sim_method)
        return sim

    def get_hpo_weight(self, hp):
        disease_list = self.get_disease(hp) 
        disease_ic_list = []
        for i in disease_list:
            disease_ic_list.append(self.get_disease_ic(i))
        disease_ic = np.mean(disease_ic_list)
        hpo_ic = self.get_hpo_ic(hp)
        weight = disease_ic + hpo_ic
        return weight
    
    def get_hpo2disease_sim(self, hp, disease):
        D = self.disease_dict_str[disease]
        if hp in D:
            return 1
        else:
            sim_list = []
            for hp_d in D:
                sim_list.append(self.get_hpo_sim(hp, hp_d))
            return np.max(sim_list)
    
    def get_hpo2disease_score(self, hp, disease):
        D = self.disease_dict_str[disease]
        if hp in D:
            return 1 * self.get_hpo_weight(hp)
        else:
            sim = self.get_hpo2disease_sim(hp, disease)
            return sim * self.get_hpo_weight(hp)
    


    def initial(self):
        for count, i in enumerate(tqdm(self.hpo_list, desc="HPO Processing")):
            if i == 'HP:0000001' or i == 'HP:0000118':
                self.hpo2disease_scores.append([0 for t in range(self.disease_len)])
                continue
            hp2disease_scores = []
            for j in self.disease_list:
                hp2disease_scores.append(self.get_hpo2disease_score(i, j))
            self.hpo2disease_scores.append(hp2disease_scores)
            self.processed_hpos.append(i)
        print('end')
        return self.hpo2disease_scores, self.processed_hpos

    def initial_split(self, start, end):
        print('total hpo len:', len(self.hpo_list))
        for count, i in enumerate(tqdm(self.hpo_list[start:end], desc="HPO Processing")):
            if i == 'HP:0000001' or i == 'HP:0000118':
                self.hpo2disease_scores.append([0 for t in range(self.disease_len)])
                self.processed_hpos.append(i)
                continue
            hp2disease_scores = []
            for count2, j in enumerate(self.disease_list):
                hp2disease_scores.append(self.get_hpo2disease_score(i, j))
            self.hpo2disease_scores.append(hp2disease_scores)
            self.processed_hpos.append(i)
        print('end')
        return self.hpo2disease_scores, self.processed_hpos
    
    def get_normal_sim(self, hp, disease):
        D = self.disease_dict_str[disease]
        if hp in D:
            return 1 
        else:
            sim = self.get_hpo2disease_sim(hp, disease)
            return sim 
        
    def initial_sim(self, start, end):
        print('total hpo len:', len(self.hpo_list))
        for count, i in enumerate(tqdm(self.hpo_list[start:end], desc="HPO Processing")):
            if i == 'HP:0000001' or i == 'HP:0000118':
                self.hpo2normal_sim.append([0 for t in range(self.disease_len)])
                self.processed_hpos_sim.append(i)
                continue
            hp2disease_sim = []
            for count2, j in enumerate(self.disease_list):
                hp2disease_sim.append(self.get_normal_sim(i, j))
            self.hpo2normal_sim.append(hp2disease_sim)
            self.processed_hpos_sim.append(i)
        print('end')
        return self.hpo2normal_sim, self.processed_hpos_sim
    
    def initial_sim_singlecore(self):
        print('total hpo len:', len(self.hpo_list))
        for count, i in enumerate(tqdm(self.hpo_list, desc="HPO Processing")):
            if i == 'HP:0000001' or i == 'HP:0000118':
                self.hpo2normal_sim.append([0 for t in range(self.disease_len)])
                self.processed_hpos_sim.append(i)
                continue
            hp2disease_sim = []
            for count2, j in enumerate(self.disease_list):
                hp2disease_sim.append(self.get_normal_sim(i, j))
            self.hpo2normal_sim.append(hp2disease_sim)
            self.processed_hpos_sim.append(i)
        print('end')
        return self.hpo2normal_sim, self.processed_hpos_sim
    
