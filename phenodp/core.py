import pickle
import pandas as pd
import numpy as np
from pyhpo import Ontology
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed

class PhenoDP:
    def __init__(self, pre_model, hp2d_sim_dict, node_embedding, PCL_HPOEncoder=None):
        if PCL_HPOEncoder is None:
            print("PCL_HPOEncoder is None")
        elif isinstance(PCL_HPOEncoder, nn.Module):
            print("PCL_HPOEncoder is a pre-trained model")
        else:
            raise ValueError("PCL_HPOEncoder must be either None or a pre-trained model")
        self.PCL_HPOEncoder = PCL_HPOEncoder
        self.ontology = pre_model.ontology
        self.node_embedding = node_embedding
        self.hp2d_sim_dict = hp2d_sim_dict
        self.disease_ic_dict = pre_model.disease_ic_dict
        self.ic_type = pre_model.ic_type
        self.disease_list = pre_model.disease_list
        self.disease_dict_int = pre_model.disease_dict_int
        self.disease_dict_str = pre_model.disease_dict_str
        self.hp2disease_sim_dict = hp2d_sim_dict
        self.hp2disease_dict = dict()
        self.disease_len = len(pre_model.disease_list)
        self.sim_method = pre_model.sim_method
        self.hp_weight_dict = pre_model.hp_weight_dict
        self.family_key_list = []
        self.hpo_list = pre_model.hpo_list
        for ahp in self.ontology.get_hpo_object(118).children:
            self.family_key_list.append(ahp.id)
        for hp in self.hpo_list:
            self.hp2disease_dict[hp] = self.get_disease(hp)
        related_list = []
        for j in self.disease_dict_str.keys():
            related_list.extend(self.disease_dict_str[j])
        related_list = list(set(related_list))
        self.related_len = len(related_list)
        self.hp_weight_sum = np.sum([self.get_hpo_weight(t) for t in related_list])
        self.ic_sum = np.sum([self.get_hpo_ic(t) for t in related_list])
        self.node_related_hpos = list(self.node_embedding.keys())

    def get_ancester(self, hpo_term):
        obj = self.ontology.get_hpo_object(hpo_term)
        parents_list = list(obj.all_parents)
        list_an = []
        for terms in parents_list:
            if terms.id != 'HP:0000001' and terms.id != 'HP:0000118':
                list_an.append(terms.id)
        return list_an

    def get_ancester_from_hps(self, hps):
        ancesters = []
        for hp in hps:
            ancesters.extend(self.get_ancester(hp))
        return np.unique(ancesters)

    def get_hpo_ic(self, hp):
        return self.ontology.get_hpo_object(hp).information_content[self.ic_type]

    def get_sim_hpopair(self, hp1, hp2):
        return self.hp2hp_sim_dict[hp1][hp2]

    def get_sim_hpo2set(self, hp1, hps):
        if hp1 in hps:
            return 1
        sim_list = []
        for hp2 in hps:
            sim_list.append(self.get_sim_hpopair(hp1, hp2))
        return np.max(sim_list)

    def is_sister_terms(self, hp, disease):
        d_hps = list(self.disease_dict_str[disease])
        if hp in d_hps:
            return 2
        for d_hp in d_hps:
            obj = self.ontology.get_hpo_object(d_hp)
            for d_hp_p in obj.parents:
                sister_terms = [t.id for t in d_hp_p.children]
                if hp in sister_terms:
                    return 1
        return 0

    def get_hpo_weight(self, hp):
        return self.hp_weight_dict[hp]
        # return self.get_hpo_ic(hp)

    def get_set_sim_list(self, hps1, hps2):
        sim_list = []
        for hp1 in hps1:
            sim_list.append(self.get_sim_hpo2set(hp1, hps2))
        return sim_list

    def get_set_scores(self, hps1, hps2):
        weight_list = []
        sim_list = np.array(self.get_set_sim_list(hps1, hps2))
        for hp in hps1:
            w = self.get_hpo_weight(hp)
            weight_list.append(w)
        weight_list = np.array(weight_list)
        norm = np.sum(weight_list)
        scores_list = weight_list * sim_list
        return np.sum(scores_list) / norm

    def get_set_scores_paral(self, hps1, d):
        weight_list = []
        sim_list = []
        for hp in hps1:
            try:
                sim_list.append(self.hp2disease_sim_dict[hp][str(d)])
            except:
                return 0
            w = self.get_hpo_weight(hp)
            weight_list.append(w)
        weight_list = np.array(weight_list)
        norm = np.sum(weight_list)
        scores_list = weight_list * sim_list
        return np.sum(scores_list) / norm

    def get_all_related_diseases(self, hps1):
        disease_list = []
        for hp in hps1:
            disease_list.extend(self.hp2disease_dict[hp])
        return np.unique(disease_list)

    def get_disease(self, hp):
        if self.ic_type == 'omim':
            return [t.id for t in self.ontology.get_hpo_object(hp).omim_diseases]
        elif self.ic_type == 'orpha':
            return [t.id for t in self.ontology.get_hpo_object(hp).orpha_diseases]

    def get_related_diseases(self, hps):
        all_ids = []
        for hp in hps:
            ids = self.hp2disease_dict[hp]
            all_ids.extend(ids)
        id_counts = pd.Series(all_ids).value_counts()
        return id_counts.index

    def filter_hps(self, hps):
        hps_list = []
        for hp in hps:
            try:
                hps_list.append(self.ontology.get_hpo_object(hp).id)
            except Exception as e:
                continue
        return hps_list

    def first_round_rank_disease(self, hps1):
        d_list = []
        sim_list = []
        related_disease = self.get_related_diseases(hps1)
        for d in tqdm(related_disease, desc="Find Candidate Diseases"):
            hps2 = self.disease_dict_str[d]
            sim = self.get_set_scores(hps1, hps2)
            sim_list.append(sim)
            d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list})
        sorted_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df

    def first_rank_disease(self, hps1, diseases):
        d_list = []
        sim_list = []
        related_disease = diseases
        for d in tqdm(related_disease, desc="Find Candidate Diseases"):
            sim = self.get_set_scores_paral(hps1, d)
            sim_list.append(sim)
            d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list})
        sorted_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df

    def get_reverse_scores(self, hps1, disease_set):
        d_list = []
        sim_list = []
        for d in tqdm(disease_set, desc="Calculating Reverse Scores"):
            hps2 = self.disease_dict_str[d]
            sim = self.get_set_scores(hps2, hps1)
            sim_list.append(sim)
            d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list})
        sorted_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df

    def get_ancester_rank_scores(self, hps1, disease_set):
        anc_hps1 = self.get_ancester_from_hps(hps1)
        d_list = []
        sim_list_inter = []
        sim_list_outer = []
        for d in tqdm(disease_set, desc="Calculating Ancester Scores and Reverse Scores"):
            hps2 = self.disease_dict_str[d]
            anc_hps2 = self.get_ancester_from_hps(hps2)
            anc_inter = np.intersect1d(anc_hps1, anc_hps2)
            total_sim_inter = 0
            total_sim_outer = 0
            sim = 0
            for hp in anc_hps1:
                total_sim_inter += self.get_hpo_weight(hp)

            for hp in anc_hps2:
                total_sim_outer += self.get_hpo_weight(hp)

            for hp in anc_inter:
                sim += self.get_hpo_weight(hp)

            sim_list_inter.append(sim / total_sim_inter)
            sim_list_outer.append(sim / total_sim_outer)
            d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list_inter})
        sorted_df_inter = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list_outer})
        sorted_df_outer = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df_inter, sorted_df_outer

    def get_embedding_vec(self, hps):
        vec = torch.zeros(256)
        count = 0
        for hp in hps:
            try:
                count += 1
                vec += self.node_embedding[hp]
            except:
                continue
        vec = vec / count
        return vec

    def get_precise_node(self, hps):
        precise_list = []
        for hp in hps:
            if len(self.ontology.get_hpo_object(hp).children) == 0:
                precise_list.append(hp)
        return precise_list

    def get_hps_vec(self, hps):
        v_list = []
        for hp in hps:
            try:
                v_list.append(self.node_embedding[hp])
            except Exception as e:
                continue
        v_list = np.vstack(v_list)
        return v_list

    def pad_or_truncate(self, vec, max_len=128):
        original_length = vec.size(0)

        if original_length > max_len:
            padded_vec = vec[:max_len]
        else:
            pad_size = max_len - original_length
            padded_vec = F.pad(vec, (0, 0, 0, pad_size))
        attention_mask = torch.zeros(max_len, dtype=torch.bool)
        attention_mask[:min(original_length, max_len)] = True
        return padded_vec, attention_mask

    def get_all_embedding_pre(self):
        with torch.no_grad():
            self.disease_encoder_outputs = {}
            all_vecs = []
            all_masks = []
            disease_to_index = {}

            # Collect all vectors and masks
            for idx, d in enumerate(tqdm(self.disease_list, desc="Processing Diseases")):
                vecs1 = []
                hps1 = self.get_precise_node(self.disease_dict_str[d])
                if len(hps1) == 0:
                    hps1 = self.disease_dict_str[d]
                vec1 = self.get_hps_vec(hps1)
                vec1 = torch.tensor(vec1)
                vecs1, mask1 = self.pad_or_truncate(vec1)
                mask1 = torch.tensor([1] + list(mask1))

                all_vecs.append(vecs1)
                all_masks.append(mask1)
                disease_to_index[d] = idx

            # Stack all vectors and masks
            all_vecs = torch.stack(all_vecs)  # Shape: [num_diseases, seq_len, input_dim]
            all_masks = torch.stack(all_masks)  # Shape: [num_diseases, seq_len + 1]
            cls_tokens, embeddings = self.model(all_vecs, all_masks)

            # Save the CLS embeddings
            for d, idx in disease_to_index.items():
                self.disease_encoder_outputs[d] = cls_tokens[idx].cpu()

    def get_vec_sim(self, vec1, vec2):
        return F.cosine_similarity(vec1, vec2, dim=0)

    def get_precise_node(self, hps):
        precise_list = []
        for hp in hps:
            if len(self.ontology.get_hpo_object(hp).children) == 0:
                precise_list.append(hp)
        return precise_list

    def get_embedding_scores(self, hps1, hps2):
        D_set = hps2
        if len(D_set) == 0:
            D_set = hps2
        hp1_list = []
        for hp1 in hps1:
            if hp1 in D_set:
                hp1_list.append(1)
            list_sim = []
            for hp2 in D_set:
                try:
                    vec1 = torch.tensor(self.node_embedding[hp1])
                    vec2 = torch.tensor(self.node_embedding[hp2])
                    sim = self.get_vec_sim(vec1, vec2)
                    list_sim.append(sim)
                except Exception as e:
                    list_sim.append(0)
            hp1_list.append(np.max(list_sim))
        return np.mean(hp1_list)

    def get_embedding_rank_scores3(self, hps, disease_set):
        d_list = []
        sim_list = []
        for d in tqdm(disease_set, desc="Calculating Embedding Similarity"):
            hps2 = self.disease_dict_str[d]
            hps3 = self.get_precise_node(hps2)
            if len(hps3) == 0:
                hps3 = hps2
            cos_sim = self.get_embedding_scores(hps, hps3)
            sim_list.append(cos_sim)
            d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list})
        sorted_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df

    def get_set_embedding_scorse(self, hps1, hps2):
        vec1 = self.get_embedding_vec(hps1)
        vec2 = self.get_embedding_vec(hps2)
        return F.cosine_similarity(vec1, vec2, dim=0)

    def get_embedding_rank_scores(self, hps, disease_set):
        d_list = []
        sim_list = []
        for d in tqdm(disease_set, desc="Calculating Embedding Similarity"):
            hps2 = self.disease_dict_str[d]
            cos_sim = self.get_set_embedding_scorse(hps, hps2).item()
            sim_list.append(cos_sim)
            d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list})
        sorted_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df

    def get_mt(self, hps, disease):
        given_hps = hps
        Q = self.get_ancester_from_hps(given_hps)
        D = self.get_ancester_from_hps(self.disease_dict_str[int(disease)])
        QD = np.intersect1d(Q, D)
        inter_ic = len(QD)

        QDn = np.setdiff1d(Q, D)
        Qonly_ic = len(QDn)

        DQn = np.setdiff1d(D, Q)
        Donly_ic = len(DQn)

        none_ic = self.related_len - inter_ic - Qonly_ic - Donly_ic
        contingency_table = np.array([[inter_ic, Qonly_ic], [Donly_ic, none_ic]])
        return contingency_table

    def get_phi_cor(self, hps, disease):
        mt = self.get_mt(hps, disease)
        a, b, c, d = mt[0, 0], mt[0, 1], mt[1, 0], mt[1, 1]
        phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        return phi

    def get_phi_ranks_scores(self, hps, disease_set, num_threads=4):
        d_list = []
        sim_list = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_disease = {executor.submit(self.get_phi_cor, hps, d): d for d in disease_set}
            for future in tqdm(as_completed(future_to_disease), total=len(disease_set), desc="Calculating Phi Scores"):
                d = future_to_disease[future]
                sim = future.result()
                sim_list.append(sim)
                d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list})
        sorted_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df

    def get_is_in_family(self, hp1, hp2):
        return self.hp2hp_family_dict[hp1][hp2]

    def get_family_key(self, hp):
        lin_list = []
        h_obj = self.ontology.get_hpo_object(hp)
        anc_list = [t.id for t in h_obj.all_parents]
        if 'HP:0000118' in anc_list:
            for a_hp in anc_list:
                if a_hp in self.family_key_list:
                    lin_list.append(a_hp)
        return lin_list

    def get_family_vec(self, hps):
        hp_family = dict()
        for hp in self.ontology.get_hpo_object(118).children:
            hp_family[hp.id] = 0
        for hp in hps:
            h_obj = self.ontology.get_hpo_object(hp)
            anc_list = [t.id for t in h_obj.all_parents]
            if 'HP:0000118' in anc_list:
                for a_hp in anc_list:
                    if a_hp in list(hp_family.keys()):
                        hp_family[a_hp] += 1
        sorted_values = [value for key, value in sorted(hp_family.items())]
        return sorted_values

    def get_family_messure(self, hps1, hps2):
        vec1 = self.get_family_vec(hps1)
        vec2 = self.get_family_vec(hps2)
        correlation = np.corrcoef(vec1, vec2)[0, 1]
        return correlation

    def get_family_messure_for_disease(self, hps1, disease):
        return self.get_family_messure(hps1, self.disease_dict_str[disease])

    def get_family_score(self, hps, disease_set):
        d_list = []
        sim_list = []
        for d in tqdm(disease_set, desc="Calculating Family Scores"):
            sim = self.get_family_messure_for_disease(hps, d)
            sim_list.append(sim)
            d_list.append(d)
        df = pd.DataFrame({'Disease': d_list, 'Similarity': sim_list})
        sorted_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        return sorted_df

    def get_mrr(self, arr):
        return np.mean(1 / (np.array(arr) + 1))

    def get_merge3(self, df, df1, df2):
        temp_df = df.sort_values(by='Disease').reset_index(drop=True)
        temp_df1 = df1.sort_values(by='Disease').reset_index(drop=True)
        temp_df2 = df2.sort_values(by='Disease').reset_index(drop=True)
        merged_df = pd.merge(temp_df, temp_df1, on='Disease', suffixes=('_df', '_df1'))
        merged_df = pd.merge(merged_df, temp_df2, on='Disease')
        merged_df.rename(columns={'Similarity': 'Similarity_df2'}, inplace=True)
        merged_df['Total_Similarity'] = (
                    merged_df['Similarity_df'] + merged_df['Similarity_df1'] + merged_df['Similarity_df2'])
        merged_df = merged_df.sort_values(by='Total_Similarity', ascending=False).reset_index(drop=True)
        return merged_df

    def get_merge2(self, df, df1):
        temp_df = df.sort_values(by='Disease').reset_index(drop=True)
        temp_df1 = df1.sort_values(by='Disease').reset_index(drop=True)
        merged_df = pd.merge(temp_df, temp_df1, on='Disease', suffixes=('_df', '_df1'))
        merged_df['Total_Similarity'] = (merged_df['Similarity_df'] + merged_df['Similarity_df1'])
        merged_df = merged_df.sort_values(by='Total_Similarity', ascending=False).reset_index(drop=True)
        return merged_df

    def run_Ranker(self, given_hps, top_n=200):
        candidate_disease = self.get_all_related_diseases(given_hps)
        IC_based = self.first_rank_disease(given_hps, candidate_disease)
        candidate_disease = IC_based.iloc[:top_n, 0].values
        Phi_based = self.get_phi_ranks_scores(given_hps, candidate_disease)
        Semantic_based = self.get_embedding_rank_scores3(given_hps, candidate_disease)
        res = self.get_merge3(IC_based, Phi_based, Semantic_based)
        res['Total_Similarity'] = res['Total_Similarity'] / 3
        res = res[['Disease', 'Total_Similarity']]
        return res

    def get_cv_values(self, res, num=3):
        data1 = res['Total_Similarity'][:num]
        mean_value = np.mean(data1)
        std_deviation = np.std(data1)
        coefficient_of_variation = (std_deviation / mean_value) * 100
        return coefficient_of_variation

    def info_nce_loss(self, emb0, emb1, emb2, temperature=0.07):
        def cosine_similarity(a, b):
            a = a.flatten()
            b = b.flatten()
            return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
        pos_similarity = cosine_similarity(emb0, emb1)
        neg_similarities = torch.tensor([cosine_similarity(emb0, neg) for neg in emb2])
        pos_similarity_exp = torch.exp(pos_similarity / temperature)
        neg_similarities_exp = torch.exp(neg_similarities / temperature)
        loss = -torch.log(pos_similarity_exp / (pos_similarity_exp + torch.sum(neg_similarities_exp)))
        return loss

    def get_PCLHPOEncoder_out(self, given_hps):
        hps = given_hps
        vec1 = self.get_hps_vec(hps)
        vec1 = torch.tensor(vec1)
        vec1, mask1 = self.pad_or_truncate(vec1)
        mask1 = torch.tensor([1] + list(mask1))
        item_len = len(torch.where(mask1 == 1)[0])
        vec1 = vec1.unsqueeze(0)
        mask1 = mask1.unsqueeze(0).float()
        self.PCL_HPOEncoder.eval()
        with torch.no_grad():
            cls, emb = self.PCL_HPOEncoder(vec1, mask1)
        temp = emb[1:item_len]
        return cls, temp.permute(1, 0, 2)[0]

    def run_Recommender(self, given_hps, target_disease, candidate_diseases, user_candidate_hps=None):
        if self.PCL_HPOEncoder is None:
            print("Please load the pre-trained model and pass it to the PhenoDP class using the PCL_HPOEncoder parameter.")
            return -1
        if user_candidate_hps is None:
            user_candidate_hps = []
            print('using default setting...')
        d_list = [target_disease]
        d_list.extend(list(candidate_diseases))
        tar_d = target_disease
        cls_list = []
        res_dlist = np.setdiff1d(d_list, tar_d)
        cls_tar, emb_tar = self.get_PCLHPOEncoder_out(self.disease_dict_str[tar_d])
        for d in res_dlist:
            cls1, emb1 = self.get_PCLHPOEncoder_out(self.disease_dict_str[d])
            cls_list.append(cls1[0])
        cls_list = torch.stack(cls_list)
        nce_loss = []
        added_hps = []
        can_hps = []
        if len(user_candidate_hps) != 0:
            print('using user define setting...')
            can_hps = user_candidate_hps
        else:
            for hp in self.disease_dict_str[tar_d]:
                count = 0
                for d in res_dlist:
                    if hp in self.disease_dict_str[d]:
                        count += 1
                    if count == 0:
                        can_hps.append(hp)
            can_hps = np.setdiff1d(can_hps, given_hps)
        for hp in tqdm(can_hps, desc="Calculating NCE Loss"):
            cls_add, emb_add = self.get_PCLHPOEncoder_out(given_hps + [hp])
            score = self.info_nce_loss(cls_add, cls_tar, cls_list).item()
            nce_loss.append(score)
            added_hps.append(hp)
        add_df = pd.DataFrame({'hp': added_hps, 'importance': 1 / np.array(nce_loss)})
        add_df = add_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        return add_df