import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import pickle

train_split = 0.8
COL = ['student_id', 'knowledgecomponent_id', 'answer_state', 'question_text', 'choice_text']

def preprocess(data_dir):

    seqs = pd.read_csv(f'{data_dir}/Transaction.csv')
    question_kc = pd.read_csv(f'{data_dir}/Question_KC_Relationships.csv')
    qs = pd.read_csv(f'{data_dir}/Questions.csv')
    ans = pd.read_csv(f'{data_dir}/Question_Choices.csv')
    q2kc = question_kc.groupby('question_id')['knowledgecomponent_id'].agg(lambda s: ','.join(s.sort_values().astype(str)))
    merged = seqs.merge(q2kc, on = 'question_id').merge(qs, left_on = 'question_id', right_on = 'id')
    merged = merged.merge(ans, left_on = ['question_id', 'answer_choice_id'], right_on = ['question_id', 'id'])
    merged = merged[COL + ['question_id']]
    merged.columns = ['user_id', 'skill_name', 'correct', 'qtxt', 'atxt', 'question_id']
    merged = merged.merge(ans[ans['is_correct']], on = 'question_id')
    merged['correct'] = merged['correct'].astype(int)
    merged = merged[['user_id', 'skill_name', 'correct', 'qtxt', 'atxt', 'choice_text']]
    merged.columns = ['user_id', 'skill_name', 'correct', 'qtxt', 'atxt', 'ans']
    data = merged

    def train_test_split(data, skill_list = None):
        np.random.seed(42)
        data = data.set_index(['user_id', 'skill_name'])
        idx = np.random.permutation(data.index.unique())
        train_idx, test_idx = idx[:int(train_split * len(idx))], idx[int(train_split * len(idx)):]
        data_train = data.loc[train_idx].reset_index()
        data_val = data.loc[test_idx].reset_index()
        return data_train, data_val

    if 'skill_name' not in data.columns:
        data.rename(columns={'skill_id': 'skill_name'}, inplace=True)
    if 'original' in data.columns:
        data = data[data['original'] == 1]

    data = data[~data['skill_name'].isna() & (data['skill_name'] != 'Special Null Skill')]

    data_train, data_val = train_test_split(data)
    print("Train-test split finished...")

    train_skills = data_train['skill_name'].unique()
    skill_dict = {sn: i for i, sn in enumerate(train_skills)}

    print("Imputing skills...")
    repl = skill_dict[data_train['skill_name'].value_counts().index[0]]
    for skill_name in set(data_val['skill_name'].unique()) - set(skill_dict):
        skill_dict[skill_name] = repl

    print("Replacing skills...")
    data_train['skill_id'] = data_train['skill_name'].apply(lambda s: skill_dict[s])
    data_val['skill_id'] = data_val['skill_name'].apply(lambda s: skill_dict[s])

    return data_train, data_val
