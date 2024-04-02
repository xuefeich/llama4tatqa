'''
vi /home/alan/.conda/envs/alan/lib/python3.10/site-packages/torch/utils/data/dataloader.py
633行改成  data = self._next_data() 并删除index

vi /home/alan/.conda/envs/alan/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py
308行改成 data=index

'''



import os
import pickle
import torch
import random
import numpy as np
from torch.utils.data import Dataset,BatchSampler,dataloader,RandomSampler

class CustomBatchSampler(BatchSampler):
    def get_num_ops(self,num_ops):
        self.num_ops = num_ops
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(self.sampler.data_source[idx])
            if len(batch) == self.batch_size:
                bsz = self.batch_size
                input_ids = torch.LongTensor(bsz, 2048)
                answer_ids = torch.zeros([bsz, 2048],dtype=torch.long)
                attention_mask = torch.LongTensor(bsz, 2048)
                token_type_ids = torch.LongTensor(bsz, 2048).fill_(0)
                paragraph_mask = torch.LongTensor(bsz, 2048)
                table_mask = torch.LongTensor(bsz, 2048)
                question_mask = torch.LongTensor(bsz, 2048)
                paragraph_index = torch.LongTensor(bsz, 2048)
                table_cell_index = torch.LongTensor(bsz, 2048)
                tag_labels = torch.LongTensor(bsz, 2048)
                operator_labels = torch.LongTensor(bsz)
                scale_labels = torch.LongTensor(bsz)
                ari_labels = torch.LongTensor([])
                selected_indexes = np.zeros([1, 21])
                opt_mask = torch.LongTensor(bsz)
                ari_ops = torch.LongTensor(bsz, self.num_ops)
                opt_labels = torch.LongTensor(bsz, self.num_ops - 1, self.num_ops - 1)
                order_labels = torch.LongTensor(bsz, self.num_ops)
                paragraph_tokens = []
                table_cell_tokens = []
                gold_answers = []
                question_ids = []
                paragraph_numbers = []
                table_cell_numbers = []
                for i in range(bsz):
                    input_ids[i] = batch[i][0]
                    answer_ids[i][:len(batch[i][23])] = batch[i][23]
                    attention_mask[i] = batch[i][1]
                    token_type_ids[i] = batch[i][2]
                    paragraph_mask[i] = batch[i][3]
                    table_mask[i] = batch[i][4]
                    paragraph_index[i] = batch[i][5]
                    opt_mask[i] = batch[i][19]
                    question_mask[i] = batch[i][22]
                    table_cell_index[i] = batch[i][6]
                    tag_labels[i] = batch[i][7]
                    operator_labels[i] = batch[i][8]
                    ari_ops[i] = torch.LongTensor(batch[i][16])
                    if len(batch[i][21]) != 0:
                        ari_labels = torch.cat((ari_labels, batch[i][18]), dim=0)
                        num = batch[i][21].shape[0]
                        sib = np.zeros([num, 21])
                        for j in range(num):
                            sib[j, 0] = i
                            try:
                                sib[j, 1:] = batch[i][21][j]
                            except:
                                print(batch[i][21][j])
                                sib[j, 1:] = batch[i][21][j][:20]
                        selected_indexes = np.concatenate((selected_indexes, sib), axis=0)
                    order_labels[i] = batch[i][20]
                    opt_labels[i] = batch[i][17]
                    scale_labels[i] = batch[i][9]
                    paragraph_tokens.append(batch[i][11])
                    table_cell_tokens.append(batch[i][12])
                    paragraph_numbers.append(batch[i][13])
                    table_cell_numbers.append(batch[i][14])
                    gold_answers.append(batch[i][10])
                    question_ids.append(batch[i][15])
                out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                             "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index,
                             "tag_labels": tag_labels,
                             "operator_labels": operator_labels, "scale_labels": scale_labels,
                             "paragraph_tokens": paragraph_tokens,
                             "table_cell_tokens": table_cell_tokens, "paragraph_numbers": paragraph_numbers,
                             "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers,
                             "question_ids": question_ids,
                             "table_mask": table_mask, "table_cell_index": table_cell_index, "ari_ops": ari_ops,
                             "ari_labels": ari_labels, "opt_labels": opt_labels, "opt_mask": opt_mask,
                             "order_labels": order_labels,
                             "selected_indexes": selected_indexes[1:], "question_mask": question_mask,"answer_ids":answer_ids
                             }
                yield out_batch
                batch = []
        if len(batch) > 0:
            bsz = self.batch_size
            input_ids = torch.LongTensor(bsz, 2048)
            answer_ids = torch.zeros([bsz, 2048],dtype=torch.long)
            attention_mask = torch.LongTensor(bsz, 2048)
            token_type_ids = torch.LongTensor(bsz, 2048).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 2048)
            table_mask = torch.LongTensor(bsz, 2048)
            question_mask = torch.LongTensor(bsz, 2048)
            paragraph_index = torch.LongTensor(bsz, 2048)
            table_cell_index = torch.LongTensor(bsz, 2048)
            tag_labels = torch.LongTensor(bsz, 2048)
            operator_labels = torch.LongTensor(bsz)
            scale_labels = torch.LongTensor(bsz)
            ari_labels = torch.LongTensor([])
            selected_indexes = np.zeros([1, 21])
            opt_mask = torch.LongTensor(bsz)
            ari_ops = torch.LongTensor(bsz, self.num_ops)
            opt_labels = torch.LongTensor(bsz, self.num_ops - 1, self.num_ops - 1)
            order_labels = torch.LongTensor(bsz, self.num_ops)
            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []
            for i in range(bsz):
                input_ids[i] = batch[i][0]
                answer_ids[i][:len(batch[i][23])] = batch[i][23]
                attention_mask[i] = batch[i][1]
                token_type_ids[i] = batch[i][2]
                paragraph_mask[i] = batch[i][3]
                table_mask[i] = batch[i][4]
                paragraph_index[i] = batch[i][5]
                opt_mask[i] = batch[i][19]
                question_mask[i] = batch[i][22]
                table_cell_index[i] = batch[i][6]
                tag_labels[i] = batch[i][7]
                operator_labels[i] = batch[i][8]
                ari_ops[i] = torch.LongTensor(batch[i][16])
                if len(batch[i][21]) != 0:
                    ari_labels = torch.cat((ari_labels, batch[i][18]), dim=0)
                    num = batch[i][21].shape[0]
                    sib = np.zeros([num, 21])
                    for j in range(num):
                        sib[j, 0] = i
                        try:
                            sib[j, 1:] = batch[i][21][j]
                        except:
                            print(batch[i][21][j])
                            sib[j, 1:] = batch[i][21][j][:20]
                    selected_indexes = np.concatenate((selected_indexes, sib), axis=0)
                order_labels[i] = batch[i][20]
                opt_labels[i] = batch[i][17]
                scale_labels[i] = batch[i][9]
                paragraph_tokens.append(batch[i][11])
                table_cell_tokens.append(batch[i][12])
                paragraph_numbers.append(batch[i][13])
                table_cell_numbers.append(batch[i][14])
                gold_answers.append(batch[i][10])
                question_ids.append(batch[i][15])

            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                         "operator_labels": operator_labels, "scale_labels": scale_labels,
                         "paragraph_tokens": paragraph_tokens,
                         "table_cell_tokens": table_cell_tokens, "paragraph_numbers": paragraph_numbers,
                         "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers,
                         "question_ids": question_ids,
                         "table_mask": table_mask, "table_cell_index": table_cell_index, "ari_ops": ari_ops,
                         "ari_labels": ari_labels, "opt_labels": opt_labels, "opt_mask": opt_mask,
                         "order_labels": order_labels,
                         "selected_indexes": selected_indexes[1:], "question_mask": question_mask,"answer_ids":answer_ids
                         }
            yield out_batch

class custom_data(Dataset):
    def __init__(self, dpath):
        with open(dpath, 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)
        all_data = []
        for item in data[:1000]:
            input_ids = torch.from_numpy(item["input_ids"])
            answer_ids = torch.from_numpy(item["answer_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            operator_labels = torch.tensor(item["operator_label"])
            scale_labels = torch.tensor(item["scale_label"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            opt_mask = item["opt_mask"]
            ari_ops = item["ari_ops"]
            opt_labels = item["opt_labels"]
            ari_labels = item["ari_labels"]
            selected_indexes = item["selected_indexes"]
            order_labels = item["order_labels"]
            question_mask = torch.from_numpy(item["question_mask"])
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                             table_cell_index, tag_labels, operator_labels, scale_labels, gold_answers,
                             paragraph_tokens, table_cell_tokens, paragraph_numbers, table_cell_numbers,
                             question_id, ari_ops, opt_labels, ari_labels, opt_mask, order_labels, selected_indexes,
                             question_mask,answer_ids
                             ))
        print("Load data size {}.".format(len(all_data)))
        self.data = all_data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    


class EvalBatchSampler(BatchSampler):
    def get_num_ops(self,num_ops):
        self.num_ops = num_ops
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(self.sampler.data_source[idx])
            if len(batch) == self.batch_size:
                bsz = self.batch_size
                input_ids = torch.LongTensor(bsz, 2048)
                attention_mask = torch.LongTensor(bsz, 2048)
                token_type_ids = torch.LongTensor(bsz, 2048).fill_(0)
                paragraph_mask = torch.LongTensor(bsz, 2048)
                table_mask = torch.LongTensor(bsz, 2048)
                paragraph_index = torch.LongTensor(bsz, 2048)
                table_cell_index = torch.LongTensor(bsz, 2048)
                tag_labels = torch.LongTensor(bsz, 2048)
                question_mask = torch.LongTensor(bsz, 2048)
                opt_mask = torch.LongTensor(bsz)
                paragraph_tokens = []
                table_cell_tokens = []
                gold_answers = []
                question_ids = []
                paragraph_numbers = []
                table_cell_numbers = []
                derivation = []
                for i in range(bsz):
                    input_ids[i] = batch[i][0]
                    attention_mask[i] = batch[i][1]
                    token_type_ids[i] = batch[i][2]
                    paragraph_mask[i] = batch[i][3]
                    table_mask[i] = batch[i][4]
                    paragraph_index[i] = batch[i][5]
                    opt_mask[i] = batch[i][14]
                    question_mask[i] = batch[i][16]
                    table_cell_index[i] = batch[i][6]
                    tag_labels[i] = batch[i][7]
                    paragraph_tokens.append(batch[i][9])
                    table_cell_tokens.append(batch[i][10])
                    paragraph_numbers.append(batch[i][11])
                    table_cell_numbers.append(batch[i][12])
                    gold_answers.append(batch[i][8])
                    question_ids.append(batch[i][13])
                    derivation.append(batch[i][15])
                out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                         "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
                         "paragraph_numbers": paragraph_numbers,
                         "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers, "question_ids": question_ids,
                         "table_mask": table_mask, "table_cell_index": table_cell_index,"opt_mask":opt_mask,"derivation":derivation,"question_mask":question_mask
                         }
                yield out_batch
                batch = []
        if len(batch) > 0:
            bsz = self.batch_size
            input_ids = torch.LongTensor(bsz, 2048)
            attention_mask = torch.LongTensor(bsz, 2048)
            token_type_ids = torch.LongTensor(bsz, 2048).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 2048)
            table_mask = torch.LongTensor(bsz, 2048)
            paragraph_index = torch.LongTensor(bsz, 2048)
            table_cell_index = torch.LongTensor(bsz, 2048)
            tag_labels = torch.LongTensor(bsz, 2048)
            question_mask = torch.LongTensor(bsz, 2048)
            opt_mask = torch.LongTensor(bsz)
            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []
            derivation = []
            for i in range(bsz):
                input_ids[i] = batch[i][0]
                attention_mask[i] = batch[i][1]
                token_type_ids[i] = batch[i][2]
                paragraph_mask[i] = batch[i][3]
                table_mask[i] = batch[i][4]
                paragraph_index[i] = batch[i][5]
                opt_mask[i] = batch[i][14]
                question_mask[i] = batch[i][16]
                table_cell_index[i] = batch[i][6]
                tag_labels[i] = batch[i][7]
                paragraph_tokens.append(batch[i][9])
                table_cell_tokens.append(batch[i][10])
                paragraph_numbers.append(batch[i][11])
                table_cell_numbers.append(batch[i][12])
                gold_answers.append(batch[i][8])
                question_ids.append(batch[i][13])
                derivation.append(batch[i][15])
            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                         "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
                         "paragraph_numbers": paragraph_numbers,
                         "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers,
                         "question_ids": question_ids,
                         "table_mask": table_mask, "table_cell_index": table_cell_index, "opt_mask": opt_mask,
                         "derivation": derivation, "question_mask": question_mask
                         }
            yield out_batch

class eval_data(Dataset):
    def __init__(self, dpath):
        with open(dpath, 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)
        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            derivation = item["derivation"]
            opt_mask = item["opt_mask"]
            question_mask = torch.from_numpy(item["question_mask"])
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                             table_cell_index, tag_labels, gold_answers, paragraph_tokens, table_cell_tokens,
                             paragraph_numbers, table_cell_numbers, question_id, opt_mask, derivation, question_mask))
        print("Load data size {}.".format(len(all_data)))
        self.data = all_data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
