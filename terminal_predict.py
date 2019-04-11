# encoding=utf-8

"""
基于命令行的在线预测方法
@Author: Macan (ma_cancan@163.com) 
"""
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import datetime

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser

args = get_args_parser()

model_dir = './ner_model'
bert_dir = './chinese_L-12_H-768_A-12'
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
is_training = False
use_one_hot_embeddings = False
batch_size = 1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
model = None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None

print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    # sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(
    vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)

import re


def predict():
    with open("./data/test.txt", encoding="utf-8") as f:
        lines = f.readlines()
    words = []
    lables = []
    with open("./data/predict.txt", mode="w", encoding="utf-8") as f:
        for line in lines:
            sen = line.strip()
            if len(sen) == 0:
                token, pred_label_result = predict_sen("".join(words))
                if len(token) == len(pred_label_result[0]):
                    real_token = []
                    for item in token:
                        if item.startswith("##"):
                            real_token.append(item[2:])
                        else:
                            real_token.append(item)
                    entities = result_to_json(real_token, pred_label_result[0])
                    predict_labels = []
                    offset = 0
                    for entity in entities:
                        for i in range(offset, entity.start):
                            predict_labels.append("O")
                        if entity.end - entity.start == 1:
                            predict_labels.append("S-" + entity.name)
                        else:
                            predict_labels.append("B-" + entity.name)
                            for i in range(entity.end - entity.start - 2):
                                predict_labels.append("I-" + entity.name)
                            predict_labels.append("E-" + entity.name)
                        offset = entity.end
                    for i in range(offset, len("".join(words))):
                        predict_labels.append("O")
                    for word, label, predict_label in zip(words, lables, predict_labels):
                        f.write(word + " " + label + " " + predict_label + "\n")
                    f.write("\n")
                else:
                    print("".join(words))
                words.clear()
                lables.clear()
                continue
            split = re.split("\\s", sen)
            if len(split) > 1:
                words.append(split[0])
                lables.append(split[1])


def predict_sen(sentence):
    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, args.max_seq_length))
        label_ids = np.reshape([feature.label_ids], (batch_size, args.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        # start = datetime.now()
        # sentence = tokenizer.tokenize(sentence)
        sentence = list(sentence)
        input_ids, input_mask, segment_ids, label_ids = convert(sentence)
        # print(input_ids)
        feed_dict = {input_ids_p: input_ids,
                     input_mask_p: input_mask}
        # run session get current feed_dict result
        pred_ids_result = sess.run([pred_ids], feed_dict)
        pred_label_result = convert_id_to_label(pred_ids_result, id2label)
        return sentence, pred_label_result


def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """

    print(id2label)
    while True:
        print('input the test sentence:')
        sentence = str(input())
        token, pred_label_result = predict_sen(sentence)
        if len(token) != len(pred_label_result[0]):
            print("Error: input token size does not match with label size,tokenSize:" + str(len(token)) +"\tlabelSize:"+ str(len(pred_label_result[0])))
            continue
        for i in range(len(token)):
            print(token[i] + "\t" + pred_label_result[0][i])


def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """

    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)
        print(', '.join(line))

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)
    print_output(loc, 'LOC')
    print_output(person, 'PER')
    print_output(org, 'ORG')


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def merge(self):
        return self.__merge

    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types

    @word.setter
    def word(self, word):
        self.__word = word

    @start.setter
    def start(self, start):
        self.__start = start

    @end.setter
    def end(self, end):
        self.__end = end

    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []

    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx + 1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append(
                        {"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append(
                        {"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


class Entity:
    def __init__(self, name, start, end, value):
        self.name = name
        self.start = start
        self.end = end
        self.value = value


def result_to_json(token, tags):
    entities = []
    previousTag = "O"
    offset = 0
    for i in range(len(tags)):
        tag = tags[i].upper()
        currentTag = tags[i].split("-")[-1]
        if previousTag != "O":
            value = "".join(token[offset:i])
            charOffset = len("".join(token[:offset]))
            if (previousTag != currentTag or tag.startswith("B") or tag.startswith("S")):
                entities.append(Entity(previousTag, charOffset, charOffset + len(value), value))
            elif i == len(tags) - 1:
                value = "".join(token[offset:len(tags)])
                charOffset = len("".join(token[:offset]))
                entities.append(Entity(previousTag, charOffset, charOffset + len(value), value))
        if tag.startswith("B") or tag.startswith("S") or (not currentTag.startswith("O") and currentTag != previousTag):
            offset = i
        previousTag = currentTag
    return entities


if __name__ == "__main__":
    predict_online()
