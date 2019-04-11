# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/30 16:47
 @Author  : MaCan (ma_cancan@163.com)
 @File    : __init__.py.py
"""


def start_server():
    from bert_base.server import BertServer
    from bert_base.server.helper import get_run_args

    args = get_run_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_map)
    # print(args)
    server = BertServer(args)
    server.start()
    server.join()


def start_client():
    pass



def train_ner():
    import os
    from bert_base.train.train_helper import get_args_parser
    from bert_base.train.bert_lstm_ner import train

    args = get_args_parser()
    args.data_dir = "./data"
    args.output_dir = "./ner_model"
    # args.init_checkpoint = "./chinese_L-12_H-768_A-12/bert_model.ckpt"
    args.init_checkpoint = ""
    args.bert_config_file = "./chinese_L-12_H-768_A-12/bert_config.json"
    args.vocab_file = "./chinese_L-12_H-768_A-12/vocab.txt"
    args.num_train_epochs = 2
    args.max_seq_length = 64
    args.do_train = True
    args.do_eval = True
    args.batch_size = 1
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    train(args=args)

    # if __name__ == '__main__':
    #     # start_server()
    #     train_ner()
