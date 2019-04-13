import argparse
import os
import threading
import time


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str)
    parser.add_argument("-output", type=str)
    parser.add_argument("-vocab", type=str)
    parser.add_argument("-thread", type=int, default=10)
    parser.add_argument("-sleep", type=float, default=5)
    return parser.parse_args()


def normalize(path):
    path = path.replace("\\", "/")
    if path.endswith("/"):
        path = path[:-1]
    return path


def preprocess_input_output(input_dir, output_dir):
    # args = get_args_parser().parse_args()
    input_dir = normalize(input_dir)
    output_dir = normalize(output_dir)
    files = [input_dir]
    isDir = False
    input_files = []
    output_files = []
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        isDir = True
    else:
        output_files.append(output_dir)
    while len(files) > 0:
        file = files[-1]
        if os.path.isdir(file):
            for root, sub_dirs, sub_files in os.walk(file):
                for sub_file in sub_dirs:
                    files.insert(0, normalize(root) + "/" + sub_file)
                for sub_file in sub_files:
                    files.insert(0, normalize(root) + "/" + sub_file)
        else:
            input_files.append(file)
            if isDir:
                output_file_name = output_dir + file[len(input_dir):]
                parent_path = os.path.dirname(output_file_name)
                if not os.path.exists(parent_path):
                    os.makedirs(parent_path)
                output_files.append(output_file_name)
        files.pop(-1)
    return input_files, output_files


def process():
    args = get_args_parser()
    input_files, output_files = preprocess_input_output(args.input, args.output)
    if len(input_files) == len(output_files):
        threads = []
        for input_file, output_file in zip(input_files, output_files):
            if len(threads) < args.thread:
                train_thread = TrainThread(input_file, output_file, args.vocab)
                train_thread.start()
                threads.append(train_thread)
            else:
                while True:
                    is_break = False
                    for i in range(len(threads) - 1, -1, -1):
                        threads.pop(i)
                        is_break = True
                    if is_break:
                        break
                    time.sleep(args.sleep)


    else:
        os.system("nohup python create_pretraining_data.py " +
                  " --input_file " + input_files +
                  " --output_file " + output_files +
                  " --vocab_file  " + args.vocab +
                  " > log 2>&1 &")


class TrainThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, input_file, output_file, vocab_file):
        threading.Thread.__init__(self)
        self.input_file = input_file
        self.output_file = output_file
        self.vocab_file = vocab_file
        self.finished = False

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        os.system("python create_pretraining_data.py " +
                  " --input_file " + self.input_file +
                  " --output_file " + self.output_file +
                  " --vocab_file  " + self.vocab_file)
        self.finished = True


if __name__ == "__main__":
    process()
