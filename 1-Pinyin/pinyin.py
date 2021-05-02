import numpy as np
import argparse
import pickle
from copy import deepcopy


# num to char: charlist
with open('data/charlist.txt', encoding='gbk') as f:
    charlist = f.readline()# only one line

# (char, char) tuple to log probability: mesh_prob_log
mesh_prob_log = np.load('data/mesh_prob_log.npy')
# char to log probability: char_prob_log
char_prob_log = np.load('data/char_prob_log.npy')

# word to num: map_word_num
with open('data/map_word_num', 'rb') as f:
    map_word_num = pickle.load(f)

def c(i: int):
    return charlist[i]
        

def say(s: str):
    words = s.lower().strip().split()

    old_ans = {}

    for wordcnt, word in enumerate(words, 1):

        if wordcnt == 1:
            old_num_prob = np.zeros(
                shape=(2, len(map_word_num[word])), dtype=np.int32)
            old_num_prob[0] = map_word_num[word]
            for i, num in enumerate(old_num_prob[0], 0):
                old_ans[i] = [num]
                old_num_prob[1][i] = char_prob_log[num]
            continue


        cur_num_prob = np.zeros(
            shape=(2, len(map_word_num[word])), dtype=np.int32)
        cur_num_prob[0] = map_word_num[word]
        cur_ans = {}

        for i, num in enumerate(cur_num_prob[0], 0):
            minlogprob = 0x3f3f3f3f
            for j, onum in enumerate(old_num_prob[0], 0):
                if old_num_prob[1][j] + mesh_prob_log[onum][num] < minlogprob:
                    chosen = j
                    minlogprob = old_num_prob[1][j] + mesh_prob_log[onum][num]

            cur_num_prob[1][i] = minlogprob

            cur_ans[i] = deepcopy(old_ans[chosen])
            cur_ans[i].append(num)

        old_num_prob = cur_num_prob
        old_ans = deepcopy(cur_ans)

    ans_num = old_ans[old_num_prob[1].argmin()]

    ans_str = ''
    for i in ans_num:
        ans_str += charlist[i]
    return ans_str

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()


    with open(args.input_file) as f:
        testcases = f.readlines()

        with open(args.output_file, 'w+') as f:
            for words in testcases:
                print(say(words), file=f)
                print(words.strip())
                print(say(words))
                
    
