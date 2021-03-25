import numpy as np
import argparse
import fileinput
import pickle
from copy import deepcopy


# num to char: charlist
with fileinput.input(files='ref/charlist.txt',
                     openhook=fileinput.hook_encoded('gbk')) as f:
    for line in f: # only one line
        charlist = line

# tuple to log probability: mesh_prob_log
mesh_prob_log = np.load('data/mesh_prob_log.npy')

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

#print(say("yi zhi ke ai de da huang gou"))
# tong ji
# gan ni niang ji bai
