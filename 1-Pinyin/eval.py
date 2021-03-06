import numpy as np
import argparse
import fileinput
import pickle
from copy import deepcopy


# num to char: charlist
with open('ref/charlist.txt', encoding='gbk') as f:
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

    with open('ref/tests.txt') as f:
        tests = f.readlines()

    sentence_correct = 0
    sentence_total = 0
    word_correct = 0
    word_total = 0

    for i in range(0, len(tests), 2):
        pred = say(tests[i])

        #print(tests[i + 1].strip())
        #print(pred)
        
        gt = tests[i + 1]
        sentence_total += 1
        sentence_wrong = False
        for p, g in zip(pred, gt):
            word_total += 1
            if p == g:
                word_correct += 1
            else:
                sentence_wrong = True
        if not sentence_wrong:
            sentence_correct += 1
            #print('correct\n')
        else:
            #print('wrong\n')
            pass
            
    print('Tests done')
    print('Sentence acc: {:.3f}'.format(sentence_correct / sentence_total))
    print('Word acc: {:.3f}'.format(word_correct / word_total))

