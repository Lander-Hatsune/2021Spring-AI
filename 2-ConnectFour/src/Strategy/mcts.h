#ifndef M_CTS_H_
#define M_CTS_H_

#include <vector>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include "Judge.h"
#include "Point.h"

using std::cerr;
using std::endl;

extern int M_, N_, noX_, noY_;

class Node {
public:
  int cnt;
  int win;
  int** board;
  bool end;
  bool my_step;
  Point step = Point(-1, -1);

  std::vector<Node*> ch;
  Node* fa;
  int* top;
  bool* actions;

  Node(int** board, int* top, bool my_step):
    top(top), my_step(my_step) {
    cnt = 0;
    win = 0;
    end = 0;
    this->board = new int *[M_];
    for (int i = 0; i < M_; i++) {
      this->board[i] = new int[N_];
      for (int j = 0; j < N_; j++) {
        this->board[i][j] = board[i][j];
      }
    }
    actions = new bool [N_];
    for (int i = 0; i < N_; i++)
      actions[i] = 0;
  }

  Node(const Node* x):
    my_step(x->my_step) {
    // step, fa, end, ch, initial
    cnt = 0;
    win = 0;
    end = 0;
    this->board = new int *[M_];
    for (int i = 0; i < M_; i++) {
      this->board[i] = new int[N_];
      for (int j = 0; j < N_; j++) {
        this->board[i][j] = x->board[i][j];
      }
    }
    this->top = new int[N_];
    for (int i = 0; i < N_; i++)
      this->top[i] = x->top[i];

    actions = new bool [N_];
    for (int i = 0; i < N_; i++)
      actions[i] = 0;
  }

  Node(const Node& x) = delete;

  ~Node() {
    for (int i = 0; i < M_; i++)
      delete[] board[i];
    delete[] top;
    delete[] actions;
  }

  int default_policy();
  void back_prop(int reward);
  Node* best_ch();
  Node* best_ch_expand();
  void describe();
  void simstep();
};

class MCT {
public:
  Node* root;
  MCT(int** board, int* top) {
    root = new Node(board, top, 1);
    root->fa = nullptr;
    cerr << "MCT: root init\n";
    srand(20010814);
  }
  ~MCT();
  static void del(Node*);
  Point search(int time);
  Node* tree_policy();
};

#endif
