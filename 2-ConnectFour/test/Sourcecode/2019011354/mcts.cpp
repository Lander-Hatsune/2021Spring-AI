#include "mcts.h"
#include "Judge.h"
#include <chrono>
#include <stack>
#include <cmath>

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void Node::describe() {
  cerr << "--------Node--Description--------\n";
  cerr << "win/cnt: " << win << "/" << cnt << endl;
  cerr << "board: \n";
  for (int i = 0; i < M_; i++) {
    for (int j = 0; j < N_; j++) {
      cerr << board[i][j] << " ";
    }
    cerr << endl;
  }

  cerr << "top: \n";
  for (int i = 0; i < N_; i++)
    cerr << top[i] << " ";
  cerr << endl;

  cerr << "actions: \n";
  for (int i = 0; i < N_; i++)
    cerr << actions[i] << " ";
  cerr << endl;
  
  cerr << "end: " << end << endl;
  cerr << "my_step: " << my_step << endl;
  cerr << "step: " << step.x << "," << step.y << endl;
  cerr << "---------------------------------\n";
}

void Node::simstep() {
  if (end) return;
  std::vector<int> choices;
  bool pre_def = false;
  for (int i = 0; i < N_; i++) {
    if (top[i] >= 0) {

      // I will win at step
      board[top[i]][i] = my_step ? 2 : 1;
      if (my_step ?
          machineWin(top[i], i, M_, N_, board) :
          userWin(top[i], i, M_, N_, board)) {
        pre_def = true;
        step.x = top[i], step.y = i;
        break;
      }
      board[top[i]][i] = 0;

      // He will win at step
      board[top[i]][i] = my_step ? 1 : 2;
      if (my_step ? 
          userWin(top[i], i, M_, N_, board) :
          machineWin(top[i], i, M_, N_, board)) {
        pre_def = true;
        board[top[i]][i] = my_step ? 2 : 1;
        step.x = top[i], step.y = i;
        break;
      }
      board[top[i]][i] = 0;

      // He will win next step
      if (top[i] >= 1 && (top[i] - 1 != noX_ || i != noY_)) {
        board[top[i]][i] = my_step ? 2 : 1;
        board[top[i] - 1][i] = my_step ? 1 : 2;
        if ((my_step) ?
            userWin(top[i] - 1, i, M_, N_, board) :
            machineWin(top[i] - 1, i, M_, N_, board)) {
            
          board[top[i]][i] = 0;
          board[top[i] - 1][i] = 0;
          continue;
          
        }
        board[top[i]][i] = 0;
        board[top[i] - 1][i] = 0;
      }

      // I will win next step
      // deprecated
      
      choices.push_back(i);
    }
  }
  
  if (!pre_def) {
    
    if (!choices.size()) {
      for (int i = 0; i < N_; i++)
        if (top[i] >= 0)
          choices.push_back(i);
    }
    if (!choices.size()) {
      end = true;
      return;
    }

    int idx = rand() % choices.size();
    int pos = choices[idx];
    step.x = top[pos], step.y = pos;
    //cerr << "default_policy: step is " << sim->top[pos] << "," << pos << endl;

    board[top[pos]][pos] = my_step ? 2 : 1;
    //cerr << "default_policy: board updated\n";
  }

  top[step.y] -= 1;
  if (step.y == noY_ && top[step.y] == noX_)
    top[step.y] -= 1;
    
  if (my_step ?
      machineWin(step.x, step.y, M_, N_, board) :
      userWin(step.x, step.y, M_, N_, board) ||
      isTie(N_, top)) {
    end = true;
  } else my_step = !my_step;
}

int Node::default_policy() {

  int reward;
  
  if (this->end) {
    if (isTie(N_, this->top)) {
      reward = 0;
    } else reward = 1;
    return reward;
  }

  Node* sim = new Node(this);
  while (true) {
    sim->simstep();
    if (sim->end) break;
  }

  //cerr << "default_policy: stepcnt: " << stepcnt << endl;


  if (isTie(N_, sim->top)) reward = 0;
  if (sim->my_step != this->my_step) reward = 1;
  else reward = 0;
  //cerr << "default_policy: reward: " << reward << endl;
  delete sim;
  return reward;
}

void Node::back_prop(int reward) {
  Node* v = this;
  while (v != nullptr) {
    v->cnt += 1;
    v->win += reward;
    reward = 1 - reward;
    v = v->fa;
  }
}
    
Node* Node::best_ch_expand() {
  double maxx = -1;
  Node* best = nullptr;

  //cerr << "best_ch_expand: win/cnt of each: " << endl;
  for (auto child: this->ch) {

    //cerr << child->win << "/" << child->cnt << endl;
    //cerr << sqrt(2 * log(this->cnt) / child->cnt) << endl;

    if ((double)child->win / child->cnt +
        sqrt(2 * log(this->cnt) / child->cnt) > maxx) {
      best = child;
      maxx =
        (double)child->win / child->cnt + 
        sqrt(2 * log(this->cnt) / child->cnt);
    }
  }
  //cerr << maxx << endl;
  return best;
}

Node* Node::best_ch() {
  double maxx = -1;
  Node* best = nullptr;

  cerr << "best_ch: win/cnt of each: " << endl;
  for (auto child: this->ch) {

    cerr << child->win << "/" << child->cnt << endl;
    if ((double)child->win / child->cnt > maxx) {
      best = child;
      maxx =
        (double)child->win / child->cnt;
    }
  }
  cerr << maxx << endl;
  return best;
}

Node* MCT::tree_policy() {
  Node* v = root;
  while (!v->end) {
    for (int i = 0; i < N_; i++) {
      if (!v->actions[i] && v->top[i] >= 0) {// has valid action(s)

        v->actions[i] = 1;
        
        // expand
        Node* v_ = new Node(v);
        v_->step = Point(v->top[i], i);
        v_->board[v->top[i]][i] = v->my_step ? 2 : 1;
        v_->top[i] -= 1;
        if (i == noY_ && v_->top[i] == noX_)
          v_->top[i] -= 1;

        v_->fa = v;
        v->ch.push_back(v_);
        v_->my_step = !v->my_step;

        if (v->my_step ?
            machineWin(v->top[i], i, M_, N_, v_->board) :
            userWin(v->top[i], i, M_, N_, v_->board) ||
            isTie(N_, v_->top)) {
          v_->end = true;
        }

        return v_;
      }
    }
    v = v->best_ch_expand();
    if (!v) return nullptr;
  }
  return v;
}

Point MCT::search(int time) {
  long long start_time = duration_cast<milliseconds>
    (system_clock::now().time_since_epoch()).count();
  long long cur_time = start_time;
  while (cur_time - start_time < time) {

    Node* vl = tree_policy();
    //cerr << "MCT: search: vl selected\n";
    //vl->describe();
    
    int reward = vl->default_policy();
    //cerr << "MCT: search: default policied\n";
    
    vl->back_prop(reward);
    //cerr << "MCT: search: back proped\n";

    cur_time = duration_cast<milliseconds>
      (system_clock::now().time_since_epoch()).count();
  }
  return root->best_ch()->step;
}

void MCT::del(Node* x) {
  if (!x) return;
  for (auto child: x->ch) {
    del(child);
  }
  delete x;
}

MCT::~MCT() {
  del(root);
}
    

