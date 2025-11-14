#include "ChessGameState.h"
#include <vector>
#include <queue>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

#ifndef CHESSHMM
#define CHESSHMM

class HMMState {
    friend class ChessHMM;

    public:
        HMMState(ChessGameState* position);
        HMMState(HMMState& parent, ChessGameState* position, bool is_selfloop = false);
        double eval_prob(py::array_t<float, py::array::c_style | py::array::forcecast> obs_probs);

        vector<HMMState*> get_children();

        bool operator<(const HMMState& other) const;
        bool operator==(const HMMState& other) const;
        
    private:
        ChessGameState* game_state;
        HMMState* parent;
        vector<HMMState*> children;
        
        int timestep;
        double prob;

        bool is_self_loop;

        double observartion_prob;
        double transition_prob();
        double parent_prob();

        double get_child_transition_prob(bool is_legal = true) const;

        bool is_children_computed;

        void compute_children();
};

class HMMStateFactory {
    public:
        static HMMState* create_state(HMMState* parent, ChessGameState* position, bool is_selfloop = false);

    private:
        struct Registry {
            std::vector<HMMState*> states;
            ~Registry();
        };

        static HMMStateFactory::Registry registry;
};

class ChessHMM {
    public:
        ChessHMM(int max_width, string fen = STARTING_POSITION_FEN);
        ~ChessHMM();

        int top_t();
        int top_bind_t();
        
        void set_probs(int timestep, py::array_t<float, py::array::c_style | py::array::forcecast> obs_probs);
        void bind(int timestep);

        string print(int timestep);

        py::array_t<int> get_history(bool include_non_bound = false);
        string get_pgn();
    private:
        HMMState* root;

        struct CompareProbs {
            bool operator()(const HMMState* a, const HMMState* b) const;
        };

        vector<multiset<HMMState*, ChessHMM::CompareProbs>*> timed_tree_states;

        int _top_bind_t;
        size_t max_width;
};

#endif