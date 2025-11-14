#include "ChessHMM.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

HMMState::HMMState(ChessGameState* position) : parent(nullptr) {
    HMMState::game_state = position;

    HMMState::timestep = 0;
    HMMState::prob = 0;
    HMMState::observartion_prob = 0;

    HMMState::is_children_computed = false;

    HMMState::is_self_loop = false;
}
HMMState::HMMState(HMMState& parent, ChessGameState* position, bool is_selfloop) : parent(&parent) {
    HMMState::game_state = position;

    HMMState::timestep = parent.timestep+1;
    HMMState::prob = 0;
    HMMState::observartion_prob = 0;

    HMMState::is_children_computed = false;

    HMMState::is_self_loop = is_selfloop;
}

namespace {
    template <typename T>
    class NumpyArray3D {
        private:
            struct Index3D {
                ssize_t i,j,k;
            };
        public:
            NumpyArray3D(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
                info = arr.request();
                if (info.ndim != 3) throw std::runtime_error("Expected a 3D NumPy array of shape (8,8,13)");
                ptr = static_cast<T*>(info.ptr);
                shape = {
                    info.shape[0],
                    info.shape[1],
                    info.shape[2],
                };
            }

            T& operator[](Index3D index) {
                return at(index);
            }

            string str() {
                string res = "[";
                for (int i = 0; i < 8; i++)
                {
                    res += "[";
                    for (int j = 0; j < 8; j++)
                    {
                        res += "[";
                        for (int k = 0; k < 13; k++)
                        {
                            res += to_string(at({i,j,k}));
                            if (k != 12) res += ", ";
                        }
                        res += "]";
                        if (j != 7) res += "\n";
                    }
                    res += "]";
                    if (i != 7) res += "\n\n";
                }
                res += "]";
                return res;
            }

            Index3D shape;
        private:
            py::buffer_info info;
            T* ptr;

            T& at(Index3D index) {
                ssize_t flat_index = index.i * (shape.j * shape.k) + index.j * shape.k + index.k;
                // ssize_t flat_index = (index.i * index.j + index.j) * index.k + index.k;
                return ptr[flat_index];
            }
    };
}

double HMMState::eval_prob(py::array_t<float, py::array::c_style | py::array::forcecast> obs_probs) {
    // TODO:
    NumpyArray3D arr(obs_probs);

    if (arr.shape.i != 8 || arr.shape.j != 8 || arr.shape.k != 13) throw invalid_argument("Numpy Shape is Invalid");

    HMMState::observartion_prob = 0;
    for (ssize_t i = 0; i < 8; i++)
    {
        for (ssize_t j = 0; j < 8; j++)
        {
            HMMState::observartion_prob += arr[{i,j,(ssize_t)(HMMState::game_state->current_position[i][j])}];
        }
    }

    // HMMState::observartion_prob = 0;
    for (ssize_t i = 0; i < 8; i++)
    {
        for (ssize_t j = 0; j < 8; j++)
        {
            uint8_t obsereved_label = 0;
            for (ssize_t k = 0; k < 13; k++)
            {
                if (arr[{i,j,k}] < arr[{i,j,obsereved_label}]) obsereved_label = (uint8_t)k;
            }

            if (
                HMMState::game_state->current_position[i][j] != HMMState::parent->game_state->current_position[i][j] || // move
                HMMState::game_state->current_position[i][j] != obsereved_label // difference to observed
            )
                HMMState::observartion_prob += arr[{i,j,(ssize_t)(HMMState::game_state->current_position[i][j])}]; // expected
        }
    }

    HMMState::prob = HMMState::observartion_prob + HMMState::transition_prob() + HMMState::parent_prob();

    return HMMState::prob;
}

vector<HMMState*> HMMState::get_children() {
    if (!HMMState::is_children_computed) HMMState::compute_children();

    return HMMState::children;
}

bool HMMState::operator<(const HMMState& other) const {
    return HMMState::prob < other.prob;
}
bool HMMState::operator==(const HMMState& other) const {
    return HMMState::prob == other.prob;
}

double HMMState::transition_prob() {
    double prob = HMMState::parent->get_child_transition_prob();
    if (!(HMMState::is_self_loop)) {
        prob += 20;
    }
    return prob;
}
double HMMState::parent_prob() {
    return HMMState::parent->prob;
}

double HMMState::get_child_transition_prob(bool is_legal) const {
    if (!HMMState::is_children_computed) throw logic_error("Children Not Computed");

    return log(HMMState::children.size());
}

void HMMState::compute_children() {
    HMMState::children.push_back(HMMStateFactory::create_state(this, HMMState::game_state, true)); // self loop

    for (ChessGameState* positions : HMMState::game_state->get_children()) {
        HMMState::children.push_back(HMMStateFactory::create_state(this, positions)); // follow up legal moves
    }
    HMMState::is_children_computed = true;
}

// --------------------------------------------------

HMMStateFactory::Registry HMMStateFactory::registry = {
    vector<HMMState*>(0)
};
HMMState* HMMStateFactory::create_state(HMMState* parent, ChessGameState* position, bool is_selfloop) {
    HMMState* new_state;
    if (parent == nullptr) new_state = new HMMState(position);
    else new_state = new HMMState(*parent, position, is_selfloop);

    HMMStateFactory::registry.states.push_back(new_state);
    return new_state;
}

HMMStateFactory::Registry::~Registry() {
    for (auto ptr : HMMStateFactory::registry.states) {
        try {
            delete ptr;
        } catch (...) {}
    }
}

// --------------------------------------------------

bool ChessHMM::CompareProbs::operator()(const HMMState* a, const HMMState* b) const {
    return a->prob < b->prob;  // smaller cost = higher priority
}

ChessHMM::ChessHMM(int max_width, string fen) {
    ChessHMM::max_width = (size_t)max_width;
    ChessHMM::_top_bind_t = 0;
    ChessHMM::root = HMMStateFactory::create_state(nullptr, GameStateFactory::create_state(fen));

    ChessHMM::timed_tree_states = vector<multiset<HMMState*, ChessHMM::CompareProbs>*>(0);
    ChessHMM::timed_tree_states.push_back(new multiset<HMMState*, ChessHMM::CompareProbs>());
    ChessHMM::timed_tree_states[0]->insert(root);
}
ChessHMM::~ChessHMM() {
    for (auto vec_ptr : ChessHMM::timed_tree_states) {
        delete vec_ptr;
    }
}

int ChessHMM::top_t() {
    return ChessHMM::timed_tree_states.size() - 1;
}
int ChessHMM::top_bind_t() {
    return ChessHMM::_top_bind_t;
}

void ChessHMM::set_probs(int timestep, py::array_t<float, py::array::c_style | py::array::forcecast> obs_probs) {

    if ((timestep != ChessHMM::top_t() && timestep != ChessHMM::top_t()+1) || timestep <= 0) throw invalid_argument("Timstep Invalid");

    if (timestep == ChessHMM::top_t()) {
        // clear next queue elements
        ChessHMM::timed_tree_states[timestep]->clear();
    } else if (timestep == ChessHMM::top_t()+1) {
        ChessHMM::timed_tree_states.push_back(new multiset<HMMState*, ChessHMM::CompareProbs>());
    }

    for (HMMState* state : *(ChessHMM::timed_tree_states[timestep-1])) {
        for (HMMState* child_state : state->get_children()) {            
            child_state->eval_prob(obs_probs);
            ChessHMM::timed_tree_states[timestep]->insert(child_state);
        }
    }

    if (ChessHMM::timed_tree_states[timestep]->size() > ChessHMM::max_width) {   
        auto it = ChessHMM::timed_tree_states[timestep]->begin();
        advance(it, ChessHMM::max_width);
        ChessHMM::timed_tree_states[timestep]->erase(it, ChessHMM::timed_tree_states[timestep]->end());
    }
}
void ChessHMM::bind(int timestep) {
    if (timestep > ChessHMM::top_t() || timestep <= ChessHMM::top_bind_t()) throw invalid_argument("Timstep Invalid");
    HMMState* state = *(ChessHMM::timed_tree_states[ChessHMM::top_t()]->begin());
    for (; state != nullptr; state = state->parent) {
        if (state->timestep <= timestep) {
            ChessHMM::timed_tree_states[state->timestep]->clear();
            ChessHMM::timed_tree_states[state->timestep]->insert(state);
        }
    }

    for (int t = timestep+1; t <= ChessHMM::top_t(); ++t) {
        // collect pointers to valid parents
        std::unordered_set<HMMState*> valid_parents;
        for (auto s : *(ChessHMM::timed_tree_states[t - 1]))
            valid_parents.insert(s);

        // filter current level
        std::multiset<HMMState*, ChessHMM::CompareProbs> pruned_level;
        for (auto s : *(ChessHMM::timed_tree_states[t])) {
            if (valid_parents.count(s->parent))
                pruned_level.insert(s);
        }

        // replace with pruned version
        ChessHMM::timed_tree_states[t]->swap(pruned_level);
    }

    ChessHMM::_top_bind_t = timestep;
}
string ChessHMM::print(int timestep) {
    if (timestep > ChessHMM::top_t() || timestep < 0) throw invalid_argument("Timstep Invalid");

    string res = "";

    for (HMMState* s : *(ChessHMM::timed_tree_states[timestep])) {
        res += "Prob: " + to_string(s->prob) + "\n\n";
        res += s->game_state->str();
        res += "\n\n\n\n";
    }

    return res;
}
py::array_t<int> ChessHMM::get_history(bool include_non_bound) {
    ssize_t N = (include_non_bound ? ChessHMM::top_t()+1 : ChessHMM::top_bind_t()+1);
    auto res = py::array_t<int>({N, (ssize_t)8, (ssize_t)8});

    auto buf = res.request();
    int* ptr = static_cast<int*>(buf.ptr);

    HMMState* state = *(ChessHMM::timed_tree_states[ChessHMM::top_t()])->begin();

    for (; state != nullptr; state = state->parent) {
        if ((!include_non_bound) && state->timestep > ChessHMM::top_bind_t()) continue;

        auto& pos = state->game_state->current_position;
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                ssize_t flat_index = (state->timestep * 8 * 8) + (i * 8) + j;
                uint8_t value = pos[i][j];
                ptr[flat_index] = static_cast<int>(value);
                // ptr[flat_index] = 2;
            }
        }
    }

    return res;
}
string ChessHMM::get_pgn() {
    return "";
}