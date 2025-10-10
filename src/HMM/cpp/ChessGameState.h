#include <string>
#include <vector>
#include <cstdint>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

#ifndef CHESSGAMESTATES
#define CHESSGAMESTATES

static const char rank_lables[] = {'1','2','3','4','5','6','7','8'};
static const char file_lables[] = {'a','b','c','d','e','f','g','h'};

struct ChessSquare {
    int rank;
    int file;

    bool operator==(const ChessSquare& other) const;
    string get_san();
};

enum PieceTypes {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
};

enum PromotionPieceType {
    PROMOTED_KNIGHT = PieceTypes::KNIGHT,
    PROMOTED_BISHOP = PieceTypes::BISHOP,
    PROMOTED_ROOK = PieceTypes::ROOK,
    PROMOTED_QUEEN = PieceTypes::QUEEN,
};

class ChessMove {
    friend class ChessGameState;
    
    public:
        ChessMove(string from_sq, string to_sq);
        ChessMove(ChessSquare from_sq, ChessSquare to_sq);
        ChessMove(ChessSquare from_sq, ChessSquare to_sq, PromotionPieceType promotion_type);

        string str() const;

        string repr() const;

        bool operator<(const ChessMove& other) const;
        bool operator==(const ChessMove& other) const;
    
    private:
        string from_san;
        string to_san;

        ChessSquare from_sq;
        ChessSquare to_sq;

        PromotionPieceType promotion_type = PromotionPieceType::PROMOTED_QUEEN; // TODO: set in constructor
};

enum GameFlags {
    WHITE_TURN = 1 << 0,

    WHITE_KING_CASTLE = 1 << 1,
    WHITE_QUEEN_CASTLE = 1 << 2,

    BLACK_KING_CASTLE = 1 << 3,
    BLACK_QUEEN_CASTLE = 1 << 4,

    EN_PASSANT = 1 << 5,

    IS_CHECK = 1 << 6,
    IS_STALEMATE = 1 << 7,
    IS_GAME_OVER = 1 << 8,

    IS_LEGAL = 1 << 9,
    CHILDREN_COMPUTED = 1 << 10,
};

struct PieceHistogram {
    uint8_t white_pawn = 0;
    uint8_t white_knight = 0;
    uint8_t white_bishop = 0;
    uint8_t white_rook = 0;
    uint8_t white_queen = 0;
    uint8_t white_king = 0;

    uint8_t black_pawn = 0;
    uint8_t black_knight = 0;
    uint8_t black_bishop = 0;
    uint8_t black_rook = 0;
    uint8_t black_queen = 0;
    uint8_t black_king = 0;

    uint8_t empty = 0;
};

static const string STARTING_POSITION_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

class ChessGameState {
    friend class HMMState;
    friend class ChessHMM;
    public:
        ChessGameState(string fen = STARTING_POSITION_FEN);
        ChessGameState(const ChessGameState& other, ChessMove move);
        ~ChessGameState();

        void eval_next();

        string get_fen();

        const ChessGameState* get_parent();

        vector<ChessMove> get_legal_moves();
        vector<ChessGameState*> get_children();

        ChessGameState* move(ChessMove move);
        
        string str();
        string repr();

    private:
        string fen;
        const ChessGameState* parent;
        uint8_t current_position[8][8];
        uint16_t flags = 0; // up to 16 flags

        ChessSquare enpassant_sq;

        PieceHistogram piece_counts;

        map<ChessMove, ChessGameState*> children;

        void parse_fen(string fen);
        bool is_valid();

        void add_child(ChessGameState* new_pos, ChessMove move);

        bool attacking_pawn(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece = PieceTypes::KING);
        bool attacking_knight(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece = PieceTypes::KING);
        bool attacking_bishop(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece = PieceTypes::KING);
        bool attacking_rook(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece = PieceTypes::KING);
        bool attacking_queen(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece = PieceTypes::KING);
        bool attacking_king(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece = PieceTypes::KING);

        bool is_attacked(ChessSquare piece_square, bool is_white);

        bool legal_to_go(ChessSquare to_sq, bool is_white);

        void legal_moves_pawn(ChessSquare piece_square, bool is_white);
        void legal_moves_knight(ChessSquare piece_square, bool is_white);
        void legal_moves_bishop(ChessSquare piece_square, bool is_white);
        void legal_moves_rook(ChessSquare piece_square, bool is_white);
        void legal_moves_queen(ChessSquare piece_square, bool is_white);
        void legal_moves_king(ChessSquare piece_square, bool is_white);

        void compute_children();
};

class GameStateFactory {
    public:
        ChessGameState* get_state();

        static ChessGameState* create_state(ChessGameState* position, ChessMove move);
        static ChessGameState* create_state(string fen);
    
    private:
        struct Registry {
            std::vector<ChessGameState*> states;
            ~Registry();
        };

        static GameStateFactory::Registry registry;
};

#endif