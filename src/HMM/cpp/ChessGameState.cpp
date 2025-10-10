#include "ChessGameState.h"
#include <iostream>
#include <stdexcept>

using namespace std;

bool ChessSquare::operator==(const ChessSquare& other) const {
        return ChessSquare::rank == other.rank &&
                ChessSquare::file == other.file;
}

string ChessSquare::get_san() {
    string res = "";
    res += file_lables[ChessSquare::file];
    res += rank_lables[ChessSquare::rank];
    return res;
}

ChessMove::ChessMove(string from_sq, string to_sq) {
    if (from_sq.length() != 2 || to_sq.length() != 2) {
        throw invalid_argument("Invalid Input");
    }

    if (!(from_sq[0] >= 97 && from_sq[0] <= 104)) {
        throw invalid_argument("Invalid Input");
    }
    if (!(to_sq[0] >= 97 && to_sq[0] <= 104)) {
        throw invalid_argument("Invalid Input");
    }

    if (!(from_sq[1] >= 49 && from_sq[1] <= 56)) {
        throw invalid_argument("Invalid Input");
    }
    if (!(to_sq[1] >= 49 && to_sq[1] <= 56)) {
        throw invalid_argument("Invalid Input");
    }

    ChessMove::from_san = from_sq;
    ChessMove::to_san = to_sq;

    ChessMove::from_sq.rank = from_sq[1]-48;
    ChessMove::from_sq.file = from_sq[0]-97;

    ChessMove::to_sq.rank = to_sq[1]-48;
    ChessMove::to_sq.file = to_sq[0]-97;
}
ChessMove::ChessMove(ChessSquare from_sq, ChessSquare to_sq) {
    ChessMove::from_san = from_sq.get_san();
    ChessMove::to_san = to_sq.get_san();

    ChessMove::from_sq = from_sq;
    ChessMove::to_sq = to_sq;
}
ChessMove::ChessMove(ChessSquare from_sq, ChessSquare to_sq, PromotionPieceType promotion_type) {
    ChessMove::from_san = from_sq.get_san();
    ChessMove::to_san = to_sq.get_san();

    ChessMove::from_sq = from_sq;
    ChessMove::to_sq = to_sq;

    ChessMove::promotion_type = promotion_type;
}

string ChessMove::str() const {
    return "(" + ChessMove::from_san + "," + ChessMove::to_san + ")";
}

string ChessMove::repr() const {
    return "<" + ChessMove::from_san + " -> " + ChessMove::to_san + ">";
}

bool ChessMove::operator<(const ChessMove& other) const {
    if (ChessMove::from_sq.rank != other.from_sq.rank) return ChessMove::from_sq.rank < other.from_sq.rank;
    if (ChessMove::from_sq.file != other.from_sq.file) return ChessMove::from_sq.file < other.from_sq.file;
    if (ChessMove::to_sq.rank != other.to_sq.rank) return ChessMove::to_sq.rank < other.to_sq.rank;
    return ChessMove::to_sq.file < other.to_sq.file;
}

bool ChessMove::operator==(const ChessMove& other) const {
    return ChessMove::from_sq == other.from_sq &&
        ChessMove::to_sq == other.to_sq;
}

// --------------------------------------------------

ChessGameState::ChessGameState(string fen) : parent(nullptr) {
    ChessGameState::parse_fen(fen);
    ChessGameState::flags |= GameFlags::IS_LEGAL;
    ChessGameState::flags &= ~GameFlags::CHILDREN_COMPUTED;
}
ChessGameState::ChessGameState(const ChessGameState& other, ChessMove move) : parent(&other) {
    ChessGameState::fen = "";
    ChessGameState::flags = other.flags;
    ChessGameState::piece_counts = other.piece_counts;

    for (int rank = 0; rank < 8; rank++)
    {
        for (int file = 0; file < 8; file++)
        {
            ChessGameState::current_position[rank][file] = other.current_position[rank][file];
        }
    }

    // apply move

    uint8_t from_piece = ChessGameState::current_position[move.from_sq.rank][move.from_sq.file];
    uint8_t to_piece = ChessGameState::current_position[move.to_sq.rank][move.to_sq.file];

    if (from_piece == 12) throw invalid_argument("Invalid Move");

    PieceTypes from_piece_type = (PieceTypes)(from_piece%6);
    // PieceTypes to_piece_type = (PieceTypes)(to_piece%6);
    bool is_white_turn = ChessGameState::flags & GameFlags::WHITE_TURN;

    if (from_piece_type == (uint8_t)PieceTypes::PAWN && (move.to_sq.rank == 0 || move.to_sq.rank == 7)) {
        // Piece Counts
        uint8_t promotion_piece = move.promotion_type + (is_white_turn ? 0 : 6);
        ChessGameState::piece_counts.empty ++;
        switch (to_piece)
        {
        case 0:
            ChessGameState::piece_counts.white_pawn--;
            break;
        case 1:
            ChessGameState::piece_counts.white_knight--;
            break;
        case 2:
            ChessGameState::piece_counts.white_bishop--;
            break;
        case 3:
            ChessGameState::piece_counts.white_rook--;
            break;
        case 4:
            ChessGameState::piece_counts.white_queen--;
            break;
        case 5:
            ChessGameState::piece_counts.white_king--;
            break;

        case 6:
            ChessGameState::piece_counts.black_pawn--;
            break;
        case 7:
            ChessGameState::piece_counts.black_knight--;
            break;
        case 8:
            ChessGameState::piece_counts.black_bishop--;
            break;
        case 9:
            ChessGameState::piece_counts.black_rook--;
            break;
        case 10:
            ChessGameState::piece_counts.black_queen--;
            break;
        case 11:
            ChessGameState::piece_counts.black_king--;
            break;
        
        case 12:
            ChessGameState::piece_counts.empty--;
            break;
        
        default:
            break;
        }
        if (is_white_turn) ChessGameState::piece_counts.white_pawn--;
        else ChessGameState::piece_counts.black_pawn--;
        switch (promotion_piece)
        {
        case 0:
            ChessGameState::piece_counts.white_pawn++;
            break;
        case 1:
            ChessGameState::piece_counts.white_knight++;
            break;
        case 2:
            ChessGameState::piece_counts.white_bishop++;
            break;
        case 3:
            ChessGameState::piece_counts.white_rook++;
            break;
        case 4:
            ChessGameState::piece_counts.white_queen++;
            break;
        case 5:
            ChessGameState::piece_counts.white_king++;
            break;

        case 6:
            ChessGameState::piece_counts.black_pawn++;
            break;
        case 7:
            ChessGameState::piece_counts.black_knight++;
            break;
        case 8:
            ChessGameState::piece_counts.black_bishop++;
            break;
        case 9:
            ChessGameState::piece_counts.black_rook++;
            break;
        case 10:
            ChessGameState::piece_counts.black_queen++;
            break;
        case 11:
            ChessGameState::piece_counts.black_king++;
            break;
        
        case 12:
            ChessGameState::piece_counts.empty++;
            break;
        
        default:
            break;
        }

        // Move Piece
        ChessGameState::current_position[move.from_sq.rank][move.from_sq.file] = 12;
        ChessGameState::current_position[move.to_sq.rank][move.to_sq.file] = promotion_piece;

        // Castling Rights
        // No Changes

        // Enpassant
        ChessGameState::flags &= ~GameFlags::EN_PASSANT;
    } else if (from_piece_type == PieceTypes::PAWN && (other.flags&GameFlags::EN_PASSANT) && move.to_sq == other.enpassant_sq) {
        // Enpassant

        // Piece Counts
        if (is_white_turn) ChessGameState::piece_counts.black_pawn--;
        else ChessGameState::piece_counts.white_pawn--;
        ChessGameState::piece_counts.empty++;

        // Move Pieces
        ChessGameState::current_position[move.from_sq.rank][move.from_sq.file] = 12;
        ChessGameState::current_position[move.to_sq.rank][move.to_sq.file] = from_piece;
        ChessGameState::current_position[move.to_sq.rank+(is_white_turn ? -1 : 1)][move.to_sq.file] = 12;

        // Castling Rights
        // No Changes

        // Enpassant
        ChessGameState::flags &= ~GameFlags::EN_PASSANT;
    } else if (from_piece_type == PieceTypes::KING && (move.to_sq.file == move.from_sq.file+2 || move.to_sq.file == move.from_sq.file-2)) {
        // Castling

        // Piece Counts
        // No Changes

        // Move Pieces
        int direction = move.to_sq.file - move.from_sq.file;
        ChessGameState::current_position[move.from_sq.rank][move.from_sq.file] = 12;
        ChessGameState::current_position[move.to_sq.rank][move.to_sq.file] = from_piece;
        ChessGameState::current_position[move.from_sq.rank][move.from_sq.file+(direction/2)] = (is_white_turn ? 3 : 9);
        ChessGameState::current_position[move.from_sq.rank][(direction > 0) ? 7 : 0] = 12;

        // Castling Rights
        if (is_white_turn) {
            ChessGameState:: flags &= ~GameFlags::WHITE_KING_CASTLE;
            ChessGameState:: flags &= ~GameFlags::WHITE_QUEEN_CASTLE;
        } else {
            ChessGameState:: flags &= ~GameFlags::BLACK_KING_CASTLE;
            ChessGameState:: flags &= ~GameFlags::BLACK_QUEEN_CASTLE;
        }

        // Enpassant
        ChessGameState::flags &= ~GameFlags::EN_PASSANT;
    } else {
        // General Case

        // Piece Counts
        ChessGameState::piece_counts.empty++;
        switch (to_piece)
        {
        case 0:
            ChessGameState::piece_counts.white_pawn--;
            break;
        case 1:
            ChessGameState::piece_counts.white_knight--;
            break;
        case 2:
            ChessGameState::piece_counts.white_bishop--;
            break;
        case 3:
            ChessGameState::piece_counts.white_rook--;
            break;
        case 4:
            ChessGameState::piece_counts.white_queen--;
            break;
        case 5:
            ChessGameState::piece_counts.white_king--;
            break;

        case 6:
            ChessGameState::piece_counts.black_pawn--;
            break;
        case 7:
            ChessGameState::piece_counts.black_knight--;
            break;
        case 8:
            ChessGameState::piece_counts.black_bishop--;
            break;
        case 9:
            ChessGameState::piece_counts.black_rook--;
            break;
        case 10:
            ChessGameState::piece_counts.black_queen--;
            break;
        case 11:
            ChessGameState::piece_counts.black_king--;
            break;
        
        case 12:
            ChessGameState::piece_counts.empty--;
            break;
        
        default:
            break;
        }
        
        // Move Piece
        ChessGameState::current_position[move.from_sq.rank][move.from_sq.file] = 12;
        ChessGameState::current_position[move.to_sq.rank][move.to_sq.file] = from_piece;
        
        // Castling
        if (from_piece_type == PieceTypes::KING) {
            if (is_white_turn) {
                ChessGameState::flags &= ~GameFlags::WHITE_KING_CASTLE;
                ChessGameState::flags &= ~GameFlags::WHITE_QUEEN_CASTLE;
            } else {
                ChessGameState::flags &= ~GameFlags::BLACK_KING_CASTLE;
                ChessGameState::flags &= ~GameFlags::BLACK_QUEEN_CASTLE;
            }
        }
        if (from_piece_type == PieceTypes::ROOK) {
            if ((move.from_sq.rank == 0 || move.from_sq.rank == 7) && (move.from_sq.file == 0)) {
                if (is_white_turn) ChessGameState::flags &= ~GameFlags::WHITE_QUEEN_CASTLE;
                else ChessGameState::flags &= ~GameFlags::BLACK_QUEEN_CASTLE;
            } else if ((move.from_sq.rank == 0 || move.from_sq.rank == 7) && (move.from_sq.file == 7)) {
                if (is_white_turn) ChessGameState::flags &= ~GameFlags::WHITE_KING_CASTLE;
                else ChessGameState::flags &= ~GameFlags::BLACK_KING_CASTLE;
            }
        }
        // Enpassant
        ChessGameState::flags &= ~GameFlags::EN_PASSANT;
        if (from_piece_type == PieceTypes::PAWN && (move.from_sq.rank == 1 || move.from_sq.rank == 6) && (move.to_sq.rank == 3 || move.to_sq.rank == 4)) {
            ChessGameState::flags |= GameFlags::EN_PASSANT;
            ChessGameState::enpassant_sq = move.to_sq;
            ChessGameState::enpassant_sq.rank += (is_white_turn ? -1 : 1);
        }
    }

    // Turn
    ChessGameState::flags ^= GameFlags::WHITE_TURN;
    if (!ChessGameState::is_valid()) throw invalid_argument("Invalid Move");
    ChessGameState::flags |= GameFlags::IS_LEGAL;
    ChessGameState::flags &= ~GameFlags::CHILDREN_COMPUTED;
}
ChessGameState::~ChessGameState() {
}

void ChessGameState::eval_next() {
    if (!(ChessGameState::flags & GameFlags::CHILDREN_COMPUTED)) ChessGameState::compute_children();
}

// double ChessGameState::eval_prob() {
//     return 0;
// }

string ChessGameState::get_fen(){
    return ChessGameState::fen;
}
// int ChessGameState::get_timestep(){
//     return ChessGameState::timstep;
// }

const ChessGameState* ChessGameState::get_parent() {
    return ChessGameState::parent;
}

vector<ChessMove> ChessGameState::get_legal_moves() {
    if (!(ChessGameState::flags & GameFlags::CHILDREN_COMPUTED)) ChessGameState::compute_children();
    vector<ChessMove> res;
    for (const auto& [mv, state] : ChessGameState::children) {
        res.push_back(mv);
    }
    return res;
}
vector<ChessGameState*> ChessGameState::get_children() {
    if (!(ChessGameState::flags & GameFlags::CHILDREN_COMPUTED)) ChessGameState::compute_children();
    vector<ChessGameState*> res;
    for (const auto& [mv, state] : ChessGameState::children) {
        res.push_back(state);
    }
    return res;
}

ChessGameState* ChessGameState::move(ChessMove move) {
    if (!(ChessGameState::flags & GameFlags::CHILDREN_COMPUTED)) ChessGameState::compute_children();

    auto it = ChessGameState::children.find(move);
    if (it == ChessGameState::children.end()) throw invalid_argument("Move Invalid");

    ChessGameState* next_pos = it->second;
    return next_pos;
}

string ChessGameState::str() {
    string res = "";
    for (int rank = 7; rank >= 0; rank--)
    {
        for (int file = 0; file < 8; file++)
        {
            uint8_t cell = ChessGameState::current_position[rank][file];
            switch (cell)
            {
            case 0:
                res += 'P';
                break;
            case 1:
                res += 'N';
                break;
            case 2:
                res += 'B';
                break;
            case 3:
                res += 'R';
                break;
            case 4:
                res += 'Q';
                break;
            case 5:
                res += 'K';
                break;
            case 6:
                res += 'p';
                break;
            case 7:
                res += 'n';
                break;
            case 8:
                res += 'b';
                break;
            case 9:
                res += 'r';
                break;
            case 10:
                res += 'q';
                break;
            case 11:
                res += 'k';
                break;
            case 12:
                res += '-';
                break;
            
            default:
                throw domain_error("Stored Position Invalid");
                break;
            }
        }
        res += '\n';
    }
    
    return res;
}

string ChessGameState::repr() {
    string res = "";

    res += "FEN: " + ChessGameState::get_fen() + "\n\n";

    res += ChessGameState::str() + "\n";

    res += "Turn: ";
    if (ChessGameState::flags & GameFlags::WHITE_TURN) res += "White\n";
    else if (!(ChessGameState::flags & GameFlags::WHITE_TURN)) res += "Black\n";

    res += "Castling Rights: ";
    if (ChessGameState::flags & GameFlags::WHITE_KING_CASTLE) res += "K";
    if (ChessGameState::flags & GameFlags::WHITE_QUEEN_CASTLE) res += "Q";
    if (ChessGameState::flags & GameFlags::BLACK_KING_CASTLE) res += "k";
    if (ChessGameState::flags & GameFlags::BLACK_QUEEN_CASTLE) res += "q";
    res += "\n";

    res += "En Passant: ";
    if (ChessGameState::flags & GameFlags::EN_PASSANT) res += enpassant_sq.get_san();
    else res += "-";
    res += "\n";

    res += "Check: ";
    if (ChessGameState::flags & GameFlags::IS_CHECK) res += "True";
    else res += "False";
    res += "\n";

    res += "Stalemate: ";
    if (ChessGameState::flags & GameFlags::IS_STALEMATE) res += "True";
    else res += "False";
    res += "\n";

    res += "Gameover: ";
    if (ChessGameState::flags & GameFlags::IS_GAME_OVER) res += "True";
    else res += "False";
    // res += "\n";

    return res;
}

void ChessGameState::parse_fen(string fen) {
    // TODO: validate fen (check king count, multiple checks, etc)
    ChessGameState::fen = fen;
    int rank = 7, file = 0;

    int is_flags = 0;
    int empty_count = 0;
    bool is_color_set = false;
    bool is_castle_set = false;
    bool is_enpassant_set = false;

    for (const char& c : fen)
    {
        if (is_flags == 0) { // board squares
            switch (c)
            {
            case '/':
                rank--;
                file = 0;
                break;

            case 'P':
                ChessGameState::current_position[rank][file] = 0;
                ChessGameState::piece_counts.white_pawn++;
                file++;
                break;
                
            case 'N':
                ChessGameState::current_position[rank][file] = 1;
                ChessGameState::piece_counts.white_knight++;
                file++;
                break;
                
            case 'B':
                ChessGameState::current_position[rank][file] = 2;
                ChessGameState::piece_counts.white_bishop++;
                file++;
                break;

            case 'R':
                ChessGameState::current_position[rank][file] = 3;
                ChessGameState::piece_counts.white_rook++;
                file++;
                break;

            case 'Q':
                ChessGameState::current_position[rank][file] = 4;
                ChessGameState::piece_counts.white_queen++;
                file++;
                break;
            
            case 'K':
                ChessGameState::current_position[rank][file] = 5;
                ChessGameState::piece_counts.white_king++;
                file++;
                break;

            case 'p':
                ChessGameState::current_position[rank][file] = 6;
                ChessGameState::piece_counts.black_pawn++;
                file++;
                break;
            
            case 'n':
                ChessGameState::current_position[rank][file] = 7;
                ChessGameState::piece_counts.black_knight++;
                file++;
                break;

            case 'b':
                ChessGameState::current_position[rank][file] = 8;
                ChessGameState::piece_counts.black_bishop++;
                file++;
                break;

            case 'r':
                ChessGameState::current_position[rank][file] = 9;
                ChessGameState::piece_counts.black_rook++;
                file++;
                break;

            case 'q':
                ChessGameState::current_position[rank][file] = 10;
                ChessGameState::piece_counts.black_queen++;
                file++;
                break;
            
            case 'k':
                ChessGameState::current_position[rank][file] = 11;
                ChessGameState::piece_counts.black_king++;
                file++;
                break;

            case ' ':
                if (!(rank == 0 && file == 8)) throw invalid_argument("FEN Invalid");
                is_flags++;
                break;
            
            default:
                if (!(c >= 49 && c <= 56)) throw invalid_argument("FEN Invalid");
                empty_count = c - 48;
                if (file + empty_count > 8) throw invalid_argument("FEN Invalid");

                for (int i = file; i < file+empty_count; i++)
                {
                    ChessGameState::current_position[rank][i] = 12;
                    ChessGameState::piece_counts.empty++;
                }

                file += empty_count;
                
                break;
            }
        } else if (is_flags == 1) { // turn
            switch (c)
            {
            case 'w':
                ChessGameState::flags |= GameFlags::WHITE_TURN;
                is_color_set = true;
                break;
                
            case 'b':
                ChessGameState::flags &= (~GameFlags::WHITE_TURN);
                is_color_set = true;
                break;

            case ' ':
                if (!is_color_set) throw invalid_argument("FEN Invalid");
                is_flags++;
                break;
            
            default:
                throw invalid_argument("FEN Invalid");
                break;
            }
        } else if (is_flags == 2) { // castling rights
            switch (c)
            {
            case '-':
            is_castle_set = true;
            break;
            
            case ' ':
                if (!is_castle_set) throw invalid_argument("FEN Invalid");
                is_flags++;
                break;
            
            case 'K':
                ChessGameState::flags |= GameFlags::WHITE_KING_CASTLE;
                is_castle_set = true;
                break;
                
            case 'Q':
                ChessGameState::flags |= GameFlags::WHITE_QUEEN_CASTLE;
                is_castle_set = true;
                break;

            case 'k':
                ChessGameState::flags |= GameFlags::BLACK_KING_CASTLE;
                is_castle_set = true;
                break;

            case 'q':
                ChessGameState::flags |= GameFlags::BLACK_QUEEN_CASTLE;
                is_castle_set = true;
                break;
            
            default:
                throw invalid_argument("FEN Invalid");
                break;
            }
        } else if (is_flags == 3) { // enpassant
            switch (c)
            {
            case '-':
                is_enpassant_set = true;
                break;

            case ' ':
                if (!is_enpassant_set) throw invalid_argument("FEN Invalid");
                is_flags++;
                break;
            
            default:
                if (c >= 97 && c <= 104) {
                    ChessGameState::enpassant_sq.file = c - 97;
                    ChessGameState::flags |= GameFlags::EN_PASSANT;
                    is_enpassant_set = true;
                }
                else if (c >= 49 && c<= 56) {
                    ChessGameState::enpassant_sq.rank = c - 48;
                    ChessGameState::flags |= GameFlags::EN_PASSANT;
                    is_enpassant_set = true;
                }
                else throw invalid_argument("FEN Invalid");
                break;
            }
        } else if (is_flags == 4) {
            // TODO: Handle ply count
        }
    }

    // TODO: check other flags (stalemate, gameover)
    if (!ChessGameState::is_valid()) throw invalid_argument("FEN Invalid");
}

bool ChessGameState::is_valid() {
    if (ChessGameState::piece_counts.white_king != 1 || ChessGameState::piece_counts.black_king != 1) return false;

    bool is_white_checked = false;
    bool is_black_checked = false;

    ChessSquare white_king_sq;
    ChessSquare black_king_sq;
    bool white_king_found = false, black_king_found = false;
    for (int rank = 0; rank < 8; rank++)
    {
        for (int file = 0; file < 8; file++)
        {
            if (ChessGameState::current_position[rank][file] == 5) {
                white_king_found = true;
                white_king_sq.rank = rank;
                white_king_sq.file = file;
            } else if (ChessGameState::current_position[rank][file] == 11) {
                black_king_found = true;
                black_king_sq.rank = rank;
                black_king_sq.file = file;
            }
            if (white_king_found && black_king_found) break;
        }
        if (white_king_found && black_king_found) break;
    }

    if (ChessGameState::is_attacked(white_king_sq, true)) is_white_checked = true;
    if (ChessGameState::is_attacked(black_king_sq, false)) is_black_checked = true;

    if (is_white_checked && is_black_checked) return false;

    bool is_white_turn = ChessGameState::flags & GameFlags::WHITE_TURN;

    if (is_white_checked && !is_white_turn) return false;
    if (is_black_checked && is_white_turn) return false;

    if (is_white_checked || is_black_checked) ChessGameState::flags |= GameFlags::IS_CHECK;

    return true;
}

void ChessGameState::add_child(ChessGameState* new_pos, ChessMove move) {
    ChessGameState::children[move] = new_pos;
}

bool is_square(ChessSquare sq) {
    return sq.rank >= 0 && sq.rank < 8 && sq.file >= 0 && sq.file < 8;
}

bool ChessGameState::attacking_pawn(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece_type) {
    uint8_t attacked_piece = (uint8_t)attacked_piece_type + (is_white ? 6 : 0);
    int rank_diff = is_white ? 1 : -1;
    if (
        is_square({piece_square.rank+rank_diff, piece_square.file+1}) &&
        ChessGameState::current_position[piece_square.rank+rank_diff][piece_square.file+1] == attacked_piece
    ) return true;

    if (
        is_square({piece_square.rank+rank_diff, piece_square.file-1}) &&
        ChessGameState::current_position[piece_square.rank+rank_diff][piece_square.file-1] == attacked_piece
    ) return true;

    return false;
}

bool ChessGameState::attacking_knight(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece_type) {
    uint8_t attacked_piece = (uint8_t)attacked_piece_type + (is_white ? 6 : 0);
    for (int new_rank = piece_square.rank-1; new_rank <= piece_square.rank+1; new_rank+=2)
    {
        for (int new_file = piece_square.file-2; new_file <= piece_square.file+2; new_file+=4)
        {
            if (is_square({new_rank, new_file}) && ChessGameState::current_position[new_rank][new_file] == attacked_piece) return true;
        }
    }
    
    for (int new_rank = piece_square.rank-2; new_rank <= piece_square.rank+2; new_rank+=4)
    {
        for (int new_file = piece_square.file-1; new_file <= piece_square.file+1; new_file+=2)
        {
            if (is_square({new_rank, new_file}) && ChessGameState::current_position[new_rank][new_file] == attacked_piece) return true;
        }
    }

    return false;
}

bool ChessGameState::attacking_bishop(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece_type) {
    uint8_t current_piece;
    uint8_t attacked_piece = (uint8_t)attacked_piece_type + (is_white ? 6 : 0);
    
    // up right
    for (int new_rank = 1; new_rank <= 7; new_rank++)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file + new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file + new_rank];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    // up left
    for (int new_rank = 1; new_rank <= 7; new_rank++)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file - new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file - new_rank];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    // down left
    for (int new_rank = -1; new_rank >= -7; new_rank--)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file + new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file + new_rank];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    // down right
    for (int new_rank = -1; new_rank >= -7; new_rank--)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file - new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file - new_rank];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    return false;
}
bool ChessGameState::attacking_rook(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece_type) {
    uint8_t current_piece;
    uint8_t attacked_piece = (uint8_t)attacked_piece_type + (is_white ? 6 : 0);
    
    // up
    for (int new_rank = 1; new_rank <= 7; new_rank++)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    // down
    for (int new_rank = -1; new_rank >= -7; new_rank--)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    // right
    for (int new_file = 1; new_file <= 7; new_file++)
    {
        if (
            is_square({piece_square.rank, piece_square.file + new_file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank][piece_square.file + new_file];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    // left
    for (int new_file = -1; new_file >= -7; new_file--)
    {
        if (
            is_square({piece_square.rank, piece_square.file + new_file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank][piece_square.file + new_file];
            if (current_piece == attacked_piece) return true;
            else if (current_piece == 12) continue;
            else break;
        }
    }

    return false;
}
bool ChessGameState::attacking_queen(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece_type) {
    if (ChessGameState::attacking_bishop(piece_square, is_white, attacked_piece_type)) return true;
    if (ChessGameState::attacking_rook(piece_square, is_white, attacked_piece_type)) return true;
    return false;
}

bool ChessGameState::attacking_king(ChessSquare piece_square, bool is_white, PieceTypes attacked_piece_type) {
    uint8_t attacked_piece = (uint8_t)attacked_piece_type + (is_white ? 6 : 0);

    for (int new_rank = -1; new_rank <= 1; new_rank++)
    {
        for (int new_file = -1; new_file <= 1; new_file++)
        {
            if (
                is_square({piece_square.rank+new_rank, piece_square.file+new_file}) &&
                ChessGameState::current_position[piece_square.rank+new_rank][piece_square.file+new_file] == attacked_piece
            ) return true;
        }
    }

    return false;
}

bool ChessGameState::is_attacked(ChessSquare piece_square, bool is_white) {
    // check pawns
    if (ChessGameState::attacking_pawn(piece_square, is_white, PieceTypes::PAWN)) return true;
    
    // check knights
    if (ChessGameState::attacking_knight(piece_square, is_white, PieceTypes::KNIGHT)) return true;
    
    // check bishops
    if (ChessGameState::attacking_bishop(piece_square, is_white, PieceTypes::BISHOP)) return true;
    
    // check rooks
    if (ChessGameState::attacking_rook(piece_square, is_white, PieceTypes::ROOK)) return true;
    
    // check queens
    if (ChessGameState::attacking_queen(piece_square, is_white, PieceTypes::QUEEN)) return true;

    return false;
}

bool ChessGameState::legal_to_go(ChessSquare to_sq, bool is_white) {
    return (is_white && ChessGameState::current_position[to_sq.rank][to_sq.file] >= 6) ||
        (!is_white && ChessGameState::current_position[to_sq.rank][to_sq.file] < 6) ||
        (ChessGameState::current_position[to_sq.rank][to_sq.file] == 12);
}

void ChessGameState::legal_moves_pawn(ChessSquare piece_square, bool is_white) {
    int rank_diff = is_white ? 1 : -1;
    ChessGameState* new_pos;
    if (
        is_square({piece_square.rank+rank_diff, piece_square.file+1}) &&
        (
            (
                ChessGameState::current_position[piece_square.rank+rank_diff][piece_square.file+1] != 12 &&
                legal_to_go({piece_square.rank+rank_diff, piece_square.file+1}, is_white)
            ) ||
            (
                ChessGameState::enpassant_sq.rank == piece_square.rank+rank_diff &&
                ChessGameState::enpassant_sq.file == piece_square.file+1 &&
                (ChessGameState::flags & GameFlags::EN_PASSANT)
            )
        )
    ) {
        if (!(piece_square.rank+rank_diff == 0 || piece_square.rank+rank_diff == 7)) {
            try {
                ChessMove move(piece_square, {piece_square.rank+rank_diff, piece_square.file+1});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        } else {
            vector<PromotionPieceType> piece_types = {PromotionPieceType::PROMOTED_KNIGHT, PromotionPieceType::PROMOTED_BISHOP, PromotionPieceType::PROMOTED_ROOK, PromotionPieceType::PROMOTED_QUEEN};

            for (PromotionPieceType promotion_type : piece_types) {
                try {
                    ChessMove move(piece_square, {piece_square.rank+rank_diff, piece_square.file+1}, promotion_type);
                    new_pos = GameStateFactory::create_state(this, move);
                    ChessGameState::add_child(new_pos, move);
                } catch (...) {}
            }
        }
    }

    if (
        is_square({piece_square.rank+rank_diff, piece_square.file-1}) &&
        (
            (
                ChessGameState::current_position[piece_square.rank+rank_diff][piece_square.file-1] != 12 &&
                legal_to_go({piece_square.rank+rank_diff, piece_square.file-1}, is_white)
            ) ||
            (
                ChessGameState::enpassant_sq.rank == piece_square.rank+rank_diff &&
                ChessGameState::enpassant_sq.file == piece_square.file-1 &&
                (ChessGameState::flags & GameFlags::EN_PASSANT)
            )
        )
    ) {
        if (!(piece_square.rank+rank_diff == 0 || piece_square.rank+rank_diff == 7)) {
            try {
                ChessMove move(piece_square, {piece_square.rank+rank_diff, piece_square.file-1});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        } else {
            vector<PromotionPieceType> piece_types = {PromotionPieceType::PROMOTED_KNIGHT, PromotionPieceType::PROMOTED_BISHOP, PromotionPieceType::PROMOTED_ROOK, PromotionPieceType::PROMOTED_QUEEN};

            for (PromotionPieceType promotion_type : piece_types) {
                try {
                    ChessMove move(piece_square, {piece_square.rank+rank_diff, piece_square.file-1}, promotion_type);
                    new_pos = GameStateFactory::create_state(this, move);
                    ChessGameState::add_child(new_pos, move);
                } catch (...) {}
            }
        }
    }

    if (
        is_square({piece_square.rank+rank_diff, piece_square.file}) &&
        ChessGameState::current_position[piece_square.rank+rank_diff][piece_square.file] == 12
    ) {
        if (!(piece_square.rank+rank_diff == 0 || piece_square.rank+rank_diff == 7)) {
            try {
                ChessMove move(piece_square, {piece_square.rank+rank_diff, piece_square.file});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        } else {
            vector<PromotionPieceType> piece_types = {PromotionPieceType::PROMOTED_KNIGHT, PromotionPieceType::PROMOTED_BISHOP, PromotionPieceType::PROMOTED_ROOK, PromotionPieceType::PROMOTED_QUEEN};

            for (PromotionPieceType promotion_type : piece_types) {
                try {
                    ChessMove move(piece_square, {piece_square.rank+rank_diff, piece_square.file}, promotion_type);
                    new_pos = GameStateFactory::create_state(this, move);
                    ChessGameState::add_child(new_pos, move);
                } catch (...) {}
            }
        }


        if (
            is_square({piece_square.rank+(2*rank_diff), piece_square.file}) &&
            ChessGameState::current_position[piece_square.rank+(2*rank_diff)][piece_square.file] == 12 &&
            (piece_square.rank == 1 || piece_square.rank == 6)
        ) {
            try {
                ChessMove move(piece_square, {piece_square.rank+(2*rank_diff), piece_square.file});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        }
    }
}
void ChessGameState::legal_moves_knight(ChessSquare piece_square, bool is_white) {
    ChessGameState* new_pos;
    for (int new_rank = piece_square.rank-1; new_rank <= piece_square.rank+1; new_rank+=2)
    {
        for (int new_file = piece_square.file-2; new_file <= piece_square.file+2; new_file+=4)
        {
            if (is_square({new_rank, new_file}) && ChessGameState::legal_to_go({new_rank, new_file}, is_white)) {
                try {
                    ChessMove move(piece_square, {new_rank, new_file});
                    new_pos = GameStateFactory::create_state(this, move);
                    ChessGameState::add_child(new_pos, move);
                } catch (...) {}
            }
        }
    }
    
    for (int new_rank = piece_square.rank-2; new_rank <= piece_square.rank+2; new_rank+=4)
    {
        for (int new_file = piece_square.file-1; new_file <= piece_square.file+1; new_file+=2)
        {
            if (is_square({new_rank, new_file}) && ChessGameState::legal_to_go({new_rank, new_file}, is_white)) {
                try {
                    ChessMove move(piece_square, {new_rank, new_file});
                    new_pos = GameStateFactory::create_state(this, move);
                    ChessGameState::add_child(new_pos, move);
                } catch (...) {}
            }
        }
    }
}
void ChessGameState::legal_moves_bishop(ChessSquare piece_square, bool is_white) {
    uint8_t current_piece;
    ChessGameState* new_pos;
    
    // up right
    for (int new_rank = 1; new_rank <= 7; new_rank++)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file + new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file + new_rank];
            if (!ChessGameState::legal_to_go({piece_square.rank + new_rank, piece_square.file + new_rank}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank+new_rank, piece_square.file+new_rank});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}

            if (current_piece != 12) break;
        }
    }

    // up left
    for (int new_rank = 1; new_rank <= 7; new_rank++)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file - new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file - new_rank];
            if (!ChessGameState::legal_to_go({piece_square.rank + new_rank, piece_square.file - new_rank}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank+new_rank, piece_square.file-new_rank});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}

            if (current_piece != 12) break;
        }
    }

    // down left
    for (int new_rank = -1; new_rank >= -7; new_rank--)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file + new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file + new_rank];
            if (!ChessGameState::legal_to_go({piece_square.rank + new_rank, piece_square.file + new_rank}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank+new_rank, piece_square.file+new_rank});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
            
            if (current_piece != 12) break;
        }
    }

    // down right
    for (int new_rank = -1; new_rank >= -7; new_rank--)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file - new_rank})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file - new_rank];
            if (!ChessGameState::legal_to_go({piece_square.rank + new_rank, piece_square.file - new_rank}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank+new_rank, piece_square.file-new_rank});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}

            if (current_piece != 12) break;
        }
    }
}
void ChessGameState::legal_moves_rook(ChessSquare piece_square, bool is_white) {
    uint8_t current_piece;
    ChessGameState* new_pos;
    
    // up
    for (int new_rank = 1; new_rank <= 7; new_rank++)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file];
            if (!ChessGameState::legal_to_go({piece_square.rank + new_rank, piece_square.file}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank+new_rank, piece_square.file});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}

            if (current_piece != 12) break;
        }
    }

    // down
    for (int new_rank = -1; new_rank >= -7; new_rank--)
    {
        if (
            is_square({piece_square.rank + new_rank, piece_square.file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank + new_rank][piece_square.file];
            if (!ChessGameState::legal_to_go({piece_square.rank + new_rank, piece_square.file}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank+new_rank, piece_square.file});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}

            if (current_piece != 12) break;
        }
    }

    // right
    for (int new_file = 1; new_file <= 7; new_file++)
    {
        if (
            is_square({piece_square.rank, piece_square.file + new_file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank][piece_square.file + new_file];
            if (!ChessGameState::legal_to_go({piece_square.rank, piece_square.file + new_file}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank, piece_square.file+new_file});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}

            if (current_piece != 12) break;
        }
    }

    // left
    for (int new_file = -1; new_file >= -7; new_file--)
    {
        if (
            is_square({piece_square.rank, piece_square.file + new_file})
        ) {
            current_piece = ChessGameState::current_position[piece_square.rank][piece_square.file + new_file];
            if (!ChessGameState::legal_to_go({piece_square.rank, piece_square.file + new_file}, is_white)) break;
            
            try {
                ChessMove move(piece_square, {piece_square.rank, piece_square.file+new_file});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}

            if (current_piece != 12) break;
        }
    }
}
void ChessGameState::legal_moves_queen(ChessSquare piece_square, bool is_white) {
    ChessGameState::legal_moves_bishop(piece_square, is_white);
    ChessGameState::legal_moves_rook(piece_square, is_white);
}
void ChessGameState::legal_moves_king(ChessSquare piece_square, bool is_white) {
    ChessGameState* new_pos;
    for (int new_rank = -1; new_rank <= 1; new_rank++)
    {
        for (int new_file = -1; new_file <= 1; new_file++)
        {
            if (
                is_square({piece_square.rank+new_rank, piece_square.file+new_file}) &&
                ChessGameState::legal_to_go({piece_square.rank+new_rank, piece_square.file+new_file}, is_white) &&
                !(new_rank == 0 && new_file == 0)
            ) {
                try {
                    ChessMove move(piece_square, {piece_square.rank+new_rank, piece_square.file+new_file});
                    new_pos = GameStateFactory::create_state(this, move);
                    ChessGameState::add_child(new_pos, move);
                } catch (...) {}
            }
        }
    }

    if (is_white && (ChessGameState::flags & GameFlags::WHITE_KING_CASTLE)) {
        if (ChessGameState::current_position[0][5] == 12 && ChessGameState::current_position[0][6] == 12) {
            try {
                ChessMove move(piece_square, {0,6});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        }
    }

    if (is_white && (ChessGameState::flags & GameFlags::WHITE_QUEEN_CASTLE)) {
        if (ChessGameState::current_position[0][3] == 12 && ChessGameState::current_position[0][2] == 12 && ChessGameState::current_position[0][1] == 12) {
            try {
                ChessMove move(piece_square, {0,2});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        }
    }

    if (!is_white && (ChessGameState::flags & GameFlags::BLACK_KING_CASTLE)) {
        if (ChessGameState::current_position[7][5] == 12 && ChessGameState::current_position[7][6] == 12) {
            try {
                ChessMove move(piece_square, {7,6});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        }
    }

    if (!is_white && (ChessGameState::flags & GameFlags::BLACK_QUEEN_CASTLE)) {
        if (ChessGameState::current_position[7][3] == 12 && ChessGameState::current_position[7][2] == 12 && ChessGameState::current_position[7][1] == 12) {
            try {
                ChessMove move(piece_square, {7,2});
                new_pos = GameStateFactory::create_state(this, move);
                ChessGameState::add_child(new_pos, move);
            } catch (...) {}
        }
    }
}

void ChessGameState::compute_children() {
    bool is_white_turn = ChessGameState::flags & GameFlags::WHITE_TURN;

    ChessGameState::children.clear();

    uint8_t cell;
    PieceTypes piece_type;
    for (int rank = 0; rank < 8; rank++)
    {
        for (int file = 0; file < 8; file++)
        {
            cell = ChessGameState::current_position[rank][file];
            if (cell == 12) continue;

            if ((is_white_turn && cell < 6) || (!is_white_turn && cell >= 6)) {
                piece_type = (PieceTypes)(cell%6);
                switch (piece_type)
                {
                case PieceTypes::PAWN:
                    ChessGameState::legal_moves_pawn({rank, file}, is_white_turn);
                    break;
                
                case PieceTypes::KNIGHT:
                    ChessGameState::legal_moves_knight({rank, file}, is_white_turn);
                    break;
                
                case PieceTypes::BISHOP:
                    ChessGameState::legal_moves_bishop({rank, file}, is_white_turn);
                    break;

                case PieceTypes::ROOK:
                    ChessGameState::legal_moves_rook({rank, file}, is_white_turn);
                    break;

                case PieceTypes::QUEEN:
                    ChessGameState::legal_moves_queen({rank, file}, is_white_turn);
                    break;
                
                case PieceTypes::KING:
                    ChessGameState::legal_moves_king({rank, file}, is_white_turn);
                    break;
                
                default:
                    break;
                }
            }
        }
    }

    ChessGameState::flags |= GameFlags::CHILDREN_COMPUTED;
}

// --------------------------------------------------

GameStateFactory::Registry GameStateFactory::registry = {
    vector<ChessGameState*>(0)
};

GameStateFactory::Registry::~Registry() {
    for (auto ptr : GameStateFactory::registry.states) {
        try {
            delete ptr;
        } catch (...) {}
    }
}

ChessGameState* GameStateFactory::get_state() {
    return NULL;
}

ChessGameState* GameStateFactory::create_state(ChessGameState* position, ChessMove move) {
    ChessGameState* new_position = new ChessGameState(*position, move);

    GameStateFactory::registry.states.push_back(new_position);

    return new_position;
}

ChessGameState* GameStateFactory::create_state(string fen) {
    ChessGameState* new_position = new ChessGameState(fen);
    GameStateFactory::registry.states.push_back(new_position);
    return new_position;
}