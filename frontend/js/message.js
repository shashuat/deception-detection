const ctx = {

}

function read_message(event) {
    content = JSON.parse(event.data);
    switch (content.type) {
        case "popup":
            read_popup_message(content.data);
            break;
        case "toast":
            read_toast_message(content.data);
            break;
        case "game-started":
            read_game_started(content.data.message);
            break;
        case "confirm-move":
            game_state.turn = game_state.turn === "w" ? "b" : "w";
            read_confirm_move(content.data.message);
            break;
        case "ai-move":
            game_state.turn = game_state.turn === "w" ? "b" : "w";
            read_ai_move(content.data.message);
            break;
        default:
            return content
    }

    return content
}

function read_popup_message(data) {
    // data
}

function read_toast_message(data) {
    // data
}

function read_game_started(data) {
    // data
    draw_game(data.FEN);
    update_game_state(data.FEN);
    new Audio('../media/game-start.mp3').play();
}

function read_confirm_move(data) {
    // data
    game_state.king_in_check = data.king_in_check;
    game_state.checkmate = data.checkmate;
    game_state.draw = data.draw;
    if (game_state.draw) stalemate();
    update_game_state(data.FEN);
    update_right_panel()
}

function read_ai_move(data) {
    // data

    game_state.king_in_check = data.king_in_check;
    game_state.checkmate = data.checkmate;
    game_state.draw = data.draw;
    if (game_state.draw !== null) stalemate();

    update_game_state(data.FEN);
    piece = d3.select(`svg#board #board-pieces [pos="${data.from}"]`);
    promote = data.promote === undefined || data.promote === null ? undefined : (game_state.turn === "w" ? data.promote.toLowerCase() : data.promote.toUpperCase());
    let rect = d3.select(`svg#board #board-boxes #${data.to}`);
    move_piece(null, rect, piece, true, promote);

    update_right_panel()
}

function update_right_panel() {
    right_panel = d3.select(".right-panel")
    white_castling = right_panel.select(".possible-castling .castling-block .castling-white > div")
    white_castling.text((game_state.castling["K"] ? "K " : "") + (game_state.castling["Q"] ? "Q" : ""))
    black_castling = right_panel.select(".possible-castling .castling-block .castling-black > div")
    black_castling.text((game_state.castling["k"] ? "k " : "") + (game_state.castling["q"] ? "q" : ""))

    right_panel.select(".possible-en-passant .en-passant").text(game_state.en_passant === null ? "-" : game_state.en_passant)
}

