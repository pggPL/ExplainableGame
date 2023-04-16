let global_predictions = null;

async function makeMove(row, col) {
    const response = await fetch('/get_move', {
        method: 'POST',
        body: JSON.stringify({ board: getBoard(), player: 0 }),
        headers: { 'Content-Type': 'application/json' },
    });
    const data = await response.json();
    return [data.row, data.col, data.result, data.value];
}

async function getPredictions() {
    const response = await fetch('/get_predictions', {
        method: 'POST',
        body: JSON.stringify({ board: getBoard(), player: 0 }),
        headers: { 'Content-Type': 'application/json' },
    });
    const data = await response.json();
    global_predictions = data.q_values;
}


function getBoard() {
    const rows = document.querySelectorAll('tr');
    return Array.from(rows).map((row) => {
        const cells = row.querySelectorAll('td');
        return Array.from(cells).map((cell) => cell.textContent || '');
    });
}

function resetBoard() {
    const rows = document.querySelectorAll('tr');

    document.getElementById("value").innerHTML = "0";
    Array.from(rows).map((row) => {
        const cells = row.querySelectorAll('td');
        Array.from(cells).map((cell) => {
            cell.textContent = '';
            cell.classList.remove('x');
            cell.classList.remove('o');
        });
    });
}

function handleClick(row, col) {
    return async function () {
        const cell = document.querySelector(`tr:nth-child(${row + 1}) td:nth-child(${col + 1})`);
        if (cell.textContent) return;
        cell.textContent = 'X';
        cell.classList.add('x');

        const [modelRow, modelCol, result, value, predictions] = await makeMove(row, col);
        const modelCell = document.querySelector(`tr:nth-child(${modelRow + 1}) td:nth-child(${modelCol + 1})`);

        document.getElementById("value").innerHTML = value;
        if (result === "win") {
            show_modal('You win!')
            resetBoard();
        } else if (result === "lose") {
            show_modal('You lose!')
            resetBoard();
        } else if (result === "draw"){
            show_modal('Draw!')
            resetBoard();
        }
        else {
            modelCell.textContent = 'O';
            modelCell.classList.add('o');
        }

    };
}

function createTable() {
    const table = document.createElement('table');
    for (let row = 0; row < 10; row++) {
        const tr = document.createElement('tr');
        for (
        let col = 0; col < 10; col++) {
            const td = document.createElement('td');
            td.addEventListener('click', handleClick(row, col));
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
    return table;
}

function show_modal(result) {
    let modal = document.getElementById("modal");
    let span = document.getElementsByClassName("close")[0];
    let modal_text = document.getElementById("modal_text");
    modal.style.display = "block";
    modal_text.innerHTML = result;

    span.onclick = function() {
        modal.style.display = "none";
    }
    window.onclick = function(event) {
        if (event.target != modal) {
            modal.style.display = "none";
        }
    }
}

async function yourMovePred() {
    let cells = document.getElementsByTagName("td");

    await getPredictions();
    let used_cells_num = 0;
    for (let i = 0; i < cells.length; i++) {
        if (cells[i].textContent !== "") {
            continue;
        }
        // Background color depends on prediction, which is float from -1 to 1
        let pred = global_predictions[used_cells_num];
        let color = Math.round((pred + 1) * 255 / 2);
        cells[i].style.backgroundColor = "rgb(" + color + ", " + color + ", " + color + ")";
        used_cells_num++
    }
}

function botMovePred() {
    let cells = document.getElementsByTagName("td");
    for (let i = 0; i < cells.length; i++) {
        cells[i].style.backgroundColor = "blue";
    }
}

function endingPred() {
    let cells = document.getElementsByTagName("td");
    for (let i = 0; i < cells.length; i++) {
        cells[i].style.backgroundColor = "green";
    }
}

function resetPred() {
    // remove back colors of all cells
    let cells = document.getElementsByTagName("td");
    for (let i = 0; i < cells.length; i++) {
        cells[i].style.backgroundColor = "white";
    }
}

// get class col5
let col5 = document.getElementsByClassName("item")[4];
col5.appendChild(createTable());

// add event listerns to buttons
document.getElementById("your_move_pred").addEventListener("mouseenter", yourMovePred);
document.getElementById("bot_move_pred").addEventListener("mouseenter", botMovePred);
document.getElementById("ending_pred").addEventListener("mouseenter", endingPred);
document.getElementById("your_move_pred").addEventListener("mouseout", resetPred);
document.getElementById("bot_move_pred").addEventListener("mouseout", resetPred);
document.getElementById("ending_pred").addEventListener("mouseout", resetPred);

